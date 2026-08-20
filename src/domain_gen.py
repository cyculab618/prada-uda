"""
Domain-Level Batch Generation Script
=====================================
自動遍歷 source domain 下所有 class 子資料夾，
對每張圖片 × 每個 target domain 生成指定數量的圖片。

--input 只需給到 domain 層，class name 自動從子資料夾名稱取得作為 text prompt。

用法：
  # real domain 下所有 class，轉成 domain 1,2,3,4,5，每張圖生成 1 張
  python domain_gen.py \
    --checkpoint ./checkpoints/clip_two_stage_stage2_best.pt \
    --input ./data/DomainNet/real \
    --domain_id 1 2 3 4 5 \
    --output ./generated/real_to_others

  # 每張圖生成 3 張，啟用 ControlNet
  python domain_gen.py \
    --checkpoint ./checkpoints/clip_two_stage_stage2_best.pt \
    --input ./data/DomainNet/real \
    --domain_id 1 2 3 4 5 \
    --num_per_image 3 \
    --use_controlnet \
    --output ./generated/real_to_others

  # 限制每個 class 最多處理 50 張，batch size 8
  python domain_gen.py \
    --checkpoint ./checkpoints/clip_two_stage_stage2_best.pt \
    --input ./data/DomainNet/real \
    --domain_id 1 2 3 4 5 \
    --max_images_per_class 50 \
    --batch_size 8 \
    --output ./generated/real_to_others

  # 只處理特定 class
  python domain_gen.py \
    --checkpoint ./checkpoints/clip_two_stage_stage2_best.pt \
    --input ./data/DomainNet/real \
    --domain_id 1 2 3 4 5 \
    --classes airplane bird cat \
    --output ./generated/real_to_others

  # 使用 style prompts 增加多樣性
  python domain_gen.py \
    --checkpoint ./checkpoints/clip_two_stage_stage2_best.pt \
    --input ./data/DomainNet/real \
    --domain_id 4 --domain_names sketch \
    --classes airplane \
    --use_controlnet --max_images_per_class 100 \
    --style_prompts \
        "pencil sketch, white background" \
        "charcoal drawing, minimal strokes" \
        "ink sketch, clean lines" \
        "rough sketch on white paper" \
        "detailed pencil drawing" \
    --output ./test_diversity/with_style

輸出結構：
  output/
    real-to-painting/
      airplane/
        img_000_gen_s0.png
        img_001_gen_s0.png
        ...
      bird/
        ...
    real-to-sketch/
      airplane/
        ...
"""

import os
import sys
import glob
import argparse
import time
import json
import random as _random
from PIL import Image

import torch

# 從 controlnet_gen.py 載入核心 class
from controlnet_gen import LearnablePromptGenerator, preprocess_canny

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff',
                    '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP', '.TIFF'}


def resolve_domain_names(checkpoint_path, domain_ids, domain_names_arg=None):
    """
    取得 domain_id -> domain_name 的對應
    優先用 --domain_names 參數，其次從 checkpoint 讀取
    """
    if domain_names_arg and len(domain_names_arg) == len(domain_ids):
        return {did: name for did, name in zip(domain_ids, domain_names_arg)}

    # 從 checkpoint 讀
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    all_names = ckpt.get("domain_names", None)

    if all_names is not None:
        mapping = {}
        for did in domain_ids:
            if did < len(all_names):
                mapping[did] = all_names[did]
            else:
                mapping[did] = f"domain_{did}"
        return mapping

    # fallback
    return {did: f"domain_{did}" for did in domain_ids}


def get_source_name(input_path):
    """
    從 input 路徑推斷 source domain 名稱
    eg. './data/DomainNet/real' -> 'real'
    eg. './data/DomainNet/real/' -> 'real'
    """
    path = input_path.rstrip("/").rstrip("\\")
    return os.path.basename(path)


def discover_classes(domain_dir, filter_classes=None):
    """
    掃描 domain 目錄下的所有 class 子資料夾
    回傳 [(class_name, class_dir, image_paths), ...]
    """
    if not os.path.isdir(domain_dir):
        raise RuntimeError(f"Domain directory not found: {domain_dir}")

    class_dirs = sorted([
        d for d in os.listdir(domain_dir)
        if os.path.isdir(os.path.join(domain_dir, d)) and not d.startswith('.')
    ])

    if filter_classes:
        filter_set = set(c.lower() for c in filter_classes)
        class_dirs = [d for d in class_dirs if d.lower() in filter_set]

    results = []
    for cls_name in class_dirs:
        cls_dir = os.path.join(domain_dir, cls_name)
        images = []
        for f in sorted(os.listdir(cls_dir)):
            if os.path.splitext(f)[1] in IMAGE_EXTENSIONS:
                images.append(os.path.join(cls_dir, f))
        # 也搜尋子資料夾（有些 DomainNet 版本有更深的結構）
        for root, _, files in os.walk(cls_dir):
            if root == cls_dir:
                continue  # 已經處理過
            for f in sorted(files):
                if os.path.splitext(f)[1] in IMAGE_EXTENSIONS:
                    images.append(os.path.join(root, f))
        # 去重
        images = sorted(set(images))
        if len(images) > 0:
            results.append((cls_name, cls_dir, images))

    return results


def pad_images(image_paths, target_count, seed=42):
    """
    將 image_paths 調整為剛好 target_count 張：
      - 如果 >= target_count：取前 target_count 張（截斷）
      - 如果 < target_count：先保留全部原圖，再從中隨機重複抽樣補滿

    回傳的 list 長度一定 == target_count。
    """
    n = len(image_paths)
    if n >= target_count:
        return image_paths[:target_count]

    # 不足 → 保留全部原圖 + 重複抽樣補滿
    rng = _random.Random(seed)
    extra_needed = target_count - n
    extra = rng.choices(image_paths, k=extra_needed)
    padded = list(image_paths) + extra
    return padded


def make_text_prompt(class_name, style_suffix=None):
    """
    將 class 資料夾名稱轉成自然語言 text prompt
    eg. 'The_Eiffel_Tower' -> 'The Eiffel Tower'
    eg. 'alarm_clock' -> 'alarm clock'

    如果有 style_suffix，則接在後面：
    eg. 'airplane' + 'pencil sketch, white background'
        -> 'airplane, pencil sketch, white background'
    """
    base = class_name.replace('_', ' ').replace('-', ' ')
    if style_suffix:
        return f"{base}, {style_suffix}"
    return base


def generate_for_class(
    generator,
    class_name,
    image_paths,
    domain_ids,
    domain_dir_names,
    num_per_image,
    batch_size,
    save_dir,
    use_controlnet,
    steps,
    cfg,
    cn_scale,
    base_seed,
    style_prompts=None,
):
    """
    對一個 class 的所有圖片做生成。

    style_prompts: None 或 list of str
      - None: 所有圖共用同一個 prompt（原始行為）
      - list: 每張圖隨機分配一個 style suffix，增加多樣性
    """
    from controlnet_gen import preprocess_canny as _preprocess_canny

    num_images = len(image_paths)
    has_styles = style_prompts is not None and len(style_prompts) > 0

    # ── 為每張圖分配 style ──────────────────────────────────────────
    if has_styles:
        rng = _random.Random(base_seed)
        # 每張圖隨機分配一個 style index
        style_assignments = [rng.randint(0, len(style_prompts) - 1)
                             for _ in range(num_images)]
        # 所有用到的 style indices（用於 cache）
        used_style_indices = sorted(set(style_assignments))
    else:
        style_assignments = [None] * num_images
        used_style_indices = [None]
    # ──────────────────────────────────────────────────────────────

    # 顯示資訊
    text_prompt_display = make_text_prompt(class_name)
    print(f"\n{'─'*60}")
    print(f"  Class: {class_name}")
    if has_styles:
        print(f"  Text prompt: \"{text_prompt_display}\" + random style suffix")
        print(f"  Style pool ({len(style_prompts)}):")
        for si, sp in enumerate(style_prompts):
            count = style_assignments.count(si)
            print(f"    [{si}] \"{sp}\" → {count} images")
    else:
        print(f"  Text prompt: \"{text_prompt_display}\"")
    print(f"  Images: {num_images}")
    print(f"  Targets: {[domain_dir_names[d] for d in domain_ids]}")
    print(f"  Num per image: {num_per_image}  |  Batch size: {batch_size}")
    print(f"{'─'*60}")

    # 建立輸出資料夾
    for did in domain_ids:
        os.makedirs(os.path.join(save_dir, domain_dir_names[did], class_name), exist_ok=True)

    # ── Prompt embedding cache ──────────────────────────────────────
    # key: (domain_id, style_index)
    # style_index = None 代表無 style suffix
    print(f"  Caching prompt embeddings...", end=" ", flush=True)
    prompt_embeds_cache = {}  # {(did, style_idx): Tensor [1, 77, 768] float16}
    with torch.no_grad():
        for did in domain_ids:
            for si in used_style_indices:
                style_suffix = style_prompts[si] if si is not None else None
                text_prompt = make_text_prompt(class_name, style_suffix)
                emb = generator.get_prompt_embedding(did, text_prompt, verbose=True)
                prompt_embeds_cache[(did, si)] = emb.half()
    num_cached = len(prompt_embeds_cache)
    print(f"done ({num_cached} embeddings)")

    # negative embedding 也只需要算一次（空字串，所有圖共用）
    neg_tokens = generator.tokenizer(
        [""],
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
    ).to(generator.device)
    with torch.no_grad():
        negative_embeds_cached = generator.pipe.text_encoder(neg_tokens.input_ids)[0]
    # ───────────────────────────────────────────────────────────────

    generated_count = 0

    for batch_start in range(0, num_images, batch_size):
        batch_paths = image_paths[batch_start: batch_start + batch_size]
        actual_bs = len(batch_paths)
        batch_style_indices = style_assignments[batch_start: batch_start + actual_bs]

        for did in domain_ids:
            domain_out_dir = os.path.join(save_dir, domain_dir_names[did], class_name)

            # 組合 batch 的 prompt embeddings（每張圖可能不同 style）
            prompt_embeds_list = []
            for i in range(actual_bs):
                si = batch_style_indices[i]
                prompt_embeds_list.append(prompt_embeds_cache[(did, si)])
            prompt_embeds_batch = torch.cat(prompt_embeds_list, dim=0)  # [bs, 77, 768]
            negative_embeds_batch = negative_embeds_cached.repeat(actual_bs, 1, 1)

            for gen_idx in range(num_per_image):
                # 每張圖各自的 seed -> 各自的 Generator
                generators = [
                    torch.Generator(device=generator.device).manual_seed(
                        base_seed + (batch_start + i) * 1000 + did * 100 + gen_idx
                    )
                    for i in range(actual_bs)
                ]

                # 直接呼叫 pipeline，一次送入整個 batch
                if use_controlnet:
                    control_images = [_preprocess_canny(p) for p in batch_paths]
                    result = generator.pipe(
                        prompt_embeds=prompt_embeds_batch,
                        negative_prompt_embeds=negative_embeds_batch,
                        image=control_images,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        controlnet_conditioning_scale=cn_scale,
                        generator=generators,
                        height=512,
                        width=512,
                    )
                else:
                    result = generator.pipe(
                        prompt_embeds=prompt_embeds_batch,
                        negative_prompt_embeds=negative_embeds_batch,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generators,
                        height=512,
                        width=512,
                    )

                # 一對一存檔
                for i, gen_img in enumerate(result.images):
                    global_img_idx = batch_start + i
                    if num_per_image == 1:
                        gen_filename = f"img_{global_img_idx:04d}_gen.png"
                    else:
                        gen_filename = f"img_{global_img_idx:04d}_gen_s{gen_idx}.png"
                    gen_img.save(os.path.join(domain_out_dir, gen_filename))
                    generated_count += 1

                del result
                torch.cuda.empty_cache()

        # 進度
        batch_end = min(batch_start + batch_size, num_images)
        if batch_end % 10 == 0 or batch_end == num_images:
            print(f"    [{batch_end}/{num_images}] {generated_count} images generated")

    # 釋放 cache
    del prompt_embeds_cache
    del negative_embeds_cached
    torch.cuda.empty_cache()

    return generated_count


def main():
    parser = argparse.ArgumentParser(description="Domain-Level Batch Generation")

    # 核心參數
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Learnable prompt checkpoint path")
    parser.add_argument("--input", type=str, required=True,
                        help="Source domain 資料夾（如 ./data/DomainNet/real）")
    parser.add_argument("--domain_id", type=int, nargs='+', required=True,
                        help="Target domain IDs（如 1 2 3 4 5）")
    parser.add_argument("--domain_names", type=str, nargs='+', default=None,
                        help="Target domain 名稱（與 --domain_id 一一對應，不指定則從 checkpoint 讀取）")
    parser.add_argument("--output", type=str, default="./generated",
                        help="輸出資料夾")

    # 生成控制
    parser.add_argument("--num_per_image", type=int, default=1,
                        help="每張 input image 生成幾張（每個 domain 各生成這麼多張）")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="一次讀取並送進 generator 的 input 圖片數量（一對一輸出）")
    parser.add_argument("--max_images_per_class", type=int, default=None,
                        help="每個 class 最多處理幾張 input image（None = 全部）。"
                             "不足時會重複抽樣補滿。")
    parser.add_argument("--classes", type=str, nargs='+', default=None,
                        help="只處理指定的 class（不指定 = 全部）")

    # Style prompts（增加多樣性）
    parser.add_argument("--style_prompts", type=str, nargs='+', default=None,
                        help="Style suffix pool，每張圖隨機分配一個接在 class name 後面。"
                             "例如: --style_prompts "
                             "'pencil sketch, white background' "
                             "'charcoal drawing, minimal strokes' "
                             "'ink sketch, clean lines'")

    # ControlNet
    parser.add_argument("--use_controlnet", action="store_true",
                        help="啟用 ControlNet")
    parser.add_argument("--controlnet_type", type=str, default="canny",
                        choices=["canny", "depth", "openpose"])

    # SD 參數
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--cn_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # 多 GPU 平行
    parser.add_argument("--reverse", action="store_true",
                        help="反轉 class 順序，從最後一個 class 開始生成（搭配第二張 GPU 使用）")
    parser.add_argument("--stop_if_exists", action="store_true",
                        help="當偵測到某個 class 的輸出目錄已有生成檔案時，停止程式"
                             "（用於兩張 GPU 對向生成，相遇時自動停止）")

    # 其他
    parser.add_argument("--resume_from_class", type=str, default=None,
                        help="從指定 class 繼續（跳過之前的 class）")

    args = parser.parse_args()

    # ================================================================
    # 掃描 classes
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  Domain-Level Batch Generation")
    print(f"{'='*70}")
    print(f"  Source domain dir: {args.input}")
    print(f"  Target domain IDs: {args.domain_id}")
    print(f"  Num per image: {args.num_per_image}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  ControlNet: {'ON' if args.use_controlnet else 'OFF'}")
    print(f"  Reverse: {'ON' if args.reverse else 'OFF'}")
    print(f"  Stop if exists: {'ON' if args.stop_if_exists else 'OFF'}")
    if args.style_prompts:
        print(f"  Style prompts: {len(args.style_prompts)} variants")
        for i, sp in enumerate(args.style_prompts):
            print(f"    [{i}] \"{sp}\"")
    else:
        print(f"  Style prompts: OFF")
    print(f"  Output: {args.output}")

    classes = discover_classes(args.input, filter_classes=args.classes)

    if len(classes) == 0:
        print(f"\n❌ No classes found in {args.input}")
        return

    # 解析 source domain name 和 target domain names
    source_name = get_source_name(args.input)
    target_name_map = resolve_domain_names(args.checkpoint, args.domain_id, args.domain_names)
    domain_dir_names = {
        did: f"{source_name}-to-{target_name_map[did]}" for did in args.domain_id
    }

    print(f"  Source domain: {source_name}")
    print(f"  Target mapping:")
    for did in args.domain_id:
        print(f"    domain_id {did} -> {target_name_map[did]} -> dir: {domain_dir_names[did]}/")

    # ================================================================
    # 套用 max_images_per_class（含重複抽樣補滿）
    # ================================================================
    if args.max_images_per_class:
        padded_classes = []
        oversampled_count = 0
        for name, d, imgs in classes:
            original_count = len(imgs)
            imgs_padded = pad_images(imgs, args.max_images_per_class, seed=args.seed)
            if original_count < args.max_images_per_class:
                oversampled_count += 1
                print(f"  ⚠ {name}: {original_count} images → "
                      f"padded to {args.max_images_per_class} "
                      f"(+{args.max_images_per_class - original_count} oversampled)")
            padded_classes.append((name, d, imgs_padded))
        classes = padded_classes

        if oversampled_count > 0:
            print(f"\n  📊 {oversampled_count} classes were oversampled to reach "
                  f"{args.max_images_per_class} images each")

    total_images = sum(len(imgs) for _, _, imgs in classes)
    total_to_generate = total_images * len(args.domain_id) * args.num_per_image

    # ================================================================
    # Reverse 順序（多 GPU 平行用）
    # ================================================================
    if args.reverse:
        classes = list(reversed(classes))
        print(f"\n  🔄 Class order REVERSED (processing from last to first)")

    print(f"\n  Classes found: {len(classes)}")
    print(f"  Total input images: {total_images}")
    print(f"  Total to generate: {total_to_generate}")
    print(f"  (= {total_images} images × {len(args.domain_id)} domains × {args.num_per_image} per)")

    if args.max_images_per_class:
        print(f"  Max images per class: {args.max_images_per_class} (with oversampling)")

    # 預覽前幾個 class
    print(f"\n  Classes preview:")
    for i, (name, _, imgs) in enumerate(classes[:10]):
        print(f"    {i+1:3d}. {name:<30} ({len(imgs)} images)")
    if len(classes) > 10:
        print(f"    ... and {len(classes) - 10} more")

    # ================================================================
    # 初始化 generator
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  Initializing Generator")
    print(f"{'='*70}")

    generator = LearnablePromptGenerator(
        checkpoint_path=args.checkpoint,
        sd_model_id=args.sd_model,
        clip_model_name=args.clip_model,
        use_controlnet=args.use_controlnet,
        controlnet_type=args.controlnet_type,
    )
    generator.pipe.set_progress_bar_config(disable=True)

    # ================================================================
    # 儲存設定（方便事後查看）
    # ================================================================
    os.makedirs(args.output, exist_ok=True)

    config = {
        "checkpoint": args.checkpoint,
        "source_domain": args.input,
        "source_name": source_name,
        "target_domain_ids": args.domain_id,
        "target_names": target_name_map,
        "domain_dir_names": domain_dir_names,
        "num_per_image": args.num_per_image,
        "batch_size": args.batch_size,
        "max_images_per_class": args.max_images_per_class,
        "oversampling": True if args.max_images_per_class else False,
        "style_prompts": args.style_prompts,
        "use_controlnet": args.use_controlnet,
        "controlnet_type": args.controlnet_type,
        "steps": args.steps,
        "cfg": args.cfg,
        "cn_scale": args.cn_scale,
        "seed": args.seed,
        "num_classes": len(classes),
        "total_input_images": total_images,
        "total_to_generate": total_to_generate,
    }
    with open(os.path.join(args.output, "generation_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ================================================================
    # 逐 class 生成
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  Starting Generation")
    print(f"{'='*70}")

    start_time = time.time()
    total_generated = 0
    skip_until_found = args.resume_from_class is not None
    skipped_classes = 0
    is_resume_class = False

    for cls_idx, (cls_name, cls_dir, cls_images) in enumerate(classes):
        # Resume 邏輯
        if skip_until_found:
            if cls_name == args.resume_from_class:
                skip_until_found = False
                is_resume_class = True
            else:
                skipped_classes += 1
                continue

        print(f"\n[{cls_idx+1}/{len(classes)}] Processing: {cls_name}")

        # ── Stop-if-exists 檢查 ──────────────────────────────────────
        if args.stop_if_exists and not is_resume_class:
            existing_found = False
            for did in args.domain_id:
                check_dir = os.path.join(args.output, domain_dir_names[did], cls_name)
                if os.path.isdir(check_dir):
                    png_files = [f for f in os.listdir(check_dir)
                                 if f.endswith('.png')]
                    if len(png_files) > 0:
                        existing_found = True
                        print(f"    ⛔ Found {len(png_files)} existing files in:")
                        print(f"       {check_dir}")
                        break
            if existing_found:
                print(f"\n  🛑 Stopping: class '{cls_name}' already has generated files.")
                print(f"     This likely means the other GPU has reached this point.")
                print(f"     Total generated before stopping: {total_generated} images")
                break
        elif is_resume_class:
            print(f"    ℹ Skipping stop_if_exists check (this is the resumed class)")
            is_resume_class = False
        # ─────────────────────────────────────────────────────────────

        count = generate_for_class(
            generator=generator,
            class_name=cls_name,
            image_paths=cls_images,
            domain_ids=args.domain_id,
            domain_dir_names=domain_dir_names,
            num_per_image=args.num_per_image,
            batch_size=args.batch_size,
            save_dir=args.output,
            use_controlnet=args.use_controlnet,
            steps=args.steps,
            cfg=args.cfg,
            cn_scale=args.cn_scale,
            base_seed=args.seed,
            style_prompts=args.style_prompts,
        )
        total_generated += count

        elapsed = time.time() - start_time
        classes_done = cls_idx + 1 - skipped_classes
        if classes_done > 0:
            avg_per_class = elapsed / classes_done
            remaining_classes = len(classes) - cls_idx - 1
            eta = avg_per_class * remaining_classes
            print(f"    ⏱ Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")

    # ================================================================
    # 完成
    # ================================================================
    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"  ✅ Generation Complete!")
    print(f"{'='*70}")
    print(f"  Total generated: {total_generated} images")
    print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hr)")
    if total_generated > 0:
        print(f"  Avg per image: {total_time/total_generated:.2f} sec")
    print(f"  Output: {args.output}/")
    print(f"\n  Structure:")
    print(f"    {args.output}/")
    for did in args.domain_id:
        print(f"      {domain_dir_names[did]}/")
        print(f"        <class_name>/")
        print(f"          img_XXXX_gen.png")


if __name__ == "__main__":
    main()