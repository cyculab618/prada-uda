"""
Learnable Prompt 生成器（支援 ControlNet 可選）
=============================================
模式 1：純 Learnable Prompt（不使用 ControlNet）
模式 2：Learnable Prompt + ControlNet

用來測試問題出在哪裡

cmd:
    (pure learnable prompt)
    python controlnet_gen.py --checkpoint {ckpt path} --output {output path} --num_images 30

    (leranable prompt + text prompt)
    python controlnet_gen.py --checkpoint {ckpt path} --output {output path} --text "{cls}" --num_images 30

    (controlnet + learnable prompt + text prompt)
    python controlnet_gen.py --checkpoint {ckpt path} --output {output path} --text "{cls}" --use_controlnet --controlnet_type {type} --num_images 30

    (negative prompt)
    python ........ --exclude_other_classes --negative "low quality, blurry, ....."
"""

import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    UniPCMultistepScheduler,
)
from transformers import CLIPModel, CLIPTokenizer
from PIL import Image
import numpy as np
import cv2
import os
import argparse


# ============================================================
# 預處理器
# ============================================================

def preprocess_canny(image_path, low_threshold=100, high_threshold=200):
    image = cv2.imread(image_path)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


CONTROLNET_MODELS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "depth": "lllyasviel/sd-controlnet-depth",
    "openpose": "lllyasviel/sd-controlnet-openpose",
}


def get_negative_prompt_from_classes(class_dir, exclude_class):
    """
    從資料夾讀取 class list，排除指定的 class，組成 negative prompt
    """
    if not os.path.exists(class_dir):
        print(f"Warning: {class_dir} not found, skip negative prompt generation")
        return ""
    
    all_classes = [d for d in os.listdir(class_dir) 
                   if os.path.isdir(os.path.join(class_dir, d))]
    all_classes = sorted(all_classes)
    
    other_classes = [c for c in all_classes if c.lower() != exclude_class.lower()]
    
    negative_prompt = ", ".join(other_classes)
    
    print(f"=== Negative Prompt Info ===")
    print(f"  Total classes: {len(all_classes)}")
    print(f"  Excluded class: '{exclude_class}'")
    print(f"  Remaining classes: {len(other_classes)}")
    print(f"  Negative prompt: {negative_prompt[:80]}...")
    print(f"============================")
    
    return negative_prompt


# ============================================================
# 主類別
# ============================================================

class LearnablePromptGenerator:
    """
    支援兩種模式：
    1. 純 Learnable Prompt（原本的效果）
    2. Learnable Prompt + ControlNet
    """
    
    def __init__(
        self,
        checkpoint_path,
        sd_model_id="runwayml/stable-diffusion-v1-5",
        clip_model_name="openai/clip-vit-large-patch14",
        use_controlnet=False,
        controlnet_type="canny",
        device="cuda"
    ):
        self.device = device
        self.hidden_dim = 768
        self.use_controlnet = use_controlnet
        self.controlnet_type = controlnet_type
        
        # ===== 載入 SD Pipeline =====
        if use_controlnet:
            print(f"Loading SD + ControlNet ({controlnet_type})...")
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_MODELS[controlnet_type],
                torch_dtype=torch.float16
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                sd_model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None
            ).to(device)
        else:
            print("Loading SD (no ControlNet)...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                sd_model_id,
                torch_dtype=torch.float16,
                safety_checker=None
            ).to(device)
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # ===== 載入 Learnable Prompt =====
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 自動偵測 state_dict 的 key 名稱（相容不同版本的 checkpoint）
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        else:
            raise KeyError(f"Checkpoint 中找不到模型權重，available keys: {list(ckpt.keys())}")
        
        # 自動偵測 num_domains 和 prompt_len
        domain_keys = sorted([k for k in state_dict if k.startswith("domain_prompts.")])
        if len(domain_keys) == 0:
            raise KeyError(f"Checkpoint 中找不到 domain_prompts，available keys: {list(state_dict.keys())[:20]}")
        
        self.num_domains = len(domain_keys)
        self.prompt_len = state_dict[domain_keys[0]].shape[0]
        
        print(f"  Detected {self.num_domains} domains, prompt_len={self.prompt_len}")
        
        # 載入所有 domain prompt
        self.domain_prompts = []
        for key in domain_keys:
            prompt = state_dict[key].to(device).float()
            self.domain_prompts.append(prompt)
            print(f"  {key}: {prompt.shape}")
        
        # ===== 載入 CLIP =====
        print(f"Loading CLIP: {clip_model_name}")
        self.clip = CLIPModel.from_pretrained(clip_model_name).to(device).float()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        self._init_special_tokens()
        print("Ready!\n")
    
    def _init_special_tokens(self):
        """初始化 SOT/EOT"""
        with torch.no_grad():
            tokens = self.tokenizer(
                "", return_tensors="pt",
                padding="max_length", max_length=77, truncation=True
            )
            input_ids = tokens["input_ids"].to(self.device)
            token_emb = self.clip.text_model.embeddings.token_embedding(input_ids)
            
            self.sot_embedding = token_emb[0, 0:1, :].clone()
            self.eot_embedding = token_emb[0, 1:2, :].clone()
            
            position_ids = torch.arange(77, device=self.device).unsqueeze(0)
            self.position_embedding = self.clip.text_model.embeddings.position_embedding(position_ids).squeeze(0)
    
    def get_prompt_embedding(self, domain_id, additional_text="", verbose=False):
        """
        取得 domain prompt 的 embedding
        
        結構: [SOT] + [additional_text] + [learnable_prompt] + [EOT] + [PAD...]
        """
        prompt = self.domain_prompts[domain_id]  # [P, hidden_dim]
        P = self.prompt_len
        
        if additional_text:
            add_tokens = self.tokenizer(
                additional_text,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"].to(self.device)
            
            add_emb = self.clip.text_model.embeddings.token_embedding(add_tokens).squeeze(0)
            N = add_emb.shape[0]
            N_orig = N
            
            used = 1 + N + P + 1
            pad_len = 77 - used
            truncated = False
            
            if pad_len < 0:
                N = 77 - 1 - P - 1
                add_emb = add_emb[:N]
                pad_len = 0
                truncated = True

            if verbose:
                tag = f" (truncated from {N_orig})" if truncated else ""
                print(f"  [embed] domain={domain_id} text='{additional_text}' | "
                    f"SOT=1 + text={N}{tag} + learnable={P} + EOT=1 "
                    f"= {1+N+P+1}  (+{pad_len} pad → 77)")
            
            pad = torch.zeros(pad_len, self.hidden_dim, device=self.device)
            # [SOT] + [text] + [learnable] + [EOT] + [PAD]
            token_emb = torch.cat([self.sot_embedding, add_emb, prompt, self.eot_embedding, pad], dim=0)
        else:
            used = 1 + P + 1
            pad_len = 77 - used
            if verbose:
                print(f"  [embed] domain={domain_id} (no text) | "
                    f"SOT=1 + learnable={P} + EOT=1 = {1+P+1}  (+{pad_len} pad → 77)")
            pad = torch.zeros(pad_len, self.hidden_dim, device=self.device)
            # [SOT] + [learnable] + [EOT] + [PAD]
            token_emb = torch.cat([self.sot_embedding, prompt, self.eot_embedding, pad], dim=0)
        
        token_emb = token_emb.unsqueeze(0)  # [1, 77, hidden_dim]
        hidden_states = token_emb + self.position_embedding.unsqueeze(0)
        
        # Causal attention mask
        causal_attention_mask = torch.triu(
            torch.full((77, 77), float('-inf'), device=self.device, dtype=hidden_states.dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # CLIP Text Transformer
        for layer in self.clip.text_model.encoder.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=causal_attention_mask
            )[0]
        
        hidden_states = self.clip.text_model.final_layer_norm(hidden_states)
        return hidden_states  # [1, 77, hidden_dim]
    
    def generate(
        self,
        domain_id,
        additional_text="",
        negative_prompt="",
        input_image=None,
        num_images=1,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        seed=None,
        height=512,
        width=512,
    ):
        """
        生成圖片
        """
        # ===== Prompt Embedding =====
        prompt_embeds = self.get_prompt_embedding(domain_id, additional_text)
        prompt_embeds = prompt_embeds.repeat(num_images, 1, 1).half()
        
        # ===== Negative Embedding =====
        neg_text = negative_prompt if negative_prompt else ""
        neg_tokens = self.tokenizer(
            [neg_text] * num_images,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            negative_embeds = self.pipe.text_encoder(neg_tokens.input_ids)[0]
        
        # ===== Generator =====
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # ===== 生成 =====
        if self.use_controlnet:
            if input_image is None:
                raise ValueError("ControlNet 模式需要提供 --input 圖片")
            
            control_image = preprocess_canny(input_image)
            
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                height=height,
                width=width,
            ).images
            
            return images, control_image
        else:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width,
            ).images
            
            return images, None
    
    def generate_all_domains(
        self,
        additional_text="",
        negative_prompt="",
        input_image=None,
        num_images=1,
        seed=42,
        controlnet_conditioning_scale=1.0,
        save_dir="./outputs"
    ):
        """生成所有 domain 的結果，每個 domain 存在不同子資料夾"""
        os.makedirs(save_dir, exist_ok=True)
        
        all_images = []
        control_image = None
        
        if self.use_controlnet and input_image is not None:
            control_image = preprocess_canny(input_image)
            control_image.save(os.path.join(save_dir, "control_image.png"))
        
        for domain_id in range(self.num_domains):
            print(f"Generating domain {domain_id}/{self.num_domains - 1}...")
            
            domain_dir = os.path.join(save_dir, f"domain_{domain_id}")
            os.makedirs(domain_dir, exist_ok=True)
            
            images, _ = self.generate(
                domain_id=domain_id,
                additional_text=additional_text,
                negative_prompt=negative_prompt,
                input_image=input_image,
                num_images=num_images,
                seed=seed,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            )
            
            for i, img in enumerate(images):
                img.save(os.path.join(domain_dir, f"{i}.png"))
            
            all_images.append(images[0])
            
            del images
            torch.cuda.empty_cache()
        
        # 拼接對比圖
        W, H = all_images[0].size
        comparison = Image.new('RGB', (W * self.num_domains, H))
        for i in range(self.num_domains):
            comparison.paste(all_images[i], (W * i, 0))
        comparison.save(os.path.join(save_dir, "comparison.png"))
        
        print(f"\nSaved to {save_dir}/")
        print(f"  - domain_0/ ~ domain_{self.num_domains - 1}/ (各 {num_images} 張)")
        print(f"  - comparison.png")
        if control_image:
            print(f"  - control_image.png")
        
        return all_images


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learnable Prompt Generator（支援 ControlNet 可選）")
    
    # 必要參數
    parser.add_argument("--checkpoint", type=str, required=True, help="Learnable prompt checkpoint")
    
    # 模式選擇
    parser.add_argument("--use_controlnet", action="store_true", 
                        help="啟用 ControlNet 模式（需要 --input）")
    parser.add_argument("--input", type=str, default=None, 
                        help="ControlNet 的輸入圖片")
    parser.add_argument("--controlnet_type", type=str, default="canny",
                        choices=["canny", "depth", "openpose"])
    
    # Prompt 參數
    parser.add_argument("--domain_id", type=int, default=None,
                        help="指定 domain，不指定則生成全部")
    parser.add_argument("--text", type=str, default="",
                        help="額外的文字描述（例如 'an airplane'）")
    
    # 生成參數
    parser.add_argument("--output", type=str, default="./outputs")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--cn_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Negative prompt 功能
    parser.add_argument("--class_dir", type=str, default=None,
                        help="DomainNet 資料夾路徑，用來讀取 class list")
    parser.add_argument("--exclude_other_classes", action="store_true",
                        help="把其他 class 加入 negative prompt")
    parser.add_argument("--negative", type=str, default="",
                        help="額外的 negative prompt")
    
    args = parser.parse_args()
    
    # ===== 自動從 input 路徑抓 class 名稱 =====
    content_class = args.text

    if not content_class and args.input:
        path_parts = args.input.replace("\\", "/").split("/")
        for i, part in enumerate(path_parts):
            if part in ["real", "painting", "sketch", "clipart", "quickdraw", "infograph"]:
                if i + 1 < len(path_parts):
                    content_class = path_parts[i + 1]
                    print(f"Auto-detected class from path: {content_class}")
                    break
    
    # ===== 處理 Negative Prompt =====
    negative_prompt = ""
    
    if args.exclude_other_classes and args.class_dir and content_class:
        negative_prompt = get_negative_prompt_from_classes(args.class_dir, content_class)
    
    if args.negative:
        if negative_prompt:
            negative_prompt = f"{negative_prompt}, {args.negative}"
        else:
            negative_prompt = args.negative
        print(f"Extra negative prompt: {args.negative}")
    
    if negative_prompt:
        print(f"Final negative prompt: {negative_prompt[:100]}...")
    
    # 初始化
    generator = LearnablePromptGenerator(
        checkpoint_path=args.checkpoint,
        use_controlnet=args.use_controlnet,
        controlnet_type=args.controlnet_type,
    )
    
    os.makedirs(args.output, exist_ok=True)
    
    text_to_use = args.text if args.text else content_class
    
    if args.domain_id is not None:
        # 生成指定 domain
        images, control_image = generator.generate(
            domain_id=args.domain_id,
            additional_text=text_to_use,
            negative_prompt=negative_prompt,
            input_image=args.input,
            num_images=args.num_images,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            controlnet_conditioning_scale=args.cn_scale,
            seed=args.seed,
        )
        
        for i, img in enumerate(images):
            img.save(os.path.join(args.output, f"output_{i}.png"))
        
        if control_image is not None:
            control_image.save(os.path.join(args.output, "control_image.png"))
        
        print(f"Saved to {args.output}/")
    
    else:
        # 生成所有 domain
        generator.generate_all_domains(
            additional_text=text_to_use,
            negative_prompt=negative_prompt,
            input_image=args.input,
            num_images=args.num_images,
            seed=args.seed,
            controlnet_conditioning_scale=args.cn_scale,
            save_dir=args.output,
        )