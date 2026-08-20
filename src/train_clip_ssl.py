"""
USB FreeMatch — DomainNet 統一訓練腳本 (CLIP Backbone 版)
====================================================================
修復項目：
  1. CLIP normalization 統一套用到 labeled / unlabeled / eval 三個 dataset
  2. eval_loader 使用 data_sampler=None, drop_last=False（避免無限迴圈）
  3. eval_batch_size 預設 256（加速 eval）
  4. 加入 eval 計時（monkey-patch）

支援三種 backbone 設定：
  1. USB 原生 ViT（原本的設定，向後相容）
  2. CLIP ViT (原始 pretrain) — baseline
  3. CLIP ViT (fine-tuned) — 實驗組

用法範例：
  # CLIP backbone — Ours (generated labeled)
  python train_clip_ssl.py --net clip_vit --lb_per_class 100 \
      --labeled_source generated \
      --generated_dir ./result/style-transfer-gen/prompt-len20/real-to/real-to-painting \
      --train_txt ./data/DomainNet/painting_train.txt \
      --test_txt ./data/DomainNet/painting_test.txt \
      --save_dir ./checkpoints/ssl-model/clip-backbone/rl2pt

  # CLIP backbone — Source baseline (real source labeled)
  python train_clip_ssl.py --net clip_vit --lb_per_class 100 \
      --labeled_source generated \
      --generated_dir ./data/DomainNet/real \
      --train_txt ./data/DomainNet/painting_train.txt \
      --test_txt ./data/DomainNet/painting_test.txt \
      --save_dir ./checkpoints/ssl-model/clip-backbone/rl2pt_lowerbound

  # CLIP backbone — Oracle (real target labeled)
  python train_clip_ssl.py --net clip_vit --lb_per_class 100 \
      --labeled_source real \
      --train_txt ./data/DomainNet/painting_train.txt \
      --test_txt ./data/DomainNet/painting_test.txt \
      --save_dir ./checkpoints/ssl-model/clip-backbone/rl2pt_oracle

  # 原始 USB ViT（向後相容）
  python train_clip_ssl.py --net vit_base_patch16_224 --lb_per_class 100 \
      --layer_decay 0.5 --pretrain_path ./vit_base_p16_224_for_usb.pth
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from semilearn import get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from semilearn.datasets.augmentation import RandAugment

# CLIP backbone
from clip_backbone import get_clip_net_builder


# ============================================================
# Normalization 常數
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


# ============================================================
# Argument Parser
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='USB FreeMatch — DomainNet (CLIP Backbone)')

    # Data
    parser.add_argument('--labeled_source', type=str, default='generated',
                        choices=['generated', 'real'],
                        help='labeled data 來源: generated (style transfer) 或 real (oracle baseline)')
    parser.add_argument('--generated_dir', type=str,
                        default='./result/style-transfer-gen/prompt-len20/real-to-painting',
                        help='generated data 資料夾路徑 (labeled_source=generated 時使用)')
    parser.add_argument('--img_root', type=str, default='./data/DomainNet',
                        help='DomainNet 圖片根目錄')
    parser.add_argument('--train_txt', type=str, default='./data/DomainNet/painting_train.txt',
                        help='官方 training split txt')
    parser.add_argument('--test_txt', type=str, default='./data/DomainNet/painting_test.txt',
                        help='官方 test split txt')
    parser.add_argument('--lb_per_class', type=int, default=100,
                        help='每 class 挑幾張當 labeled')
    # --- 生成圖自評 (boundary) ---
    parser.add_argument('--gen_holdout_ratio', type=float, default=0.0,
                        help='>0 時:從 generated 每類切出此比例當 holdout 做「生成圖自評」(boundary)，'
                             '其餘最多取 lb_per_class 當 labeled train。holdout 與 train 不重疊。0=關閉(原行為)。'
                             '僅在 labeled_source=generated 時有效。')
    parser.add_argument('--gen_holdout_seed', type=int, default=None,
                        help='gen holdout 切分用 seed(預設沿用 --seed)')

    # Model — 原始 USB ViT
    parser.add_argument('--net', type=str, default='vit_base_patch16_224',
                        help='backbone 名稱 (USB 內建的 or "clip_vit")')
    parser.add_argument('--pretrain_path', type=str, default='./vit_base_p16_224_for_usb.pth',
                        help='pretrained weights 路徑 (USB ViT 用)')

    # Model — CLIP backbone（--net clip_vit 時使用）
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-large-patch14',
                        help='CLIP model 名稱')
    parser.add_argument('--clip_checkpoint', type=str, default=None,
                        help='two_stage.py 的 checkpoint path（None = 原始 CLIP pretrain）')
    parser.add_argument('--freeze_vision_encoder', action='store_true', default=True,
                        help='凍結 CLIP vision encoder（只訓練 projection + cls head）')
    parser.add_argument('--no_freeze_vision_encoder', action='store_true', default=False,
                        help='不凍結 CLIP vision encoder（全部都訓練）')
    parser.add_argument('--freeze_vision_projection', action='store_true', default=False,
                    help='額外凍結 CLIP vision_projection (預設只凍結 vision encoder)')

    # LoRA
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='在 CLIP vision encoder 上使用 LoRA adapter')
    parser.add_argument('--lora_rank', type=int, default=4,
                        help='LoRA rank (default: 4)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout (default: 0.1)')
    parser.add_argument('--lora_target_modules', type=str, default=None,
                        help='LoRA target modules (comma-separated, default: q_proj,v_proj)')

    # MLP Head
    parser.add_argument('--mlp_head', action='store_true', default=False,
                        help='用 MLP 取代 Linear classification head')
    parser.add_argument('--mlp_hidden_dim', type=int, default=1024,
                        help='MLP hidden dimension (default: 1024)')
    parser.add_argument('--mlp_dropout', type=float, default=0.1,
                        help='MLP dropout (default: 0.1)')

    # Training
    parser.add_argument('--num_train_iter', type=int, default=15000)
    parser.add_argument('--num_eval_iter', type=int, default=2200)
    parser.add_argument('--num_log_iter', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--uratio', type=int, default=2)
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--ent_loss_ratio', type=float, default=0.001)
    parser.add_argument('--img_size', type=int, default=224)

    # SSL
    parser.add_argument('--algorithm', type=str, default='freematch',
                        help='SSL algorithm (freematch, fixmatch, softmatch, etc.)')
    parser.add_argument('--no_ssl', action='store_true',
                        help='純 supervised（不用 unlabeled loss），用來量化 SSL 的貢獻')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/ssl-model/clip-backbone',
                        help='模型儲存路徑')

    return parser.parse_args()


# ============================================================
# Dataset 工具
# ============================================================
def build_class_mapping_from_txt(txt_path):
    """從官方 txt 建立 class_to_idx mapping。"""
    class_to_idx = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(' ', 1)
            parts = rel_path.split('/')
            if len(parts) >= 2:
                class_name = parts[1]
                label_int = int(label)
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = label_int
    return class_to_idx


def load_txt_split(txt_path, img_root):
    """讀取官方 split txt，回傳 (paths, targets)"""
    paths, targets = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(' ', 1)
            full_path = os.path.join(img_root, rel_path)
            paths.append(full_path)
            targets.append(int(label))
    return paths, np.array(targets)


def scan_labeled_with_mapping(root_dir, class_to_idx):
    """掃描 generated data 資料夾，用官方 class_to_idx 對齊。"""
    paths, targets = [], []
    skipped_classes = []

    all_dirs = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    matched = 0
    for class_name in all_dirs:
        if class_name in class_to_idx:
            idx = class_to_idx[class_name]
            matched += 1
        elif class_name.replace('-', '_') in class_to_idx:
            idx = class_to_idx[class_name.replace('-', '_')]
            matched += 1
            print(f"  [auto-fix] '{class_name}' → '{class_name.replace('-', '_')}'")
        else:
            skipped_classes.append(class_name)
            continue

        class_dir = os.path.join(root_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                paths.append(os.path.join(class_dir, fname))
                targets.append(idx)

    if skipped_classes:
        print(f"  WARNING: {len(skipped_classes)} classes not in official mapping (skipped):")
        for name in skipped_classes[:10]:
            print(f"    {name}")

    print(f"  Matched {matched}/{len(all_dirs)} classes to official mapping")
    return paths, np.array(targets)


def subsample_labeled(paths, targets, per_class, seed=42):
    """從 labeled data 中每個 class 隨機挑 per_class 張"""
    rng = random.Random(seed)

    class_indices = {}
    for i, t in enumerate(targets):
        t_int = int(t)
        if t_int not in class_indices:
            class_indices[t_int] = []
        class_indices[t_int].append(i)

    selected = []
    too_few = []

    for c in sorted(class_indices.keys()):
        indices = class_indices[c]
        rng.shuffle(indices)
        if len(indices) < per_class:
            selected.extend(indices)
            too_few.append((c, len(indices)))
        else:
            selected.extend(indices[:per_class])

    if too_few:
        print(f"  WARNING: {len(too_few)} classes have fewer than {per_class} images")

    sel_paths   = [paths[i] for i in selected]
    sel_targets = targets[selected]
    return sel_paths, sel_targets


def split_labeled_unlabeled(paths, targets, per_class, seed=42):
    """從同一個 dataset 中分出 labeled 和 unlabeled。"""
    rng = random.Random(seed)

    class_indices = {}
    for i, t in enumerate(targets):
        t_int = int(t)
        if t_int not in class_indices:
            class_indices[t_int] = []
        class_indices[t_int].append(i)

    lb_selected = []
    ulb_selected = []
    too_few = []

    for c in sorted(class_indices.keys()):
        indices = class_indices[c][:]
        rng.shuffle(indices)
        if len(indices) <= per_class:
            lb_selected.extend(indices)
            too_few.append((c, len(indices)))
        else:
            lb_selected.extend(indices[:per_class])
            ulb_selected.extend(indices[per_class:])

    if too_few:
        print(f"  WARNING: {len(too_few)} classes have fewer than {per_class} images")

    lb_paths   = [paths[i] for i in lb_selected]
    lb_targets = targets[lb_selected]
    ulb_paths   = [paths[i] for i in ulb_selected]
    ulb_targets = targets[ulb_selected]

    return lb_paths, lb_targets, ulb_paths, ulb_targets


def split_generated_holdout(paths, targets, holdout_ratio, lb_per_class, seed=42):
    """從 generated data 每個 class 切出 holdout(生成圖自評用)。
    每類: shuffle(seeded) → 前 n_hold 張當 holdout、其餘最多取 lb_per_class 張當 labeled train。
    保證 holdout 與 train 完全不重疊。回傳 (lb_paths, lb_targets, hold_paths, hold_targets)。
    """
    rng = random.Random(seed)

    class_indices = {}
    for i, t in enumerate(targets):
        class_indices.setdefault(int(t), []).append(i)

    lb_sel, hold_sel, too_few = [], [], []
    for c in sorted(class_indices.keys()):
        idxs = class_indices[c][:]
        rng.shuffle(idxs)
        n = len(idxs)
        n_hold = max(1, int(round(holdout_ratio * n))) if n > 1 else 0
        hold = idxs[:n_hold]
        train_pool = idxs[n_hold:]
        if len(train_pool) > lb_per_class:
            train = train_pool[:lb_per_class]
        else:
            train = train_pool
            if len(train_pool) < lb_per_class:
                too_few.append((c, len(train_pool)))
        lb_sel.extend(train)
        hold_sel.extend(hold)

    if too_few:
        print(f"  WARNING: {len(too_few)} classes have < {lb_per_class} train imgs after holdout")

    lb_paths   = [paths[i] for i in lb_sel]
    lb_targets = targets[lb_sel]
    h_paths    = [paths[i] for i in hold_sel]
    h_targets  = targets[hold_sel]
    return lb_paths, lb_targets, h_paths, h_targets


class LazySSLDataset(Dataset):
    """On-the-fly 圖片讀取的 SSL Dataset"""
    def __init__(self, algorithm, paths, targets, num_classes,
                 img_size=224, mode='labeled', mean=None, std=None):
        self.paths = paths
        self.targets = np.array(targets)
        self.num_classes = num_classes
        self.mode = mode
        self.algorithm = algorithm

        # 使用傳入的 mean/std，預設 ImageNet
        if mean is None:
            mean = IMAGENET_MEAN
        if std is None:
            std = IMAGENET_STD

        self.mean = mean
        self.std = std

        resize_size = int(img_size / 0.875)

        self.weak_transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.strong_transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        target = self.targets[idx]

        if self.mode == 'unlabeled':
            return {
                'idx_ulb': idx,
                'x_ulb_w': self.weak_transform(img),
                'x_ulb_s': self.strong_transform(img),
                'y_ulb': target,
            }
        elif self.mode == 'eval':
            return {
                'idx_lb': idx,
                'x_lb': self.eval_transform(img),
                'y_lb': target,
            }
        else:
            return {
                'idx_lb': idx,
                'x_lb': self.weak_transform(img),
                'y_lb': target,
            }


# ============================================================
# 主程式
# ============================================================
if __name__ == '__main__':
    args = parse_args()

    # gen holdout seed 預設沿用 --seed
    if args.gen_holdout_seed is None:
        args.gen_holdout_seed = args.seed

    # 處理 freeze 參數
    if args.no_freeze_vision_encoder:
        args.freeze_vision_encoder = False

    # LoRA mode 自動 freeze encoder（LoRA 會自己處理 trainable）
    if args.use_lora:
        args.freeze_vision_encoder = True

    # 判斷是否使用 CLIP backbone
    use_clip = (args.net == 'clip_vit')

    # ---- 決定 normalization ----
    if use_clip:
        norm_mean, norm_std = CLIP_MEAN, CLIP_STD
        norm_name = "CLIP"
    else:
        norm_mean, norm_std = IMAGENET_MEAN, IMAGENET_STD
        norm_name = "ImageNet"

    # ---- 印出實驗設定 ----
    print("=" * 60)
    print("Experiment Config:")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    if use_clip:
        print(f"\n  >>> Using CLIP Vision Backbone <<<")
        if args.clip_checkpoint:
            print(f"  >>> Mode: Fine-tuned (two-stage) <<<")
        else:
            print(f"  >>> Mode: Original CLIP pretrain (baseline) <<<")
        if args.use_lora:
            print(f"  >>> LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout} <<<")
        if args.mlp_head:
            print(f"  >>> MLP Head: hidden={args.mlp_hidden_dim}, dropout={args.mlp_dropout} <<<")
    else:
        print(f"\n  >>> Using USB native backbone: {args.net} <<<")

    print(f"  >>> Normalization: {norm_name} <<<")
    print(f"      mean={norm_mean}")
    print(f"      std={norm_std}")
    print("=" * 60)

    # ---- [1] 從官方 txt 建立 class mapping ----
    print("\n" + "=" * 60)
    print("[1/5] Building class mapping from official txt...")
    print("=" * 60)

    class_to_idx = build_class_mapping_from_txt(args.train_txt)
    NUM_CLASSES = max(class_to_idx.values()) + 1
    print(f"  Official classes: {len(class_to_idx)} names, {NUM_CLASSES} class IDs (0–{NUM_CLASSES-1})")

    # ---- [2] 準備 labeled / unlabeled ----
    print("\n" + "=" * 60)

    gen_hold_paths, gen_hold_targets = [], []   # 生成圖自評 holdout(預設空)

    if args.labeled_source == 'generated':
        print("[2/5] Scanning labeled (generated) data...")
        print("=" * 60)

        all_lb_paths, all_lb_targets = scan_labeled_with_mapping(args.generated_dir, class_to_idx)
        print(f"  Found {len(all_lb_paths)} images")

        if args.gen_holdout_ratio and args.gen_holdout_ratio > 0:
            lb_paths, lb_targets, gen_hold_paths, gen_hold_targets = split_generated_holdout(
                all_lb_paths, all_lb_targets,
                holdout_ratio=args.gen_holdout_ratio,
                lb_per_class=args.lb_per_class,
                seed=args.gen_holdout_seed,
            )
            print(f"  [gen-holdout] ratio={args.gen_holdout_ratio} seed={args.gen_holdout_seed}")
            print(f"  Train labeled (generated): {len(lb_paths)}")
            print(f"  Gen holdout  (self-eval):  {len(gen_hold_paths)}")
        else:
            lb_paths, lb_targets = subsample_labeled(
                all_lb_paths, all_lb_targets,
                per_class=args.lb_per_class,
                seed=args.seed,
            )
            print(f"  Selected {len(lb_paths)} labeled ({args.lb_per_class}/class)")

        # unlabeled = 完整的 train.txt
        ulb_paths, ulb_targets = load_txt_split(args.train_txt, args.img_root)

    elif args.labeled_source == 'real':
        print("[2/5] Splitting real data into labeled / unlabeled...")
        print("=" * 60)

        all_train_paths, all_train_targets = load_txt_split(args.train_txt, args.img_root)
        print(f"  Total train: {len(all_train_paths)} images")

        lb_paths, lb_targets, ulb_paths, ulb_targets = split_labeled_unlabeled(
            all_train_paths, all_train_targets,
            per_class=args.lb_per_class,
            seed=args.seed,
        )

    print(f"  Labeled:   {len(lb_paths)} images")
    print(f"  Unlabeled: {len(ulb_paths)} images")

    # ---- [3] 讀取 eval split ----
    print("\n" + "=" * 60)
    print("[3/5] Loading eval split...")
    print("=" * 60)

    val_paths, val_targets = load_txt_split(args.test_txt, args.img_root)
    print(f"  Eval ({args.test_txt}): {len(val_paths)} images")

    missing = 0
    for p in lb_paths[:50] + ulb_paths[:50] + val_paths[:50]:
        if not os.path.exists(p):
            missing += 1
    if missing > 0:
        print(f"  WARNING: {missing}/150 sampled files not found!")
        print(f"  Example path: {lb_paths[0]}")
    else:
        print(f"  File check passed")

    # ---- [4] USB Config & Dataset ----
    print("\n" + "=" * 60)
    print("[4/5] Setting up config and model...")
    print("=" * 60)

    NUM_LABELS = len(lb_paths)

    # ---- 決定 net_builder ----
    if use_clip:
        lora_targets = None
        if args.lora_target_modules:
            lora_targets = [m.strip() for m in args.lora_target_modules.split(',')]

        net_builder = get_clip_net_builder(
            clip_model_name=args.clip_model_name,
            two_stage_checkpoint=args.clip_checkpoint,
            freeze_vision_encoder=args.freeze_vision_encoder,
            freeze_vision_projection=args.freeze_vision_projection,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=lora_targets,
            mlp_head=args.mlp_head,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_dropout=args.mlp_dropout,
        )
        use_pretrain = False
        pretrain_path = None

        clip_tag = "clip_finetuned" if args.clip_checkpoint else "clip_pretrain"
        lora_tag = f"_lora_r{args.lora_rank}" if args.use_lora else ""
        mlp_tag = f"_mlp{args.mlp_hidden_dim}" if args.mlp_head else ""
        fp_tag = "_fp" if args.freeze_vision_projection else ""
        save_name = (f'{args.algorithm}_{args.labeled_source}_{clip_tag}{lora_tag}{mlp_tag}{fp_tag}'
                     f'_lb{args.lb_per_class}_ld{args.layer_decay}'
                     f'{"_nossl" if args.no_ssl else ""}')
    else:
        net_builder = get_net_builder(args.net, from_name=False)
        use_pretrain = True
        pretrain_path = args.pretrain_path

        save_name = (f'{args.algorithm}_{args.labeled_source}_{args.net}'
                     f'_lb{args.lb_per_class}_ld{args.layer_decay}'
                     f'{"_nossl" if args.no_ssl else ""}')

    config = {
        'algorithm': args.algorithm,
        'net': args.net if not use_clip else 'clip_vit',
        'use_pretrain': use_pretrain,
        'pretrain_path': pretrain_path if pretrain_path else '',

        'epoch': 1,
        'num_train_iter': args.num_train_iter,
        'num_eval_iter': args.num_eval_iter,
        'num_log_iter': args.num_log_iter,

        'optim': 'AdamW',
        'lr': args.lr,
        'layer_decay': args.layer_decay,
        'batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,

        'dataset': 'none',
        'num_labels': NUM_LABELS,
        'num_classes': NUM_CLASSES,
        'img_size': args.img_size,
        'crop_ratio': 0.875,
        'data_dir': '.',
        'ulb_samples_per_class': None,

        'hard_label': True,
        'T': 0.5,
        'ema_p': 0.999,
        'ent_loss_ratio': 0.0 if args.no_ssl else args.ent_loss_ratio,
        'uratio': args.uratio,
        'ulb_loss_ratio': 0.0 if args.no_ssl else args.ulb_loss_ratio,

        'gpu': args.gpu,
        'world_size': 1,
        'distributed': False,
        'num_workers': args.num_workers,

        'save_dir': args.save_dir,
        'save_name': save_name,
    }

    config = get_config(config)

    config.ulb_dest_len = len(ulb_paths)
    config.lb_dest_len = len(lb_paths)

    # ---- 建立 Dataset（統一 normalization）----
    print(f"\n  Building datasets with {norm_name} normalization...")

    train_lb_dataset = LazySSLDataset(
        config.algorithm, lb_paths, lb_targets,
        NUM_CLASSES, img_size=args.img_size, mode='labeled',
        mean=norm_mean, std=norm_std,
    )
    train_ulb_dataset = LazySSLDataset(
        config.algorithm, ulb_paths, ulb_targets,
        NUM_CLASSES, img_size=args.img_size, mode='unlabeled',
        mean=norm_mean, std=norm_std,
    )
    eval_dataset = LazySSLDataset(
        config.algorithm, val_paths, val_targets,
        NUM_CLASSES, img_size=args.img_size, mode='eval',
        mean=norm_mean, std=norm_std,
    )

    print(f"  train_lb:  {len(train_lb_dataset)} (norm={norm_name})")
    print(f"  train_ulb: {len(train_ulb_dataset)} (norm={norm_name})")
    print(f"  eval:      {len(eval_dataset)} (norm={norm_name})")

    # ---- 建立 DataLoader ----
    # 注意：train loader 用預設 RandomSampler
    #        eval loader 必須用 data_sampler=None 避免無限迴圈
    train_lb_loader  = get_data_loader(config, train_lb_dataset, config.batch_size)
    train_ulb_loader = get_data_loader(config, train_ulb_dataset, int(config.batch_size * config.uratio))
    eval_loader      = get_data_loader(config, eval_dataset, config.eval_batch_size,
                                        data_sampler=None, drop_last=False)

    print(f"  train_lb_loader:  batch_size={config.batch_size}, sampler=RandomSampler")
    print(f"  train_ulb_loader: batch_size={int(config.batch_size * config.uratio)}, sampler=RandomSampler")
    print(f"  eval_loader:      batch_size={config.eval_batch_size}, sampler=None (sequential)")

    # ---- 生成圖自評 (gen-holdout) loader（mode='eval'，同 normalization / transform，可比）----
    gen_holdout_loader = None
    if len(gen_hold_paths) > 0:
        gen_holdout_dataset = LazySSLDataset(
            config.algorithm, gen_hold_paths, gen_hold_targets,
            NUM_CLASSES, img_size=args.img_size, mode='eval',
            mean=norm_mean, std=norm_std,
        )
        gen_holdout_loader = get_data_loader(config, gen_holdout_dataset, config.eval_batch_size,
                                             data_sampler=None, drop_last=False)
        print(f"  gen_holdout_loader: {len(gen_holdout_dataset)} imgs, batch_size={config.eval_batch_size} (norm={norm_name})")


    algorithm = get_algorithm(
        config,
        net_builder,
        tb_log=None,
        logger=None,
    )

    # ---- 儲存 config ----
    exp_save_dir = os.path.join(args.save_dir, save_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    config_save_path = os.path.join(exp_save_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        config_dict = vars(args)
        config_dict['normalization'] = norm_name
        config_dict['norm_mean'] = norm_mean
        config_dict['norm_std'] = norm_std
        json.dump(config_dict, f, indent=2)
    print(f"  Config saved to {config_save_path}")

    # ---- Monkey-patch evaluate 加上計時 + 生成圖自評(gen-holdout)----
    _original_evaluate = algorithm.evaluate

    _gho_best = {'acc': -1.0, 'it': -1}

    def _eval_gen_holdout():
        """每次主 eval 後，順手用同一個 model eval 一次 gen-holdout（boundary）。best-effort、不影響主流程。"""
        if gen_holdout_loader is None:
            return
        try:
            if not hasattr(algorithm, 'loader_dict') or algorithm.loader_dict is None:
                algorithm.loader_dict = {}
            algorithm.loader_dict['gen_holdout'] = gen_holdout_loader
            res = _original_evaluate(eval_dest='gen_holdout')
            acc = None
            if isinstance(res, dict):
                acc = res.get('gen_holdout/top-1-acc')
                if acc is None:
                    for k, v in res.items():
                        if 'top-1-acc' in k:
                            acc = v
                            break
            if acc is not None:
                acc = float(acc)
                cur_it = int(getattr(algorithm, 'it', -1))
                if acc > _gho_best['acc']:
                    _gho_best['acc'] = acc
                    _gho_best['it'] = cur_it
                print(f"  >>> GEN_HOLDOUT/top-1-acc: {acc:.4f}  "
                      f"GEN_HOLDOUT_BEST_ACC: {_gho_best['acc']:.4f}, at {_gho_best['it']} iters")
            else:
                print(f"  >>> GEN_HOLDOUT/top-1-acc: {acc}")
        except Exception as e:
            print(f"  [gen_holdout eval skipped: {e}]")

    def _timed_evaluate(*args_eval, **kwargs_eval):
        t0 = time.time()
        print(f"\n⏱️  Eval started at {time.strftime('%H:%M:%S')}")
        result = _original_evaluate(*args_eval, **kwargs_eval)
        # 只在「主 target eval」之後跑 gen-holdout（eval_dest 非 gen_holdout 時）
        if kwargs_eval.get('eval_dest', 'eval') != 'gen_holdout':
            _eval_gen_holdout()
        elapsed = time.time() - t0
        print(f"⏱️  Eval finished at {time.strftime('%H:%M:%S')} ({elapsed:.1f}s = {elapsed/60:.1f}min)")
        return result
    algorithm.evaluate = _timed_evaluate

    # ---- [5] 訓練 ----
    print("\n" + "=" * 60)
    if use_clip:
        clip_mode = "fine-tuned" if args.clip_checkpoint else "original pretrain"
        print(f"[5/5] Training (CLIP {clip_mode}, {args.labeled_source} labeled, "
              f"{args.algorithm}{'  [SUPERVISED ONLY]' if args.no_ssl else ''})...")
    else:
        print(f"[5/5] Training ({args.labeled_source} labeled, {args.algorithm}"
              f"{'  [SUPERVISED ONLY - no SSL]' if args.no_ssl else ''})...")
    print(f"  Normalization: {norm_name}")
    print(f"  lr={args.lr}, iter={args.num_train_iter}, eval_every={args.num_eval_iter}")
    print(f"  1 labeled epoch ≈ {len(lb_paths) // args.batch_size} iter")
    print("=" * 60)

    trainer = Trainer(config, algorithm)
    trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)

    print("\n" + "=" * 60)
    print("Final evaluation:")
    print("=" * 60)
    trainer.evaluate(eval_loader)

    # ---- 生成圖自評 final（boundary 的保證數字；即使上面 monkey-patch 失敗也有這個）----
    if gen_holdout_loader is not None:
        print("\n" + "=" * 60)
        print("Final GEN-HOLDOUT evaluation (生成圖自評 / boundary):")
        print("=" * 60)
        try:
            trainer.evaluate(gen_holdout_loader)
        except Exception as e:
            print(f"  [final gen_holdout eval failed: {e}]")
        print(f"GEN_HOLDOUT_BEST_ACC: {_gho_best['acc']:.4f}, at {_gho_best['it']} iters")

    print(f"\nDone! Model saved at {exp_save_dir}")