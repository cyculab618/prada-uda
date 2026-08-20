"""
CLIP Domain Prompt Tuning - Two Stage 版本 (修復版 v2)

架構：
Stage 1: 訓練 Domain Classifier
  - Loss: Classification only
  - 更新: domain_classifier
  - 凍結: prompts, vision_projection, logit_scale

Stage 2: 聯合訓練 Learnable Prompts + Vision Projection + Domain Classifier
  - Loss: Contrastive + λ * Classification
  - 更新: prompts, vision_projection, domain_classifier
  - 凍結: logit_scale
  - Classification loss 作為正則化，防止 projection 特徵漂移

用法：
    python two_stage.py --data_dir ./data/DomainNet --domains real,painting,sketch,clipart
    python two_stage.py --data_dir ./data/DomainNet --domains real,painting,sketch --augment --amp
    python two_stage.py --data_dir ./data/DomainNet --domains real,painting,sketch --mixup_alpha 0.4
    python two_stage.py --data_dir ./data/DomainNet --domains real,painting,sketch --lambda_cls 0.3
    python two_stage.py --data_dir ./data/DomainNet --domains real,painting,sketch --resume ./checkpoints/clip_two_stage_latest.pt
    python two_stage.py --data_dir ./data/DomainNet --list_domains
    python two_stage.py test
"""

import os
import glob
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

# 嘗試載入 wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️ wandb not installed, logging disabled")


# ============================================================
# Dataset
# ============================================================

class MultiDomainDataset(Dataset):
    """
    多 Domain 資料集（平衡取樣）
    """
    
    def __init__(self, data_dir, processor, domain_names, augment=False, domain_images=None, verbose=True):
        self.data_dir = data_dir
        self.processor = processor
        self.domain_names = domain_names
        self.num_domains = len(domain_names)
        self.augment = augment
        
        if verbose:
            print(f"\n📂 Loading multi-domain dataset from: {data_dir}")
            print(f"   Domains ({self.num_domains}): {domain_names}")
            print(f"   Augmentation: {'ON 🔥' if augment else 'OFF'}")
        
        if domain_images is not None:
            self.domain_images = domain_images
            if verbose:
                for domain_id, domain_name in enumerate(domain_names):
                    print(f"   - {domain_name} (id={domain_id}): {len(domain_images[domain_id])} images")
        else:
            self.domain_images = []
            for domain_id, domain_name in enumerate(domain_names):
                # xie, eg. ./data/DomainNet/real
                domain_dir = os.path.join(data_dir, domain_name)
                
                if not os.path.isdir(domain_dir):
                    raise RuntimeError(f"Domain directory not found: {domain_dir}")
                
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    # xie, 「**」這邊是所有的意思 => 「*」當前所有；「**」下一層所
                    images.extend(glob.glob(os.path.join(domain_dir, '**', ext), recursive=True))
                
                if len(images) == 0:
                    raise RuntimeError(f"No images found in {domain_dir}")
                
                self.domain_images.append(images)
                if verbose:
                    print(f"   - {domain_name} (id={domain_id}): {len(images)} images")
        
        self.domain_sizes = [len(imgs) for imgs in self.domain_images]
        self.min_domain_size = min(self.domain_sizes)
        self.total_images = sum(self.domain_sizes)
        
        if verbose:
            print(f"   Total: {self.total_images} images")
            print(f"   Min domain size: {self.min_domain_size} (用於平衡取樣)")
        
        if augment:
            self.aug_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=15),
                T.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            ])
        else:
            self.aug_transform = None
    
    def __len__(self):
        return self.min_domain_size * self.num_domains
    
    def __getitem__(self, idx):
        domain_id = idx % self.num_domains
        domain_images = self.domain_images[domain_id]
        img_idx = torch.randint(0, len(domain_images), (1,)).item()
        img_path = domain_images[img_idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.aug_transform is not None:
            image = self.aug_transform(image)
        
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "domain_id": domain_id
        }


# ============================================================
# MixUp
# ============================================================

def mixup_within_domain(pixel_values, domain_labels, alpha=0.4):
    """同 domain 內做 MixUp"""
    batch_size = pixel_values.size(0)
    device = pixel_values.device
    
    mixed_pixels = pixel_values.clone()
    
    lam = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(device)
    lam = lam.view(-1, 1, 1, 1)
    
    for domain_id in domain_labels.unique():
        mask = (domain_labels == domain_id)
        indices = torch.where(mask)[0]
        
        if len(indices) < 2:
            continue
        
        perm = indices[torch.randperm(len(indices))]
        
        mixed_pixels[indices] = (
            lam[indices] * pixel_values[indices] +
            (1 - lam[indices]) * pixel_values[perm]
        )
    
    return mixed_pixels, domain_labels


# ============================================================
# Model
# ============================================================

class CLIPDomainPromptTuner(nn.Module):
    """
    CLIP Domain Prompt Tuning - Two Stage 版本
    """
    
    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14",
        prompt_len=4,
        num_domains=2,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.prompt_len = prompt_len
        self.num_domains = num_domains
        self.model_name = model_name
        
        print(f"\n🔧 Loading CLIP: {model_name}")
        # xie, put whole model's checkpoints on the GPU VRAM, ie. whole nn.Parameter, different model architectures contain differenet nn.Parameter set
        self.clip = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # xie, 這邊只的是每一個transformer block的 input、output最終都是hidden_size, eg. ViT-L/14, each transformer block (768,3072) (3072,768)
        self.hidden_dim = self.clip.text_model.config.hidden_size
        # xie, text & visual projection output feature dim was the same
        self.embed_dim = self.clip.text_projection.out_features
        self.ctx_len = self.clip.text_model.config.max_position_embeddings
        
        print(f"   Hidden dim: {self.hidden_dim}, Embed dim: {self.embed_dim}")
        print(f"   Context length: {self.ctx_len}")
        
        # Learnable Prompts
        # xie, 「*0.01」since text encoder's token embedding value range (+-0.01~0.05), randn()是常態分佈大約+-1~3
        # xie, 在parameter裡面加入for，是python的list comprehesion syntax，若等價java syntax的話，無非就是在最外層做for然後body add
        self.domain_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(prompt_len, self.hidden_dim, device=device) * 0.01)
            for _ in range(num_domains)
        ])
        print(f"   ✅ Created {num_domains} learnable prompts (each {prompt_len} tokens)")
        
        # 永遠凍結 Text Encoder
        for param in self.clip.text_model.parameters():
            param.requires_grad = False
        for param in self.clip.text_projection.parameters():
            param.requires_grad = False
        print("   ❄️  Text Encoder + Text Projection: FROZEN (永遠)")
        
        # 永遠凍結 ViT Encoder
        for param in self.clip.vision_model.parameters():
            param.requires_grad = False
        print("   ❄️  ViT Encoder: FROZEN (永遠)")
        
        # Vision Projection (初始凍結)
        for param in self.clip.visual_projection.parameters():
            param.requires_grad = False
        print("   ❄️  Vision Projection: FROZEN (Stage 2 解凍)")
        
        # Prompts 初始凍結
        for param in self.domain_prompts.parameters():
            param.requires_grad = False
        print("   ❄️  Domain Prompts: FROZEN (Stage 2 解凍)")
        
        # logit_scale 初始凍結
        self.clip.logit_scale.requires_grad = False
        print("   ❄️  logit_scale: FROZEN (永遠)")
        
        # Domain Classifier
        self.domain_classifier = nn.Linear(self.embed_dim, num_domains).to(device)
        nn.init.xavier_uniform_(self.domain_classifier.weight)
        nn.init.zeros_(self.domain_classifier.bias)
        print(f"   ✅ Domain Classifier: Linear({self.embed_dim}, {num_domains})")
        
        # Special Tokens
        self._init_special_tokens()
        self._print_param_summary()
    
    def _init_special_tokens(self):
        """初始化 SOT/EOT embeddings"""
        with torch.no_grad():
            tokens = self.tokenizer(
                "", return_tensors="pt",
                padding="max_length", max_length=self.ctx_len, truncation=True
            )
            input_ids = tokens["input_ids"].to(self.device)
            
            token_emb = self.clip.text_model.embeddings.token_embedding(input_ids)
            self.register_buffer("sot_embedding", token_emb[0, 0:1, :].clone())
            self.register_buffer("eot_embedding", token_emb[0, 1:2, :].clone())
            
            position_ids = torch.arange(self.ctx_len, device=self.device).unsqueeze(0)
            pos_emb = self.clip.text_model.embeddings.position_embedding(position_ids)
            self.register_buffer("position_embedding", pos_emb.squeeze(0).clone())
        
        print("   ✅ Special tokens (SOT/EOT) initialized")
        
        # 偵測 CLIPEncoderLayer 的 API 版本
        import inspect
        layer_sig = inspect.signature(self.clip.text_model.encoder.layers[0].forward)
        self._has_causal_param = 'causal_attention_mask' in layer_sig.parameters
        if self._has_causal_param:
            print("   ✅ Detected transformers API: causal_attention_mask supported")
        else:
            print("   ✅ Detected transformers API: attention_mask only")
    
    def _print_param_summary(self):
        """印出參數概要"""
        total_params = sum(p.numel() for p in self.parameters())
        # xie, parameters() is nn.Module裡面的method, 只要使用的class有inheritance nn.Module就可以直接使用 (public static)
        prompt_params = sum(p.numel() for p in self.domain_prompts)
        cls_params = sum(p.numel() for p in self.domain_classifier.parameters())
        proj_params = sum(p.numel() for p in self.clip.visual_projection.parameters())
        vit_params = sum(p.numel() for p in self.clip.vision_model.parameters())
        text_params = sum(p.numel() for p in self.clip.text_model.parameters())
        text_proj_params = sum(p.numel() for p in self.clip.text_projection.parameters())
        
        print(f"\n   === 參數概要 ===")
        print(f"   Domain Prompts:    {prompt_params:>12,}")
        print(f"   Domain Classifier: {cls_params:>12,}")
        print(f"   Vision Projection: {proj_params:>12,}")
        print(f"   ViT Encoder:       {vit_params:>12,} (永遠凍結)")
        print(f"   Text Encoder:      {text_params:>12,} (永遠凍結)")
        print(f"   Text Projection:   {text_proj_params:>12,} (永遠凍結)")
        print(f"   logit_scale:       {1:>12,}")
        print(f"   ─────────────────────────────")
        print(f"   Total:             {total_params:>12,}")
    
    def print_trainable_params(self, stage_name=""):
        """印出目前可訓練的參數"""
        print(f"\n   === {stage_name} 可訓練參數 ===")
        
        trainable_total = 0
        
        prompt_params = sum(p.numel() for p in self.domain_prompts)
        prompt_trainable = sum(p.numel() for p in self.domain_prompts if p.requires_grad)
        status = "🔥 TRAINABLE" if prompt_trainable > 0 else "❄️  FROZEN"
        print(f"   Domain Prompts:    {prompt_trainable:>10,} / {prompt_params:>10,}  {status}")
        trainable_total += prompt_trainable
        
        cls_params = sum(p.numel() for p in self.domain_classifier.parameters())
        cls_trainable = sum(p.numel() for p in self.domain_classifier.parameters() if p.requires_grad)
        status = "🔥 TRAINABLE" if cls_trainable > 0 else "❄️  FROZEN"
        print(f"   Domain Classifier: {cls_trainable:>10,} / {cls_params:>10,}  {status}")
        trainable_total += cls_trainable
        
        proj_params = sum(p.numel() for p in self.clip.visual_projection.parameters())
        proj_trainable = sum(p.numel() for p in self.clip.visual_projection.parameters() if p.requires_grad)
        status = "🔥 TRAINABLE" if proj_trainable > 0 else "❄️  FROZEN"
        print(f"   Vision Projection: {proj_trainable:>10,} / {proj_params:>10,}  {status}")
        trainable_total += proj_trainable
        
        scale_trainable = 1 if self.clip.logit_scale.requires_grad else 0
        status = "🔥 TRAINABLE" if scale_trainable > 0 else "❄️  FROZEN"
        print(f"   logit_scale:       {scale_trainable:>10,} / {1:>10,}  {status}")
        trainable_total += scale_trainable
        
        vit_params = sum(p.numel() for p in self.clip.vision_model.parameters())
        text_params = sum(p.numel() for p in self.clip.text_model.parameters())
        text_proj_params = sum(p.numel() for p in self.clip.text_projection.parameters())
        print(f"   ViT Encoder:       {0:>10,} / {vit_params:>10,}  ❄️  FROZEN (永遠)")
        print(f"   Text Encoder:      {0:>10,} / {text_params:>10,}  ❄️  FROZEN (永遠)")
        print(f"   Text Projection:   {0:>10,} / {text_proj_params:>10,}  ❄️  FROZEN (永遠)")
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   ─────────────────────────────────────────────────")
        print(f"   Total Trainable:   {trainable_total:>10,} / {total_params:>10,}  ({trainable_total/total_params*100:.4f}%)")
    
    def encode_text_prompt(self, domain_id):
        """編碼單個 domain 的 learnable prompt（相容新舊版 transformers）"""
        prompt = self.domain_prompts[domain_id]
        P = self.prompt_len
        used = 1 + P + 1
        pad_len = self.ctx_len - used
        
        sot = self.sot_embedding
        eot = self.eot_embedding
        pad = torch.zeros(pad_len, self.hidden_dim, device=self.device, dtype=prompt.dtype)
        
        token_emb = torch.cat([sot, prompt, eot, pad], dim=0).unsqueeze(0)
        hidden_states = token_emb + self.position_embedding.unsqueeze(0).to(token_emb.dtype)
        
        # Causal mask: [1, 1, seq_len, seq_len]
        causal_mask = torch.triu(
            torch.full((self.ctx_len, self.ctx_len), float('-inf'), device=self.device, dtype=hidden_states.dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        for layer in self.clip.text_model.encoder.layers:
            if self._has_causal_param:
                # 舊版 transformers: 有 causal_attention_mask 參數
                layer_out = layer(
                    hidden_states,
                    attention_mask=None,
                    causal_attention_mask=causal_mask,
                )
            else:
                # 新版 transformers: 只有 attention_mask
                layer_out = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                )
            hidden_states = layer_out[0]
        
        hidden_states = self.clip.text_model.final_layer_norm(hidden_states)
        pooled = hidden_states[:, 1 + P, :]
        text_feat = self.clip.text_projection(pooled)
        
        return text_feat
    
    def get_text_prototypes(self):
        """取得所有 domain 的 text prototypes"""
        protos = []
        for i in range(self.num_domains):
            proto = self.encode_text_prompt(i).squeeze(0)
            protos.append(proto)
        protos = torch.stack(protos, dim=0)
        return F.normalize(protos, dim=-1)
    
    def encode_image(self, pixel_values):       # xie, pixel_values, [batch, 3, img_h, img_w]
        """編碼圖片"""
         # xie, image encoder
         #  257 = 1 (cls token) + 256 (patch tokens)
         #  256 = (224 /14) x (224/14) = 16 x 16
         #  1024 = Vit-L/14 本身的hidden_dim
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output           # xie, [32, 1024]
        image_feat = self.clip.visual_projection(pooled)        # xie, projection [1024,768], image_feat [32,768]
        img_norm = F.normalize(image_feat, dim=-1)              # xie, norm用來做contrastive
        img_raw = image_feat
        return img_norm, img_raw
    
    def forward(self, pixel_values):
        """完整 forward pass"""
        img_norm, img_raw = self.encode_image(pixel_values)
        text_protos = self.get_text_prototypes()
        logit_scale = self.clip.logit_scale.exp().clamp(max=100.0)
        return img_norm, img_raw, text_protos, logit_scale


# ============================================================
# Freeze / Unfreeze Utilities
# ============================================================

def freeze_params(module_or_params):
    """凍結參數"""
    if hasattr(module_or_params, 'parameters'):
        params = module_or_params.parameters()
    else:
        params = module_or_params
    for p in params:
        p.requires_grad = False


def unfreeze_params(module_or_params):
    """解凍參數"""
    if hasattr(module_or_params, 'parameters'):
        params = module_or_params.parameters()
    else:
        params = module_or_params
    for p in params:
        p.requires_grad = True


def setup_stage1(model):
    """設定 Stage 1: 只訓練 Domain Classifier"""
    print("\n" + "=" * 60)
    print("🔧 Setting up Stage 1: Train Domain Classifier")
    print("=" * 60)
    
    # xie, object attribute 「requires_grad = False」
    freeze_params(model.clip.vision_model)
    freeze_params(model.clip.text_model)
    freeze_params(model.clip.text_projection)
    
    freeze_params(model.domain_prompts)
    freeze_params(model.clip.visual_projection)
    model.clip.logit_scale.requires_grad = False
    
    unfreeze_params(model.domain_classifier)
    
    model.print_trainable_params("Stage 1")


def setup_stage2(model):
    """設定 Stage 2: 聯合訓練 Prompts + Projection + Classifier"""
    print("\n" + "=" * 60)
    print("🔧 Setting up Stage 2: Train Prompts + Projection + Classifier")
    print("=" * 60)
    
    freeze_params(model.clip.vision_model)
    freeze_params(model.clip.text_model)
    freeze_params(model.clip.text_projection)
    model.clip.logit_scale.requires_grad = False
    
    # 解凍三組參數：prompts, projection, classifier
    unfreeze_params(model.domain_prompts)
    unfreeze_params(model.clip.visual_projection)
    unfreeze_params(model.domain_classifier)
    
    model.print_trainable_params("Stage 2")


# ============================================================
# Loss Functions
# ============================================================

def contrastive_loss(image_features, text_prototypes, domain_labels, logit_scale):
    """
    CLIP 風格雙向對比損失
    """
    # Similarity [B, num_domains]
    logits = logit_scale * (image_features @ text_prototypes.T)
    
    # i2t: image 預測 domain
    """
    xie, i2t的原因, 是因為matrix, 一張image對應多個text prototype

                text_proto_0  text_proto_1  text_proto_2  ...
        img_0 [    0.8,          0.1,          0.05,     ...]
        img_1 [    0.2,          0.7,          0.1,      ...]
        img_2 [    0.1,          0.05,         0.9,      ...]

    """
    loss_i2t = F.cross_entropy(logits, domain_labels)
    
    # t2i: domain prototype 預測哪些 image 屬於它
    # xie, 然後做了"轉置"自然就變成t2i
    logits_t2i = logits.T  # [num_domains, B]
    
    loss_t2i = torch.tensor(0.0, device=image_features.device)
    
    for d in range(text_prototypes.shape[0]):
        mask = (domain_labels == d)
        if mask.sum() == 0:
            continue
        pos_logits = logits_t2i[d][mask]
        all_logits = logits_t2i[d]
        # Multi-positive NCE
        loss_t2i = loss_t2i - (torch.logsumexp(pos_logits, 0) - torch.logsumexp(all_logits, 0))
    
    loss_t2i = loss_t2i / text_prototypes.shape[0]
    
    return (loss_i2t + loss_t2i) / 2, logits


def classification_loss(image_features, domain_labels, classifier):
    """Domain classification loss"""
    logits = classifier(image_features)
    loss = F.cross_entropy(logits, domain_labels)           # xie, pytorch的F.cross_entropy API它內部會先對參數做softmax才會去算loss
    return loss, logits


# ============================================================
# Training Functions
# ============================================================

def train_stage1_epoch(model, dataloader, optimizer, scaler, args, epoch):
    """Stage 1 訓練一個 epoch: 只訓練 Domain Classifier"""
    # xie, train(), 使用dropout, ie.在trainning的時候，隨機捨棄神經元，每次都會用不同的排列組合在學習，降低依賴特定神經元的風險
    model.train()
    # xie, eval(), 不使用dropout
    # xie, 所以train、eval與backward甚麼的無關，backward與否是取決於「被放進optimizer的參數」，而optimizer的參數條件往往來源於，requires_grad的條件判斷；所以requires_grad=True如果不放進optimizer一樣不會被更新，但實務上不常這樣做，除非debugg
    model.clip.text_model.eval()
    model.clip.vision_model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0
    
    # xie, 產生進度條的module, 這邊是constructor, new object with dataloader => 所以才會看到每個epoch有多少個batch size，然後進度條
    pbar = tqdm(dataloader, desc=f"[Stage1] Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(model.device)
        domain_labels = batch["domain_id"].to(model.device)
        
        # xie, grad是累加的, 所以進入下一個batch的時候需要清空
        optimizer.zero_grad()
        
        # xie, python with對應到java的try
        # xie, 然後這邊的「torch.amp.autocast(...)」 => 這邊是說，進入try catch的時候，將float切換至16bit加速運算，然後!!由於是try catch，所以結束之後又恢復回float32
        # xie, 所以在這邊stage1的時候，amp只影響，encode & classification loss的type scale
        with torch.amp.autocast('cuda', enabled=args.amp):
            img_norm, img_raw = model.encode_image(pixel_values)
            loss, logits = classification_loss(img_raw, domain_labels, model.domain_classifier)
        
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 5:
                print(f"\n⚠️ [Step {step}] Loss is nan/inf! Skipping...")
            continue
        
        # xie, 以loss作為出發點, 沿著計算圖往回走, 計算每個requires_grad=True參數的梯度 => 然後optimizer.step()會再用這些grad更新參數
        scaler.scale(loss).backward()
        
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = (preds == domain_labels).sum().item()
        
        batch_size = len(domain_labels)
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size
        
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct/batch_size*100:.1f}%"
        )
        
        if HAS_WANDB and not args.no_wandb and step % args.log_step == 0 and step > 0:
            wandb.log({
                "stage1/step_loss": loss.item(),
                "stage1/step_acc": correct / batch_size * 100,
                "stage1/global_step": step + (epoch - 1) * len(dataloader),
            })
    
    if total_samples == 0:
        print(f"\n❌ [Epoch {epoch}] All {nan_count} batches had nan/inf loss!")
        return {"loss": float('nan'), "acc": 0.0}
    
    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples * 100,
    }


def train_stage2_epoch(model, dataloader, optimizer, scaler, args, epoch):
    """
    Stage 2 訓練一個 epoch: 聯合訓練 Prompts + Projection + Classifier
    Loss = contrastive_loss + lambda_cls * classification_loss
    
    梯度流向：
      - domain_prompts:      ← contrastive loss only (計算圖自然分離)
      - visual_projection:   ← contrastive loss + cls loss (共享參數，兩邊協商)
      - domain_classifier:   ← cls loss only (計算圖自然分離)
    """
    model.train()
    model.clip.text_model.eval()
    model.clip.vision_model.eval()
    
    total_loss = 0.0
    total_loss_con = 0.0
    total_loss_cls = 0.0
    total_correct_con = 0
    total_correct_cls = 0
    total_samples = 0
    nan_count = 0
    
    pbar = tqdm(dataloader, desc=f"[Stage2] Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(model.device)
        domain_labels = batch["domain_id"].to(model.device)
        
        # MixUp (只在 Stage 2 使用)
        if args.mixup_alpha > 0:
            pixel_values, domain_labels = mixup_within_domain(
                pixel_values, domain_labels, alpha=args.mixup_alpha
            )
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=args.amp):
            img_norm, img_raw, text_protos, logit_scale = model(pixel_values)           # xie, model() equal to forward()
            
            # 兩個 loss
            loss_con, logits_con = contrastive_loss(img_norm, text_protos, domain_labels, logit_scale)
            loss_cls, logits_cls = classification_loss(img_raw, domain_labels, model.domain_classifier)
            
            # 混合 loss：一次 backward，梯度由計算圖自然分配
            # xie, 這邊要去了解甚麼是計算圖 => 但結論就是，縱使混和了各自的loss也只會找到各自的貢獻者，不會混淆
            loss = loss_con + args.lambda_cls * loss_cls
        
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 5:
                print(f"\n⚠️ [Step {step}] Loss is nan/inf! Skipping...")
                print(f"   loss_con: {loss_con.item():.4f}, loss_cls: {loss_cls.item():.4f}")
                print(f"   logit_scale: {logit_scale.item():.4f}")
            continue
        
        # xie, loss依據計算圖，找到requires_grad=True參數，並計算他們各自的gradient，等待後續optimizer.step()時更新parameters
        scaler.scale(loss).backward()
        
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            preds_con = logits_con.argmax(dim=1)
            preds_cls = logits_cls.argmax(dim=1)
            correct_con = (preds_con == domain_labels).sum().item()
            correct_cls = (preds_cls == domain_labels).sum().item()
        
        batch_size = len(domain_labels)
        total_loss += loss.item() * batch_size
        total_loss_con += loss_con.item() * batch_size
        total_loss_cls += loss_cls.item() * batch_size
        total_correct_con += correct_con
        total_correct_cls += correct_cls
        total_samples += batch_size
        
        pbar.set_postfix(
            L=f"{loss.item():.3f}",
            con=f"{loss_con.item():.3f}",
            cls=f"{loss_cls.item():.3f}",
            acc_c=f"{correct_con/batch_size*100:.0f}%",
            acc_d=f"{correct_cls/batch_size*100:.0f}%",
            s=f"{logit_scale.item():.1f}"
        )
        
        if HAS_WANDB and not args.no_wandb and step % args.log_step == 0 and step > 0:
            wandb.log({
                "stage2/step_loss": loss.item(),
                "stage2/step_loss_con": loss_con.item(),
                "stage2/step_loss_cls": loss_cls.item(),
                "stage2/step_acc_con": correct_con / batch_size * 100,
                "stage2/step_acc_cls": correct_cls / batch_size * 100,
                "stage2/logit_scale": logit_scale.item(),
                "stage2/global_step": step + (epoch - 1) * len(dataloader),
            })
    
    if total_samples == 0:
        print(f"\n❌ [Epoch {epoch}] All {nan_count} batches had nan/inf loss!")
        return {"loss": float('nan'), "loss_con": float('nan'), "loss_cls": float('nan'),
                "acc_con": 0.0, "acc_cls": 0.0}
    
    if nan_count > 0:
        print(f"\n⚠️ [Epoch {epoch}] {nan_count}/{len(dataloader)} batches had nan/inf loss")
    
    return {
        "loss": total_loss / total_samples,
        "loss_con": total_loss_con / total_samples,
        "loss_cls": total_loss_cls / total_samples,
        "acc_con": total_correct_con / total_samples * 100,
        "acc_cls": total_correct_cls / total_samples * 100,
    }


# ============================================================
# Evaluation Functions
# ============================================================

@torch.no_grad()
def evaluate_stage1(model, dataloader, args):
    """Stage 1 評估: Classifier accuracy"""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Eval [Classifier]"):
        pixel_values = batch["pixel_values"].to(model.device)
        domain_labels = batch["domain_id"].to(model.device)
        
        with torch.amp.autocast('cuda', enabled=args.amp):
            img_norm, img_raw = model.encode_image(pixel_values)
            loss, logits = classification_loss(img_raw, domain_labels, model.domain_classifier)
        
        preds = logits.argmax(dim=1)
        
        batch_size = len(domain_labels)
        total_loss += loss.item() * batch_size
        total_correct += (preds == domain_labels).sum().item()
        total_samples += batch_size
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(domain_labels.cpu().tolist())
    
    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples * 100,
        "preds": all_preds,
        "labels": all_labels,
    }


@torch.no_grad()
def evaluate_stage2(model, dataloader, args):
    """Stage 2 評估: Contrastive + Classifier accuracy"""
    model.eval()
    
    total_loss = 0.0
    total_loss_con = 0.0
    total_loss_cls = 0.0
    total_correct_con = 0
    total_correct_cls = 0
    total_samples = 0
    all_preds_con = []
    all_preds_cls = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Eval [Stage2]"):
        pixel_values = batch["pixel_values"].to(model.device)
        domain_labels = batch["domain_id"].to(model.device)
        
        with torch.amp.autocast('cuda', enabled=args.amp):
            img_norm, img_raw, text_protos, logit_scale = model(pixel_values)
            loss_con, logits_con = contrastive_loss(img_norm, text_protos, domain_labels, logit_scale)
            loss_cls, logits_cls = classification_loss(img_raw, domain_labels, model.domain_classifier)
            loss = loss_con + args.lambda_cls * loss_cls
        
        preds_con = logits_con.argmax(dim=1)
        preds_cls = logits_cls.argmax(dim=1)
        
        batch_size = len(domain_labels)
        total_loss += loss.item() * batch_size
        total_loss_con += loss_con.item() * batch_size
        total_loss_cls += loss_cls.item() * batch_size
        total_correct_con += (preds_con == domain_labels).sum().item()
        total_correct_cls += (preds_cls == domain_labels).sum().item()
        total_samples += batch_size
        
        all_preds_con.extend(preds_con.cpu().tolist())
        all_preds_cls.extend(preds_cls.cpu().tolist())
        all_labels.extend(domain_labels.cpu().tolist())
    
    return {
        "loss": total_loss / total_samples,
        "loss_con": total_loss_con / total_samples,
        "loss_cls": total_loss_cls / total_samples,
        "acc_con": total_correct_con / total_samples * 100,
        "acc_cls": total_correct_cls / total_samples * 100,
        "preds_con": all_preds_con,
        "preds_cls": all_preds_cls,
        "labels": all_labels,
    }


@torch.no_grad()
def evaluate_both(model, dataloader, args):
    """最終評估（與 evaluate_stage2 相同，保留向後相容）"""
    return evaluate_stage2(model, dataloader, args)


# ============================================================
# Utilities
# ============================================================

def compute_confusion_matrix(preds, labels, num_classes):
    """計算 confusion matrix"""
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for p, l in zip(preds, labels):
        matrix[l, p] += 1
    return matrix


def print_confusion_matrix(conf_matrix, domain_names, title="Confusion Matrix"):
    """印出 confusion matrix"""
    num_domains = len(domain_names)
    
    print(f"\n📊 {title}:")
    print("   " + " " * 10 + "".join([f"{d[:8]:>10}" for d in domain_names]) + "  (predicted)")
    
    for i, name in enumerate(domain_names):
        row = f"   {name[:8]:>8}  "
        for j in range(num_domains):
            row += f"{conf_matrix[i, j].item():>10}"
        print(row)
    print("   (actual)")


def print_per_domain_accuracy(preds, labels, domain_names, title="Per-domain Accuracy"):
    """印出每個 domain 的準確率"""
    num_domains = len(domain_names)
    conf_matrix = compute_confusion_matrix(preds, labels, num_domains)
    
    print(f"\n📊 {title}:")
    print(f"   {'Domain':<12} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print("   " + "-" * 46)
    
    total_correct = 0
    total_samples = 0
    
    for i, name in enumerate(domain_names):
        total = conf_matrix[i].sum().item()
        correct = conf_matrix[i, i].item()
        acc = correct / total * 100 if total > 0 else 0
        print(f"   {name:<12} {correct:>10} {total:>10} {acc:>11.2f}%")
        total_correct += correct
        total_samples += total
    
    print("   " + "-" * 46)
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"   {'Overall':<12} {total_correct:>10} {total_samples:>10} {overall_acc:>11.2f}%")


@torch.no_grad()
def visualize_prompt_similarity(model, domain_names):
    """視覺化不同 domain prompt 之間的相似度"""
    model.eval()
    
    protos = model.get_text_prototypes()
    sim_matrix = protos @ protos.T
    
    print("\n" + "=" * 60)
    print("📊 Domain Prompt Similarity Matrix (Cosine)")
    print("=" * 60)
    
    header = "          " + "".join([f"{d[:8]:>10}" for d in domain_names])
    print(header)
    
    for i, name in enumerate(domain_names):
        row = f"{name[:8]:>8}  "
        for j in range(len(domain_names)):
            sim = sim_matrix[i, j].item()
            if i == j:
                row += f"{'1.000':>10}"
            else:
                row += f"{sim:>10.3f}"
        print(row)
    
    num_domains = len(domain_names)
    if num_domains > 1:
        mask = ~torch.eye(num_domains, dtype=torch.bool, device=sim_matrix.device)
        avg_off_diag = sim_matrix[mask].mean().item()
        min_off_diag = sim_matrix[mask].min().item()
        max_off_diag = sim_matrix[mask].max().item()
        
        print(f"\n   Off-diagonal similarity:")
        print(f"     Average: {avg_off_diag:.4f}")
        print(f"     Min:     {min_off_diag:.4f}")
        print(f"     Max:     {max_off_diag:.4f}")
        print(f"   (Lower average = more distinct prompts)")
    
    return sim_matrix


def list_available_domains(data_dir):
    """列出可用的 domain"""
    print(f"\n📂 Available domains in {data_dir}:")
    print("=" * 60)
    
    if not os.path.isdir(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return
    
    all_domains = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ])
    
    if len(all_domains) == 0:
        print("   No domain directories found.")
        return
    
    total_images = 0
    for i, domain_name in enumerate(all_domains):
        domain_dir = os.path.join(data_dir, domain_name)
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(glob.glob(os.path.join(domain_dir, '**', ext), recursive=True))
        
        print(f"   {i+1:3d}. {domain_name:<20} ({len(images):>6} images)")
        total_images += len(images)
    
    print("=" * 60)
    print(f"   Total: {len(all_domains)} domains, {total_images} images")


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, stage, args, domain_names, save_path, extra=None):
    """儲存 checkpoint"""
    ckpt = {
        "stage": stage,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc,
        "domain_names": domain_names,
        "model_name": model.model_name,
        "prompt_len": model.prompt_len,
        "num_domains": model.num_domains,
        "args": vars(args),
    }
    if extra is not None:
        ckpt.update(extra)
    torch.save(ckpt, save_path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CLIP Domain Prompt Tuning - Two Stage")
    
    # Data
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--domains", type=str, default=None)
    parser.add_argument("--list_domains", action="store_true")
    parser.add_argument("--augment", action="store_true")
    
    # Model
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--prompt_len", type=int, default=4)
    
    # Training - Stage 1
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage1_lr", type=float, default=1e-4)
    
    # Training - Stage 2
    parser.add_argument("--stage2_epochs", type=int, default=15)
    parser.add_argument("--stage2_lr_prompt", type=float, default=1e-3)
    parser.add_argument("--stage2_lr_proj", type=float, default=1e-5)
    parser.add_argument("--stage2_lr_cls", type=float, default=1e-4,
                        help="Stage 2 classifier learning rate (通常沿用 stage1_lr)")
    parser.add_argument("--lambda_cls", type=float, default=0.3,
                        help="Classification loss weight in Stage 2 (0.1~0.5 recommended)")
    
    # Training - Common
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    
    # Save & Log
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default="clip_two_stage")
    parser.add_argument("--log_step", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path (e.g., ./checkpoints/clip_two_stage_latest.pt)")
    
    args = parser.parse_args()
    
    # List Domains Mode
    if args.list_domains:
        if args.data_dir is None:
            print("❌ Error: --data_dir is required for --list_domains")
            return
        list_available_domains(args.data_dir)
        return
    
    # 檢查必要參數
    # xie, 若空 ie. None就抱錯 & stop code
    if args.data_dir is None or args.domains is None:
        print("❌ Error: --data_dir and --domains are required for training")
        return
    
    # Setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # xie, eg. 「real,painting,...」
    domain_names = [d.strip() for d in args.domains.split(",")]
    num_domains = len(domain_names)
    
    if num_domains < 2:
        print("❌ Error: At least 2 domains are required")
        return
    
    # xie, create save dir if not exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print Config
    # xie, vscode support emoji, just see it as a string letter, directly use it
    print("\n" + "=" * 70)
    print("🚀 CLIP Domain Prompt Tuning - Two Stage (Joint Training v2)")
    print("=" * 70)
    print(f"   Device:          {device}")
    print(f"   Model:           {args.model_name}")
    print(f"   Domains ({num_domains}):     {domain_names}")
    print(f"   Prompt length:   {args.prompt_len}")
    print(f"   Batch size:      {args.batch_size}")
    print(f"   Stage 1 epochs:  {args.stage1_epochs} (lr={args.stage1_lr})")
    print(f"   Stage 2 epochs:  {args.stage2_epochs} (lr_prompt={args.stage2_lr_prompt}, lr_proj={args.stage2_lr_proj}, lr_cls={args.stage2_lr_cls})")
    print(f"   Lambda cls:      {args.lambda_cls}")
    print(f"   Augmentation:    {'ON 🔥' if args.augment else 'OFF'}")
    print(f"   MixUp:           {'ON (alpha=' + str(args.mixup_alpha) + ', Stage2 only) 🔀' if args.mixup_alpha > 0 else 'OFF'}")
    print(f"   AMP:             {'ON ⚡' if args.amp else 'OFF'}")
    print(f"   Grad clip:       {args.grad_clip}")
    if args.resume:
        print(f"   Resume from:     {args.resume}")
    
    # WandB
    if HAS_WANDB and not args.no_wandb:
        wandb.init(
            project="clip-domain-prompt",
            name=args.run_name,
            config=vars(args),
            tags=["two-stage-joint", f"{num_domains}-domains"]
        )
        print(f"   WandB:           ON ✅")
    else:
        print(f"   WandB:           OFF")
    
    # Model
    model = CLIPDomainPromptTuner(
        model_name=args.model_name,
        prompt_len=args.prompt_len,
        num_domains=num_domains,
        device=device,
    )
    
    # Dataset
    print("\n" + "=" * 70)
    print("📂 Loading Dataset")
    print("=" * 70)
    
    # xie, whole data, assume 'm'
    train_dataset = MultiDomainDataset(
        data_dir=args.data_dir,
        processor=model.processor,
        domain_names=domain_names,
        augment=args.augment,
        verbose=True
    )
    
    val_dataset = MultiDomainDataset(
        data_dir=args.data_dir,
        processor=model.processor,
        domain_names=domain_names,
        augment=False,
        domain_images=train_dataset.domain_images,
        verbose=False
    )
    
    # xie, pick up 'n', ie. batch size images from dataset => 1 epoch mean checkout once datset => therefore, 1 epoch equals to m/n batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\n   Train samples/epoch: {len(train_dataset)}")
    print(f"   Val samples/epoch:   {len(val_dataset)}")
    print(f"   Train batches/epoch: {len(train_loader)}")
    print(f"   Val batches/epoch:   {len(val_loader)}")
    
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    
    # ================================================================
    # RESUME LOGIC
    # ================================================================
    resume_stage = 0   # 0 = 從頭開始
    resume_epoch = 0
    best_s1_acc = 0.0
    best_s2_acc_con = 0.0
    best_s2_acc_cls = 0.0
    
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            print(f"❌ Resume checkpoint not found: {args.resume}")
            return
        
        print(f"\n🔄 Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        # 載入模型權重
        model.load_state_dict(ckpt["model_state_dict"])
        
        resume_stage = ckpt["stage"]
        resume_epoch = ckpt["epoch"]
        best_s1_acc = ckpt.get("best_s1_acc", ckpt.get("best_acc", 0.0))
        best_s2_acc_con = ckpt.get("best_s2_acc_con", 0.0)
        best_s2_acc_cls = ckpt.get("best_s2_acc_cls", 0.0)
        
        # 如果是 Stage 1 checkpoint，best_acc 就是 s1_acc
        if resume_stage == 1:
            best_s1_acc = ckpt.get("best_acc", best_s1_acc)
        
        print(f"   Resumed stage: {resume_stage}, epoch: {resume_epoch}")
        print(f"   best_s1_acc: {best_s1_acc:.2f}%")
        if resume_stage == 2:
            print(f"   best_s2_acc_con: {best_s2_acc_con:.2f}%")
            print(f"   best_s2_acc_cls: {best_s2_acc_cls:.2f}%")
    
    # ================================================================
    # STAGE 1: Train Domain Classifier
    # ================================================================
    if resume_stage < 2:
        # 需要跑 Stage 1（從頭或 resume Stage 1 中途）
        s1_start_epoch = (resume_epoch + 1) if resume_stage == 1 else 1
        
        if s1_start_epoch <= args.stage1_epochs:
            print("\n")
            print("=" * 70)
            print("📌 STAGE 1: Training Domain Classifier")
            print(f"   Epochs: {s1_start_epoch}~{args.stage1_epochs}")
            print(f"   Learning rate: {args.stage1_lr}")
            print("=" * 70)
            
            setup_stage1(model)
            
            # xie, 通常model訓練的步驟：
            # 1) model = MyModel(), 建立model
            # 2) setup_stage(model), 設定模型參數
            # 3) optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            # 4) scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
            #
            # (上面都屬於建立object)
            # ================================== 
            #
            # 5) for epoch in range(epochs):        # training
            #       train(model,optimizer)
            #       scheduler.step()
            # xie, optimizer的parameters()只訓練requires_grad == True
            optimizer_s1 = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.stage1_lr,
                weight_decay=args.weight_decay
            )

            # xie, scheduler自動調整LR直到LR變成0
            scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_s1, T_max=args.stage1_epochs
            )
            
            # Resume optimizer/scheduler state
            if resume_stage == 1 and "optimizer_state_dict" in ckpt:
                optimizer_s1.load_state_dict(ckpt["optimizer_state_dict"])
                if ckpt.get("scheduler_state_dict") is not None:
                    scheduler_s1.load_state_dict(ckpt["scheduler_state_dict"])
                print(f"   ✅ Resumed Stage 1 optimizer & scheduler")
            
            # xie, 這裡才是開始trainning
            for epoch in range(s1_start_epoch, args.stage1_epochs + 1):
                train_metrics = train_stage1_epoch(
                    model, train_loader, optimizer_s1, scaler, args, epoch
                )
                scheduler_s1.step()
                
                val_metrics = evaluate_stage1(model, val_loader, args)
                
                print(f"\n📈 Stage1 Epoch {epoch}/{args.stage1_epochs}")
                print(f"   Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%")
                print(f"   Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}%")
                print(f"   LR: {scheduler_s1.get_last_lr()[0]:.2e}")
                
                if HAS_WANDB and not args.no_wandb:
                    wandb.log({
                        "stage1/epoch": epoch,
                        "stage1/train_loss": train_metrics['loss'],
                        "stage1/train_acc": train_metrics['acc'],
                        "stage1/val_loss": val_metrics['loss'],
                        "stage1/val_acc": val_metrics['acc'],
                        "stage1/lr": scheduler_s1.get_last_lr()[0],
                    })
                
                if val_metrics['acc'] > best_s1_acc:
                    best_s1_acc = val_metrics['acc']
                    print(f"   ✅ New best Stage1 accuracy: {best_s1_acc:.2f}%")
                    
                    save_checkpoint(
                        model, optimizer_s1, scheduler_s1, epoch, best_s1_acc,
                        stage=1, args=args, domain_names=domain_names,
                        save_path=os.path.join(args.save_dir, f"{args.run_name}_stage1_best.pt"),
                        extra={"best_s1_acc": best_s1_acc}
                    )
                
                # Stage 1 也存 latest（方便 resume）
                save_checkpoint(
                    model, optimizer_s1, scheduler_s1, epoch, best_s1_acc,
                    stage=1, args=args, domain_names=domain_names,
                    save_path=os.path.join(args.save_dir, f"{args.run_name}_latest.pt"),
                    extra={"best_s1_acc": best_s1_acc}
                )
        
        # Stage 1 完成
        print("\n" + "=" * 70)
        print("✅ Stage 1 Complete!")
        print(f"   Best classifier accuracy: {best_s1_acc:.2f}%")
        print("=" * 70)
        
        s1_final = evaluate_stage1(model, val_loader, args)
        print_confusion_matrix(
            compute_confusion_matrix(s1_final['preds'], s1_final['labels'], num_domains),
            domain_names,
            title="Stage 1 Confusion Matrix (Classifier)"
        )
        print_per_domain_accuracy(
            s1_final['preds'], s1_final['labels'], domain_names,
            title="Stage 1 Per-domain Accuracy"
        )
    else:
        print(f"\n⏭️  Skipping Stage 1 (resumed from Stage 2, best_s1_acc={best_s1_acc:.2f}%)")
    
    # ================================================================
    # STAGE 2: Joint Training - Prompts + Projection + Classifier
    # ================================================================
    s2_start_epoch = (resume_epoch + 1) if resume_stage == 2 else 1
    
    print("\n")
    print("=" * 70)
    print("📌 STAGE 2: Joint Training - Prompts + Projection + Classifier")
    print(f"   Epochs: {s2_start_epoch}~{args.stage2_epochs}")
    print(f"   Learning rates: prompt={args.stage2_lr_prompt}, proj={args.stage2_lr_proj}, cls={args.stage2_lr_cls}")
    print(f"   Loss = contrastive + {args.lambda_cls} * classification")
    if args.mixup_alpha > 0:
        print(f"   MixUp alpha: {args.mixup_alpha}")
    print("=" * 70)
    
    setup_stage2(model)
    
    optimizer_s2 = torch.optim.AdamW([
        {
            'params': list(model.domain_prompts.parameters()),
            'lr': args.stage2_lr_prompt,
            'weight_decay': 0,
            'name': 'prompts'
        },
        {
            'params': list(model.clip.visual_projection.parameters()),
            'lr': args.stage2_lr_proj,
            'weight_decay': args.weight_decay,
            'name': 'vision_projection'
        },
        {
            'params': list(model.domain_classifier.parameters()),
            'lr': args.stage2_lr_cls,
            'weight_decay': args.weight_decay,
            'name': 'domain_classifier'
        },
    ])
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=args.stage2_epochs
    )
    
    # Resume optimizer/scheduler state for Stage 2
    if resume_stage == 2 and "optimizer_state_dict" in ckpt:
        optimizer_s2.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler_s2.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"   ✅ Resumed Stage 2 optimizer & scheduler from epoch {resume_epoch}")
    
    for epoch in range(s2_start_epoch, args.stage2_epochs + 1):
        train_metrics = train_stage2_epoch(
            model, train_loader, optimizer_s2, scaler, args, epoch
        )
        scheduler_s2.step()
        
        val_metrics = evaluate_stage2(model, val_loader, args)
        
        print(f"\n📈 Stage2 Epoch {epoch}/{args.stage2_epochs}")
        print(f"   Train - Loss: {train_metrics['loss']:.4f} (con: {train_metrics['loss_con']:.4f}, cls: {train_metrics['loss_cls']:.4f})")
        print(f"           Acc:  con={train_metrics['acc_con']:.2f}%, cls={train_metrics['acc_cls']:.2f}%")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f} (con: {val_metrics['loss_con']:.4f}, cls: {val_metrics['loss_cls']:.4f})")
        print(f"           Acc:  con={val_metrics['acc_con']:.2f}%, cls={val_metrics['acc_cls']:.2f}%")
        print(f"   LR (prompt): {optimizer_s2.param_groups[0]['lr']:.2e}")
        
        if HAS_WANDB and not args.no_wandb:
            wandb.log({
                "stage2/epoch": epoch,
                "stage2/train_loss": train_metrics['loss'],
                "stage2/train_loss_con": train_metrics['loss_con'],
                "stage2/train_loss_cls": train_metrics['loss_cls'],
                "stage2/train_acc_con": train_metrics['acc_con'],
                "stage2/train_acc_cls": train_metrics['acc_cls'],
                "stage2/val_loss": val_metrics['loss'],
                "stage2/val_loss_con": val_metrics['loss_con'],
                "stage2/val_loss_cls": val_metrics['loss_cls'],
                "stage2/val_acc_con": val_metrics['acc_con'],
                "stage2/val_acc_cls": val_metrics['acc_cls'],
                "stage2/lr_prompt": optimizer_s2.param_groups[0]['lr'],
                "stage2/lr_proj": optimizer_s2.param_groups[1]['lr'],
                "stage2/lr_cls": optimizer_s2.param_groups[2]['lr'],
            })
        
        if epoch % 5 == 0 or epoch == 1:
            sim_matrix = visualize_prompt_similarity(model, domain_names)
            
            if HAS_WANDB and not args.no_wandb:
                mask = ~torch.eye(num_domains, dtype=torch.bool, device=sim_matrix.device)
                avg_off_diag = sim_matrix[mask].mean().item()
                wandb.log({"stage2/prompt_sim_avg": avg_off_diag})
        
        s2_extra = {
            "best_s1_acc": best_s1_acc,
            "best_s2_acc_con": best_s2_acc_con,
            "best_s2_acc_cls": best_s2_acc_cls,
        }
        
        # 以 contrastive accuracy 為主要指標存 best checkpoint
        if val_metrics['acc_con'] > best_s2_acc_con:
            best_s2_acc_con = val_metrics['acc_con']
            s2_extra["best_s2_acc_con"] = best_s2_acc_con
            print(f"   ✅ New best Stage2 contrastive acc: {best_s2_acc_con:.2f}%")
            
            save_checkpoint(
                model, optimizer_s2, scheduler_s2, epoch, best_s2_acc_con,
                stage=2, args=args, domain_names=domain_names,
                save_path=os.path.join(args.save_dir, f"{args.run_name}_stage2_best.pt"),
                extra=s2_extra
            )
        
        if val_metrics['acc_cls'] > best_s2_acc_cls:
            best_s2_acc_cls = val_metrics['acc_cls']
            s2_extra["best_s2_acc_cls"] = best_s2_acc_cls
        
        save_checkpoint(
            model, optimizer_s2, scheduler_s2, epoch, best_s2_acc_con,
            stage=2, args=args, domain_names=domain_names,
            save_path=os.path.join(args.save_dir, f"{args.run_name}_latest.pt"),
            extra=s2_extra
        )
    
    # ================================================================
    # FINAL EVALUATION
    # ================================================================
    print("\n")
    print("=" * 70)
    print("📊 FINAL EVALUATION")
    print("=" * 70)
    
    final_metrics = evaluate_both(model, val_loader, args)
    
    print(f"\n🎯 Final Results:")
    print(f"   Contrastive Accuracy: {final_metrics['acc_con']:.2f}% (best: {best_s2_acc_con:.2f}%)")
    print(f"   Classifier Accuracy:  {final_metrics['acc_cls']:.2f}% (best S1: {best_s1_acc:.2f}%, best S2: {best_s2_acc_cls:.2f}%)")
    
    print_confusion_matrix(
        compute_confusion_matrix(final_metrics['preds_con'], final_metrics['labels'], num_domains),
        domain_names,
        title="Final Confusion Matrix (Contrastive)"
    )
    
    print_confusion_matrix(
        compute_confusion_matrix(final_metrics['preds_cls'], final_metrics['labels'], num_domains),
        domain_names,
        title="Final Confusion Matrix (Classifier)"
    )
    
    print(f"\n📊 Per-domain Accuracy Comparison:")
    print(f"   {'Domain':<12} {'Contrastive':>14} {'Classifier':>14}")
    print("   " + "-" * 42)
    
    conf_con = compute_confusion_matrix(final_metrics['preds_con'], final_metrics['labels'], num_domains)
    conf_cls = compute_confusion_matrix(final_metrics['preds_cls'], final_metrics['labels'], num_domains)
    
    for i, name in enumerate(domain_names):
        total = conf_con[i].sum().item()
        correct_con = conf_con[i, i].item()
        correct_cls = conf_cls[i, i].item()
        
        acc_con = correct_con / total * 100 if total > 0 else 0
        acc_cls = correct_cls / total * 100 if total > 0 else 0
        print(f"   {name:<12} {acc_con:>13.2f}% {acc_cls:>13.2f}%")
    
    visualize_prompt_similarity(model, domain_names)
    
    # Summary
    print("\n")
    print("=" * 70)
    print("🏆 TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Stage 1 (Classifier) best acc:    {best_s1_acc:.2f}%")
    print(f"   Stage 2 (Contrastive) best acc:   {best_s2_acc_con:.2f}%")
    print(f"   Stage 2 (Classifier) best acc:    {best_s2_acc_cls:.2f}%")
    print(f"\n💾 Saved Checkpoints:")
    print(f"   Stage1 Best: {os.path.join(args.save_dir, args.run_name + '_stage1_best.pt')}")
    print(f"   Stage2 Best: {os.path.join(args.save_dir, args.run_name + '_stage2_best.pt')}")
    print(f"   Latest:      {os.path.join(args.save_dir, args.run_name + '_latest.pt')}")
    
    if HAS_WANDB and not args.no_wandb:
        wandb.log({
            "final/acc_con": final_metrics['acc_con'],
            "final/acc_cls": final_metrics['acc_cls'],
            "final/best_s1_acc": best_s1_acc,
            "final/best_s2_acc_con": best_s2_acc_con,
            "final/best_s2_acc_cls": best_s2_acc_cls,
        })
        wandb.finish()
    
    print("\n✅ Done!")


# ============================================================
# Quick Test
# ============================================================

def test_gradient():
    """測試梯度流動"""
    print("\n" + "=" * 70)
    print("🧪 Gradient Flow Test (Two Stage - Joint Training v2)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_domains = 4
    
    print(f"\n📦 Creating model on {device}...")
    model = CLIPDomainPromptTuner(
        model_name="openai/clip-vit-base-patch32",
        prompt_len=4,
        num_domains=num_domains,
        device=device,
    )
    
    batch_size = 8
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device)
    domain_labels = torch.randint(0, num_domains, (batch_size,), device=device)
    
    print(f"\n📦 Fake batch:")
    print(f"   Batch size: {batch_size}")
    print(f"   Labels: {domain_labels.tolist()}")
    
    # Test Stage 1
    print("\n" + "=" * 70)
    print("📌 Testing Stage 1 (Classifier only)")
    print("=" * 70)
    
    setup_stage1(model)
    
    img_norm, img_raw = model.encode_image(pixel_values)
    loss_cls, logits_cls = classification_loss(img_raw, domain_labels, model.domain_classifier)
    
    print(f"\n📊 Classification Loss: {loss_cls.item():.4f}")
    
    loss_cls.backward()
    
    print("\n🔍 Gradient Check (Stage 1):")
    
    cls_grads = [p.grad for p in model.domain_classifier.parameters() if p.grad is not None]
    print(f"   [Classifier] {len(cls_grads)} params have gradients {'✅' if len(cls_grads) > 0 else '❌'}")
    
    prompt_grads = [p.grad for p in model.domain_prompts if p.grad is not None]
    print(f"   [Prompts] {len(prompt_grads)} params have gradients {'✅' if len(prompt_grads) == 0 else '❌'} (should be 0)")
    
    proj_grads = [p.grad for p in model.clip.visual_projection.parameters() if p.grad is not None]
    print(f"   [Projection] {len(proj_grads)} params have gradients {'✅' if len(proj_grads) == 0 else '❌'} (should be 0)")
    
    model.zero_grad()
    
    # Test Stage 2 (Joint Training)
    print("\n" + "=" * 70)
    print("📌 Testing Stage 2 (Joint: Prompts + Projection + Classifier)")
    print("=" * 70)
    
    setup_stage2(model)
    
    img_norm, img_raw, text_protos, logit_scale = model(pixel_values)
    loss_con, logits_con = contrastive_loss(img_norm, text_protos, domain_labels, logit_scale)
    loss_cls, logits_cls = classification_loss(img_raw, domain_labels, model.domain_classifier)
    
    lambda_cls = 0.3
    total_loss = loss_con + lambda_cls * loss_cls
    
    print(f"\n📊 Losses:")
    print(f"   Contrastive:    {loss_con.item():.4f}")
    print(f"   Classification: {loss_cls.item():.4f}")
    print(f"   Total (λ={lambda_cls}): {total_loss.item():.4f}")
    print(f"   logit_scale:    {logit_scale.item():.4f}")
    
    total_loss.backward()
    
    print("\n🔍 Gradient Check (Stage 2 - Joint):")
    
    # Prompts: 應該只收到 contrastive loss 的梯度
    prompt_grads = [(i, p.grad.norm().item()) for i, p in enumerate(model.domain_prompts) if p.grad is not None]
    print(f"   [Prompts] {len(prompt_grads)} params have gradients {'✅' if len(prompt_grads) == num_domains else '❌'}")
    for i, norm in prompt_grads:
        print(f"     prompt[{i}] grad norm: {norm:.6f} (from contrastive loss)")
    
    # Projection: 應該收到兩邊的梯度
    proj_grads = [(name, p.grad.norm().item()) for name, p in model.clip.visual_projection.named_parameters() if p.grad is not None]
    print(f"   [Projection] {len(proj_grads)} params have gradients {'✅' if len(proj_grads) > 0 else '❌'}")
    for name, norm in proj_grads:
        print(f"     {name} grad norm: {norm:.6f} (from BOTH losses)")
    
    # Classifier: 應該只收到 cls loss 的梯度
    cls_grads = [(name, p.grad.norm().item()) for name, p in model.domain_classifier.named_parameters() if p.grad is not None]
    print(f"   [Classifier] {len(cls_grads)} params have gradients {'✅' if len(cls_grads) > 0 else '❌'}")
    for name, norm in cls_grads:
        print(f"     {name} grad norm: {norm:.6f} (from cls loss)")
    
    # logit_scale: 應該沒有梯度
    print(f"   [logit_scale] has gradient: {'❌ unexpected!' if model.clip.logit_scale.grad is not None else '✅ None (frozen)'}")
    
    # ViT: 應該沒有梯度
    vit_grads = [p.grad for p in model.clip.vision_model.parameters() if p.grad is not None]
    print(f"   [ViT] {len(vit_grads)} params have gradients {'✅' if len(vit_grads) == 0 else '❌'} (should be 0)")
    
    # 驗證 projection 確實收到兩邊梯度（對比測試）
    print("\n" + "=" * 70)
    print("🧪 Verifying Projection receives gradients from BOTH losses")
    print("=" * 70)
    
    model.zero_grad()
    
    # 只用 contrastive loss
    img_norm2, img_raw2, text_protos2, logit_scale2 = model(pixel_values)
    loss_con_only, _ = contrastive_loss(img_norm2, text_protos2, domain_labels, logit_scale2)
    loss_con_only.backward()
    
    proj_grad_con_only = {name: p.grad.clone() for name, p in model.clip.visual_projection.named_parameters() if p.grad is not None}
    cls_grad_from_con = {name: p.grad.clone() if p.grad is not None else None for name, p in model.domain_classifier.named_parameters()}
    
    model.zero_grad()
    
    # 只用 cls loss
    img_norm3, img_raw3 = model.encode_image(pixel_values)
    loss_cls_only, _ = classification_loss(img_raw3, domain_labels, model.domain_classifier)
    loss_cls_only.backward()
    
    proj_grad_cls_only = {name: p.grad.clone() for name, p in model.clip.visual_projection.named_parameters() if p.grad is not None}
    
    print(f"\n   Projection gradient sources:")
    for name in proj_grad_con_only:
        con_norm = proj_grad_con_only[name].norm().item()
        cls_norm = proj_grad_cls_only.get(name, torch.zeros(1)).norm().item()
        print(f"     {name}:")
        print(f"       from contrastive: {con_norm:.6f}")
        print(f"       from cls:         {cls_norm:.6f}")
        print(f"       ✅ Both contribute" if con_norm > 0 and cls_norm > 0 else "       ⚠️ One source missing")
    
    print(f"\n   Classifier gradient from contrastive loss:")
    for name, grad in cls_grad_from_con.items():
        has_grad = grad is not None and grad.norm().item() > 0
        print(f"     {name}: {'⚠️ unexpected!' if has_grad else '✅ None (correctly isolated)'}")
    
    # Test MixUp
    print("\n" + "=" * 70)
    print("🧪 MixUp Test")
    print("=" * 70)
    
    mixed_pixels, mixed_labels = mixup_within_domain(pixel_values, domain_labels, alpha=0.4)
    pixel_diff = (mixed_pixels - pixel_values).abs().mean().item()
    
    print(f"   Pixel difference: {pixel_diff:.6f} {'✅' if pixel_diff > 0 else '⚠️'}")
    print(f"   Labels unchanged: {'✅' if (mixed_labels == domain_labels).all() else '❌'}")
    
    visualize_prompt_similarity(model, [f"domain_{i}" for i in range(num_domains)])
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_gradient()
    else:
        main()