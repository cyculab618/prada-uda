"""
CLIP Vision Backbone for USB Semi-Supervised Learning
======================================================
將 CLIP 的 vision_model + visual_projection 包裝成 USB 相容的 backbone。

支援三種模式：
  1. 原始 CLIP pretrain 權重 — frozen encoder（baseline）
  2. two_stage.py fine-tuned 的 checkpoint — frozen encoder（實驗組）
  3. LoRA — encoder 加低秩 adapter，只訓練 adapter + projection + head

用法：
  from clip_backbone import get_clip_net_builder

  # 原始 CLIP（frozen encoder）
  net_builder = get_clip_net_builder(clip_model_name="openai/clip-vit-large-patch14")

  # LoRA（encoder 加 adapter）
  net_builder = get_clip_net_builder(
      clip_model_name="openai/clip-vit-large-patch14",
      use_lora=True, lora_rank=4, lora_alpha=16, lora_dropout=0.1,
  )
"""

import re
import torch
import torch.nn as nn
from transformers import CLIPModel


class CLIPVisionBackbone(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-large-patch14",
        two_stage_checkpoint=None,
        freeze_vision_encoder=True,
        freeze_vision_projection=False,
        num_classes=345,
        use_lora=False,
        lora_rank=4,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=None,
        mlp_head=False,
        mlp_hidden_dim=1024,
        mlp_dropout=0.1,
    ):
        super().__init__()

        print(f"\n{'='*60}")
        print(f"  CLIPVisionBackbone")
        print(f"{'='*60}")
        print(f"  CLIP model: {clip_model_name}")
        print(f"  Two-stage checkpoint: {two_stage_checkpoint or 'None (original pretrain)'}")
        print(f"  Freeze vision encoder: {freeze_vision_encoder}")
        print(f"  Freeze vision projection: {freeze_vision_projection}")
        print(f"  Num classes: {num_classes}")

        # 載入 CLIP
        clip_model = CLIPModel.from_pretrained(clip_model_name)

        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection

        self.embed_dim = clip_model.visual_projection.out_features
        self.hidden_dim = clip_model.vision_model.config.hidden_size
        self.num_features = self.embed_dim

        # Classification head
        if mlp_head:
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden_dim, num_classes),
            )
            head_params = sum(p.numel() for p in self.head.parameters())
            print(f"  Vision hidden dim: {self.hidden_dim}")
            print(f"  Projection output dim (num_features): {self.embed_dim}")
            print(f"  Classification head: MLP({self.embed_dim}→{mlp_hidden_dim}→{num_classes})")
            print(f"  MLP head params: {head_params:,}")
        else:
            self.head = nn.Linear(self.embed_dim, num_classes)
            print(f"  Vision hidden dim: {self.hidden_dim}")
            print(f"  Projection output dim (num_features): {self.embed_dim}")
            print(f"  Classification head: Linear({self.embed_dim}, {num_classes})")

        # 如果有 two_stage checkpoint，載入 fine-tuned 權重
        if two_stage_checkpoint is not None:
            self._load_two_stage_weights(two_stage_checkpoint)

        # ---- LoRA mode ----
        self.use_lora = use_lora
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha, lora_dropout, lora_target_modules)
        elif freeze_vision_encoder:
            # 傳統 frozen mode
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print(f"  ❄️  Vision encoder: FROZEN")
            print(f"  🔥 Visual projection: TRAINABLE")
            print(f"  🔥 Classification head: TRAINABLE")
        else:
            print(f"  🔥 Vision encoder: TRAINABLE (full)")
            print(f"  🔥 Visual projection: TRAINABLE")
            print(f"  🔥 Classification head: TRAINABLE")

        if freeze_vision_projection:
            for param in self.visual_projection.parameters():
                param.requires_grad = False
            print(f"  ❄️  Visual projection: FROZEN (overridden)")

        self._print_param_summary()

    def _apply_lora(self, rank, alpha, dropout, target_modules):
        """在 vision_model 上套用 LoRA"""
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        print(f"\n  === Applying LoRA ===")
        print(f"  Rank: {rank}")
        print(f"  Alpha: {alpha}")
        print(f"  Dropout: {dropout}")
        print(f"  Target modules: {target_modules}")

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
        )

        # 先凍結所有 vision_model 參數
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # 套用 LoRA（會自動把 target modules 的 adapter 設為 trainable）
        self.vision_model = get_peft_model(self.vision_model, lora_config)

        # 確保 projection 和 head 是 trainable
        for param in self.visual_projection.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True

        # 印出 LoRA 資訊
        lora_params = sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)
        print(f"  LoRA adapter params: {lora_params:,}")
        print(f"  ❄️  Vision encoder base: FROZEN")
        print(f"  🔥 LoRA adapters (Q, V): TRAINABLE")
        print(f"  🔥 Visual projection: TRAINABLE")
        print(f"  🔥 Classification head: TRAINABLE")

    def _load_two_stage_weights(self, checkpoint_path):
        """從 two_stage.py 的 checkpoint 載入 vision_model + visual_projection 權重"""
        print(f"\n  Loading two-stage checkpoint: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        vision_state = {}
        projection_state = {}

        for key, value in state_dict.items():
            if key.startswith("clip.vision_model."):
                new_key = key.replace("clip.vision_model.", "vision_model.")
                vision_state[new_key] = value
            elif key.startswith("clip.visual_projection."):
                new_key = key.replace("clip.visual_projection.", "visual_projection.")
                projection_state[new_key] = value

        missing, unexpected = self.load_state_dict(
            {**vision_state, **projection_state},
            strict=False,
        )

        print(f"  ✅ Loaded vision_model: {len(vision_state)} params")
        print(f"  ✅ Loaded visual_projection: {len(projection_state)} params")

        if missing:
            print(f"  ⚠️  Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"      {k}")
        if unexpected:
            print(f"  ⚠️  Unexpected keys: {len(unexpected)}")

        stage = ckpt.get("stage", "?")
        epoch = ckpt.get("epoch", "?")
        domain_names = ckpt.get("domain_names", [])
        print(f"  Checkpoint info: stage={stage}, epoch={epoch}")
        if domain_names:
            print(f"  Domains: {domain_names}")

    def _print_param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        # LoRA 專用統計
        if self.use_lora:
            lora_params = sum(
                p.numel() for n, p in self.vision_model.named_parameters()
                if p.requires_grad
            )
            base_params = sum(
                p.numel() for n, p in self.vision_model.named_parameters()
                if not p.requires_grad
            )
            proj_params = sum(p.numel() for p in self.visual_projection.parameters())
            head_params = sum(p.numel() for p in self.head.parameters())

            print(f"\n  === Parameter Summary (LoRA) ===")
            print(f"  Vision encoder base:  {base_params:>12,} ❄️")
            print(f"  LoRA adapters:        {lora_params:>12,} 🔥")
            print(f"  Visual projection:    {proj_params:>12,} 🔥")
            print(f"  Classification head:  {head_params:>12,} 🔥")
            print(f"  Total:                {total:>12,}")
            print(f"  Trainable:            {trainable:>12,} ({trainable/total*100:.2f}%)")
            print(f"  Frozen:               {frozen:>12,}")
        else:
            print(f"\n  === Parameter Summary ===")
            print(f"  Vision encoder:    {sum(p.numel() for p in self.vision_model.parameters()):>12,}")
            print(f"  Visual projection: {sum(p.numel() for p in self.visual_projection.parameters()):>12,}")
            print(f"  Classification head: {sum(p.numel() for p in self.head.parameters()):>12,}")
            print(f"  Total:             {total:>12,}")
            print(f"  Trainable:         {trainable:>12,} ({trainable/total*100:.2f}%)")
            print(f"  Frozen:            {frozen:>12,}")
        print(f"{'='*60}\n")

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if only_fc:
            return self.head(x)

        vision_out = self.vision_model(pixel_values=x)

        # peft 包裝後 output 格式可能不同
        if hasattr(vision_out, 'pooler_output'):
            pooled = vision_out.pooler_output
        elif isinstance(vision_out, tuple):
            pooled = vision_out[1] if len(vision_out) > 1 else vision_out[0][:, 0, :]
        else:
            pooled = vision_out.last_hidden_state[:, 0, :]

        feat = self.visual_projection(pooled)

        if only_feat:
            return feat

        logits = self.head(feat)
        return {"logits": logits, "feat": feat}

    def group_matcher(self, coarse=False):
        if self.use_lora:
            # LoRA mode: 把 lora adapter 歸到對應的 layer group
            return dict(
                stem=r'^vision_model\.base_model\.model\.embeddings',
                blocks=[
                    (r'^vision_model\.base_model\.model\.encoder\.layers\.(\d+)', None),
                    (r'^vision_model\.base_model\.model\.(?:post_layernorm|pre_layrnorm)', (99999,)),
                    (r'^visual_projection', (99999,)),
                    (r'^head', (99999,)),
                ],
            )
        else:
            return dict(
                stem=r'^vision_model\.embeddings',
                blocks=[
                    (r'^vision_model\.encoder\.layers\.(\d+)', None),
                    (r'^vision_model\.(?:post_layernorm|pre_layrnorm)', (99999,)),
                    (r'^visual_projection', (99999,)),
                    (r'^head', (99999,)),
                ],
            )

    def no_weight_decay(self):
        no_wd = set()
        for name, _ in self.named_parameters():
            if 'bias' in name or 'layernorm' in name.lower() or 'layer_norm' in name.lower():
                no_wd.add(name)
            if 'position_embedding' in name or 'class_embedding' in name:
                no_wd.add(name)
        return no_wd


def get_clip_net_builder(
    clip_model_name="openai/clip-vit-large-patch14",
    two_stage_checkpoint=None,
    freeze_vision_encoder=True,
    freeze_vision_projection=False,
    use_lora=False,
    lora_rank=4,
    lora_alpha=16,
    lora_dropout=0.1,
    lora_target_modules=None,
    mlp_head=False,
    mlp_hidden_dim=1024,
    mlp_dropout=0.1,
):
    def _builder(num_classes=None, pretrained=True, pretrain_path=None, **kwargs):
        model = CLIPVisionBackbone(
            clip_model_name=clip_model_name,
            two_stage_checkpoint=two_stage_checkpoint,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_vision_projection=freeze_vision_projection,
            num_classes=num_classes if num_classes is not None else 345,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            mlp_head=mlp_head,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_dropout=mlp_dropout,
        )
        return model

    return _builder


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  CLIPVisionBackbone Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test 1: Original CLIP pretrain (frozen)
    print("\n[Test 1] Original CLIP pretrain (frozen)")
    model = CLIPVisionBackbone(
        clip_model_name="openai/clip-vit-base-patch32",
        two_stage_checkpoint=None,
        freeze_vision_encoder=True,
    ).to(device)

    x = torch.randn(4, 3, 224, 224, device=device)
    out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Logits: {out['logits'].shape}")
    print(f"  Feat:   {out['feat'].shape}")
    print(f"  num_features: {model.num_features}")
    print(f"  ✅ Test 1 passed!")

    # Test 2: group_matcher
    print("\n[Test 2] group_matcher")
    matcher = model.group_matcher()
    print(f"  Matcher keys: {list(matcher.keys())}")
    print(f"  ✅ Test 2 passed!")

    # Test 3: LoRA mode
    print("\n[Test 3] LoRA mode")
    try:
        model_lora = CLIPVisionBackbone(
            clip_model_name="openai/clip-vit-base-patch32",
            two_stage_checkpoint=None,
            freeze_vision_encoder=True,
            use_lora=True,
            lora_rank=4,
        ).to(device)

        out_lora = model_lora(x)
        print(f"  Logits: {out_lora['logits'].shape}")
        print(f"  Feat:   {out_lora['feat'].shape}")

        lora_params = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_lora.parameters())
        print(f"  Trainable: {lora_params:,} / {total_params:,}")
        print(f"  ✅ Test 3 passed!")
    except ImportError:
        print(f"  ⚠️  peft not installed, skipping LoRA test")

    # Test 4: USB layer_decay integration
    print("\n[Test 4] USB param_groups_layer_decay integration")
    try:
        from semilearn.nets.utils import param_groups_layer_decay
        no_wd = model.no_weight_decay()
        param_groups = param_groups_layer_decay(
            model, lr=5e-4, weight_decay=1e-4,
            no_weight_decay_list=no_wd, layer_decay=0.5
        )
        print(f"  Generated {len(param_groups)} param groups")
        print(f"  ✅ Test 4 passed!")
    except ImportError:
        print(f"  ⚠️  semilearn not installed, skipping integration test")

    # Test 5: net_builder interface
    print("\n[Test 5] net_builder interface")
    builder = get_clip_net_builder(
        clip_model_name="openai/clip-vit-base-patch32",
        two_stage_checkpoint=None,
    )
    model2 = builder(num_classes=345, pretrained=True, pretrain_path="dummy.pth")
    print(f"  num_features: {model2.num_features}")
    print(f"  ✅ Test 5 passed!")

    print("\n" + "=" * 60)
    print("  ✅ All tests passed!")
    print("=" * 60)