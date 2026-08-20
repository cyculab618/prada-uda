#!/usr/bin/env bash
# ============================================================
# Stage 1：訓練 per-domain learnable style prompt + domain classifier
#          （two_stage.py，CLIP ViT-L/14 凍結）
# 產出： ${SAVE_DIR}/mini_DomainNet_len${PROMPT_LEN}_stage2_best.pt
# ============================================================
set -euo pipefail

# ┌────────────────────────────────────────────────────────┐
# │  學弟要改的地方（依你要重現的實驗組別填）                 │
# └────────────────────────────────────────────────────────┘
PROMPT_LEN=20                                   # ← prompt 長度。實驗跑過 4 和 20；主實驗(gen_len20)用 20
DATA_DIR="./data/mini-DomainNet/train"          # ← mini-DomainNet 訓練影像根目錄（底下是 real/painting/sketch/clipart）
SAVE_DIR="./checkpoints/two_stage_mini/prompt_len${PROMPT_LEN}"   # ← checkpoint 存這（會自動帶 prompt_len）
# ────────────────────────────────────────────────────────────

# ↓↓↓ 以下為固定實驗參數，不用改（與論文設定一致）↓↓↓
MODEL_NAME="openai/clip-vit-large-patch14"
DOMAINS="real,painting,sketch,clipart"          # domain 順序 → id: real=0,painting=1,sketch=2,clipart=3
STAGE1_EPOCHS=5
STAGE1_LR=1e-4
STAGE2_EPOCHS=15
STAGE2_LR_PROMPT=1e-3
STAGE2_LR_PROJ=1e-5
STAGE2_LR_CLS=1e-4
LAMBDA_CLS=0.3
BATCH_SIZE=32
MIXUP_ALPHA=0.5                                 # 注意：argparse 預設是 0.0，實驗用 0.5
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SEED=42

echo "=========================================="
echo " Stage 1  prompt_len=${PROMPT_LEN}"
echo " data: ${DATA_DIR}"
echo " save: ${SAVE_DIR}"
echo "=========================================="

python two_stage.py \
  --data_dir "${DATA_DIR}" \
  --domains "${DOMAINS}" \
  --model_name "${MODEL_NAME}" \
  --prompt_len "${PROMPT_LEN}" \
  --stage1_epochs "${STAGE1_EPOCHS}" \
  --stage1_lr "${STAGE1_LR}" \
  --stage2_epochs "${STAGE2_EPOCHS}" \
  --stage2_lr_prompt "${STAGE2_LR_PROMPT}" \
  --stage2_lr_proj "${STAGE2_LR_PROJ}" \
  --stage2_lr_cls "${STAGE2_LR_CLS}" \
  --lambda_cls "${LAMBDA_CLS}" \
  --batch_size "${BATCH_SIZE}" \
  --mixup_alpha "${MIXUP_ALPHA}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --grad_clip "${GRAD_CLIP}" \
  --amp \
  --save_dir "${SAVE_DIR}" \
  --seed "${SEED}"

echo "Stage 1 完成 → checkpoint 在 ${SAVE_DIR}"
