#!/usr/bin/env bash
# ============================================================
# Stage 2 (ControlNet 版)：learnable prompt + 輸入影像(Canny) → 生成
#   吃 source 影像的結構，產出 source×target 排列組合（12 方向）
#   用 domain_gen.py（自動遍歷 source 下所有 class，對每個 target 生成）
#   → 這是四象限主實驗 (gen_len20) 用的生成方式
# ============================================================
set -euo pipefail

# ┌──────────── 要改的地方 ────────────┐
PROMPT_LEN=20
CKPT="./checkpoints/two_stage_mini/prompt_len${PROMPT_LEN}/mini_DomainNet_len${PROMPT_LEN}_stage2_best.pt"
DATA_DIR="./data/mini-DomainNet/train"          # source 影像根目錄
OUT_ROOT="./result/generation/controlnet/mini_domainNet/gen_len${PROMPT_LEN}"
# └────────────────────────────────────┘

# ── 固定參數（與 config 一致）──
MAX_PER_CLASS=100
NUM_PER_IMAGE=1
BATCH_SIZE=20
STEPS=30; CFG=7.5; CN_SCALE=1.0; SEED=42

# 每個 source 生成到所有 4 個 target（domain_gen.py 的 --domain_id 可多給）
for SRC in real painting sketch clipart; do
  echo "===== ControlNet 生成： ${SRC} → 全部 target ====="
  python domain_gen.py \
    --checkpoint "${CKPT}" \
    --input "${DATA_DIR}/${SRC}" \
    --domain_id 0 1 2 3 \
    --domain_names real painting sketch clipart \
    --max_images_per_class "${MAX_PER_CLASS}" \
    --num_per_image "${NUM_PER_IMAGE}" \
    --batch_size "${BATCH_SIZE}" \
    --use_controlnet --controlnet_type canny \
    --steps "${STEPS}" --cfg "${CFG}" --cn_scale "${CN_SCALE}" \
    --seed "${SEED}" \
    --output "${OUT_ROOT}/${SRC}-to"
done
echo "ControlNet 生成完成 → ${OUT_ROOT}/<src>-to/<src>-to-<tgt>/"
