#!/usr/bin/env bash
# ============================================================
# Stage 2 (SDM 版)：純 learnable prompt，無輸入影像 → 生成
#   沒有 source input，只靠 prompt 生成該域風格
#   → 只有 4 種生成（4 個 domain 風格，非排列組合）
#   用 controlnet_gen.py，不給 --domain_id、不加 --use_controlnet
#   （--domain_id 留空時，程式自動遍歷所有 domain 產 domain_0~3）
# ============================================================
set -euo pipefail

# ┌──────────── 要改的地方 ────────────┐
PROMPT_LEN=4                                     # SDM 實驗你用的 len（eval_prompt 那批是 len4）
CKPT="./checkpoints/two_stage_mini/prompt_len${PROMPT_LEN}/mini_DomainNet_len${PROMPT_LEN}_stage2_best.pt"
OUT_ROOT="./result/generation/sdm/prompt_len${PROMPT_LEN}"
NUM_IMAGES=100                                   # 每個 domain 生成幾張
# └────────────────────────────────────┘

STEPS=30; CFG=7.5; SEED=42

# 不給 --domain_id → controlnet_gen.py 自動產 domain_0 ~ domain_3（4 種域風格）
# 不加 --use_controlnet → 純 learnable prompt（SDM）
echo "===== SDM 生成：4 種域風格（純 prompt，無 input）====="
python controlnet_gen.py \
  --checkpoint "${CKPT}" \
  --num_images "${NUM_IMAGES}" \
  --steps "${STEPS}" --cfg "${CFG}" --seed "${SEED}" \
  --output "${OUT_ROOT}"
echo "SDM 生成完成 → ${OUT_ROOT}/domain_0 ~ domain_3/"
echo "注意：domain_0=real, 1=painting, 2=sketch, 3=clipart"
