#!/usr/bin/env bash
# ============================================================
# 總控：Stage1 → Stage2 → Stage3
# ============================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "########## Stage 1：訓練 learnable prompt ##########"
bash "${HERE}/stage1_train_prompt.sh"

echo "########## Stage 2：生成資料 ##########"
# 主實驗（四象限）用 ControlNet 版；若要 SDM 版改跑 stage2_sdm.sh
bash "${HERE}/stage2_controlnet.sh"

echo "########## Stage 3：下游訓練（四象限）##########"
for Q in ours_ssl naive_ssl ours_fsl naive_fsl; do
  echo "===== 象限 ${Q} ====="
  QUADRANT="${Q}" bash "${HERE}/stage3_train.sh"
done

echo "########## 全部完成 ##########"
