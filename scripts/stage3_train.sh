#!/usr/bin/env bash
# ============================================================
# Stage 3：下游分類器訓練（train_clip_ssl.py + clip_backbone.py）
# 四象限 × 12 方向 = 48 runs
#   Ours  = 生成圖當 labeled       Naive = 真實 source 當 labeled
#   SSL   = flexmatch              FSL   = --no_ssl (+ --use_lora)
# 含「跳過已完成」防呆：重複執行只補沒跑的
# ============================================================
set -uo pipefail

# ┌────────────────────────────────────────────────────────┐
# │  學弟要改的地方                                          │
# └────────────────────────────────────────────────────────┘
QUADRANT="${QUADRANT:-ours_ssl}"     # ← 要跑哪個象限： ours_ssl / naive_ssl / ours_fsl / naive_fsl（可用環境變數覆蓋）
PROMPT_LEN=20           # ← 生成圖的 prompt 長度（對應 Stage2）
GEN_ROOT="./result/generation/controlnet/mini_domainNet/gen_len${PROMPT_LEN}"  # ← Ours 的生成圖根目錄
REAL_ROOT="./data/DomainNet"       # ← Naive 的真實 source 根目錄（完整 DomainNet）
DATA_DIR="./data/DomainNet"        # ← mini_<domain>_{train,test}.txt 放這
IMG_ROOT="./data/DomainNet"        # ← txt 內相對路徑的根
CKPT_ROOT="/tmp/stage3_ckpt"       # ← checkpoint 存這（建議本地/tmp，避免網路磁碟寫入失敗）
LOG_ROOT="./result/log"            # ← log 存這
# ────────────────────────────────────────────────────────────

# ↓↓↓ 固定實驗參數（與論文一致，四象限共用）↓↓↓
NET="clip_vit"
SSL_ALGO="flexmatch"               # SSL 用 flexmatch（原論文 freematch，替代實驗 flexmatch）
NUM_TRAIN_ITER=5600
NUM_EVAL_ITER=400
LB_PER_CLASS=100
SEED=42
# ── 各象限的差異設定 ──
case "${QUADRANT}" in
  ours_ssl)   IS_OURS=1; EXTRA=""                        ; BATCH=16 ; LOGDIR="flexmatch"        ;;
  naive_ssl)  IS_OURS=0; EXTRA=""                        ; BATCH=16 ; LOGDIR="flexmatch_naive"  ;;
  ours_fsl)   IS_OURS=1; EXTRA="--no_ssl --use_lora"     ; BATCH=8  ; LOGDIR="lora_ours"        ;;  # LoRA 因顯存 batch=8
  naive_fsl)  IS_OURS=0; EXTRA="--no_ssl --use_lora"     ; BATCH=8  ; LOGDIR="lora_naive"       ;;
  *) echo "QUADRANT 只能是 ours_ssl/naive_ssl/ours_fsl/naive_fsl"; exit 1 ;;
esac
# 註：linear-probe FSL（原論文那版）= 把 EXTRA 改成 "--no_ssl"（不加 --use_lora），BATCH 用 16

# 12 方向： src tgt tag
DIRECTIONS=(
  "real painting rl2pt" "real sketch rl2sk" "real clipart rl2cl"
  "painting real pt2rl" "painting sketch pt2sk" "painting clipart pt2cl"
  "sketch real sk2rl" "sketch painting sk2pt" "sketch clipart sk2cl"
  "clipart real cl2rl" "clipart painting cl2pt" "clipart sketch cl2sk"
)

echo "=========================================="
echo " Stage 3  象限=${QUADRANT}  (batch=${BATCH})"
echo "=========================================="

for d in "${DIRECTIONS[@]}"; do
  set -- $d; src=$1; tgt=$2; tag=$3

  # labeled 來源：Ours=生成圖，Naive=真實 source
  if [ "${IS_OURS}" -eq 1 ]; then
    gen_dir="${GEN_ROOT}/${src}-to/${src}-to-${tgt}"
  else
    gen_dir="${REAL_ROOT}/${src}"
  fi

  train_txt="${DATA_DIR}/mini_${tgt}_train.txt"
  test_txt="${DATA_DIR}/mini_${tgt}_test.txt"
  log_dir="${LOG_ROOT}/${LOGDIR}/${tgt}2/${tag}"
  save_dir="${CKPT_ROOT}/${LOGDIR}/${tag}"
  mkdir -p "${log_dir}" "${save_dir}"
  log_file="${log_dir}/${tag}.log"

  # 跳過已完成
  if [ -f "${log_file}" ] && grep -q "Training finished" "${log_file}"; then
    echo "  [${tag}] 已完成，跳過。"; continue
  fi

  echo "  [${tag}] ${src} -> ${tgt}"
  python train_clip_ssl.py \
    --net "${NET}" \
    --algorithm "${SSL_ALGO}" \
    ${EXTRA} \
    --batch_size "${BATCH}" \
    --num_train_iter "${NUM_TRAIN_ITER}" \
    --num_eval_iter "${NUM_EVAL_ITER}" \
    --lb_per_class "${LB_PER_CLASS}" \
    --labeled_source generated \
    --generated_dir "${gen_dir}" \
    --img_root "${IMG_ROOT}" \
    --train_txt "${train_txt}" \
    --test_txt "${test_txt}" \
    --save_dir "${save_dir}" \
    --seed "${SEED}" \
    > "${log_file}" 2>&1

  echo "  [${tag}] done → ${log_file}"
done

echo "象限 ${QUADRANT} 處理完畢。抓 acc： grep -rH 'Best acc' ${LOG_ROOT}/${LOGDIR}/"
