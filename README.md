# Generative UDA Pipeline — 交接文件

論文主題：基於風格感知提示學習之生成式非監督式域適應影像分類研究

生成式無監督域適應（Generative Unsupervised Domain Adaptation）。
三階段：學 learnable style prompt → 生成 target 風格資料 → 下游分類訓練。
資料集：mini-DomainNet（126 類，4 域：real / painting / sketch / clipart）。

---

## 0. 環境（最重要，先弄對再說）

> ⚠️ 環境沒對齊，LoRA / semilearn 會各種炸。以下版本是實測能跑的組合。

Docker base image：`[TODO: 填你的 base image，例如 nvcr.io/nvidia/pytorch:24.xx-py3]`

核心套件版本（**三兄弟必須一致，且都是 +cu128**）：

| 套件 | 版本 |
|---|---|
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| torchaudio | 2.10.0+cu128 |
| transformers | 4.36.2 |
| peft | 0.11.1 |
| semilearn | 0.3.2 |

安裝（新環境）：
```bash
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.36.2" "peft==0.11.1" "semilearn==0.3.2"
```

驗證：
```bash
python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.cuda.is_available())"
python -c "import semilearn, transformers, peft; print('all OK')"
```

**踩過的坑（別重蹈）：**
- 新機器 driver 是 CUDA 13，但 **driver 向下相容**，照裝 cu128 就好，不用裝 CUDA 13 的 torch。
- torchaudio 版本不對會噴 `libcudart.so.13`；torchvision 不對會噴 `torchvision::nms does not exist`。→ 三兄弟版本對齊 +cu128 就解決。
- semilearn import 時會連帶 import torchaudio，所以 torchaudio 壞了整個 semilearn 就載不進來。

---

## 1. 資料擺放

```
data/
├── mini-DomainNet/train/<domain>/<class>/*.jpg   # source 影像（Stage2 生成用）
├── DomainNet/<domain>/<class>/*.jpg              # 完整 DomainNet（Stage3 naive 的真實 source）
└── DomainNet/mini_<domain>_{train,test}.txt      # splits（已保留，直接用）
```

- splits（`.txt`）和 benchmark 影像**都已保留**，學弟不用重生成。
- `.txt` 內是相對路徑（`real/aircraft_carrier/xxx.jpg 0`），配合 `--img_root ./data/DomainNet` 拼出實際路徑。
- 位置：`[TODO: 這些檔案實際放在哪，例如 //192.168.0.203/open_clip 掛到 /Project]`

---

## 2. Pipeline 總覽

```
Stage 1                Stage 2                    Stage 3
two_stage.py    →      生成資料            →      train_clip_ssl.py
學 style prompt        ┌ ControlNet: domain_gen.py（12 種排列，吃 input）
                       └ SDM: controlnet_gen.py（4 種域風格，純 prompt 無 input）
                                                  四象限 × 12 方向
```

一鍵跑全部：`bash run_all.sh`（stage1 → stage2_controlnet → stage3 四象限）
或各階段單獨跑：`bash stage1_train_prompt.sh` 等。

---

## 3. Stage 1 — 訓練 learnable prompt（`stage1_train_prompt.sh`）

單純訓練 per-domain learnable style prompt + domain classifier，CLIP ViT-L/14 凍結。

**要改**：`PROMPT_LEN`（實驗跑過 4 和 20，主實驗用 20）、`DATA_DIR`、`SAVE_DIR`。

**非 default 但實驗用的（已寫死在 bash）**：
- `mixup_alpha = 0.5`（argparse 預設 0.0）
- `--amp`（開混合精度，預設不開）
- 其餘（epochs 5/15、lr、lambda_cls 0.3、batch 32、seed 42）= 你的 config 值

產出：`${SAVE_DIR}/mini_DomainNet_len<PROMPT_LEN>_stage2_best.pt`
參考結果：best_s2_acc_cls ≈ 90.1%

---

## 4. Stage 2 — 生成資料（兩種方式）

### 4a. ControlNet（`stage2_controlnet.sh`）→ 主實驗用這個
- `domain_gen.py`，吃 source 影像的 Canny 邊緣 + learnable prompt。
- 產出 source×target **12 種排列**：`gen_len20/<src>-to/<src>-to-<tgt>/`
- 固定參數：steps 30、cfg 7.5、cn_scale 1.0、max_per_class 100、seed 42。

### 4b. SDM（`stage2_sdm.sh`）→ 純 prompt，無 input
- `controlnet_gen.py`，**不給 `--domain_id`、不加 `--use_controlnet`**。
- 只產 **4 種域風格**（domain_0~3），因為沒有 source input。
- `[TODO: SDM 輸出是 domain_0~3/，但實驗路徑是 sdm/prompt_len4/real-to-xxx/，
   中間是否有重命名步驟？如果有，補在這裡]`

---

## 5. Stage 3 — 下游訓練（`stage3_train.sh`）

USB / semilearn 框架訓練 CLIP 分類器。四象限 × 12 方向 = 48 runs。

| 象限 | labeled 來源 | 設定 | batch |
|---|---|---|---|
| ours_ssl | 生成圖 | SSL (flexmatch) | 16 |
| naive_ssl | 真實 source | SSL (flexmatch) | 16 |
| ours_fsl | 生成圖 | FSL (`--no_ssl --use_lora`) | 8 |
| naive_fsl | 真實 source | FSL (`--no_ssl --use_lora`) | 8 |

**要改**：`QUADRANT`（選象限）、`PROMPT_LEN`、路徑。

**關鍵參數（已寫死，別用 default）：**
- `num_train_iter = 5600`（⚠️ 預設是 15000，差很多）
- `num_eval_iter = 400`
- `batch_size`：SSL 16 / LoRA 8（LoRA 因顯存降到 8）
- `lb_per_class = 100`、`seed = 42`、`--net clip_vit`
- 其餘（lr 5e-4、layer_decay 0.5、uratio 2...）= default

**FSL 兩種訓練策略**：
- Linear probe（原論文）：`--no_ssl`（不加 --use_lora），batch 16
- LoRA（新增對照）：`--no_ssl --use_lora`，batch 8，rank 4 / alpha 16

**跑法**：一次一個象限。跑完整四象限就四個 QUADRANT 都跑。
含「跳過已完成」防呆，重複執行只補沒跑的。

**抓結果**：
```bash
grep -rH "Best acc" ./result/log/<logdir>/ | sort
# logdir: flexmatch / flexmatch_naive / lora_ours / lora_naive
```

---

## 6. 已知問題 / 注意事項

- **網路磁碟寫 checkpoint 會間歇失敗**（`RuntimeError: basic_ios::clear: iostream error`）。
  → Stage3 的 `CKPT_ROOT` 建議設本地 `/tmp`，log 才寫網路磁碟。訓練照跑，只有存 checkpoint 那步會炸。
- **完成判斷看 `grep "Training finished"`**，不要看 batch log 的 "done"（那是無條件印的，crash 也會印）。
- **Naive 的真實 source** 取自完整 DomainNet，部分類別不足 100 張（如 painting ~11479），論文需註明。
- **兩種刻意的設定差異**（需在論文說明）：LoRA batch=8（其餘16）；FSL 用 `--no_ssl` 使 SSL 演算法退化為純監督，故 FSL 下 flexmatch/freematch 等價。

---

## 7. 實驗背景（這批數據為何要跑）

`[TODO: 補上背景，例如：回應 APSIPA 審查意見「alternative SSL methods」→ 跑 flexmatch；
口委要求「多一種 FSL」→ 跑 LoRA。四象限對應論文 Table [X]。]`

核心發現：
- SSL 下 Ours ≈ Naive（生成優勢被 SSL 抹平）
- FSL 下 Ours > Naive（生成優勢顯現）

---

## 8. 檔案清單

| 檔案 | 用途 |
|---|---|
| `two_stage.py` | Stage 1 |
| `domain_gen.py` | Stage 2 ControlNet 批量生成 |
| `controlnet_gen.py` | Stage 2 SDM / 單方向測試 |
| `train_clip_ssl.py` | Stage 3 下游訓練 |
| `clip_backbone.py` | CLIP backbone 封裝（含 LoRA），被 train_clip_ssl.py 呼叫 |
| `eval_prompt_domain.py` | 評估：生成圖餵回 Stage1 域分類器（Problem-1 分析） |
| `stage1_train_prompt.sh` / `stage2_*.sh` / `stage3_train.sh` / `run_all.sh` | 各階段執行腳本 |