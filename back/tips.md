
# git command tips
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Qihongfei-hub/My_nanochat.git
git push -u origin main


#
import swanlab as wandb
# qhf 
import swanlab as wandb
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

##
(my_project_env) C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat>pip install wandb==0.18.3
(my_project_env) C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat>pip install swanlab==0.6.8
##




# 1 python prepare 
python -m pip install --upgrade pip 



# 2.1 create and active the Virtual Env

##
conda create -n my_project_env python=3.11
location C:\Users\hongf\miniconda3.1\envs\my_project_env

##
conda activate my_project_env
conda deactivate

## PowerShell
C:\Users\hongf\Documents\WindowsPowerShell\profile.ps1  


# 2.2 creat and active the Virtual Env
python -m venv .venv

## 激活环境
.\.venv\Scripts\Activate.ps1 




# 3. Virtual Env check 
(my_project_env) PS C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat> conda env list

## conda environments:

 * -> active
 + -> frozen
base                     C:\Users\hongf\miniconda3.1
llama_hw                 C:\Users\hongf\miniconda3.1\envs\llama_hw
my_project_env       *   C:\Users\hongf\miniconda3.1\envs\my_project_env




# 4.Pytorch install  
## issue? not used
## Command 
python -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-cache-dir 
###
Downloading https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp311-cp311-win_amd64.whl (2862.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━ 2.8/2.9 GB 2.7 MB/s eta 0:00:42

## Pytorch install  result check 
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
### 
(my_project_env) PS C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat> python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
PyTorch version: 2.9.1+cu128
CUDA available: True
CUDA version: 12.8

##
(my_project_env) PS C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat> conda list | findstr torch
torch                      2.9.1+cu128      pypi_0           pypi



# 5 安装项目
##  pip install -e .    #  not used
## 
Write-Host "安装基本依赖..."
python -m pip install "datasets>=4.0.0" "fastapi>=0.117.1" "ipykernel>=7.1.0" "kernels>=0.11.7" "matplotlib>=3.10.8" "psutil>=7.1.0" "python-dotenv>=1.2.1" "regex>=2025.9.1" "rustbpe>=0.1.0" "scipy>=1.15.3" "setuptools>=80.9.0" "tabulate>=0.9.0" "tiktoken>=0.11.0" "tokenizers>=0.22.0" "transformers>=4.57.3" "uvicorn>=0.36.0" "wandb>=0.21.3" "zstandard>=0.25.0"



# 5. run the pre-train of NanaChat
## command 
nvidia-smi

# pq size  85   1900*16384= 31M, on average 2000,
## Total number of training tokens: 360,448,000,  360M/31M=12 file
step 01900/22000 (8.64%) | loss: 4.197877 | lrm: 1.00 | dt: 451.72ms | tok/sec: 36,270 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 83 | total time: 14.28m | eta: 151.9m
step 02000/22000 (9.09%) | loss: 4.149134 | lrm: 1.00 | dt: 448.21ms | tok/sec: 36,554 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 3 | total time: 15.03m | eta: 151.1m

step 03800/22000 (17.27%) | loss: 3.891526 | lrm: 1.00 | dt: 450.96ms | tok/sec: 36,331 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 80 | total time: 28.57m | eta: 137.2m
step 03900/22000 (17.73%) | loss: 3.939152 | lrm: 1.00 | dt: 448.36ms | tok/sec: 36,542 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 1 | total time: 29.32m | eta: 136.4m

step 07700/22000 (35.00%) | loss: 3.771260 | lrm: 1.00 | dt: 451.04ms | tok/sec: 36,324 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 81 | total time: 57.86m | eta: 107.6m
step 07800/22000 (35.45%) | loss: 3.764075 | lrm: 0.99 | dt: 448.15ms | tok/sec: 36,559 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 3 | total time: 58.61m | eta: 106.8m

## last steps
step 21800/22000 (99.09%) | loss: 3.455044 | lrm: 0.06 | dt: 449.69ms | tok/sec: 36,434 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 28 | total time: 163.47m | eta: 1.5m
step 21900/22000 (99.55%) | loss: 3.477977 | lrm: 0.06 | dt: 448.95ms | tok/sec: 36,493 | bf16_mfu: 0.00 | epoch: 1 Step 22000 | Validation bpb: 1.057228

## for 4th train
python -m scripts.base_train --depth=32 --save-every=4000 --num-iterations=200000 --run=dummy --head-dim=32 --window-pattern=L --max-seq-len=512 --device-batch-size=8 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=4000 --log-every=50 --eval-every=2000 --max-seq-len=512 --warmup-steps=2000 --warmdown-ratio=0.9 --aspect-ratio=64

python -m scripts.base_train --depth=32 --save-every=4000 --num-iterations=800000 --run=dummy --head-dim=32 --window-pattern=L --max-seq-len=512 --device-batch-size=8 --total-batch-size=8192 --eval-tokens=524288 --core-metric-every=-1 --sample-every=4000 --log-every=50 --eval-every=2000 --max-seq-len=512 --warmup-steps=4000 --warmdown-ratio=0.9 --aspect-ratio=64

--run=Nanochat

## for 3rd train
python -m scripts.base_train --depth=16 --save-every=4000 --num-iterations=200000 --run=dummy --head-dim=64 --window-pattern=L --max-seq-len=512 --device-batch-size=8 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=4000 --log-every=50 --eval-every=2000 --max-seq-len=512 --warmup-steps=2000 --warmdown-ratio=0.9 --aspect-ratio=64


0  NVIDIA GeForce RTX 4070 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   77C    P0             93W /   94W |    7427MiB /   8188MiB |

#
python -m scripts.base_train --depth=48 --save-every=10000 --num-iterations=320000 --run=dummy --head-dim=64 --window-pattern=L --max-seq-len=512 --device-batch-size=2 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=8000 --log-every=1 --eval-every=8000 --max-seq-len=512 --warmup-steps=6000 --warmdown-ratio=0.9 --aspect-ratio=32 --run=nanochat



## not tried yet

python -m scripts.base_train --depth=16 --save-every=4000 --num-iterations=80000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=8 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=800 --log-every=100 --eval-every=800 --max-seq-len=512 --warmup-steps=2000 --warmdown-ratio=0.65


=> `--warmup-steps=2000`  change from 700 to 2000
=> --warmdown-ratio=0.65  keep same 
=> `max-seq-len=512   (1024 will be out of memory)` or token speed will be very slow 1700 token vs 11K
=> `device-batch-size=8, 16 will be out of memory` 
   


## 
## `resume  :2026/4/15  11:00`
python -m scripts.base_train --resume-from-step 20000 --run resume01 --depth=16 --save-every=4000 --num-iterations=80000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=8 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=1600 --log-every=300 --eval-every=1600 



Vocab size: 32,768
Model config:
{
  "sequence_len": 512,
  "vocab_size": 32768,
  "n_layer": 16,
  "n_head": 6,
  "n_kv_head": 6,
  "n_embd": 768,
  "window_pattern": "SSSL"
}
Resuming optimization from step 16000
Parameter counts:
wte                     : 25,165,824
value_embeds            : 201,326,592
lm_head                 : 25,165,824
transformer_matrices    : 113,246,784
scalars                 : 32
total                   : 364,905,056
total                   : 364,905,056
Estimated FLOPs per token: 8.776616e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.028592 for depth 16
Scaling the LR for the AdamW parameters ∝1/√(768/768) = 1.000000
Using user-provided number of iterations: 80,000
Total number of training tokens: 1,310,720,000
Tokens : Scaling params ratio: 9.47
Total training FLOPs estimate: 1.150369e+18
Estimated FLOPs per token: 8.776616e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.028592 for depth 16
Scaling the LR for the AdamW parameters ∝1/√(768/768) = 1.000000
Tokens / micro-batch / rank: 8 x 512 = 4,096
Tokens / micro-batch: 4,096
Total batch size 16,384 => gradient accumulation steps: 4




python -m scripts.base_train --depth=16 --save-every=4000 --num-iterations=80000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=8 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=800 --log-every=100 --eval-every=800 --max-seq-len=512


## fb8 can not be used, 21:30~ 9:53 (3600) ~ 10:03 (4800)  ~ 10:13 (6100)   ~ 10:23 (7400)
python -m scripts.base_train --depth=16 --save-every=4000 --num-iterations=80000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=16 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=800 --log-every=100 --eval-every=800 --max-seq-len=512

7381MiB /   8188MiB
Parameter counts:
wte                     : 12,582,912
value_embeds            : 75,497,472
lm_head                 : 12,582,912
transformer_matrices    : 21,234,096
scalars                 : 24
total                   : 121,897,416
Estimated FLOPs per token: 2.205968e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.049497 for depth 12
Scaling the LR for the AdamW parameters ∝1/√(384/768) = 1.414214
Using user-provided number of iterations: 22,000
Total number of training tokens: 360,448,000
Tokens : Scaling params ratio: 10.66
Total training FLOPs estimate: 7.951366e+16
Tokens / micro-batch / rank: 16 x 512 = 8,192
Tokens / micro-batch: 8,192
Total batch size 16,384 => gradient accumulation steps: 2
step 00200/22000 (0.91%) | loss: 6.588317 | lrm: 0.29 | dt: 452.39ms | tok/sec: 36,216 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 10 | total time: 1.44m | eta: 165.3m


##
python -m scripts.base_train --depth=12 --save-every=4000 --num-iterations=22000 --run=dummy --head-dim=64 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=16 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=800 --log-every=100 --eval-every=800

7381MiB /   8188MiB 
tok/sec: 35,720
Total batch size 16,384 => gradient accumulation steps: 2
Total number of training tokens: 360,448,000
Tokens : Scaling params ratio: 10.66

step 00400/22000 (1.82%) | loss: 5.116461 | lrm: 1.00 | dt: 458.67ms | tok/sec: 35,720 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 18 | total time: 2.95m | eta: 163.2m



## note
##device-batch-size=32  
GPU: 7859MiB /   8188MiB 
tok/sec: 32,358 -> tok/sec: 5,257   decrease 




##
python -m scripts.base_train --depth=12 --save-every=4000 --num-iterations=22000 --run=dummy --head-dim=64 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=16 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=800 --log-every=100
{
  "sequence_len": 512,
  "vocab_size": 32768,
  "n_layer": 12,
  "n_head": 6,
  "n_kv_head": 6,
  "n_embd": 384,
  "window_pattern": "SSSL"
}
Parameter counts:
wte                     : 12,582,912
value_embeds            : 75,497,472
lm_head                 : 12,582,912
transformer_matrices    : 21,234,096
scalars                 : 24
total                   : 121,897,416
Estimated FLOPs per token: 2.205968e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.049497 for depth 12
Scaling the LR for the AdamW parameters ∝1/√(384/768) = 1.414214
Using user-provided number of iterations: 22,000
Total number of training tokens: 360,448,000
Tokens : Scaling params ratio: 10.66
Total training FLOPs estimate: 7.951366e+16
Tokens / micro-batch / rank: 16 x 512 = 8,192
Tokens / micro-batch: 8,192
Total batch size 16,384 => gradient accumulation steps: 2

1 pq: 0 rg: 57 | total time: 9.78m | eta: 156.9m



##
python -m scripts.base_train --depth=12 --save-every=2000 --num-iterations=20000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=16 --total-batch-size=16384 --eval-tokens=5242880 --core-metric-every=-1 --sample-every=500 --log-every=5

lm_head                 : 12,582,912
transformer_matrices    : 21,233,880
Tokens : Scaling params ratio: 9.69
Total batch size 16,384 => gradient accumulation steps: 2
7377MiB /   8188MiB  
step 00120/20000 (0.60%) | loss: 6.146648 | lrm: 1.00 | dt: 454.87ms | tok/sec: 36,019 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 6 | total time: 0.83m | eta: 150.9m


##
python -m scripts.base_train --depth=12 --save-every=2000 --num-iterations=20000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=32 --total-batch-size=16384 --eval-tokens=5242880 --core-metric-every=-1 --sample-every=500 --log-every=50

lm_head                 : 12,582,912
transformer_matrices    : 21,233,880
Total number of training tokens: 327,680,000
Tokens : Scaling params ratio: `9.69`
Total batch size 16,384 => gradient accumulation steps: 1
tok/sec: 5,902
 GPU: 7859MiB /   8188MiB


##
python -m scripts.base_train --depth=12 --save-every=2000 --num-iterations=35000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=16 --total-batch-size=32768 --eval-tokens=5242880 --core-metric-every=-1 --sample-every=500 --log-every=50
wte                     : 12,582,912
value_embeds            : 75,497,472
lm_head                 : 12,582,912
transformer_matrices    : 21,233,880
scalars                 : 24
total                   : 121,897,200

Total number of training tokens: 1,146,880,000
Tokens : Scaling params ratio: 33.91
step 00050/35000 (0.14%) | loss: 6.710575 | lrm: 1.00 | dt: 871.22ms | tok/sec: 37,611 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 5 | total time: 0.58m | eta: 507.0m



##
python -m scripts.base_train --depth=12 --save-every=2000 --num-iterations=35000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=32 --total-batch-size=32768 --eval-tokens=5242880 --core-metric-every=-1 --sample-every=500 --log-every=50

wte                     : 12,582,912
value_embeds            : 75,497,472
lm_head                 : 12,582,912
transformer_matrices    : 21,233,880
scalars                 : 24

token:
step 00000/35000 (0.00%) | loss: 10.397563 | lrm: 0.03 | dt: 14554.83ms | tok/sec: 2,251 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m


##  180m 
python -m scripts.base_train --depth=12 --save-every=5000 --num-iterations=35000 --run=dummy --head-dim=64 --window-pattern=SSSL --max-seq-len=512  --device-batch-size=16 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=500


##  2320.1m 
python -m scripts.base_train --depth=12 --save-every=5000 --num-iterations=35000 --run=dummy --head-dim=128 --window-pattern=L  --max-seq-len=512  --device-batch-size=16 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=500




### out of memory  
python -m scripts.base_train --depth=16 --save-every=5000 --num-iterations=35000 --run=dummy --head-dim=128 --window-pattern=L  --max-seq-len=512  --device-batch-size=32 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=500


{
  "sequence_len": 512,
  "vocab_size": 32768,
  "n_layer": 16,
  "n_head": 8,
  "n_kv_head": 8,
  "n_embd": 1024,
  "window_pattern": "L"
}
wte                     : 33,554,432
value_embeds            : 268,435,456
lm_head                 : 33,554,432
transformer_matrices    : 201,327,360
scalars                 : 32
total                   : 536,871,712

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Of the allocated memory `12.76` GiB is allocated by PyTorch, and 13.87 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)




##
## resume pretrain  
python scripts/base_train.py --resume-from-step 1500 --run resume01 --save-every=100 --log-every=10 --num-iterations=14000 --run=dummy --depth=12 --head-dim=64 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=32 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=50





##
bash runs/speedrun.sh for GPU
bash .\runs\runcpu.sh for CPU


##
## `disable torch.compile`

refer code-issue.md

## data set 
C:\Users\hongqi\.cache\nanochat\base_data_climbmix 
- pq / rg 的含义：
  - `pq`：Parquet 文件的索引（第几个 parquet 文件，`pq_idx`）.  
  - `rg`：当前 Parquet 文件内的 Row Group 索引（`rg_idx`）.  

- 关于 5000 次迭代到底“加载了多少数据”：
  - 脚本用的度量是“tokens”.计算公式：total_tokens = total_batch_size * num_iterations.
  - 你日志里已经显示：`Total number of training tokens: 81,920,000`，这来自 16,384 tokens/step × 5,000 steps = 81,920,000 tokens.
  - 若以序列数（每序列长度 = `max_seq_len` = 512）来算：训练中等价于 81,920,000 / 512 = 160,000 序列（每个序列对应模型看到的一个长度-512 的输入行）.
  - 每个训练 step 实际处理的序列数 = total_batch_size / max_seq_len = 16,384 / 512 = 32 序列/step（在单进程/单 rank 情况下；DDP 时每 rank 按分片比例处理）.

- 一个文件加载了多少数据：`4220* 16378 token = 70M`
  mfu: 0.00 | epoch: 1 pq: 2 rg: 1 | total time: 603.52m | eta: 111.0m
  step 04226/05000 (84.52%) | loss: 3.207484 | lrm: 0.28 | dt: 7540.05ms | tok/sec: 2,172 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 2 | total time: 603.65m | eta: 110.8m




.pt file
Saved model parameters to: `C:\Users\hongqi\.cache\nanochat\base_checkpoints\d3\model_000500.pt`

## evaluation data
Downloading https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip... 
C:/Users/hongqi/.cache/nanochat/eval_bundle.zip


## data 
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
下载SFT 数据集 到
C:\Users\hongqi\myenv\nano_GPT\nanochat\tasks\data\smol-smoltalk

- SmolTalk  
  - 来源：HuggingFace 数据集 id `HuggingFaceTB/smol-smoltalk`（https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk）  
   https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/tree/refs%2Fconvert%2Fparquet/default/train

- CustomJSON（identity_conversations） 
   https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl  

- MMLU  
  - 来源：HuggingFace 数据集 id `cais/mmlu`（https://huggingface.co/datasets/cais/mmlu）  
 
 ## ######
 https://huggingface.co/datasets/cais/mmlu/tree/refs%2Fconvert%2Fparquet

- GSM8K  
  - 来源：HuggingFace 数据集 id `openai/gsm8k`（https://huggingface.co/datasets/openai/gsm8k）  
  ###
  https://huggingface.co/datasets/openai/gsm8k/tree/main/main

- SpellingBee / SimpleSpelling  
  - 来源（词表）：`WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"`（脚本使用该词表）

  ##
  https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt

- val_dataset  
  - 验证集由脚本中组合生成：`SmolTalk(split="test")` + `MMLU(subset="all", split="test", stop=5200)` + `GSM8K(subset="main", split="test", stop=420)`.因此它们的来源与上面对应（HuggingFace 或本地缓存）.


`Results written to: C:\Users\hongqi\.cache\nanochat\base_eval\base_model_000500.`csv    

##
##  评估
这条命令默认会跑 3 类评估，因为你没有传 --eval，所以 base_eval.py 使用默认值 core,bpb,sample.执行流程在 scripts/base_eval.py 里

--model-tag 和 --step 你都没传，所以默认加载最近的 base checkpoint.
同时读出 checkpoint 里的 sequence_len，后面 BPB 评估会用到.


### 跑 sample 评估
因为默认包含 sample，它会先做一次定性检查.
用固定的几条 prompt，比如 “The capital of France is” 这类句子，调用 Engine.generate_batch(...) 生成文本.
然后再做 8 条无条件采样.
这部分只是让你直观看模型会说什么，不打分


### 跑 bpb 评估
它会分别在 train 和 val 两个 split 上评估.
数据加载不是 SFT 的对话数据，而是预训练同款 dataloader：tokenizing_distributed_data_loader_bos_bestfit(...)，也就是从预训练 parquet 文本里取样,tokenize,打包成 (x, y).
指标是 BPB，定义在 nanochat/loss_eval.py：
对每个 target token 计算交叉熵 loss；
再按这个 token 对应的 UTF-8 字节数加权；
最后算 bits per byte，这样就不太受 tokenizer 词表大小影响.

传了 --device-batch-size=1 --split-tokens=16384，所以每个 split 实际会评估多少步取决于模型的 sequence_len：
tokens_per_step = device_batch_size * sequence_len * world_size
如果你这个 checkpoint 是 sequence_len=512，单卡下就是 1 * 512 * 1 = 512
那么 steps = 16384 / 512 = 32
也就是说，train 跑 32 个 batch，val 也跑 32 个 batch


###
跑 core 评估
这是主要的离散任务准确率评估.
它先确保 eval_bundle 存在；如果本地没有，会下载 eval_bundle.zip，里面包含 core.yaml,任务数据和随机基线.
core.yaml 里定义了一组 ICL 任务，每个任务有：
任务类型：multiple_choice / schema / language_modeling
few-shot 数量
对应的数据文件
对每个任务：
读入全部样本；
用固定随机种子 1337 打乱；


因为你传了 --max-per-task=16，所以每个任务只取前 16 个样本做快速评估；
再调用 evaluate_task(...) 去算准确率.
evaluate_task(...) 在 `nanochat/core_eval.py` 里：
对每个样本构造 few-shot prompt；
multiple_choice / schema：分别为每个选项拼出完整 prompt；
前向计算每个选项 continuation 部分的平均 loss；
选择 loss 最小的选项作为预测；
看是否等于 gold label.

language_modeling 类型则直接检查 continuation token 是否逐个预测正确.
每个任务得到一个 accuracy 后，还会结合 eval_meta_data.csv 里的随机基线，算一个 `centered score`：

##
为什么用中心化得分（centered）

去除“随机猜测”偏差：accuracy 减去 0.01 * random_baseline 把随机水平（例如多选题的 25%）移除，centered=0 即等于随机猜测表现.
统一刻度（0 → 1）：除以 (1 - baseline) 后，centered=1 表示完美，0 表示随机，便于直观比较.
跨任务可比：不同任务的随机基线不同（选项数,答案分布），中心化后把难度差异标准化，平均/合并多个任务更公平.
易解释：例如 centered=0.5 表示“在随机与完美之间走了一半”.
实现细节：代码里用 0.01 * random_baseline 是因为 random_baseline 存的是百分比（如 25），需转成小数（0.25）.
限制/注意事项：

当随机基线接近 1（极端少见）时分母很小，得分会不稳定.
若基线估计不准或任务不适合“随机基线”比较（例如非均匀采样），该方法会误导.
不包含样本量/置信度信息——小样本任务波动大，合并时应注意权重或置信区间.


###
python -m  scripts.base_eval --device-batch-size=16   => 
python -m  scripts.base_eval --device-batch-size=16 --max-seq-len=2048


这是您训练模型的 **CORE 评估** 结果！让我分析一下：

## 各任务表现

| 任务 | 准确率 | 评价 |
|------|--------|------|
| **piqa** | 60.34% | ⭐ 非常好 |
| **winograd** | 54.58% | ⭐ 很好 |
| **winogrande** | 52.57% | ⭐ 很好 |
| **copa** | 46.00% | 不错 |
| **arc_easy** | 43.86% | 不错 |
| **bigbench_cs_algorithms** | 40.23% | 不错 |
| **commonsense_qa** | 33.09% | 一般 |
| **arc_challenge** | 24.57% | 较难 |
| **agi_eval_lsat_ar** | 26.96% | 较难 |
| **hellaswag** | 28.26% | 一般 |
| **openbook_qa** | 25.60% | 一般 |
| **lambada_openai** | 18.36% | 一般 |
| **bigbench_operators** | 6.19% | 难 |
| **bigbench_dyck_languages** | 0.00% | ❌ 无法完成 |
| **bigbench_repeat_copy_logic** | 0.00% | ❌ 无法完成 |

## 总结

- **最佳表现**：piqa (60.34%)、winograd (54.58%)、winogrande (52.57%)
- **较弱任务**：涉及编程/算法逻辑的任务（bigbench 系列）
- **整体水平**：大约在 GPT-2 水平（~30-40%），部分任务超过 GPT-2

模型训练得不错！物理常识推理（piqa）和因果推理（copa）表现突出。



##
GPT-2 是 OpenAI 于 2019 年发布的大型语言模型，以下是其水平定位：

## 模型规模
- 参数数量 ：1.5B（15亿）
- 层数 ：48层
- 注意力头数 ：25个
- 上下文长度 ：1024 tokens
## 能力水平
GPT-2 在 2019 年发布时是一个突破性的模型，但以今天的眼光来看：


##



##
python -m scripts.chat_sft  --device-batch-size=16


## new model :  4:00
python -m scripts.base_train --depth=16 --save-every=4000 --num-iterations=80000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=8 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=800 --log-every=100 --eval-every=800 --max-seq-len=512

python -m scripts.base_train --depth=16 --save-every=4000 --num-iterations=80000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=8 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=800 --log-every=100 --eval-every=800 --max-seq-len=1024


Vocab size: 32,768
Model config:
{
  "sequence_len": 512,
  "vocab_size": 32768,
  "n_layer": 16,
  "n_head": 6,
  "n_kv_head": 6,
  "n_embd": 768,
  "window_pattern": "SSSL"
}
Parameter counts:
wte                     : 25,165,824
value_embeds            : 201,326,592
lm_head                 : 25,165,824
transformer_matrices    : 113,246,784
scalars                 : 32
total                   : 364,905,056
Estimated FLOPs per token: 8.776616e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.028592 for depth 16
Scaling the LR for the AdamW parameters ∝1/√(768/768) = 1.000000
Using user-provided number of iterations: 80,000
Total number of training tokens: 1,310,720,000
Tokens : Scaling params ratio: 9.47
Total training FLOPs estimate: 1.150369e+18
Tokens / micro-batch / rank: 8 x 512 = 4,096
Tokens / micro-batch: 4,096
Total batch size 16,384 => gradient accumulation steps: 4
Step 00000 | Validation bpb: 3.175789
step 00000/80000 (0.00%) | loss: 10.397202 | lrm: 0.00 | dt: 1152.62ms | tok/sec: 14,214 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00100/80000 (0.12%) | loss: 7.604645 | lrm: 0.14 | dt: 1503.10ms | tok/sec: 10,900 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 5 | total time: 2.18m | eta: 1939.5m

## Lr warmup
step 00200/80000 (0.25%) | loss: 6.464067 | lrm: 0.29 | dt: 1447.66ms | tok/sec: 11,317 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 10 | total time: 4.62m | eta: 1939.6m
step 00300/80000 (0.38%) | loss: 5.919112 | `lrm: 0.43` | dt: 1448.27ms | tok/sec: 11,312 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 14 | total time: 7.04m | eta: 1935.1m
step 00400/80000 (0.50%) | loss: 5.578771 | `lrm: 0.57` | dt: 1451.43ms | tok/sec: 11,288 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 18 | total time: 9.47m | eta: 1932.6m
step 00500/80000 (0.62%) | loss: 5.238528 | `lrm: 0.72` | dt: 1454.44ms | tok/sec: 11,264 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 22 | total time: 11.89m | eta: 1929.3m
step 00600/80000 (0.75%) | loss: 5.032299 | `lrm: 0.86` | dt: 1450.40ms | tok/sec: 11,296 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 27 | total time: 14.32m | eta: 1926.5m
step 00700/80000 (0.88%) | loss: 4.780718 | `lrm: 1.00 `| dt: 1460.67ms | tok/sec: 11,216 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 31 | total time: 16.74m | eta: 1923.5m


## parameter optimization 


parser.add_argument("--warmup-steps", type=int, default=700, help="number of steps for LR warmup") # 40-》700
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")

=> --warmup-steps=2000
=> --warmdown-ratio=0.1
` optimization??`
python -m scripts.base_train --depth=12 --save-every=4000 --num-iterations=22000 --run=dummy --head-dim=64 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=16 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=800 --log-every=100 --eval-every=800 --max-seq-len=1024 --warmup-steps=2000 --warmdown-ratio=0.65

=> --warmup-steps=`2000`
=> --warmdown-ratio=0.65
=> --max-seq-len=`1024`



## train -2 总结
loss ~3.0     之前是~3.4



##
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=60, `temperature=0.75`)  `#qhf`
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, `temperature=0.75`)  `#qhf`
            for sample in uncond:
                sample_str = tokenizer.decode(sample)

##

 #parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
parser.add_argument('--split-tokens', type=int, default=1*524288, help='Number of tokens to evaluate per split for BPB')

##
python -m scripts.base_eval --device-batch-size=16 --max-seq-len=2048 




##

                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░


## 3rd train 
## 
## ###############################
###
##  ********************************************************
##  ******************************************************