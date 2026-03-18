
(my_project_env) C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat>python -m scripts.base_train --depth=16 --save-every=4000 --num-iterations=200000 --run=dummy --head-dim=64 --window-pattern=L --max-seq-len=512 --device-batch-size=8 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=4000 --log-every=50 --eval-every=2000 --max-seq-len=512 --warmup-steps=2000 --warmdown-ratio=0.9 --aspect-ratio=64
C:\Users\hongf\miniconda3.1\envs\my_project_env\Lib\site-packages\torch\cuda\__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
  
  02100/200000 (1.05%) | loss: `3.601834` 
  step 04000/200000 (2.00%) | loss: 3.241555 | lrm: 1.00
  step 08000/200000 (4.00%) | loss: 3.162331 | lrm: 1.00
  step 10000/200000 (5.00%) | loss: `3.035736 `| lrm: 1.00
  step 20000/200000 (10.00%) | loss: `3.005026` | lrm: 1.00 
  
  

                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    
Autodetected device type: cuda
2026-03-16 22:43:15,107 - nanochat.common - INFO - Distributed world size: 1
2026-03-16 22:43:15,107 - nanochat.common - WARNING - Peak flops undefined for: NVIDIA GeForce RTX 4070 Laptop GPU, MFU will show as 0%
GPU: NVIDIA GeForce RTX 4070 Laptop GPU | Peak FLOPS (BF16): inf
COMPUTE_DTYPE: torch.bfloat16 (auto-detected: CUDA SM 89 (bf16 supported))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Vocab size: 8,192
Model config:
{
  "sequence_len": 512,
  "vocab_size": 8192,
  "n_layer": 16,
  "n_head": 16,
  "n_kv_head": 16,
  "n_embd": 1024,
  "window_pattern": "L"
}
Parameter counts:
wte                     : 8,388,608
value_embeds            : 67,108,864
lm_head                 : 8,388,608
transformer_matrices    : 201,328,128
scalars                 : 32
total                   : 285,214,240
Estimated FLOPs per token: 1.358964e+09
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.021531 for depth 16
Scaling the LR for the AdamW parameters ∝1/√(1024/768) = 0.866025
Using user-provided number of iterations: 200,000
Total number of training tokens: 3,276,800,000
Tokens : Scaling params ratio: 15.62
Total training FLOPs estimate: 4.453052e+18
Tokens / micro-batch / rank: 8 x 512 = 4,096
Tokens / micro-batch: 4,096
Total batch size 16,384 => gradient accumulation steps: 4
Step 00000 | Validation bpb: 3.228247
step 00000/200000 (0.00%) | loss: 9.012936 | lrm: 0.00 | dt: 1436.54ms | tok/sec: 11,405 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00050/200000 (0.03%) | loss: 8.966355 | lrm: 0.03 | dt: 1343.33ms | tok/sec: 12,196 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 3 | total time: 0.89m | eta: 4451.6m
step 00100/200000 (0.05%) | loss: 8.702426 | lrm: 0.05 | dt: 1343.52ms | tok/sec: 12,194 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 5 | total time: 2.01m | eta: 4468.5m
step 00150/200000 (0.07%) | loss: 7.078608 | lrm: 0.08 | dt: 1365.74ms | tok/sec: 11,996 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 7 | total time: 3.13m | eta: 4474.6m
step 00200/200000 (0.10%) | loss: 6.357165 | lrm: 0.10 | dt: 1349.65ms | tok/sec: 12,139 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 9 | total time: 4.26m | eta: 4478.6m
step 00250/200000 (0.12%) | loss: 6.016930 | lrm: 0.13 | dt: 1352.37ms | tok/sec: 12,115 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 11 | total time: 5.39m | eta: 4482.3m
step 00300/200000 (0.15%) | loss: 5.751772 | lrm: 0.15 | dt: 1345.24ms | tok/sec: 12,179 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 13 | total time: 6.51m | eta: 4484.4m
step 00350/200000 (0.17%) | loss: 5.546904 | lrm: 0.18 | dt: 1351.60ms | tok/sec: 12,121 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 15 | total time: 7.64m | eta: 4485.2m
step 00400/200000 (0.20%) | loss: 5.364014 | lrm: 0.20 | dt: 1349.40ms | tok/sec: 12,141 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 17 | total time: 8.76m | eta: 4485.8m
step 00450/200000 (0.23%) | loss: 5.261083 | lrm: 0.23 | dt: 1350.08ms | tok/sec: 12,135 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 19 | total time: 9.89m | eta: 4484.5m
step 00500/200000 (0.25%) | loss: 5.123309 | lrm: 0.25 | dt: 1365.43ms | tok/sec: 11,999 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 21 | total time: 11.01m | eta: 4482.1m
step 00550/200000 (0.28%) | loss: 4.984000 | lrm: 0.28 | dt: 1341.86ms | tok/sec: 12,209 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 23 | total time: 12.13m | eta: 4479.7m
step 00600/200000 (0.30%) | loss: 4.868056 | lrm: 0.30 | dt: 1346.02ms | tok/sec: 12,172 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 25 | total time: 13.25m | eta: 4477.3m
step 00650/200000 (0.33%) | loss: 4.836062 | lrm: 0.33 | dt: 1340.55ms | tok/sec: 12,221 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 27 | total time: 14.37m | eta: 4474.9m
step 00700/200000 (0.35%) | loss: 4.675088 | lrm: 0.35 | dt: 1342.60ms | tok/sec: 12,203 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 29 | total time: 15.48m | eta: 4472.6m
step 00750/200000 (0.38%) | loss: 4.602054 | lrm: 0.38 | dt: 1341.71ms | tok/sec: 12,211 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 31 | total time: 16.60m | eta: 4470.0m
 pq: 0 rg: 23 | total time: 12.13m | eta: 4479.7m
step 00600/200000 (0.30%) | loss: 4.868056 | lrm: 0.30 | dt: 1346.02ms | tok/sec: 12,172 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 25 | total time: 13.25m | eta: 4477.3m
step 00650/200000 (0.33%) | loss: 4.836062 | lrm: 0.33 | dt: 1340.55ms | tok/sec: 12,221 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 27 | total time: 14.37m | eta: 4474.9m
step 00700/200000 (0.35%) | loss: 4.675088 | lrm: 0.35 | dt: 1342.60ms | tok/sec: 12,203 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 29 | total time: 15.48m | eta: 4472.6m
step 00750/200000 (0.38%) | loss: 4.602054 | lrm: 0.38 | dt: 1341.71ms | tok/sec: 12,211 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 31 | total time: 16.60m | eta: 4470.0m
step 00700/200000 (0.35%) | loss: 4.675088 | lrm: 0.35 | dt: 1342.60ms | tok/sec: 12,203 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 29 | total time: 15.48m | eta: 4472.6m
step 00750/200000 (0.38%) | loss: 4.602054 | lrm: 0.38 | dt: 1341.71ms | tok/sec: 12,211 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 31 | total time: 16.60m | eta: 4470.0m
 pq: 0 rg: 29 | total time: 15.48m | eta: 4472.6m
step 00750/200000 (0.38%) | loss: 4.602054 | lrm: 0.38 | dt: 1341.71ms | tok/sec: 12,211 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 31 | total time: 16.60m | eta: 4470.0m
step 00800/200000 (0.40%) | loss: 4.593319 | lrm: 0.40 | dt: 1344.51ms | tok/sec: 12,185 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 34 | total time: 17.72m | eta: 4468.2m
 pq: 0 rg: 31 | total time: 16.60m | eta: 4470.0m
step 00800/200000 (0.40%) | loss: 4.593319 | lrm: 0.40 | dt: 1344.51ms | tok/sec: 12,185 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 34 | total time: 17.72m | eta: 4468.2m
step 00800/200000 (0.40%) | loss: 4.593319 | lrm: 0.40 | dt: 1344.51ms | tok/sec: 12,185 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 34 | total time: 17.72m | eta: 4468.2m
step 00850/200000 (0.42%) | loss: 4.505882 | lrm: 0.43 | dt: 1342.45ms | tok/sec: 12,204 | bf16_mfu: 0.00 | epoch: 1step 00850/200000 (0.42%) | loss: 4.505882 | lrm: 0.43 | dt: 1342.45ms | tok/sec: 12,204 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 36 | total time: 18.84m | eta: 4466.2m
step 00900/200000 (0.45%) | loss: 4.453351 | lrm: 0.45 | dt: 1352.26ms | tok/sec: 12,115 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 38 | total time: 19.96m | eta: 4464.7m
step 00950/200000 (0.47%) | loss: 4.328526 | lrm: 0.48 | dt: 1338.63ms | tok/sec: 12,239 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 40 | total time: 21.08m | eta: 4463.1m
step 01000/200000 (0.50%) | loss: 4.207698 | lrm: 0.50 | dt: 1335.93ms | tok/sec: 12,264 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 42 | total time: 22.19m | eta: 4461.1m
step 01050/200000 (0.53%) | loss: 4.099958 | lrm: 0.53 | dt: 1337.98ms | tok/sec: 12,245 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 44 | total time: 23.31m | eta: 4458.9m
 pq: 0 rg: 42 | total time: 22.19m | eta: 4461.1m
step 01050/200000 (0.53%) | loss: 4.099958 | lrm: 0.53 | dt: 1337.98ms | tok/sec: 12,245 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 44 | total time: 23.31m | eta: 4458.9m
step 01100/200000 (0.55%) | loss: 3.997190 | lrm: 0.55 | dt: 1335.48ms | tok/sec: 12,268 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 46 | total time: 24.42m | eta: 4456.8m
step 01150/200000 (0.57%) | loss: 3.986260 | lrm: 0.58 | dt: 1334.82ms | tok/sec: 12,274 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 48 | total time: 25.54m | eta: 4454.8m
step 01200/200000 (0.60%) | loss: 4.014175 | lrm: 0.60 | dt: 1335.07ms | tok/sec: 12,271 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 50 | total time: 26.66m | eta: 4453.1m
step 01250/200000 (0.62%) | loss: 3.975892 | lrm: 0.63 | dt: 1341.27ms | tok/sec: 12,215 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 52 | total time: 27.77m | eta: 4451.4m
step 01300/200000 (0.65%) | loss: 3.899507 | lrm: 0.65 | dt: 1334.51ms | tok/sec: 12,277 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 54 | total time: 28.89m | eta: 4449.6m
step 01350/200000 (0.68%) | loss: 3.919706 | lrm: 0.68 | dt: 1337.92ms | tok/sec: 12,245 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 56 | total time: 30.00m | eta: 4448.0m
step 01350/200000 (0.68%) | loss: 3.919706 | lrm: 0.68 | dt: 1337.92ms | tok/sec: 12,245 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 56 | total time: 30.00m | eta: 4448.0m
step 01400/200000 (0.70%) | loss: 3.826542 | lrm: 0.70 | dt: 1339.33ms | tok/sec: 12,232 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 58 | total time: 31.12m | eta: 4446.3m
step 01450/200000 (0.72%) | loss: 3.807975 | lrm: 0.73 | dt: 1337.06ms | tok/sec: 12,253 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 60 | total time: 32.24m | eta: 4444.7m
step 01450/200000 (0.72%) | loss: 3.807975 | lrm: 0.73 | dt: 1337.06ms | tok/sec: 12,253 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 60 | total time: 32.24m | eta: 4444.7m
step 01500/200000 (0.75%) | loss: 3.845757 | lrm: 0.75 | dt: 1338.17ms | tok/sec: 12,243 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 62 | total time: 33.35m | eta: 4443.0m
step 01550/200000 (0.78%) | loss: 3.787417 | lrm: 0.78 | dt: 1338.59ms | tok/sec: 12,239 | bf16_mfu: 0.00 | epoch: 1step 01550/200000 (0.78%) | loss: 3.787417 | lrm: 0.78 | dt: 1338.59ms | tok/sec: 12,239 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 64 | total time: 34.47m | eta: 4441.6m
step 01600/200000 (0.80%) | loss: 3.738852 | lrm: 0.80 | dt: 1341.07ms | tok/sec: 12,217 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 66 | total time: 35.58m | eta: 4440.0m
step 01650/200000 (0.82%) | loss: 3.693751 | lrm: 0.83 | dt: 1336.84ms | tok/sec: 12,255 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 68 | total time: 36.70m | eta: 4438.3m
step 01700/200000 (0.85%) | loss: 3.798705 | lrm: 0.85 | dt: 1336.94ms | tok/sec: 12,254 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 70 | total time: 37.81m | eta: 4436.7m
 pq: 0 rg: 64 | total time: 34.47m | eta: 4441.6m
step 01600/200000 (0.80%) | loss: 3.738852 | lrm: 0.80 | dt: 1341.07ms | tok/sec: 12,217 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 66 | total time: 35.58m | eta: 4440.0m
step 01650/200000 (0.82%) | loss: 3.693751 | lrm: 0.83 | dt: 1336.84ms | tok/sec: 12,255 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 68 | total time: 36.70m | eta: 4438.3m
step 01700/200000 (0.85%) | loss: 3.798705 | lrm: 0.85 | dt: 1336.94ms | tok/sec: 12,254 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 70 | total time: 37.81m | eta: 4436.7m
step 01750/200000 (0.88%) | loss: 3.723012 | lrm: 0.88 | dt: 1330.82ms | tok/sec: 12,311 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 72 | total time: 38.93m | eta: 4435.0m
step 01800/200000 (0.90%) | loss: 3.760702 | lrm: 0.90 | dt: 1334.13ms | tok/sec: 12,280 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 74 | total time: 40.04m | eta: 4433.3m
step 01850/200000 (0.93%) | loss: 3.644269 | lrm: 0.93 | dt: 1339.36ms | tok/sec: 12,232 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 76 | total time: 41.15m | eta: 4431.6m
step 01900/200000 (0.95%) | loss: 3.654337 | lrm: 0.95 | dt: 1341.21ms | tok/sec: 12,215 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 78 | total time: 42.27m | eta: 4430.0m
step 01950/200000 (0.97%) | loss: 3.703812 | lrm: 0.98 | dt: 1333.98ms | tok/sec: 12,281 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 81 | total time: 43.38m | eta: 4429.1m
Step 02000 | Validation bpb: 1.329710
step 02000/200000 (1.00%) | loss: 3.710056 | lrm: 1.00 | dt: 1415.18ms | tok/sec: 11,577 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 83 | total time: 44.50m | eta: 4427.7m
step 02050/200000 (1.02%) | loss: 3.684593 | lrm: 1.00 | dt: 1337.57ms | tok/sec: 12,249 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 1 | total time: 45.62m | eta: 4426.4m
step 02100/200000 (1.05%) | loss: `3.601834` | lrm: 1.00 | dt: 1335.76ms | tok/sec: 12,265 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 3 | total time: 46.73m | eta: 4425.1m
step 02150/200000 (1.07%) | loss: 3.685702 | lrm: 1.00 | dt: 1332.79ms | tok/sec: 12,293 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 5 | total time: 47.85m | eta: 4423.4m
step 02200/200000 (1.10%) | loss: 3.616113 | lrm: 1.00 | dt: 1335.95ms | tok/sec: 12,263 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 7 | total time: 48.96m | eta: 4421.7m
step 02250/200000 (1.12%) | loss: 3.577833 | lrm: 1.00 | dt: 1334.04ms | tok/sec: 12,281 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 9 | total time: 50.07m | eta: 4420.0m
step 02300/200000 (1.15%) | loss: 3.591042 | lrm: 1.00 | dt: 1333.78ms | tok/sec: 12,283 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 11 | total time: 51.18m | eta: 4418.4m

#
step 02350/200000 (1.18%) | loss: 3.591939 | lrm: 1.00 | dt: 1341.72ms | tok/sec: 12,211 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 13 | total time: 52.30m | eta: 4417.5m
step 02400/200000 (1.20%) | loss: 3.596543 | lrm: 1.00 | dt: 1339.74ms | tok/sec: 12,229 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 15 | total time: 53.42m | eta: 4416.7m
step 02450/200000 (1.23%) | loss: 3.549227 | lrm: 1.00 | dt: 1332.82ms | tok/sec: 12,292 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 17 | total time: 54.54m | eta: 4415.4m
step 02500/200000 (1.25%) | loss: 3.554144 | lrm: 1.00 | dt: 1412.72ms | tok/sec: 11,597 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 19 | total time: 55.65m | eta: 4414.3m
step 02550/200000 (1.27%) | loss: 3.532918 | lrm: 1.00 | dt: 1355.61ms | tok/sec: 12,086 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 21 | total time: 56.78m | eta: 4413.5m
step 02600/200000 (1.30%) | loss: 3.552083 | lrm: 1.00 | dt: 1327.71ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 23 | total time: 57.89m | eta: 4412.5m
step 02650/200000 (1.32%) | loss: 3.543668 | lrm: 1.00 | dt: 1328.76ms | tok/sec: 12,330 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 25 | total time: 59.00m | eta: 4410.8m
step 02700/200000 (1.35%) | loss: 3.518818 | lrm: 1.00 | dt: 1326.31ms | tok/sec: 12,353 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 27 | total time: 60.11m | eta: 4409.1m
step 02750/200000 (1.38%) | loss: 3.482265 | lrm: 1.00 | dt: 1333.37ms | tok/sec: 12,287 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 29 | total time: 61.22m | eta: 4407.3m
step 02800/200000 (1.40%) | loss: 3.505946 | lrm: 1.00 | dt: 1328.54ms | tok/sec: 12,332 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 31 | total time: 62.33m | eta: 4405.6m
step 02850/200000 (1.43%) | loss: 3.453892 | lrm: 1.00 | dt: 1339.31ms | tok/sec: 12,233 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 33 | total time: 63.44m | eta: 4403.9m
step 02900/200000 (1.45%) | loss: 3.453563 | lrm: 1.00 | dt: 1327.85ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 35 | total time: 64.55m | eta: 4402.2m
step 02950/200000 (1.48%) | loss: 3.454194 | lrm: 1.00 | dt: 1331.18ms | tok/sec: 12,307 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 37 | total time: 65.66m | eta: 4400.5m
step 03000/200000 (1.50%) | loss: 3.533205 | lrm: 1.00 | dt: 1330.06ms | tok/sec: 12,318 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 39 | total time: 66.76m | eta: 4398.8m
step 03050/200000 (1.52%) | loss: 3.486776 | lrm: 1.00 | dt: 1329.42ms | tok/sec: 12,324 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 41 | total time: 67.87m | eta: 4397.2m
step 03100/200000 (1.55%) | loss: 3.465331 | lrm: 1.00 | dt: 1329.62ms | tok/sec: 12,322 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 43 | total time: 68.98m | eta: 4395.4m
step 03150/200000 (1.57%) | loss: 3.398905 | lrm: 1.00 | dt: 1327.10ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 45 | total time: 70.09m | eta: 4393.8m
step 03200/200000 (1.60%) | loss: 3.372569 | lrm: 1.00 | dt: 1327.69ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 48 | total time: 71.19m | eta: 4392.2m
step 03250/200000 (1.62%) | loss: 3.422089 | lrm: 1.00 | dt: 1328.09ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 50 | total time: 72.30m | eta: 4390.7m
step 03300/200000 (1.65%) | loss: 3.427064 | lrm: 1.00 | dt: 1331.36ms | tok/sec: 12,306 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 52 | total time: 73.41m | eta: 4389.2m
step 03350/200000 (1.68%) | loss: 3.406117 | lrm: 1.00 | dt: 1327.22ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 54 | total time: 74.52m | eta: 4387.7m
step 03400/200000 (1.70%) | loss: 3.391292 | lrm: 1.00 | dt: 1325.99ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 56 | total time: 75.63m | eta: 4386.2m
step 03450/200000 (1.73%) | loss: 3.426140 | lrm: 1.00 | dt: 1321.60ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 58 | total time: 76.74m | eta: 4384.5m
step 03500/200000 (1.75%) | loss: 3.412732 | lrm: 1.00 | dt: 1328.48ms | tok/sec: 12,332 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 60 | total time: 77.84m | eta: 4382.8m
step 03550/200000 (1.77%) | loss: 3.350393 | lrm: 1.00 | dt: 1325.91ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 62 | total time: 78.95m | eta: 4381.2m
step 03600/200000 (1.80%) | loss: 3.387587 | lrm: 1.00 | dt: 1323.60ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 64 | total time: 80.05m | eta: 4379.5m
step 03650/200000 (1.82%) | loss: 3.355096 | lrm: 1.00 | dt: 1332.81ms | tok/sec: 12,292 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 66 | total time: 81.16m | eta: 4377.9m
step 03700/200000 (1.85%) | loss: 3.421419 | lrm: 1.00 | dt: 1324.32ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 68 | total time: 82.26m | eta: 4376.2m
step 03750/200000 (1.88%) | loss: 3.326296 | lrm: 1.00 | dt: 1324.47ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 70 | total time: 83.37m | eta: 4374.7m
step 03800/200000 (1.90%) | loss: 3.369105 | lrm: 1.00 | dt: 1327.04ms | tok/sec: 12,346 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 72 | total time: 84.47m | eta: 4373.1m
step 03850/200000 (1.93%) | loss: 3.377586 | lrm: 1.00 | dt: 1321.42ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 74 | total time: 85.58m | eta: 4371.5m
step 03900/200000 (1.95%) | loss: 3.323622 | lrm: 1.00 | dt: 1324.76ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 76 | total time: 86.68m | eta: 4369.8m
step 03950/200000 (1.98%) | loss: 3.339211 | lrm: 1.00 | dt: 1326.86ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 78 | total time: 87.79m | eta: 4368.3m
Step 04000 | Validation bpb: 1.210445
<|bos|>The capital of France is the capital of France, the first major crude oil company in Europe. A major crude oil company, the French cr
<|bos|>The chemical symbol of gold is the symbol of the gold value. It is considered the symbol of the value.
The number of gold values varies from the number
<|bos|>If yesterday was Friday, then tomorrow will be the time to start thinking about it. Let's start with a thought. It's hard to see why this is. The
<|bos|>The opposite of hot is the hotter the air will reach. Targets will cool down more quickly in the event the hot air flows into
<|bos|>The planets of the solar system are: 1. The planets and the other planets are the center of energy. The planets of the solar system are
<|bos|>My favorite color is the green or blue, and green is the green. It's brown, brown, red, white, blue, blue,
<|bos|>If 5*x + 3 = 13, then x is 5*x = 5*x = 3*x = 5*x = 3*x =
2026-03-17 00:13:03,501 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_004000.pt
2026-03-17 00:13:03,502 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_004000.json
2026-03-17 00:13:04,911 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_004000_rank0.pt
step 04000/200000 (2.00%) | loss: 3.241555 | lrm: 1.00 | dt: 1533.77ms | tok/sec: 10,682 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 80 | total time: 88.90m | eta: 4366.8m
step 04050/200000 (2.02%) | loss: 3.332905 | lrm: 1.00 | dt: 1324.22ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 82 | total time: 90.00m | eta: 4365.4m
step 04100/200000 (2.05%) | loss: 3.290788 | lrm: 1.00 | dt: 1322.13ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 0 | total time: 91.11m | eta: 4363.9m
step 04150/200000 (2.08%) | loss: 3.334688 | lrm: 1.00 | dt: 1324.50ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 2 | total time: 92.21m | eta: 4362.3m
step 04200/200000 (2.10%) | loss: 3.293175 | lrm: 1.00 | dt: 1325.14ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 4 | total time: 93.32m | eta: 4360.8m
step 04250/200000 (2.12%) | loss: 3.328678 | lrm: 1.00 | dt: 1349.38ms | tok/sec: 12,141 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 6 | total time: 94.43m | eta: 4359.6m
step 04300/200000 (2.15%) | loss: 3.315060 | lrm: 1.00 | dt: 1325.27ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 9 | total time: 95.54m | eta: 4358.2m
step 04350/200000 (2.17%) | loss: 3.242168 | lrm: 1.00 | dt: 1336.51ms | tok/sec: 12,258 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 11 | total time: 96.65m | eta: 4356.8m
step 04400/200000 (2.20%) | loss: 3.366221 | lrm: 1.00 | dt: 1323.49ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 13 | total time: 97.75m | eta: 4355.5m
step 04450/200000 (2.23%) | loss: 3.298253 | lrm: 1.00 | dt: 1321.57ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 15 | total time: 98.86m | eta: 4354.1m
step 04500/200000 (2.25%) | loss: 3.252968 | lrm: 1.00 | dt: 1324.44ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 17 | total time: 99.97m | eta: 4352.7m
step 04550/200000 (2.27%) | loss: 3.288515 | lrm: 1.00 | dt: 1324.60ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 19 | total time: 101.07m | eta: 4351.3m
step 04600/200000 (2.30%) | loss: 3.326643 | lrm: 1.00 | dt: 1324.48ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 21 | total time: 102.18m | eta: 4349.8m
step 04650/200000 (2.33%) | loss: 3.272766 | lrm: 1.00 | dt: 1329.07ms | tok/sec: 12,327 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 23 | total time: 103.28m | eta: 4348.3m
step 04700/200000 (2.35%) | loss: 3.247944 | lrm: 1.00 | dt: 1322.50ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 25 | total time: 104.39m | eta: 4346.8m
step 04750/200000 (2.38%) | loss: 3.296179 | lrm: 1.00 | dt: 1320.89ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 27 | total time: 105.49m | eta: 4345.3m
step 04800/200000 (2.40%) | loss: 3.249655 | lrm: 1.00 | dt: 1322.22ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 29 | total time: 106.59m | eta: 4343.8m
step 04850/200000 (2.42%) | loss: 3.258101 | lrm: 1.00 | dt: 1325.15ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 31 | total time: 107.70m | eta: 4342.4m
step 04900/200000 (2.45%) | loss: 3.279696 | lrm: 1.00 | dt: 1323.53ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 33 | total time: 108.80m | eta: 4340.9m
step 04950/200000 (2.48%) | loss: 3.283401 | lrm: 1.00 | dt: 1320.94ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 35 | total time: 109.90m | eta: 4339.4m
step 05000/200000 (2.50%) | loss: 3.253808 | lrm: 1.00 | dt: 1327.35ms | tok/sec: 12,343 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 37 | total time: 111.01m | eta: 4338.0m
step 05050/200000 (2.52%) | loss: 3.249418 | lrm: 1.00 | dt: 1322.22ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 39 | total time: 112.12m | eta: 4336.7m
step 05100/200000 (2.55%) | loss: 3.155301 | lrm: 1.00 | dt: 1326.17ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 41 | total time: 113.22m | eta: 4335.3m
step 05150/200000 (2.58%) | loss: 3.298358 | lrm: 1.00 | dt: 1323.40ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 43 | total time: 114.32m | eta: 4333.9m
step 05200/200000 (2.60%) | loss: 3.222347 | lrm: 1.00 | dt: 1323.28ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 45 | total time: 115.43m | eta: 4332.4m
step 05250/200000 (2.62%) | loss: 3.244853 | lrm: 1.00 | dt: 1318.24ms | tok/sec: 12,428 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 47 | total time: 116.53m | eta: 4331.0m
step 05300/200000 (2.65%) | loss: 3.268108 | lrm: 1.00 | dt: 1323.74ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 49 | total time: 117.63m | eta: 4329.6m
step 05350/200000 (2.67%) | loss: 3.206997 | lrm: 1.00 | dt: 1330.54ms | tok/sec: 12,313 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 51 | total time: 118.76m | eta: 4328.9m



#
step 05400/200000 (2.70%) | loss: 3.215006 | lrm: 1.00 | dt: 1356.94ms | tok/sec: 12,074 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 53 | total time: 119.87m | eta: 4327.8m
step 05450/200000 (2.73%) | loss: 3.225896 | lrm: 1.00 | dt: 1329.95ms | tok/sec: 12,319 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 55 | total time: 120.98m | eta: 4326.7m
step 05500/200000 (2.75%) | loss: 3.188925 | lrm: 1.00 | dt: 1327.53ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 57 | total time: 122.09m | eta: 4325.4m
step 05550/200000 (2.77%) | loss: 3.176772 | lrm: 1.00 | dt: 1330.98ms | tok/sec: 12,309 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 60 | total time: 123.20m | eta: 4324.2m
step 05600/200000 (2.80%) | loss: 3.222322 | lrm: 1.00 | dt: 1330.38ms | tok/sec: 12,315 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 62 | total time: 124.31m | eta: 4322.9m
step 05650/200000 (2.83%) | loss: 3.192452 | lrm: 1.00 | dt: 1326.10ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 64 | total time: 125.41m | eta: 4321.7m
step 05700/200000 (2.85%) | loss: 3.154389 | lrm: 1.00 | dt: 1325.16ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 66 | total time: 126.52m | eta: 4320.4m
step 05750/200000 (2.88%) | loss: 3.283577 | lrm: 1.00 | dt: 1329.31ms | tok/sec: 12,325 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 68 | total time: 127.63m | eta: 4319.1m
step 05800/200000 (2.90%) | loss: 3.254617 | lrm: 1.00 | dt: 1326.40ms | tok/sec: 12,352 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 70 | total time: 128.73m | eta: 4317.8m
step 05850/200000 (2.92%) | loss: 3.209286 | lrm: 1.00 | dt: 1325.35ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 72 | total time: 129.84m | eta: 4316.5m
step 05900/200000 (2.95%) | loss: 3.210749 | lrm: 1.00 | dt: 1325.27ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 74 | total time: 130.95m | eta: 4315.2m
step 05950/200000 (2.98%) | loss: 3.235054 | lrm: 1.00 | dt: 1324.74ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 76 | total time: 132.06m | eta: 4314.0m
Step 06000 | Validation bpb: 1.163700
step 06000/200000 (3.00%) | loss: 3.169483 | lrm: 1.00 | dt: 1434.02ms | tok/sec: 11,425 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 78 | total time: 133.16m | eta: 4312.8m
step 06050/200000 (3.02%) | loss: 3.226744 | lrm: 1.00 | dt: 1323.40ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 80 | total time: 134.27m | eta: 4311.7m
step 06100/200000 (3.05%) | loss: 3.256812 | lrm: 1.00 | dt: 1322.52ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 82 | total time: 135.38m | eta: 4310.4m
step 06150/200000 (3.08%) | loss: 3.186723 | lrm: 1.00 | dt: 1328.06ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 1 | total time: 136.49m | eta: 4309.2m
step 06200/200000 (3.10%) | loss: 3.182638 | lrm: 1.00 | dt: 1325.04ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 3 | total time: 137.59m | eta: 4307.9m
step 06250/200000 (3.12%) | loss: 3.281235 | lrm: 1.00 | dt: 1321.07ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 5 | total time: 138.70m | eta: 4306.5m
step 06300/200000 (3.15%) | loss: 3.181302 | lrm: 1.00 | dt: 1320.97ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 7 | total time: 139.80m | eta: 4305.1m
step 06350/200000 (3.17%) | loss: 3.111651 | lrm: 1.00 | dt: 1325.52ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 9 | total time: 140.90m | eta: 4303.7m
step 06400/200000 (3.20%) | loss: 3.183112 | lrm: 1.00 | dt: 1322.89ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 11 | total time: 142.00m | eta: 4302.4m
step 06450/200000 (3.23%) | loss: 3.178796 | lrm: 1.00 | dt: 1324.10ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 13 | total time: 143.11m | eta: 4301.0m
step 06500/200000 (3.25%) | loss: 3.147900 | lrm: 1.00 | dt: 1321.64ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 15 | total time: 144.21m | eta: 4299.7m
step 06550/200000 (3.27%) | loss: 3.188440 | lrm: 1.00 | dt: 1320.69ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 17 | total time: 145.31m | eta: 4298.3m
step 06600/200000 (3.30%) | loss: 3.184038 | lrm: 1.00 | dt: 1321.51ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 19 | total time: 146.41m | eta: 4296.9m
step 06650/200000 (3.33%) | loss: 3.103078 | lrm: 1.00 | dt: 1320.24ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 21 | total time: 147.52m | eta: 4295.5m
step 06700/200000 (3.35%) | loss: 3.121187 | lrm: 1.00 | dt: 1324.13ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 23 | total time: 148.62m | eta: 4294.2m
step 06750/200000 (3.38%) | loss: 3.169606 | lrm: 1.00 | dt: 1321.46ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 25 | total time: 149.72m | eta: 4292.9m
step 06800/200000 (3.40%) | loss: 3.160898 | lrm: 1.00 | dt: 1330.66ms | tok/sec: 12,312 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 27 | total time: 150.83m | eta: 4291.5m
step 06850/200000 (3.42%) | loss: 3.155626 | lrm: 1.00 | dt: 1318.44ms | tok/sec: 12,426 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 29 | total time: 151.93m | eta: 4290.2m
step 06900/200000 (3.45%) | loss: 3.216773 | lrm: 1.00 | dt: 1319.98ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 32 | total time: 153.03m | eta: 4288.8m
step 06950/200000 (3.48%) | loss: 3.206837 | lrm: 1.00 | dt: 1324.50ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 34 | total time: 154.13m | eta: 4287.5m
step 07000/200000 (3.50%) | loss: 3.227001 | lrm: 1.00 | dt: 1325.83ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 36 | total time: 155.24m | eta: 4286.2m
step 07050/200000 (3.52%) | loss: 3.169776 | lrm: 1.00 | dt: 1321.47ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 38 | total time: 156.34m | eta: 4284.9m
step 07100/200000 (3.55%) | loss: 3.166035 | lrm: 1.00 | dt: 1321.62ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 40 | total time: 157.44m | eta: 4283.6m
step 07150/200000 (3.58%) | loss: 3.204118 | lrm: 1.00 | dt: 1323.88ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 42 | total time: 158.55m | eta: 4282.3m
step 07200/200000 (3.60%) | loss: 3.175864 | lrm: 1.00 | dt: 1322.22ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 44 | total time: 159.65m | eta: 4281.0m
step 07250/200000 (3.62%) | loss: 3.186117 | lrm: 1.00 | dt: 1323.15ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 46 | total time: 160.75m | eta: 4279.7m
step 07300/200000 (3.65%) | loss: 3.181051 | lrm: 1.00 | dt: 1325.41ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 48 | total time: 161.86m | eta: 4278.5m
step 07350/200000 (3.67%) | loss: 3.123119 | lrm: 1.00 | dt: 1324.67ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 50 | total time: 162.96m | eta: 4277.2m
step 07400/200000 (3.70%) | loss: 3.107512 | lrm: 1.00 | dt: 1318.95ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 52 | total time: 164.06m | eta: 4275.9m
step 07450/200000 (3.73%) | loss: 3.180693 | lrm: 1.00 | dt: 1323.96ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 54 | total time: 165.17m | eta: 4274.6m
step 07500/200000 (3.75%) | loss: 3.093797 | lrm: 1.00 | dt: 1321.79ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 56 | total time: 166.27m | eta: 4273.3m
step 07550/200000 (3.77%) | loss: 3.102874 | lrm: 1.00 | dt: 1323.83ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 58 | total time: 167.37m | eta: 4272.0m
step 07600/200000 (3.80%) | loss: 3.130881 | lrm: 1.00 | dt: 1324.67ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 60 | total time: 168.48m | eta: 4270.7m
step 07650/200000 (3.83%) | loss: 3.159053 | lrm: 1.00 | dt: 1323.37ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 62 | total time: 169.58m | eta: 4269.4m
step 07700/200000 (3.85%) | loss: 3.032515 | lrm: 1.00 | dt: 1319.04ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 64 | total time: 170.68m | eta: 4268.1m
step 07750/200000 (3.88%) | loss: 3.139714 | lrm: 1.00 | dt: 1321.02ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 66 | total time: 171.78m | eta: 4266.8m
step 07800/200000 (3.90%) | loss: 3.055480 | lrm: 1.00 | dt: 1321.52ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 68 | total time: 172.89m | eta: 4265.6m
step 07850/200000 (3.92%) | loss: 3.152125 | lrm: 1.00 | dt: 1324.14ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 70 | total time: 173.99m | eta: 4264.3m
step 07900/200000 (3.95%) | loss: 3.049871 | lrm: 1.00 | dt: 1322.27ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 72 | total time: 175.09m | eta: 4263.0m
step 07950/200000 (3.98%) | loss: 3.139507 | lrm: 1.00 | dt: 1322.53ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 74 | total time: 176.20m | eta: 4261.8m
Step 08000 | Validation bpb: 1.136531
<|bos|>The capital of France is the capital of the European Union. Due to the rise in access to the market, France's economy is being lost
<|bos|>The chemical symbol of gold is the gold's symbol. It has a great deal of symbolic value, including gold. The gold symbol is also called gold
<|bos|>If yesterday was Friday, then tomorrow will be the first time we've gone back to Dominican Republic. If you don't get tomorrow,
<|bos|>The opposite of hot is the heat loss from a hot object as it moves through the air. The hot object can be directly connected to the hot object
<|bos|>The planets of the solar system are: 1) the planet inhabits the planet's coldest temperature in its solar system. 2) the planet in
<|bos|>My favorite color is red, red, or purple. What many people like to do is "paint in" a room. I don't
<|bos|>If 5*x + 3 = 13, then x is the number of times a=cos(\textendor{c}^{3+1} ==
2026-03-17 01:41:57,065 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_008000.pt
2026-03-17 01:41:57,066 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_008000.json
2026-03-17 01:41:58,492 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_008000_rank0.pt
step 08000/200000 (4.00%) | loss: 3.162331 | lrm: 1.00 | dt: 1580.97ms | tok/sec: 10,363 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 76 | total time: 177.30m | eta: 4260.6m
step 08050/200000 (4.03%) | loss: 3.083595 | lrm: 1.00 | dt: 1324.48ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 78 | total time: 178.41m | eta: 4259.4m
step 08100/200000 (4.05%) | loss: 3.168265 | lrm: 1.00 | dt: 1322.84ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 81 | total time: 179.51m | eta: 4258.1m
step 08150/200000 (4.08%) | loss: 3.178610 | lrm: 1.00 | dt: 1322.93ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 1 | total time: 180.61m | eta: 4256.9m
step 08200/200000 (4.10%) | loss: 3.099278 | lrm: 1.00 | dt: 1320.10ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 3 | total time: 181.72m | eta: 4255.6m
step 08250/200000 (4.12%) | loss: 3.097748 | lrm: 1.00 | dt: 1343.24ms | tok/sec: 12,197 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 5 | total time: 182.82m | eta: 4254.3m
step 08300/200000 (4.15%) | loss: 3.177000 | lrm: 1.00 | dt: 1323.18ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 7 | total time: 183.92m | eta: 4253.1m
step 08350/200000 (4.17%) | loss: 3.095734 | lrm: 1.00 | dt: 1322.70ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 9 | total time: 185.03m | eta: 4251.8m
step 08400/200000 (4.20%) | loss: 3.157037 | lrm: 1.00 | dt: 1321.84ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 11 | total time: 186.13m | eta: 4250.6m
step 08450/200000 (4.22%) | loss: 3.116138 | lrm: 1.00 | dt: 1325.76ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 13 | total time: 187.23m | eta: 4249.4m
step 08500/200000 (4.25%) | loss: 3.067950 | lrm: 1.00 | dt: 1324.49ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 15 | total time: 188.34m | eta: 4248.1m
step 08550/200000 (4.28%) | loss: 3.143234 | lrm: 1.00 | dt: 1340.19ms | tok/sec: 12,225 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 17 | total time: 189.44m | eta: 4247.0m
step 08600/200000 (4.30%) | loss: 3.103723 | lrm: 1.00 | dt: 1323.47ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 19 | total time: 190.55m | eta: 4245.8m
step 08650/200000 (4.33%) | loss: 3.148774 | lrm: 1.00 | dt: 1322.03ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 21 | total time: 191.65m | eta: 4244.5m
step 08700/200000 (4.35%) | loss: 3.058840 | lrm: 1.00 | dt: 1325.46ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 23 | total time: 192.76m | eta: 4243.3m
step 08750/200000 (4.38%) | loss: 3.126102 | lrm: 1.00 | dt: 1318.63ms | tok/sec: 12,424 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 25 | total time: 193.86m | eta: 4242.1m
step 08800/200000 (4.40%) | loss: 3.179175 | lrm: 1.00 | dt: 1322.31ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 27 | total time: 194.96m | eta: 4240.8m
step 08850/200000 (4.42%) | loss: 3.080850 | lrm: 1.00 | dt: 1322.88ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 29 | total time: 196.06m | eta: 4239.5m
step 08900/200000 (4.45%) | loss: 3.108343 | lrm: 1.00 | dt: 1318.41ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 31 | total time: 197.17m | eta: 4238.3m
step 08950/200000 (4.47%) | loss: 3.181426 | lrm: 1.00 | dt: 1321.96ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 33 | total time: 198.27m | eta: 4237.1m
step 09000/200000 (4.50%) | loss: 3.092531 | lrm: 1.00 | dt: 1326.65ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 35 | total time: 199.37m | eta: 4235.9m
step 09050/200000 (4.53%) | loss: 3.120997 | lrm: 1.00 | dt: 1322.62ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 37 | total time: 200.48m | eta: 4234.6m
step 09100/200000 (4.55%) | loss: 3.045994 | lrm: 1.00 | dt: 1319.72ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 39 | total time: 201.58m | eta: 4233.4m
step 09150/200000 (4.58%) | loss: 3.155200 | lrm: 1.00 | dt: 1323.95ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 41 | total time: 202.68m | eta: 4232.2m
step 09200/200000 (4.60%) | loss: 3.185110 | lrm: 1.00 | dt: 1320.72ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 43 | total time: 203.78m | eta: 4230.9m
step 09250/200000 (4.62%) | loss: 3.146000 | lrm: 1.00 | dt: 1317.86ms | tok/sec: 12,432 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 45 | total time: 204.89m | eta: 4229.7m
step 09300/200000 (4.65%) | loss: 3.104702 | lrm: 1.00 | dt: 1324.24ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 47 | total time: 205.99m | eta: 4228.5m
step 09350/200000 (4.67%) | loss: 3.102255 | lrm: 1.00 | dt: 1323.38ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 50 | total time: 207.09m | eta: 4227.2m
step 09400/200000 (4.70%) | loss: 3.073798 | lrm: 1.00 | dt: 1320.81ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 52 | total time: 208.20m | eta: 4226.0m
step 09450/200000 (4.72%) | loss: 3.070213 | lrm: 1.00 | dt: 1322.79ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 54 | total time: 209.30m | eta: 4224.8m
step 09500/200000 (4.75%) | loss: 3.028829 | lrm: 1.00 | dt: 1324.27ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 56 | total time: 210.40m | eta: 4223.6m
step 09550/200000 (4.78%) | loss: 3.134032 | lrm: 1.00 | dt: 1322.89ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 58 | total time: 211.51m | eta: 4222.3m
step 09600/200000 (4.80%) | loss: 3.134561 | lrm: 1.00 | dt: 1320.68ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 60 | total time: 212.61m | eta: 4221.1m
step 09650/200000 (4.83%) | loss: 3.102967 | lrm: 1.00 | dt: 1323.39ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 62 | total time: 213.71m | eta: 4219.9m
step 09700/200000 (4.85%) | loss: 3.117637 | lrm: 1.00 | dt: 1317.55ms | tok/sec: 12,435 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 64 | total time: 214.81m | eta: 4218.7m
step 09750/200000 (4.88%) | loss: 3.161123 | lrm: 1.00 | dt: 1320.34ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 66 | total time: 215.92m | eta: 4217.5m
step 09800/200000 (4.90%) | loss: 2.992919 | lrm: 1.00 | dt: 1324.83ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 68 | total time: 217.02m | eta: 4216.2m
step 09850/200000 (4.92%) | loss: 3.095703 | lrm: 1.00 | dt: 1321.35ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 70 | total time: 218.12m | eta: 4215.0m
step 09900/200000 (4.95%) | loss: 3.088143 | lrm: 1.00 | dt: 1322.27ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 72 | total time: 219.22m | eta: 4213.8m
step 09950/200000 (4.97%) | loss: 3.042953 | lrm: 1.00 | dt: 1326.58ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 74 | total time: 220.33m | eta: 4212.6m
Step 10000 | Validation bpb: 1.117319
step 10000/200000 (5.00%) | loss: `3.035736 `| lrm: 1.00 | dt: 1420.80ms | tok/sec: 11,531 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 76 | total time: 221.43m | eta: 4211.4m
step 10050/200000 (5.03%) | loss: 3.044729 | lrm: 1.00 | dt: 1324.41ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 78 | total time: 222.54m | eta: 4210.3m
step 10100/200000 (5.05%) | loss: 3.069374 | lrm: 1.00 | dt: 1324.61ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 80 | total time: 223.64m | eta: 4209.1m
step 10150/200000 (5.08%) | loss: 3.108456 | lrm: 1.00 | dt: 1320.09ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 0 | total time: 224.74m | eta: 4207.8m
step 10200/200000 (5.10%) | loss: 3.100946 | lrm: 1.00 | dt: 1323.08ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 2 | total time: 225.85m | eta: 4206.6m
step 10250/200000 (5.12%) | loss: 3.052332 | lrm: 1.00 | dt: 1326.88ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 4 | total time: 226.95m | eta: 4205.4m
step 10300/200000 (5.15%) | loss: 3.082705 | lrm: 1.00 | dt: 1322.30ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 6 | total time: 228.05m | eta: 4204.2m
step 10350/200000 (5.17%) | loss: 3.095321 | lrm: 1.00 | dt: 1321.80ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 8 | total time: 229.16m | eta: 4203.0m
step 10400/200000 (5.20%) | loss: 3.059853 | lrm: 1.00 | dt: 1325.07ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 10 | total time: 230.26m | eta: 4201.8m
step 10450/200000 (5.22%) | loss: 3.075342 | lrm: 1.00 | dt: 1321.54ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 12 | total time: 231.36m | eta: 4200.6m
step 10500/200000 (5.25%) | loss: 3.088703 | lrm: 1.00 | dt: 1324.79ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 15 | total time: 232.46m | eta: 4199.4m
step 10550/200000 (5.28%) | loss: 3.098246 | lrm: 1.00 | dt: 1323.84ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 17 | total time: 233.57m | eta: 4198.2m
step 10600/200000 (5.30%) | loss: 2.988033 | lrm: 1.00 | dt: 1326.21ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 19 | total time: 234.67m | eta: 4197.0m
step 10650/200000 (5.33%) | loss: 3.022786 | lrm: 1.00 | dt: 1321.68ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 21 | total time: 235.77m | eta: 4195.9m
step 10700/200000 (5.35%) | loss: 3.084746 | lrm: 1.00 | dt: 1324.15ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 23 | total time: 236.88m | eta: 4194.7m
step 10750/200000 (5.38%) | loss: 2.999996 | lrm: 1.00 | dt: 1324.15ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 25 | total time: 237.98m | eta: 4193.5m
step 10800/200000 (5.40%) | loss: 3.057863 | lrm: 1.00 | dt: 1323.34ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 27 | total time: 239.09m | eta: 4192.4m
step 10850/200000 (5.42%) | loss: 3.135000 | lrm: 1.00 | dt: 1324.04ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 29 | total time: 240.19m | eta: 4191.2m
step 10900/200000 (5.45%) | loss: 3.066289 | lrm: 1.00 | dt: 1324.62ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 31 | total time: 241.30m | eta: 4190.0m
step 10950/200000 (5.47%) | loss: 3.028038 | lrm: 1.00 | dt: 1324.75ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 33 | total time: 242.40m | eta: 4188.8m
step 11000/200000 (5.50%) | loss: 3.099428 | lrm: 1.00 | dt: 1327.22ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 35 | total time: 243.50m | eta: 4187.7m
step 11050/200000 (5.53%) | loss: 3.086980 | lrm: 1.00 | dt: 1326.21ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 37 | total time: 244.61m | eta: 4186.5m
step 11100/200000 (5.55%) | loss: 3.022269 | lrm: 1.00 | dt: 1323.05ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 39 | total time: 245.71m | eta: 4185.3m
step 11150/200000 (5.58%) | loss: 3.034335 | lrm: 1.00 | dt: 1318.87ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 41 | total time: 246.82m | eta: 4184.1m
step 11200/200000 (5.60%) | loss: 3.038867 | lrm: 1.00 | dt: 1327.17ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 43 | total time: 247.92m | eta: 4182.9m
step 11250/200000 (5.62%) | loss: 3.100849 | lrm: 1.00 | dt: 1326.49ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 45 | total time: 249.02m | eta: 4181.7m
step 11300/200000 (5.65%) | loss: 2.978733 | lrm: 1.00 | dt: 1322.55ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 47 | total time: 250.12m | eta: 4180.5m
step 11350/200000 (5.67%) | loss: 3.092977 | lrm: 1.00 | dt: 1321.58ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 49 | total time: 251.22m | eta: 4179.3m
step 11400/200000 (5.70%) | loss: 3.033050 | lrm: 1.00 | dt: 1323.09ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 51 | total time: 252.33m | eta: 4178.1m
step 11450/200000 (5.72%) | loss: 3.078104 | lrm: 1.00 | dt: 1320.67ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 53 | total time: 253.43m | eta: 4177.0m
step 11500/200000 (5.75%) | loss: 3.044814 | lrm: 1.00 | dt: 1321.72ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 55 | total time: 254.54m | eta: 4175.8m
step 11550/200000 (5.78%) | loss: 3.058640 | lrm: 1.00 | dt: 1323.51ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 57 | total time: 255.64m | eta: 4174.6m
step 11600/200000 (5.80%) | loss: 3.032605 | lrm: 1.00 | dt: 1324.96ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 59 | total time: 256.74m | eta: 4173.4m
step 11650/200000 (5.83%) | loss: 3.009534 | lrm: 1.00 | dt: 1322.95ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 62 | total time: 257.84m | eta: 4172.3m
step 11700/200000 (5.85%) | loss: 3.094373 | lrm: 1.00 | dt: 1323.65ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 64 | total time: 258.95m | eta: 4171.1m
step 11750/200000 (5.88%) | loss: 3.048626 | lrm: 1.00 | dt: 1321.30ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 66 | total time: 260.05m | eta: 4169.8m
step 11800/200000 (5.90%) | loss: 3.082619 | lrm: 1.00 | dt: 1318.86ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 68 | total time: 261.15m | eta: 4168.7m
step 11850/200000 (5.92%) | loss: 3.036229 | lrm: 1.00 | dt: 1319.18ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 70 | total time: 262.26m | eta: 4167.5m
step 11900/200000 (5.95%) | loss: 3.015166 | lrm: 1.00 | dt: 1325.47ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 72 | total time: 263.36m | eta: 4166.3m
step 11950/200000 (5.97%) | loss: 3.014982 | lrm: 1.00 | dt: 1320.85ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 74 | total time: 264.46m | eta: 4165.1m
Step 12000 | Validation bpb: 1.103155
<|bos|>The capital of France is the Caesarea Chat Grammar, which translates to "Grammar." It is also the
<|bos|>The chemical symbol of gold is the symbol of the chemical symbol SiO2. The chemical symbol of gold is the symbol of the chemical symbol of the
<|bos|>If yesterday was Friday, then tomorrow will be the day of the year that I could finally show up in the world of dance.
For many people who are in the
<|bos|>The opposite of hot is the cold of the day. When your heat is still hot, the hot molecules are not attracted to the cold air.

<|bos|>The planets of the solar system are: 1. The planets of the solar system are: 2. The planets of the system are: 3
<|bos|>My favorite color is black. That's because the darker the color, the lighter the color.
I think it's the color that the h
<|bos|>If 5*x + 3 = 13, then x is 0.5, so y is 0.5, y is 0.5, so x is 0
2026-03-17 03:10:41,812 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_012000.pt
2026-03-17 03:10:41,814 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_012000.json
2026-03-17 03:10:43,190 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_012000_rank0.pt
step 12000/200000 (6.00%) | loss: 2.985322 | lrm: 1.00 | dt: 1574.12ms | tok/sec: 10,408 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 76 | total time: 265.57m | eta: 4164.0m
step 12050/200000 (6.03%) | loss: 3.018081 | lrm: 1.00 | dt: 1323.06ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 78 | total time: 266.67m | eta: 4162.9m
step 12100/200000 (6.05%) | loss: 3.019755 | lrm: 1.00 | dt: 1320.38ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 80 | total time: 267.78m | eta: 4161.7m
step 12150/200000 (6.08%) | loss: 3.072513 | lrm: 1.00 | dt: 1323.07ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 82 | total time: 268.88m | eta: 4160.5m
step 12200/200000 (6.10%) | loss: 3.025433 | lrm: 1.00 | dt: 1319.86ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 1 | total time: 269.98m | eta: 4159.4m
step 12250/200000 (6.12%) | loss: 3.130235 | lrm: 1.00 | dt: 1319.11ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 3 | total time: 271.08m | eta: 4158.2m
step 12300/200000 (6.15%) | loss: 3.053639 | lrm: 1.00 | dt: 1319.61ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 5 | total time: 272.19m | eta: 4157.0m
step 12350/200000 (6.17%) | loss: 3.022874 | lrm: 1.00 | dt: 1323.36ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 7 | total time: 273.29m | eta: 4155.8m
step 12400/200000 (6.20%) | loss: 3.073105 | lrm: 1.00 | dt: 1320.92ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 9 | total time: 274.39m | eta: 4154.6m
step 12450/200000 (6.22%) | loss: 2.990782 | lrm: 1.00 | dt: 1324.64ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 11 | total time: 275.49m | eta: 4153.5m
step 12500/200000 (6.25%) | loss: 3.066507 | lrm: 1.00 | dt: 1321.07ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 13 | total time: 276.60m | eta: 4152.3m
step 12550/200000 (6.28%) | loss: 2.936495 | lrm: 1.00 | dt: 1323.93ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 15 | total time: 277.70m | eta: 4151.1m
step 12600/200000 (6.30%) | loss: 3.088483 | lrm: 1.00 | dt: 1317.29ms | tok/sec: 12,437 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 17 | total time: 278.80m | eta: 4149.9m
step 12650/200000 (6.33%) | loss: 2.955520 | lrm: 1.00 | dt: 1321.37ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 19 | total time: 279.90m | eta: 4148.7m
step 12700/200000 (6.35%) | loss: 3.030208 | lrm: 1.00 | dt: 1322.80ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 21 | total time: 281.01m | eta: 4147.6m
step 12750/200000 (6.38%) | loss: 3.069637 | lrm: 1.00 | dt: 1322.10ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 23 | total time: 282.11m | eta: 4146.4m
step 12800/200000 (6.40%) | loss: 3.025853 | lrm: 1.00 | dt: 1317.84ms | tok/sec: 12,432 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 25 | total time: 283.21m | eta: 4145.2m
step 12850/200000 (6.42%) | loss: 3.001773 | lrm: 1.00 | dt: 1326.10ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 28 | total time: 284.31m | eta: 4144.0m
step 12900/200000 (6.45%) | loss: 3.010016 | lrm: 1.00 | dt: 1321.43ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 30 | total time: 285.42m | eta: 4142.8m
step 12950/200000 (6.47%) | loss: 2.938941 | lrm: 1.00 | dt: 1320.63ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 32 | total time: 286.52m | eta: 4141.7m
step 13000/200000 (6.50%) | loss: 2.976406 | lrm: 1.00 | dt: 1319.66ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 34 | total time: 287.62m | eta: 4140.5m
step 13050/200000 (6.53%) | loss: 3.039866 | lrm: 1.00 | dt: 1319.99ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 36 | total time: 288.72m | eta: 4139.3m
step 13100/200000 (6.55%) | loss: 3.059580 | lrm: 1.00 | dt: 1325.49ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 38 | total time: 289.82m | eta: 4138.1m
step 13150/200000 (6.58%) | loss: 3.015046 | lrm: 1.00 | dt: 1319.18ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 40 | total time: 290.93m | eta: 4136.9m
step 13200/200000 (6.60%) | loss: 3.039918 | lrm: 1.00 | dt: 1320.43ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 42 | total time: 292.03m | eta: 4135.8m
step 13250/200000 (6.62%) | loss: 3.027372 | lrm: 1.00 | dt: 1322.98ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 44 | total time: 293.13m | eta: 4134.6m
step 13300/200000 (6.65%) | loss: 2.954279 | lrm: 1.00 | dt: 1323.43ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 46 | total time: 294.23m | eta: 4133.4m
step 13350/200000 (6.67%) | loss: 3.046074 | lrm: 1.00 | dt: 1318.38ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 48 | total time: 295.34m | eta: 4132.3m
step 13400/200000 (6.70%) | loss: 3.107068 | lrm: 1.00 | dt: 1335.36ms | tok/sec: 12,269 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 50 | total time: 296.44m | eta: 4131.1m
step 13450/200000 (6.72%) | loss: 2.983684 | lrm: 1.00 | dt: 1320.81ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 52 | total time: 297.54m | eta: 4130.0m
step 13500/200000 (6.75%) | loss: 3.028856 | lrm: 1.00 | dt: 1321.06ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 54 | total time: 298.65m | eta: 4128.8m
step 13550/200000 (6.78%) | loss: 2.987543 | lrm: 1.00 | dt: 1322.01ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 56 | total time: 299.75m | eta: 4127.6m
step 13600/200000 (6.80%) | loss: 3.007644 | lrm: 1.00 | dt: 1320.94ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 58 | total time: 300.85m | eta: 4126.4m
step 13650/200000 (6.83%) | loss: 3.051239 | lrm: 1.00 | dt: 1320.02ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 60 | total time: 301.95m | eta: 4125.3m
step 13700/200000 (6.85%) | loss: 2.997412 | lrm: 1.00 | dt: 1325.02ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 62 | total time: 303.06m | eta: 4124.1m
step 13750/200000 (6.88%) | loss: 3.056966 | lrm: 1.00 | dt: 1324.51ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 64 | total time: 304.16m | eta: 4123.0m
step 13800/200000 (6.90%) | loss: 3.015902 | lrm: 1.00 | dt: 1321.97ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 66 | total time: 305.26m | eta: 4121.8m
step 13850/200000 (6.92%) | loss: 3.035555 | lrm: 1.00 | dt: 1321.20ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 68 | total time: 306.36m | eta: 4120.6m
step 13900/200000 (6.95%) | loss: 2.963630 | lrm: 1.00 | dt: 1321.22ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 70 | total time: 307.46m | eta: 4119.4m
step 13950/200000 (6.97%) | loss: 2.933881 | lrm: 1.00 | dt: 1326.80ms | tok/sec: 12,348 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 72 | total time: 308.57m | eta: 4118.3m
Step 14000 | Validation bpb: 1.092670
step 14000/200000 (7.00%) | loss: 2.991344 | lrm: 1.00 | dt: 1420.01ms | tok/sec: 11,537 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 74 | total time: 309.67m | eta: 4117.1m
step 14050/200000 (7.03%) | loss: 2.989738 | lrm: 1.00 | dt: 1326.53ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 76 | total time: 310.77m | eta: 4116.0m
step 14100/200000 (7.05%) | loss: 3.075892 | lrm: 1.00 | dt: 1321.03ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 78 | total time: 311.88m | eta: 4114.8m
step 14150/200000 (7.08%) | loss: 2.959904 | lrm: 1.00 | dt: 1321.81ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 81 | total time: 312.98m | eta: 4113.7m
step 14200/200000 (7.10%) | loss: 3.010315 | lrm: 1.00 | dt: 1321.75ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 1 | total time: 314.08m | eta: 4112.5m
step 14250/200000 (7.12%) | loss: 3.089072 | lrm: 1.00 | dt: 1322.74ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 3 | total time: 315.19m | eta: 4111.4m
step 14300/200000 (7.15%) | loss: 2.951582 | lrm: 1.00 | dt: 1330.86ms | tok/sec: 12,310 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 5 | total time: 316.29m | eta: 4110.2m
step 14350/200000 (7.17%) | loss: 3.058574 | lrm: 1.00 | dt: 1320.59ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 7 | total time: 317.39m | eta: 4109.0m
step 14400/200000 (7.20%) | loss: 3.040030 | lrm: 1.00 | dt: 1322.26ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 9 | total time: 318.50m | eta: 4107.9m
step 14450/200000 (7.22%) | loss: 3.066267 | lrm: 1.00 | dt: 1327.12ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 11 | total time: 319.60m | eta: 4106.7m
step 14500/200000 (7.25%) | loss: 2.993747 | lrm: 1.00 | dt: 1323.50ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 13 | total time: 320.70m | eta: 4105.6m
step 14550/200000 (7.28%) | loss: 3.025108 | lrm: 1.00 | dt: 1322.72ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 15 | total time: 321.80m | eta: 4104.4m
step 14600/200000 (7.30%) | loss: 2.942146 | lrm: 1.00 | dt: 1321.37ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 17 | total time: 322.91m | eta: 4103.4m
step 14650/200000 (7.33%) | loss: 2.956187 | lrm: 1.00 | dt: 1322.56ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 19 | total time: 324.02m | eta: 4102.2m
step 14700/200000 (7.35%) | loss: 2.939376 | lrm: 1.00 | dt: 1321.58ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 21 | total time: 325.12m | eta: 4101.1m
step 14750/200000 (7.38%) | loss: 2.987782 | lrm: 1.00 | dt: 1324.54ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 23 | total time: 326.22m | eta: 4099.9m
step 14800/200000 (7.40%) | loss: 2.998833 | lrm: 1.00 | dt: 1323.14ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 25 | total time: 327.32m | eta: 4098.7m
step 14850/200000 (7.42%) | loss: 2.994693 | lrm: 1.00 | dt: 1321.21ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 27 | total time: 328.43m | eta: 4097.6m
step 14900/200000 (7.45%) | loss: 3.054115 | lrm: 1.00 | dt: 1321.83ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 29 | total time: 329.53m | eta: 4096.5m
step 14950/200000 (7.47%) | loss: 2.967560 | lrm: 1.00 | dt: 1321.76ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 31 | total time: 330.63m | eta: 4095.3m
step 15000/200000 (7.50%) | loss: 3.037015 | lrm: 1.00 | dt: 1329.27ms | tok/sec: 12,325 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 33 | total time: 331.74m | eta: 4094.2m
step 15050/200000 (7.53%) | loss: 3.020217 | lrm: 1.00 | dt: 1323.77ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 35 | total time: 332.85m | eta: 4093.1m
step 15100/200000 (7.55%) | loss: 2.952432 | lrm: 1.00 | dt: 1320.91ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 37 | total time: 333.95m | eta: 4091.9m
step 15150/200000 (7.58%) | loss: 2.914043 | lrm: 1.00 | dt: 1335.83ms | tok/sec: 12,265 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 39 | total time: 335.05m | eta: 4090.8m
step 15200/200000 (7.60%) | loss: 2.985267 | lrm: 1.00 | dt: 1320.29ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 41 | total time: 336.16m | eta: 4089.6m
step 15250/200000 (7.62%) | loss: 2.974314 | lrm: 1.00 | dt: 1317.34ms | tok/sec: 12,437 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 43 | total time: 337.26m | eta: 4088.5m
step 15300/200000 (7.65%) | loss: 2.983345 | lrm: 1.00 | dt: 1321.22ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 45 | total time: 338.36m | eta: 4087.3m
step 15350/200000 (7.67%) | loss: 2.963960 | lrm: 1.00 | dt: 1320.00ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 47 | total time: 339.46m | eta: 4086.2m
step 15400/200000 (7.70%) | loss: 2.994250 | lrm: 1.00 | dt: 1324.92ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 50 | total time: 340.57m | eta: 4085.0m
step 15450/200000 (7.72%) | loss: 2.953158 | lrm: 1.00 | dt: 1323.72ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 52 | total time: 341.67m | eta: 4083.9m
step 15500/200000 (7.75%) | loss: 2.974885 | lrm: 1.00 | dt: 1323.06ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 54 | total time: 342.77m | eta: 4082.7m
step 15550/200000 (7.78%) | loss: 2.988198 | lrm: 1.00 | dt: 1320.72ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 56 | total time: 343.87m | eta: 4081.6m
step 15600/200000 (7.80%) | loss: 2.967212 | lrm: 1.00 | dt: 1325.53ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 58 | total time: 344.98m | eta: 4080.4m
step 15650/200000 (7.83%) | loss: 2.990274 | lrm: 1.00 | dt: 1328.14ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 60 | total time: 346.08m | eta: 4079.3m
step 15700/200000 (7.85%) | loss: 2.999576 | lrm: 1.00 | dt: 1320.63ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 62 | total time: 347.18m | eta: 4078.1m
step 15750/200000 (7.88%) | loss: 2.986594 | lrm: 1.00 | dt: 1322.18ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 64 | total time: 348.29m | eta: 4077.0m
step 15800/200000 (7.90%) | loss: 2.986720 | lrm: 1.00 | dt: 1317.77ms | tok/sec: 12,433 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 66 | total time: 349.39m | eta: 4075.8m
step 15850/200000 (7.92%) | loss: 2.947776 | lrm: 1.00 | dt: 1321.60ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 68 | total time: 350.49m | eta: 4074.7m
step 15900/200000 (7.95%) | loss: 2.924183 | lrm: 1.00 | dt: 1321.12ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 70 | total time: 351.59m | eta: 4073.5m
step 15950/200000 (7.97%) | loss: 2.968161 | lrm: 1.00 | dt: 1320.08ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 72 | total time: 352.70m | eta: 4072.4m
Step 16000 | Validation bpb: 1.084572
<|bos|>The capital of France is the capital of the United States (last only 8.26% of the total population in Europe) and 0
<|bos|>The chemical symbol of gold is the gold cation. It is not easy to distinguish it from gold from silver, but there is a common reference that
<|bos|>If yesterday was Friday, then tomorrow will be the day of the month of July. With the start of the sunny, sunny days, you have to take a
<|bos|>The opposite of hot is the hot. The heat is transferred to the water from the boiler through the condenser and from the hot water
<|bos|>The planets of the solar system are: the solar system, or solar planets, or solar planets, or solar planets, or orbiters. S
<|bos|>My favorite color is red, so I can't see my friend have a red one. One is a little bit brown and black, the other
<|bos|>If 5*x + 3 = 13, then x is 13+8+13
You also need to check if you are going to do this before the new 4x
2026-03-17 04:39:25,193 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_016000.pt
2026-03-17 04:39:25,195 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_016000.json
2026-03-17 04:39:26,468 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_016000_rank0.pt
step 16000/200000 (8.00%) | loss: 3.004752 | lrm: 1.00 | dt: 1522.60ms | tok/sec: 10,760 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 74 | total time: 353.80m | eta: 4071.3m
step 16050/200000 (8.03%) | loss: 2.988719 | lrm: 1.00 | dt: 1319.30ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 76 | total time: 354.91m | eta: 4070.2m
step 16100/200000 (8.05%) | loss: 3.013835 | lrm: 1.00 | dt: 1325.07ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 78 | total time: 356.01m | eta: 4069.0m
step 16150/200000 (8.07%) | loss: 2.997160 | lrm: 1.00 | dt: 1325.95ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 80 | total time: 357.12m | eta: 4067.9m
step 16200/200000 (8.10%) | loss: 2.978905 | lrm: 1.00 | dt: 1325.22ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 0 | total time: 358.22m | eta: 4066.8m
step 16250/200000 (8.12%) | loss: 2.962118 | lrm: 1.00 | dt: 1324.12ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 2 | total time: 359.32m | eta: 4065.6m
step 16300/200000 (8.15%) | loss: 2.910467 | lrm: 1.00 | dt: 1323.30ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 4 | total time: 360.43m | eta: 4064.5m
step 16350/200000 (8.18%) | loss: 3.020998 | lrm: 1.00 | dt: 1320.95ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 6 | total time: 361.53m | eta: 4063.3m
step 16400/200000 (8.20%) | loss: 2.997888 | lrm: 1.00 | dt: 1324.42ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 8 | total time: 362.63m | eta: 4062.2m
step 16450/200000 (8.22%) | loss: 2.927414 | lrm: 1.00 | dt: 1322.60ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 10 | total time: 363.74m | eta: 4061.1m
step 16500/200000 (8.25%) | loss: 2.985644 | lrm: 1.00 | dt: 1322.62ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 12 | total time: 364.84m | eta: 4059.9m
step 16550/200000 (8.28%) | loss: 2.962952 | lrm: 1.00 | dt: 1336.19ms | tok/sec: 12,261 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 14 | total time: 365.94m | eta: 4058.8m
step 16600/200000 (8.30%) | loss: 2.962135 | lrm: 1.00 | dt: 1318.08ms | tok/sec: 12,430 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 16 | total time: 367.05m | eta: 4057.6m
step 16650/200000 (8.32%) | loss: 2.888707 | lrm: 1.00 | dt: 1318.09ms | tok/sec: 12,430 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 18 | total time: 368.15m | eta: 4056.5m
step 16700/200000 (8.35%) | loss: 2.915539 | lrm: 1.00 | dt: 1321.69ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 21 | total time: 369.25m | eta: 4055.3m
step 16750/200000 (8.38%) | loss: 2.955974 | lrm: 1.00 | dt: 1323.95ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 23 | total time: 370.35m | eta: 4054.2m
step 16800/200000 (8.40%) | loss: 3.028729 | lrm: 1.00 | dt: 1322.17ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 25 | total time: 371.46m | eta: 4053.1m
step 16850/200000 (8.43%) | loss: 2.942616 | lrm: 1.00 | dt: 1321.01ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 27 | total time: 372.56m | eta: 4051.9m
step 16900/200000 (8.45%) | loss: 3.024129 | lrm: 1.00 | dt: 1322.27ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 29 | total time: 373.66m | eta: 4050.8m
step 16950/200000 (8.47%) | loss: 2.994789 | lrm: 1.00 | dt: 1323.60ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 31 | total time: 374.76m | eta: 4049.6m
step 17000/200000 (8.50%) | loss: 2.958905 | lrm: 1.00 | dt: 1325.28ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 33 | total time: 375.87m | eta: 4048.5m
step 17050/200000 (8.53%) | loss: 2.945403 | lrm: 1.00 | dt: 1322.44ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 35 | total time: 376.97m | eta: 4047.3m
step 17100/200000 (8.55%) | loss: 2.982975 | lrm: 1.00 | dt: 1322.49ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 37 | total time: 378.07m | eta: 4046.2m
step 17150/200000 (8.57%) | loss: 2.962763 | lrm: 1.00 | dt: 1321.94ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 39 | total time: 379.17m | eta: 4045.0m
step 17200/200000 (8.60%) | loss: 2.938371 | lrm: 1.00 | dt: 1321.09ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 41 | total time: 380.28m | eta: 4043.9m
step 17250/200000 (8.62%) | loss: 2.964217 | lrm: 1.00 | dt: 1322.45ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 43 | total time: 381.38m | eta: 4042.8m
step 17300/200000 (8.65%) | loss: 2.967980 | lrm: 1.00 | dt: 1324.74ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 45 | total time: 382.49m | eta: 4041.6m
step 17350/200000 (8.68%) | loss: 2.939195 | lrm: 1.00 | dt: 1323.60ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 47 | total time: 383.59m | eta: 4040.5m
step 17400/200000 (8.70%) | loss: 2.902521 | lrm: 1.00 | dt: 1322.04ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 49 | total time: 384.69m | eta: 4039.4m
step 17450/200000 (8.72%) | loss: 2.988243 | lrm: 1.00 | dt: 1328.27ms | tok/sec: 12,334 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 51 | total time: 385.79m | eta: 4038.2m
step 17500/200000 (8.75%) | loss: 2.942939 | lrm: 1.00 | dt: 1322.63ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 53 | total time: 386.90m | eta: 4037.1m
step 17550/200000 (8.78%) | loss: 2.941686 | lrm: 1.00 | dt: 1321.92ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 55 | total time: 388.00m | eta: 4036.0m
step 17600/200000 (8.80%) | loss: 2.974065 | lrm: 1.00 | dt: 1321.71ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 57 | total time: 389.10m | eta: 4034.8m
step 17650/200000 (8.82%) | loss: 2.911376 | lrm: 1.00 | dt: 1323.99ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 59 | total time: 390.21m | eta: 4033.7m
step 17700/200000 (8.85%) | loss: 2.926254 | lrm: 1.00 | dt: 1326.86ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 61 | total time: 391.31m | eta: 4032.5m
step 17750/200000 (8.88%) | loss: 2.976351 | lrm: 1.00 | dt: 1324.90ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 63 | total time: 392.41m | eta: 4031.4m
step 17800/200000 (8.90%) | loss: 2.999019 | lrm: 1.00 | dt: 1323.32ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 65 | total time: 393.52m | eta: 4030.3m
step 17850/200000 (8.93%) | loss: 2.904949 | lrm: 1.00 | dt: 1322.87ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 67 | total time: 394.62m | eta: 4029.1m
step 17900/200000 (8.95%) | loss: 3.035499 | lrm: 1.00 | dt: 1323.02ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 69 | total time: 395.72m | eta: 4028.0m
step 17950/200000 (8.97%) | loss: 2.948629 | lrm: 1.00 | dt: 1325.31ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 72 | total time: 396.83m | eta: 4026.9m
Step 18000 | Validation bpb: 1.076995
step 18000/200000 (9.00%) | loss: 2.876945 | lrm: 1.00 | dt: 1403.39ms | tok/sec: 11,674 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 74 | total time: 397.93m | eta: 4025.8m
step 18050/200000 (9.03%) | loss: 2.987120 | lrm: 1.00 | dt: 1324.49ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 76 | total time: 399.04m | eta: 4024.6m
step 18100/200000 (9.05%) | loss: 2.972928 | lrm: 1.00 | dt: 1323.03ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 78 | total time: 400.14m | eta: 4023.5m
step 18150/200000 (9.07%) | loss: 2.886154 | lrm: 1.00 | dt: 1323.38ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 80 | total time: 401.24m | eta: 4022.4m
step 18200/200000 (9.10%) | loss: 3.002739 | lrm: 1.00 | dt: 1322.91ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 82 | total time: 402.35m | eta: 4021.3m
step 18250/200000 (9.12%) | loss: 2.930908 | lrm: 1.00 | dt: 1323.64ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 1 | total time: 403.45m | eta: 4020.1m
step 18300/200000 (9.15%) | loss: 2.972565 | lrm: 1.00 | dt: 1319.95ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 3 | total time: 404.55m | eta: 4019.0m
step 18350/200000 (9.18%) | loss: 2.996603 | lrm: 1.00 | dt: 1322.12ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 5 | total time: 405.66m | eta: 4017.9m
step 18400/200000 (9.20%) | loss: 2.899164 | lrm: 1.00 | dt: 1320.31ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 7 | total time: 406.76m | eta: 4016.7m
step 18450/200000 (9.22%) | loss: 2.962000 | lrm: 1.00 | dt: 1321.75ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 9 | total time: 407.86m | eta: 4015.6m
step 18500/200000 (9.25%) | loss: 2.972528 | lrm: 1.00 | dt: 1323.60ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 11 | total time: 408.96m | eta: 4014.4m
step 18550/200000 (9.28%) | loss: 2.958721 | lrm: 1.00 | dt: 1331.04ms | tok/sec: 12,309 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 13 | total time: 410.06m | eta: 4013.3m
step 18600/200000 (9.30%) | loss: 2.950543 | lrm: 1.00 | dt: 1321.95ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 15 | total time: 411.17m | eta: 4012.2m
step 18650/200000 (9.32%) | loss: 2.932014 | lrm: 1.00 | dt: 1323.67ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 17 | total time: 412.27m | eta: 4011.0m
step 18700/200000 (9.35%) | loss: 3.002708 | lrm: 1.00 | dt: 1318.36ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 19 | total time: 413.37m | eta: 4009.9m
step 18750/200000 (9.38%) | loss: 2.992075 | lrm: 1.00 | dt: 1320.28ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 21 | total time: 414.48m | eta: 4008.8m
step 18800/200000 (9.40%) | loss: 2.960290 | lrm: 1.00 | dt: 1324.42ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 23 | total time: 415.58m | eta: 4007.6m
step 18850/200000 (9.43%) | loss: 2.937233 | lrm: 1.00 | dt: 1321.92ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 25 | total time: 416.68m | eta: 4006.5m
step 18900/200000 (9.45%) | loss: 3.008810 | lrm: 1.00 | dt: 1321.37ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 27 | total time: 417.79m | eta: 4005.3m
step 18950/200000 (9.47%) | loss: 2.973542 | lrm: 1.00 | dt: 1324.13ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 29 | total time: 418.89m | eta: 4004.2m
step 19000/200000 (9.50%) | loss: 2.932901 | lrm: 1.00 | dt: 1319.07ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 31 | total time: 419.99m | eta: 4003.1m
step 19050/200000 (9.53%) | loss: 2.950378 | lrm: 1.00 | dt: 1322.80ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 33 | total time: 421.09m | eta: 4001.9m
step 19100/200000 (9.55%) | loss: 2.890304 | lrm: 1.00 | dt: 1319.27ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 35 | total time: 422.19m | eta: 4000.8m
step 19150/200000 (9.57%) | loss: 2.947457 | lrm: 1.00 | dt: 1323.20ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 38 | total time: 423.30m | eta: 3999.7m
step 19200/200000 (9.60%) | loss: 2.965404 | lrm: 1.00 | dt: 1322.74ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 40 | total time: 424.40m | eta: 3998.5m
step 19250/200000 (9.62%) | loss: 2.945499 | lrm: 1.00 | dt: 1323.80ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 42 | total time: 425.50m | eta: 3997.4m
step 19300/200000 (9.65%) | loss: 3.003456 | lrm: 1.00 | dt: 1316.98ms | tok/sec: 12,440 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 44 | total time: 426.60m | eta: 3996.2m
step 19350/200000 (9.68%) | loss: 2.981239 | lrm: 1.00 | dt: 1318.42ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 46 | total time: 427.71m | eta: 3995.1m
step 19400/200000 (9.70%) | loss: 2.948472 | lrm: 1.00 | dt: 1321.45ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 48 | total time: 428.81m | eta: 3994.0m
step 19450/200000 (9.72%) | loss: 2.946696 | lrm: 1.00 | dt: 1326.83ms | tok/sec: 12,348 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 50 | total time: 429.91m | eta: 3992.8m
step 19500/200000 (9.75%) | loss: 2.935067 | lrm: 1.00 | dt: 1320.79ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 52 | total time: 431.01m | eta: 3991.7m
step 19550/200000 (9.78%) | loss: 2.950649 | lrm: 1.00 | dt: 1322.95ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 54 | total time: 432.12m | eta: 3990.6m
step 19600/200000 (9.80%) | loss: 2.981954 | lrm: 1.00 | dt: 1340.37ms | tok/sec: 12,223 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 56 | total time: 433.22m | eta: 3989.4m
step 19650/200000 (9.82%) | loss: 2.991354 | lrm: 1.00 | dt: 1319.19ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 58 | total time: 434.32m | eta: 3988.3m
step 19700/200000 (9.85%) | loss: 2.973752 | lrm: 1.00 | dt: 1322.29ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 60 | total time: 435.42m | eta: 3987.1m
step 19750/200000 (9.88%) | loss: 2.915413 | lrm: 1.00 | dt: 1321.37ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 62 | total time: 436.53m | eta: 3986.0m
step 19800/200000 (9.90%) | loss: 2.981176 | lrm: 1.00 | dt: 1324.16ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 64 | total time: 437.63m | eta: 3984.9m
step 19850/200000 (9.93%) | loss: 2.901318 | lrm: 1.00 | dt: 1320.20ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 66 | total time: 438.73m | eta: 3983.7m
step 19900/200000 (9.95%) | loss: 2.916566 | lrm: 1.00 | dt: 1319.25ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 68 | total time: 439.83m | eta: 3982.6m
step 19950/200000 (9.97%) | loss: 2.938417 | lrm: 1.00 | dt: 1321.56ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 70 | total time: 440.94m | eta: 3981.5m
Step 20000 | Validation bpb: 1.070066
<|bos|>The capital of France is the Cuy÷ne-2, which is a four-parallel and complex series of four parallel sectors
<|bos|>The chemical symbol of gold is the symbol of the metal itself, not the symbol of the metal itself. A gold atom is considered as the symbol of the
<|bos|>If yesterday was Friday, then tomorrow will be the day of the year for Sir Ashley's wedding cake. In the same way that the celebrated
<|bos|>The opposite of hot is the opposite of cold. As far as the food is concerned, in the United States and Canada, it's the same.
<|bos|>The planets of the solar system are: the solar system is the smallest star in the solar system and the Earth is the smallest of them.
The star in
<|bos|>My favorite color is black. You can use it with any color you like, and no matter if you have a blue background or a green background
<|bos|>If 5*x + 3 = 13, then x is the same as 2*y (not quite right, but then this is the 2*2/3rd
2026-03-17 06:08:08,492 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_020000.pt
2026-03-17 06:08:08,492 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_020000.json
2026-03-17 06:08:09,748 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_020000_rank0.pt
step 20000/200000 (10.00%) | loss: `3.005026` | lrm: 1.00 | dt: 1511.48ms | tok/sec: 10,839 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 72 | total time: 442.04m | eta: 3980.4m
step 20050/200000 (10.03%) | loss: `2.942849 | lrm: 1.00 | dt: 1324.21ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 74 | total time: 443.15m | eta: 3979.3m
step 20100/200000 (10.05%) | loss: 2.877598 | lrm: 1.00 | dt: 1322.84ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 76 | total time: 444.26m | eta: 3978.2m
step 20150/200000 (10.07%) | loss: 2.942606 | lrm: 1.00 | dt: 1324.87ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 78 | total time: 445.36m | eta: 3977.0m
step 20200/200000 (10.10%) | loss: 2.906587 | lrm: 1.00 | dt: 1320.09ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 80 | total time: 446.46m | eta: 3975.9m
step 20250/200000 (10.12%) | loss: 2.901471 | lrm: 1.00 | dt: 1324.43ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 82 | total time: 447.56m | eta: 3974.8m
step 20300/200000 (10.15%) | loss: 2.957007 | lrm: 1.00 | dt: 1324.40ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 2 | total time: 448.67m | eta: 3973.7m
step 20350/200000 (10.18%) | loss: 2.922709 | lrm: 1.00 | dt: 1324.74ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 4 | total time: 449.77m | eta: 3972.5m
step 20400/200000 (10.20%) | loss: 3.027287 | lrm: 1.00 | dt: 1325.77ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 6 | total time: 450.88m | eta: 3971.4m
step 20450/200000 (10.22%) | loss: 2.920944 | lrm: 1.00 | dt: 1318.97ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 8 | total time: 451.98m | eta: 3970.3m
step 20500/200000 (10.25%) | loss: 2.905603 | lrm: 1.00 | dt: 1322.81ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 10 | total time: 453.08m | eta: 3969.2m
step 20550/200000 (10.28%) | loss: 2.923840 | lrm: 1.00 | dt: 1319.08ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 12 | total time: 454.18m | eta: 3968.0m
step 20600/200000 (10.30%) | loss: 2.956197 | lrm: 1.00 | dt: 1321.81ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 14 | total time: 455.29m | eta: 3966.9m
step 20650/200000 (10.32%) | loss: 2.896507 | lrm: 1.00 | dt: 1318.53ms | tok/sec: 12,425 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 16 | total time: 456.39m | eta: 3965.8m
step 20700/200000 (10.35%) | loss: 2.946815 | lrm: 1.00 | dt: 1320.18ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 18 | total time: 457.49m | eta: 3964.6m
step 20750/200000 (10.38%) | loss: 2.910952 | lrm: 1.00 | dt: 1326.04ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 20 | total time: 458.59m | eta: 3963.5m
step 20800/200000 (10.40%) | loss: 2.956913 | lrm: 1.00 | dt: 1321.77ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 22 | total time: 459.70m | eta: 3962.4m
step 20850/200000 (10.43%) | loss: 2.930463 | lrm: 1.00 | dt: 1318.34ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 24 | total time: 460.80m | eta: 3961.2m
step 20900/200000 (10.45%) | loss: 2.981168 | lrm: 1.00 | dt: 1320.27ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 26 | total time: 461.90m | eta: 3960.1m
step 20950/200000 (10.47%) | loss: 2.958851 | lrm: 0.99 | dt: 1320.84ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 28 | total time: 463.01m | eta: 3959.0m
step 21000/200000 (10.50%) | loss: 2.923148 | lrm: 0.99 | dt: 1322.80ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 30 | total time: 464.11m | eta: 3957.9m
step 21050/200000 (10.53%) | loss: 2.937348 | lrm: 0.99 | dt: 1320.94ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 32 | total time: 465.21m | eta: 3956.7m
step 21100/200000 (10.55%) | loss: 2.917298 | lrm: 0.99 | dt: 1321.13ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 34 | total time: 466.32m | eta: 3955.6m
step 21150/200000 (10.57%) | loss: 2.921691 | lrm: 0.99 | dt: 1324.05ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 36 | total time: 467.42m | eta: 3954.5m
step 21200/200000 (10.60%) | loss: 2.983577 | lrm: 0.99 | dt: 1317.71ms | tok/sec: 12,433 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 38 | total time: 468.52m | eta: 3953.4m
step 21250/200000 (10.62%) | loss: 3.014524 | lrm: 0.99 | dt: 1322.70ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 40 | total time: 469.63m | eta: 3952.2m
step 21300/200000 (10.65%) | loss: 2.974675 | lrm: 0.99 | dt: 1323.95ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 42 | total time: 470.73m | eta: 3951.1m
step 21350/200000 (10.68%) | loss: 2.953529 | lrm: 0.99 | dt: 1321.09ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 44 | total time: 471.83m | eta: 3950.0m
step 21400/200000 (10.70%) | loss: 2.992871 | lrm: 0.99 | dt: 1323.72ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 46 | total time: 472.93m | eta: 3948.8m
step 21450/200000 (10.72%) | loss: 2.916995 | lrm: 0.99 | dt: 1320.96ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 48 | total time: 474.03m | eta: 3947.7m
step 21500/200000 (10.75%) | loss: 2.918133 | lrm: 0.99 | dt: 1323.78ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 51 | total time: 475.14m | eta: 3946.6m
step 21550/200000 (10.78%) | loss: 2.839564 | lrm: 0.99 | dt: 1320.21ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 53 | total time: 476.24m | eta: 3945.4m
step 21600/200000 (10.80%) | loss: 2.890490 | lrm: 0.99 | dt: 1324.28ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 55 | total time: 477.34m | eta: 3944.3m
step 21650/200000 (10.82%) | loss: 2.897224 | lrm: 0.99 | dt: 1320.61ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 57 | total time: 478.44m | eta: 3943.2m
step 21700/200000 (10.85%) | loss: 2.981329 | lrm: 0.99 | dt: 1322.07ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 59 | total time: 479.54m | eta: 3942.0m
step 21750/200000 (10.88%) | loss: 2.916128 | lrm: 0.99 | dt: 1320.22ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 61 | total time: 480.65m | eta: 3940.9m
step 21800/200000 (10.90%) | loss: 2.853533 | lrm: 0.99 | dt: 1321.62ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 63 | total time: 481.75m | eta: 3939.8m
step 21850/200000 (10.93%) | loss: 2.859659 | lrm: 0.99 | dt: 1322.96ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 65 | total time: 482.85m | eta: 3938.7m
step 21900/200000 (10.95%) | loss: 3.001064 | lrm: 0.99 | dt: 1325.09ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 67 | total time: 483.95m | eta: 3937.5m
step 21950/200000 (10.97%) | loss: 2.918938 | lrm: 0.99 | dt: 1338.53ms | tok/sec: 12,240 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 69 | total time: 485.06m | eta: 3936.4m
Step 22000 | Validation bpb: 1.064105
step 22000/200000 (11.00%) | loss: 2.902545 | lrm: 0.99 | dt: 1430.73ms | tok/sec: 11,451 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 71 | total time: 486.17m | eta: 3935.3m
step 22050/200000 (11.03%) | loss: 2.887911 | lrm: 0.99 | dt: 1323.43ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 73 | total time: 487.27m | eta: 3934.2m
step 22100/200000 (11.05%) | loss: 2.875035 | lrm: 0.99 | dt: 1323.17ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 75 | total time: 488.38m | eta: 3933.1m
step 22150/200000 (11.07%) | loss: 2.951334 | lrm: 0.99 | dt: 1323.07ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 77 | total time: 489.48m | eta: 3932.0m
step 22200/200000 (11.10%) | loss: 2.857379 | lrm: 0.99 | dt: 1318.87ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 79 | total time: 490.58m | eta: 3930.8m
step 22250/200000 (11.12%) | loss: 2.932418 | lrm: 0.99 | dt: 1320.48ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 81 | total time: 491.68m | eta: 3929.7m
step 22300/200000 (11.15%) | loss: 2.943601 | lrm: 0.99 | dt: 1320.44ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 1 | total time: 492.78m | eta: 3928.6m
step 22350/200000 (11.18%) | loss: 2.893871 | lrm: 0.99 | dt: 1321.42ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 3 | total time: 493.89m | eta: 3927.4m
step 22400/200000 (11.20%) | loss: 2.900383 | lrm: 0.99 | dt: 1323.43ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 5 | total time: 494.99m | eta: 3926.3m
step 22450/200000 (11.22%) | loss: 2.947594 | lrm: 0.99 | dt: 1327.09ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 7 | total time: 496.09m | eta: 3925.2m
1 pq: 11 rg: 1 | total time: 492.78m | eta: 3928.6m
step 22350/200000 (11.18%) | loss: 2.893871 | lrm: 0.99 | dt: 1321.42ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 3 | total time: 493.89m | eta: 3927.4m
step 22400/200000 (11.20%) | loss: 2.900383 | lrm: 0.99 | dt: 1323.43ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 5 | total time: 494.99m | eta: 3926.3m
step 22450/200000 (11.22%) | loss: 2.947594 | lrm: 0.99 | dt: 1327.09ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 7 | total time: 496.09m | eta: 3925.2m
1 pq: 11 rg: 5 | total time: 494.99m | eta: 3926.3m
step 22450/200000 (11.22%) | loss: 2.947594 | lrm: 0.99 | dt: 1327.09ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 7 | total time: 496.09m | eta: 3925.2m
1 pq: 11 rg: 7 | total time: 496.09m | eta: 3925.2m
step 22500/200000 (11.25%) | loss: 2.958352 | lrm: 0.99 | dt: 1318.25ms | tok/sec: 12,428 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 9 | total time: 497.20m | eta: 3924.1m
step 22550/200000 (11.28%) | loss: 2.964766 | lrm: 0.99 | dt: 1320.81ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: step 22550/200000 (11.28%) | loss: 2.964766 | lrm: 0.99 | dt: 1320.81ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 11 | total time: 498.30m | eta: 3922.9m
step 22600/200000 (11.30%) | loss: 2.942962 | lrm: 0.99 | dt: 1320.68ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 13 | total time: 499.40m | eta: 3921.8m
step 22650/200000 (11.32%) | loss: 2.940700 | lrm: 0.99 | dt: 1318.80ms | tok/sec: 12,423 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 15 | total time: 500.50m | eta: 3920.7m
step 22700/200000 (11.35%) | loss: 2.877098 | lrm: 0.99 | dt: 1318.88ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 18 | total time: 501.61m | eta: 3919.6m
step 22750/200000 (11.38%) | loss: 2.869529 | lrm: 0.99 | dt: 1319.81ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 20 | total time: 502.71m | eta: 3918.4m
step 22800/200000 (11.40%) | loss: 2.937425 | lrm: 0.99 | dt: 1322.15ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 22 | total time: 503.81m | eta: 3917.3m
step 22850/200000 (11.43%) | loss: 2.869228 | lrm: 0.98 | dt: 1322.86ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 24 | total time: 504.91m | eta: 3916.2m
step 22900/200000 (11.45%) | loss: 2.927951 | lrm: 0.98 | dt: 1319.53ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 26 | total time: 506.01m | eta: 3915.0m
step 22950/200000 (11.47%) | loss: 2.852520 | lrm: 0.98 | dt: 1324.28ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 28 | total time: 507.12m | eta: 3914.0m
step 23000/200000 (11.50%) | loss: 2.952655 | lrm: 0.98 | dt: 1321.94ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 30 | total time: 508.23m | eta: 3912.8m
step 23050/200000 (11.53%) | loss: 2.901928 | lrm: 0.98 | dt: 1320.93ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 32 | total time: 509.33m | eta: 3911.7m
step 23100/200000 (11.55%) | loss: 2.861756 | lrm: 0.98 | dt: 1322.28ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 34 | total time: 510.43m | eta: 3910.6m


#
step 23100/200000 (11.55%) | loss: 2.861756 | lrm: 0.98 | dt: 1322.28ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 34 | total time: 510.43m | eta: 3910.6m
step 23150/200000 (11.57%) | loss: 2.948940 | lrm: 0.98 | dt: 1332.83ms | tok/sec: 12,292 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 36 | total time: 511.56m | eta: 3909.6m
step 23200/200000 (11.60%) | loss: 2.934469 | lrm: 0.98 | dt: 1340.79ms | tok/sec: 12,219 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 38 | total time: 512.67m | eta: 3908.6m
step 23250/200000 (11.62%) | loss: 2.891483 | lrm: 0.98 | dt: 1325.24ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 40 | total time: 513.77m | eta: 3907.5m
step 23300/200000 (11.65%) | loss: 2.940039 | lrm: 0.98 | dt: 1325.55ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 42 | total time: 514.88m | eta: 3906.4m
step 23350/200000 (11.68%) | loss: 2.914623 | lrm: 0.98 | dt: 1325.23ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 44 | total time: 515.98m | eta: 3905.3m
step 23400/200000 (11.70%) | loss: 2.867964 | lrm: 0.98 | dt: 1324.13ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 46 | total time: 517.09m | eta: 3904.2m
step 23450/200000 (11.72%) | loss: 2.923058 | lrm: 0.98 | dt: 1331.04ms | tok/sec: 12,309 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 48 | total time: 518.20m | eta: 3903.1m
step 23500/200000 (11.75%) | loss: 2.928369 | lrm: 0.98 | dt: 1324.67ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 50 | total time: 519.30m | eta: 3902.0m
step 23550/200000 (11.78%) | loss: 2.945899 | lrm: 0.98 | dt: 1323.13ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 52 | total time: 520.41m | eta: 3900.9m
step 23600/200000 (11.80%) | loss: 2.860064 | lrm: 0.98 | dt: 1324.46ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 54 | total time: 521.51m | eta: 3899.7m
step 23650/200000 (11.82%) | loss: 2.951050 | lrm: 0.98 | dt: 1326.25ms | tok/sec: 12,353 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 56 | total time: 522.62m | eta: 3898.6m
step 23700/200000 (11.85%) | loss: 2.938054 | lrm: 0.98 | dt: 1323.76ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 58 | total time: 523.72m | eta: 3897.5m
step 23750/200000 (11.88%) | loss: 2.952499 | lrm: 0.98 | dt: 1325.28ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 60 | total time: 524.83m | eta: 3896.4m
step 23800/200000 (11.90%) | loss: 2.930411 | lrm: 0.98 | dt: 1338.40ms | tok/sec: 12,241 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 63 | total time: 525.94m | eta: 3895.3m
step 23850/200000 (11.93%) | loss: 2.890015 | lrm: 0.98 | dt: 1329.32ms | tok/sec: 12,325 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 65 | total time: 527.04m | eta: 3894.2m
step 23900/200000 (11.95%) | loss: 3.016321 | lrm: 0.98 | dt: 1323.04ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 67 | total time: 528.15m | eta: 3893.1m
step 23950/200000 (11.97%) | loss: 2.911058 | lrm: 0.98 | dt: 1336.98ms | tok/sec: 12,254 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 69 | total time: 529.26m | eta: 3892.1m
Step 24000 | Validation bpb: 1.058118
<|bos|>The capital of France is the capital of the country of Saint-Andrew, a 32-metre-wide town situated about 
<|bos|>The chemical symbol of gold is gold. Most of us don't know or understand the symbol of gold. In fact, gold is considered to be one of
<|bos|>If yesterday was Friday, then tomorrow will be the day of the month of August. He might be the last word of the month to be called in this month of the
<|bos|>The opposite of hot is that the person who has gone the dog so far has not gone.
"Sometimes we use to do what we need and
<|bos|>The planets of the solar system are: 1. The sun is at the helm of the planet: 2. The sun is the solar system's biggest
<|bos|>My favorite color is black, the color of the moon is clear, and the color of the planet is green. So it is a pretty easy
<|bos|>If 5*x + 3 = 13, then x is the number of times that the person needs to put up a new line. 5 is the number of times that a person
2026-03-17 07:36:56,402 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_024000.pt
2026-03-17 07:36:56,402 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_024000.json
2026-03-17 07:36:57,656 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_024000_rank0.pt
step 24000/200000 (12.00%) | loss: 2.842140 | lrm: 0.98 | dt: 1517.61ms | tok/sec: 10,795 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 71 | total time: 530.36m | eta: 3891.0m
step 24050/200000 (12.03%) | loss: 2.920920 | lrm: 0.98 | dt: 1319.72ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 73 | total time: 531.47m | eta: 3889.9m
step 24100/200000 (12.05%) | loss: 2.881770 | lrm: 0.98 | dt: 1322.76ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 75 | total time: 532.58m | eta: 3888.8m
step 24150/200000 (12.07%) | loss: 2.789579 | lrm: 0.98 | dt: 1322.28ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 77 | total time: 533.68m | eta: 3887.6m
step 24200/200000 (12.10%) | loss: 2.883508 | lrm: 0.98 | dt: 1321.66ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 79 | total time: 534.78m | eta: 3886.5m
step 24250/200000 (12.12%) | loss: 2.932358 | lrm: 0.98 | dt: 1319.91ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 81 | total time: 535.88m | eta: 3885.4m
step 24300/200000 (12.15%) | loss: 2.911118 | lrm: 0.98 | dt: 1327.52ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 1 | total time: 536.99m | eta: 3884.2m
step 24350/200000 (12.18%) | loss: 2.881656 | lrm: 0.98 | dt: 1324.57ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 3 | total time: 538.09m | eta: 3883.1m
step 24400/200000 (12.20%) | loss: 2.881117 | lrm: 0.98 | dt: 1320.95ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 5 | total time: 539.19m | eta: 3882.0m
step 24450/200000 (12.22%) | loss: 2.880328 | lrm: 0.98 | dt: 1322.38ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 7 | total time: 540.29m | eta: 3880.9m
step 24500/200000 (12.25%) | loss: 2.923064 | lrm: 0.98 | dt: 1318.12ms | tok/sec: 12,429 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 9 | total time: 541.40m | eta: 3879.7m
step 24550/200000 (12.28%) | loss: 2.898639 | lrm: 0.98 | dt: 1325.86ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 11 | total time: 542.50m | eta: 3878.6m
step 24600/200000 (12.30%) | loss: 2.928585 | lrm: 0.98 | dt: 1318.39ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 13 | total time: 543.60m | eta: 3877.5m
step 24650/200000 (12.32%) | loss: 2.856000 | lrm: 0.98 | dt: 1320.78ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 15 | total time: 544.70m | eta: 3876.4m
step 24700/200000 (12.35%) | loss: 2.963860 | lrm: 0.98 | dt: 1316.64ms | tok/sec: 12,443 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 17 | total time: 545.80m | eta: 3875.2m
step 24750/200000 (12.38%) | loss: 2.915310 | lrm: 0.97 | dt: 1319.74ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 19 | total time: 546.90m | eta: 3874.1m
step 24800/200000 (12.40%) | loss: 2.901258 | lrm: 0.97 | dt: 1320.45ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 21 | total time: 548.00m | eta: 3872.9m
step 24850/200000 (12.43%) | loss: 2.899168 | lrm: 0.97 | dt: 1319.64ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 23 | total time: 549.10m | eta: 3871.8m
step 24900/200000 (12.45%) | loss: 2.912140 | lrm: 0.97 | dt: 1319.95ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 26 | total time: 550.21m | eta: 3870.7m
step 24950/200000 (12.47%) | loss: 2.874341 | lrm: 0.97 | dt: 1320.07ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 28 | total time: 551.31m | eta: 3869.5m
step 25000/200000 (12.50%) | loss: 2.869657 | lrm: 0.97 | dt: 1323.14ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 30 | total time: 552.41m | eta: 3868.4m
step 25050/200000 (12.53%) | loss: 2.864011 | lrm: 0.97 | dt: 1321.72ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 32 | total time: 553.51m | eta: 3867.3m
step 25100/200000 (12.55%) | loss: 2.888014 | lrm: 0.97 | dt: 1319.93ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 34 | total time: 554.62m | eta: 3866.2m
step 25150/200000 (12.57%) | loss: 2.900064 | lrm: 0.97 | dt: 1319.89ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 36 | total time: 555.72m | eta: 3865.0m
step 25200/200000 (12.60%) | loss: 2.916372 | lrm: 0.97 | dt: 1319.46ms | tok/sec: 12,417 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 38 | total time: 556.82m | eta: 3863.9m
step 25250/200000 (12.62%) | loss: 2.893046 | lrm: 0.97 | dt: 1315.12ms | tok/sec: 12,458 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 40 | total time: 557.92m | eta: 3862.8m
step 25300/200000 (12.65%) | loss: 2.906962 | lrm: 0.97 | dt: 1319.22ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 42 | total time: 559.02m | eta: 3861.7m
step 25350/200000 (12.68%) | loss: 2.925254 | lrm: 0.97 | dt: 1320.41ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 44 | total time: 560.12m | eta: 3860.5m
step 25400/200000 (12.70%) | loss: 2.916320 | lrm: 0.97 | dt: 1321.95ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 46 | total time: 561.22m | eta: 3859.4m
step 25450/200000 (12.72%) | loss: 2.961467 | lrm: 0.97 | dt: 1322.03ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 48 | total time: 562.32m | eta: 3858.2m
step 25500/200000 (12.75%) | loss: 2.924659 | lrm: 0.97 | dt: 1323.49ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 50 | total time: 563.43m | eta: 3857.1m
step 25550/200000 (12.78%) | loss: 2.845069 | lrm: 0.97 | dt: 1319.91ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 52 | total time: 564.53m | eta: 3856.0m
step 25600/200000 (12.80%) | loss: 2.894229 | lrm: 0.97 | dt: 1319.66ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 54 | total time: 565.63m | eta: 3854.9m
step 25650/200000 (12.82%) | loss: 2.906481 | lrm: 0.97 | dt: 1318.53ms | tok/sec: 12,425 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 56 | total time: 566.73m | eta: 3853.7m
step 25700/200000 (12.85%) | loss: 2.944280 | lrm: 0.97 | dt: 1323.98ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 58 | total time: 567.83m | eta: 3852.6m
step 25750/200000 (12.88%) | loss: 2.851961 | lrm: 0.97 | dt: 1316.39ms | tok/sec: 12,446 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 60 | total time: 568.93m | eta: 3851.5m
step 25800/200000 (12.90%) | loss: 2.898941 | lrm: 0.97 | dt: 1317.27ms | tok/sec: 12,437 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 62 | total time: 570.03m | eta: 3850.3m
step 25850/200000 (12.93%) | loss: 2.954118 | lrm: 0.97 | dt: 1317.49ms | tok/sec: 12,435 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 64 | total time: 571.13m | eta: 3849.2m
step 25900/200000 (12.95%) | loss: 2.931024 | lrm: 0.97 | dt: 1322.28ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 66 | total time: 572.24m | eta: 3848.1m
step 25950/200000 (12.97%) | loss: 2.932786 | lrm: 0.97 | dt: 1337.27ms | tok/sec: 12,251 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 69 | total time: 573.34m | eta: 3846.9m
Step 26000 | Validation bpb: 1.053625
step 26000/200000 (13.00%) | loss: 2.914069 | lrm: 0.97 | dt: 1429.04ms | tok/sec: 11,465 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 71 | total time: 574.46m | eta: 3845.9m
step 26050/200000 (13.03%) | loss: 2.876795 | lrm: 0.97 | dt: 1322.41ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 73 | total time: 575.56m | eta: 3844.8m
step 26100/200000 (13.05%) | loss: 2.936231 | lrm: 0.97 | dt: 1323.67ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 75 | total time: 576.67m | eta: 3843.7m
step 26150/200000 (13.07%) | loss: 2.910558 | lrm: 0.97 | dt: 1322.10ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 77 | total time: 577.77m | eta: 3842.6m
step 26200/200000 (13.10%) | loss: 2.882121 | lrm: 0.97 | dt: 1325.06ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 79 | total time: 578.87m | eta: 3841.4m
step 26250/200000 (13.12%) | loss: 2.853498 | lrm: 0.97 | dt: 1322.39ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 81 | total time: 579.99m | eta: 3840.4m
step 26300/200000 (13.15%) | loss: 2.897289 | lrm: 0.97 | dt: 1321.93ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 0 | total time: 581.09m | eta: 3839.3m
step 26350/200000 (13.18%) | loss: 2.972522 | lrm: 0.97 | dt: 1326.59ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 2 | total time: 582.19m | eta: 3838.2m
step 26400/200000 (13.20%) | loss: 2.830870 | lrm: 0.97 | dt: 1322.10ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 4 | total time: 583.30m | eta: 3837.1m
step 26450/200000 (13.22%) | loss: 2.901861 | lrm: 0.97 | dt: 1321.74ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 6 | total time: 584.40m | eta: 3836.0m
step 26500/200000 (13.25%) | loss: 2.799782 | lrm: 0.97 | dt: 1328.19ms | tok/sec: 12,335 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 8 | total time: 585.50m | eta: 3834.8m
step 26550/200000 (13.28%) | loss: 2.835982 | lrm: 0.97 | dt: 1319.89ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 10 | total time: 586.61m | eta: 3833.7m
step 26600/200000 (13.30%) | loss: 2.956354 | lrm: 0.97 | dt: 1326.41ms | tok/sec: 12,352 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 12 | total time: 587.71m | eta: 3832.6m
step 26650/200000 (13.32%) | loss: 2.928038 | lrm: 0.96 | dt: 1339.24ms | tok/sec: 12,233 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 14 | total time: 588.82m | eta: 3831.5m
step 26700/200000 (13.35%) | loss: 2.925483 | lrm: 0.96 | dt: 1320.83ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 16 | total time: 589.92m | eta: 3830.4m
step 26750/200000 (13.38%) | loss: 2.803825 | lrm: 0.96 | dt: 1327.54ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 18 | total time: 591.04m | eta: 3829.4m
step 26800/200000 (13.40%) | loss: 2.968678 | lrm: 0.96 | dt: 1338.60ms | tok/sec: 12,239 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 20 | total time: 592.15m | eta: 3828.3m
step 26850/200000 (13.43%) | loss: 2.934183 | lrm: 0.96 | dt: 1329.90ms | tok/sec: 12,319 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 22 | total time: 593.26m | eta: 3827.3m
step 26900/200000 (13.45%) | loss: 2.894952 | lrm: 0.96 | dt: 1330.63ms | tok/sec: 12,312 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 24 | total time: 594.37m | eta: 3826.2m
step 26950/200000 (13.47%) | loss: 2.872224 | lrm: 0.96 | dt: 1336.04ms | tok/sec: 12,263 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 27 | total time: 595.49m | eta: 3825.1m
step 27000/200000 (13.50%) | loss: 3.006520 | lrm: 0.96 | dt: 1329.49ms | tok/sec: 12,323 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 29 | total time: 596.60m | eta: 3824.1m
step 27050/200000 (13.53%) | loss: 2.907751 | lrm: 0.96 | dt: 1330.90ms | tok/sec: 12,310 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 31 | total time: 597.71m | eta: 3823.0m
step 27100/200000 (13.55%) | loss: 2.918845 | lrm: 0.96 | dt: 1333.87ms | tok/sec: 12,283 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 33 | total time: 598.82m | eta: 3821.9m
1 pq: 13 rg: 33 | total time: 598.82m | eta: 3821.9m
step 27150/200000 (13.57%) | loss: 2.797862 | lrm: 0.96 | dt: 1333.22ms | tok/sec: 12,289 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 35 | total time: 599.94m | eta: 3820.9m
step 27200/200000 (13.60%) | loss: 2.905906 | lrm: 0.96 | dt: 1327.45ms | tok/sec: 12,342 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 37 | total time: 601.05m | eta: 3819.8m
step 27250/200000 (13.62%) | loss: 2.846754 | lrm: 0.96 | dt: 1332.30ms | tok/sec: 12,297 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 39 | total time: 602.15m | eta: 3818.7m


# 9:05

step 27300/200000 (13.65%) | loss: 2.963330 | lrm: 0.96 | dt: 1329.01ms | tok/sec: 12,327 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 41 | total time: 603.27m | eta: 3817.7m
step 27350/200000 (13.68%) | loss: 2.870887 | lrm: 0.96 | dt: 1328.48ms | tok/sec: 12,332 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 43 | total time: 604.38m | eta: 3816.6m
step 27400/200000 (13.70%) | loss: 2.923213 | lrm: 0.96 | dt: 1328.33ms | tok/sec: 12,334 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 45 | total time: 605.49m | eta: 3815.5m
step 27450/200000 (13.72%) | loss: 2.872645 | lrm: 0.96 | dt: 1338.30ms | tok/sec: 12,242 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 47 | total time: 606.59m | eta: 3814.4m
step 27500/200000 (13.75%) | loss: 2.929259 | lrm: 0.96 | dt: 1336.67ms | tok/sec: 12,257 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 49 | total time: 607.71m | eta: 3813.4m
step 27550/200000 (13.78%) | loss: 2.917924 | lrm: 0.96 | dt: 1334.72ms | tok/sec: 12,275 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 51 | total time: 608.82m | eta: 3812.3m
step 27600/200000 (13.80%) | loss: 2.954995 | lrm: 0.96 | dt: 1340.26ms | tok/sec: 12,224 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 53 | total time: 609.93m | eta: 3811.2m
step 27650/200000 (13.82%) | loss: 2.904145 | lrm: 0.96 | dt: 1324.41ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 55 | total time: 611.04m | eta: 3810.1m
step 27700/200000 (13.85%) | loss: 2.926403 | lrm: 0.96 | dt: 1328.17ms | tok/sec: 12,335 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 57 | total time: 612.14m | eta: 3809.0m
step 27750/200000 (13.88%) | loss: 2.814189 | lrm: 0.96 | dt: 1322.56ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 59 | total time: 613.25m | eta: 3807.9m
step 27800/200000 (13.90%) | loss: 2.841298 | lrm: 0.96 | dt: 1330.45ms | tok/sec: 12,314 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 61 | total time: 614.36m | eta: 3806.8m
step 27850/200000 (13.93%) | loss: 2.938557 | lrm: 0.96 | dt: 1332.54ms | tok/sec: 12,295 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 63 | total time: 615.46m | eta: 3805.7m
step 27900/200000 (13.95%) | loss: 2.869588 | lrm: 0.96 | dt: 1332.03ms | tok/sec: 12,300 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 65 | total time: 616.58m | eta: 3804.7m
step 27950/200000 (13.97%) | loss: 2.798633 | lrm: 0.96 | dt: 1328.16ms | tok/sec: 12,335 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 67 | total time: 617.69m | eta: 3803.6m
Step 28000 | Validation bpb: 1.049610
<|bos|>The capital of France is the city of Los Angeles, LA. The city is named after the Los Angeles City of Los
<|bos|>The chemical symbol of gold is a symbol of the element's name, while the symbol of the element is usually a chemical symbol, a small number of element
<|bos|>If yesterday was Friday, then tomorrow will be the 15th of February in the Kimberley, tomorrow.
You are not in the right place
<|bos|>`The opposite of hot is the cold` is the opposite of hot is the hot is the cold is the heat is the heat is the heat is the heat
<|bos|>The planets of the solar system are: 1. The solar system, known as the solar system, is the only planet with the mass 13.7 times
<|bos|>`My favorite color is red, or red`. If you haven't considered it, you shouldn't do it.
Depending on the source of
<|bos|>If 5*x + 3 = 13, then x is the number of times the ball rolls against the wall. The average mass of the ball is 1/2 m.
2026-03-17 09:05:52,209 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_028000.pt
2026-03-17 09:05:52,211 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_028000.json
2026-03-17 09:05:53,441 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_028000_rank0.pt
step 28000/200000 (14.00%) | loss: 2.842793 | lrm: 0.96 | dt: 1552.81ms | tok/sec: 10,551 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 69 | total time: 618.81m | eta: 3802.6m


#
step 34150/200000 (17.07%) | loss: 2.818102 | lrm: 0.93 | dt: 1321.90ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 72 | total time: 754.65m | eta: 3666.1m
step 34200/200000 (17.10%) | loss: 2.848883 | lrm: 0.93 | dt: 1324.20ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 74 | total time: 755.76m | eta: 3664.9m
step 34250/200000 (17.12%) | loss: 2.837072 | lrm: 0.92 | dt: 1322.06ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 76 | total time: 756.86m | eta: 3663.8m
step 34300/200000 (17.15%) | loss: 2.909770 | lrm: 0.92 | dt: 1328.97ms | tok/sec: 12,328 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 78 | total time: 757.97m | eta: 3662.7m
step 34350/200000 (17.18%) | loss: 2.876832 | lrm: 0.92 | dt: 1320.44ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 80 | total time: 759.07m | eta: 3661.6m
step 34400/200000 (17.20%) | loss: 2.891734 | lrm: 0.92 | dt: 1323.23ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 82 | total time: 760.17m | eta: 3660.5m
step 34450/200000 (17.23%) | loss: 2.830871 | lrm: 0.92 | dt: 1323.97ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 1 | total time: 761.27m | eta: 3659.4m
step 34500/200000 (17.25%) | loss: 2.815002 | lrm: 0.92 | dt: 1320.89ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 3 | total time: 762.38m | eta: 3658.3m
step 34550/200000 (17.27%) | loss: 2.795748 | lrm: 0.92 | dt: 1328.23ms | tok/sec: 12,335 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 5 | total time: 763.48m | eta: 3657.2m
step 34600/200000 (17.30%) | loss: 2.894006 | lrm: 0.92 | dt: 1327.17ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 7 | total time: 764.59m | eta: 3656.1m
step 34650/200000 (17.32%) | loss: 2.877214 | lrm: 0.92 | dt: 1317.87ms | tok/sec: 12,432 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 9 | total time: 765.69m | eta: 3654.9m
step 34700/200000 (17.35%) | loss: 2.841357 | lrm: 0.92 | dt: 1323.33ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 11 | total time: 766.79m | eta: 3653.8m
step 34750/200000 (17.38%) | loss: 2.901376 | lrm: 0.92 | dt: 1320.74ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 13 | total time: 767.90m | eta: 3652.7m
step 34800/200000 (17.40%) | loss: 2.868163 | lrm: 0.92 | dt: 1324.17ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 15 | total time: 769.00m | eta: 3651.6m
step 34850/200000 (17.43%) | loss: 2.870739 | lrm: 0.92 | dt: 1326.61ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 17 | total time: 770.10m | eta: 3650.5m
step 34900/200000 (17.45%) | loss: 2.829792 | lrm: 0.92 | dt: 1325.18ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 19 | total time: 771.21m | eta: 3649.4m
step 34950/200000 (17.48%) | loss: 2.904001 | lrm: 0.92 | dt: 1327.06ms | tok/sec: 12,346 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 21 | total time: 772.31m | eta: 3648.3m
step 35000/200000 (17.50%) | loss: 2.828891 | lrm: 0.92 | dt: 1327.97ms | tok/sec: 12,337 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 23 | total time: 773.42m | eta: 3647.1m
step 35050/200000 (17.52%) | loss: 2.916121 | lrm: 0.92 | dt: 1322.30ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 25 | total time: 774.52m | eta: 3646.1m
step 35100/200000 (17.55%) | loss: 2.818414 | lrm: 0.92 | dt: 1324.61ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 27 | total time: 775.63m | eta: 3644.9m
step 35150/200000 (17.57%) | loss: 2.823545 | lrm: 0.92 | dt: 1322.18ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 29 | total time: 776.73m | eta: 3643.8m
step 35200/200000 (17.60%) | loss: 2.793837 | lrm: 0.92 | dt: 1324.19ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 31 | total time: 777.83m | eta: 3642.7m
step 35250/200000 (17.62%) | loss: 2.826685 | lrm: 0.92 | dt: 1321.34ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 34 | total time: 778.94m | eta: 3641.6m
step 35300/200000 (17.65%) | loss: 2.755002 | lrm: 0.92 | dt: 1337.10ms | tok/sec: 12,253 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 36 | total time: 780.04m | eta: 3640.5m
step 35350/200000 (17.68%) | loss: 2.860842 | lrm: 0.92 | dt: 1323.23ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 38 | total time: 781.15m | eta: 3639.4m
step 35400/200000 (17.70%) | loss: 2.774363 | lrm: 0.92 | dt: 1329.47ms | tok/sec: 12,323 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 40 | total time: 782.25m | eta: 3638.3m
step 35450/200000 (17.73%) | loss: 2.855682 | lrm: 0.92 | dt: 1325.23ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 42 | total time: 783.35m | eta: 3637.2m
step 35500/200000 (17.75%) | loss: 2.824942 | lrm: 0.92 | dt: 1319.96ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 44 | total time: 784.46m | eta: 3636.0m
step 35550/200000 (17.77%) | loss: 2.845517 | lrm: 0.92 | dt: 1341.82ms | tok/sec: 12,210 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 46 | total time: 785.56m | eta: 3634.9m
step 35600/200000 (17.80%) | loss: 2.862226 | lrm: 0.92 | dt: 1322.33ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 48 | total time: 786.67m | eta: 3633.8m
step 35650/200000 (17.82%) | loss: 2.814233 | lrm: 0.92 | dt: 1345.90ms | tok/sec: 12,173 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 50 | total time: 787.77m | eta: 3632.7m
step 35700/200000 (17.85%) | loss: 2.853442 | lrm: 0.92 | dt: 1322.59ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 52 | total time: 788.87m | eta: 3631.6m
step 35750/200000 (17.88%) | loss: 2.839114 | lrm: 0.92 | dt: 1325.86ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 54 | total time: 789.98m | eta: 3630.5m
step 35800/200000 (17.90%) | loss: 2.783548 | lrm: 0.92 | dt: 1330.65ms | tok/sec: 12,312 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 56 | total time: 791.08m | eta: 3629.4m
step 35850/200000 (17.93%) | loss: 2.856965 | lrm: 0.92 | dt: 1323.43ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 58 | total time: 792.19m | eta: 3628.3m
step 35900/200000 (17.95%) | loss: 2.815640 | lrm: 0.92 | dt: 1333.31ms | tok/sec: 12,288 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 60 | total time: 793.29m | eta: 3627.2m
step 35950/200000 (17.98%) | loss: 2.836409 | lrm: 0.92 | dt: 1325.83ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 62 | total time: 794.40m | eta: 3626.1m
Step 36000 | Validation bpb: 1.036053
<|bos|>The capital of France is the capital of the United States, and all land is divided into various components that are considered to be separate. The city of
<|bos|>The chemical symbol of gold is 92.5, which is one of the most common gold nanoparticles on the market, making it a widely
<|bos|>If yesterday was Friday, then tomorrow will be Friday. The first day was Friday, and the first day was Easter, and we will have a full day
<|bos|>The opposite of hot is the hot opposite of cool. When an object is hot, the first thing the object has to do is cool it and then
<|bos|>The planets of the solar system are: Mars, Jupiter, Saturn, Saturn, Mars 2022

Today I will
<|bos|>My favorite color is blue. That's why I always used it with my green background or yellow on my wall. When my wife came to
<|bos|>If 5*x + 3 = 13, then x is the number of coefficients you
get. If you had 10 coefficients of the same 5
2026-03-17 12:03:32,907 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_036000.pt
2026-03-17 12:03:32,909 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_036000.json
2026-03-17 12:03:34,169 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_036000_rank0.pt
step 36000/200000 (18.00%) | loss: 2.794431 | lrm: 0.92 | dt: 1561.01ms | tok/sec: 10,495 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 64 | total time: 795.51m | eta: 3625.0m
step 36050/200000 (18.02%) | loss: 2.913776 | lrm: 0.92 | dt: 1322.36ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 66 | total time: 796.62m | eta: 3623.9m
step 36100/200000 (18.05%) | loss: 2.912122 | lrm: 0.92 | dt: 1321.73ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 68 | total time: 797.72m | eta: 3622.8m
step 36150/200000 (18.07%) | loss: 2.863011 | lrm: 0.91 | dt: 1325.76ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 70 | total time: 798.82m | eta: 3621.7m
step 36200/200000 (18.10%) | loss: 2.866914 | lrm: 0.91 | dt: 1326.46ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 72 | total time: 799.93m | eta: 3620.6m
step 36250/200000 (18.12%) | loss: 2.809712 | lrm: 0.91 | dt: 1325.57ms | tok/sec: 12,359 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 74 | total time: 801.04m | eta: 3619.5m
step 36300/200000 (18.15%) | loss: 2.796394 | lrm: 0.91 | dt: 1325.05ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 76 | total time: 802.14m | eta: 3618.4m
step 36350/200000 (18.18%) | loss: 2.837793 | lrm: 0.91 | dt: 1325.23ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 78 | total time: 803.24m | eta: 3617.2m
step 36400/200000 (18.20%) | loss: 2.863927 | lrm: 0.91 | dt: 1324.95ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 81 | total time: 804.35m | eta: 3616.1m
step 36450/200000 (18.23%) | loss: 2.879965 | lrm: 0.91 | dt: 1326.30ms | tok/sec: 12,353 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 0 | total time: 805.45m | eta: 3615.0m
step 36500/200000 (18.25%) | loss: 2.853595 | lrm: 0.91 | dt: 1327.86ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 2 | total time: 806.56m | eta: 3613.9m
step 36550/200000 (18.27%) | loss: 2.845059 | lrm: 0.91 | dt: 1324.67ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 4 | total time: 807.66m | eta: 3612.8m
step 36600/200000 (18.30%) | loss: 2.837419 | lrm: 0.91 | dt: 1323.81ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 6 | total time: 808.76m | eta: 3611.7m
step 36650/200000 (18.32%) | loss: 2.830210 | lrm: 0.91 | dt: 1323.03ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 8 | total time: 809.87m | eta: 3610.6m
step 36700/200000 (18.35%) | loss: 2.825041 | lrm: 0.91 | dt: 1354.90ms | tok/sec: 12,092 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 10 | total time: 810.97m | eta: 3609.5m
step 36750/200000 (18.38%) | loss: 2.850081 | lrm: 0.91 | dt: 1326.76ms | tok/sec: 12,348 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 12 | total time: 812.08m | eta: 3608.4m
step 36800/200000 (18.40%) | loss: 2.897615 | lrm: 0.91 | dt: 1326.08ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 14 | total time: 813.19m | eta: 3607.3m
step 36850/200000 (18.43%) | loss: 2.909034 | lrm: 0.91 | dt: 1325.81ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 16 | total time: 814.29m | eta: 3606.2m
step 36900/200000 (18.45%) | loss: 2.833240 | lrm: 0.91 | dt: 1324.89ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 18 | total time: 815.39m | eta: 3605.1m
step 36950/200000 (18.48%) | loss: 2.857090 | lrm: 0.91 | dt: 1323.32ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 20 | total time: 816.50m | eta: 3604.0m
step 37000/200000 (18.50%) | loss: 2.848207 | lrm: 0.91 | dt: 1322.36ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 22 | total time: 817.60m | eta: 3602.8m
step 37050/200000 (18.52%) | loss: 2.871623 | lrm: 0.91 | dt: 1322.93ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 24 | total time: 818.71m | eta: 3601.7m
step 37100/200000 (18.55%) | loss: 2.821850 | lrm: 0.91 | dt: 1327.28ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 26 | total time: 819.81m | eta: 3600.6m
step 37150/200000 (18.57%) | loss: 2.741335 | lrm: 0.91 | dt: 1322.14ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 28 | total time: 820.91m | eta: 3599.5m
step 37200/200000 (18.60%) | loss: 2.880844 | lrm: 0.91 | dt: 1323.71ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 30 | total time: 822.02m | eta: 3598.4m
step 37250/200000 (18.62%) | loss: 2.846492 | lrm: 0.91 | dt: 1322.79ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 32 | total time: 823.12m | eta: 3597.3m
step 37300/200000 (18.65%) | loss: 2.869873 | lrm: 0.91 | dt: 1320.75ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 34 | total time: 824.22m | eta: 3596.2m
step 37350/200000 (18.68%) | loss: 2.750058 | lrm: 0.91 | dt: 1325.72ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 36 | total time: 825.33m | eta: 3595.1m
step 37400/200000 (18.70%) | loss: 2.874882 | lrm: 0.91 | dt: 1329.53ms | tok/sec: 12,323 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 38 | total time: 826.43m | eta: 3594.0m
step 37450/200000 (18.73%) | loss: 2.823144 | lrm: 0.91 | dt: 1325.70ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 40 | total time: 827.54m | eta: 3592.8m
step 37500/200000 (18.75%) | loss: 2.875950 | lrm: 0.91 | dt: 1327.89ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 42 | total time: 828.64m | eta: 3591.7m
step 37550/200000 (18.77%) | loss: 2.869078 | lrm: 0.91 | dt: 1325.20ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 44 | total time: 829.75m | eta: 3590.6m
step 37600/200000 (18.80%) | loss: 2.762898 | lrm: 0.91 | dt: 1318.80ms | tok/sec: 12,423 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 46 | total time: 830.85m | eta: 3589.5m
step 37650/200000 (18.82%) | loss: 2.877352 | lrm: 0.91 | dt: 1323.70ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 49 | total time: 831.95m | eta: 3588.4m
step 37700/200000 (18.85%) | loss: 2.842808 | lrm: 0.91 | dt: 1328.14ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 51 | total time: 833.06m | eta: 3587.3m
step 37750/200000 (18.88%) | loss: 2.867603 | lrm: 0.91 | dt: 1323.19ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 53 | total time: 834.16m | eta: 3586.2m
step 37800/200000 (18.90%) | loss: 2.861944 | lrm: 0.91 | dt: 1325.15ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 55 | total time: 835.27m | eta: 3585.1m
step 37850/200000 (18.93%) | loss: 2.837034 | lrm: 0.91 | dt: 1325.23ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 57 | total time: 836.37m | eta: 3584.0m
step 37900/200000 (18.95%) | loss: 2.824546 | lrm: 0.91 | dt: 1327.55ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 59 | total time: 837.47m | eta: 3582.9m
step 37950/200000 (18.98%) | loss: 2.817284 | lrm: 0.91 | dt: 1324.38ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 61 | total time: 838.58m | eta: 3581.8m
Step 38000 | Validation bpb: 1.033379
step 38000/200000 (19.00%) | loss: 2.854203 | lrm: 0.91 | dt: 1427.94ms | tok/sec: 11,473 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 63 | total time: 839.69m | eta: 3580.7m
step 38050/200000 (19.02%) | loss: 2.871283 | lrm: 0.90 | dt: 1347.61ms | tok/sec: 12,157 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 65 | total time: 840.79m | eta: 3579.6m
step 38100/200000 (19.05%) | loss: 2.858152 | lrm: 0.90 | dt: 1320.79ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 67 | total time: 841.90m | eta: 3578.5m
step 38150/200000 (19.07%) | loss: 2.784824 | lrm: 0.90 | dt: 1322.93ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 69 | total time: 843.00m | eta: 3577.4m
step 38200/200000 (19.10%) | loss: 2.775939 | lrm: 0.90 | dt: 1322.21ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 71 | total time: 844.11m | eta: 3576.2m
step 38250/200000 (19.12%) | loss: 2.820767 | lrm: 0.90 | dt: 1322.55ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 73 | total time: 845.21m | eta: 3575.1m
step 38300/200000 (19.15%) | loss: 2.804276 | lrm: 0.90 | dt: 1324.19ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 75 | total time: 846.32m | eta: 3574.0m
step 38350/200000 (19.18%) | loss: 2.758256 | lrm: 0.90 | dt: 1324.20ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 77 | total time: 847.42m | eta: 3572.9m
step 38400/200000 (19.20%) | loss: 2.880097 | lrm: 0.90 | dt: 1326.74ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 79 | total time: 848.53m | eta: 3571.8m
step 38450/200000 (19.23%) | loss: 2.814614 | lrm: 0.90 | dt: 1325.64ms | tok/sec: 12,359 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 81 | total time: 849.63m | eta: 3570.7m
step 38500/200000 (19.25%) | loss: 2.827912 | lrm: 0.90 | dt: 1327.16ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 1 | total time: 850.74m | eta: 3569.6m
step 38550/200000 (19.27%) | loss: 2.855373 | lrm: 0.90 | dt: 1325.81ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 3 | total time: 851.84m | eta: 3568.5m
step 38600/200000 (19.30%) | loss: 2.744876 | lrm: 0.90 | dt: 1324.30ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 5 | total time: 852.95m | eta: 3567.4m
step 38650/200000 (19.32%) | loss: 2.812396 | lrm: 0.90 | dt: 1325.60ms | tok/sec: 12,359 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 7 | total time: 854.05m | eta: 3566.3m
step 38700/200000 (19.35%) | loss: 2.837401 | lrm: 0.90 | dt: 1321.15ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 9 | total time: 855.15m | eta: 3565.2m
step 38750/200000 (19.38%) | loss: 2.859184 | lrm: 0.90 | dt: 1321.76ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 11 | total time: 856.26m | eta: 3564.1m
step 38800/200000 (19.40%) | loss: 2.805038 | lrm: 0.90 | dt: 1362.52ms | tok/sec: 12,024 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 13 | total time: 857.36m | eta: 3563.0m
step 38850/200000 (19.43%) | loss: 2.785158 | lrm: 0.90 | dt: 1323.16ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 15 | total time: 858.47m | eta: 3561.8m
step 38900/200000 (19.45%) | loss: 2.862522 | lrm: 0.90 | dt: 1323.02ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 17 | total time: 859.57m | eta: 3560.7m
step 38950/200000 (19.48%) | loss: 2.830491 | lrm: 0.90 | dt: 1324.67ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 20 | total time: 860.68m | eta: 3559.6m
step 39000/200000 (19.50%) | loss: 2.841967 | lrm: 0.90 | dt: 1325.22ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 22 | total time: 861.78m | eta: 3558.5m
step 39050/200000 (19.52%) | loss: 2.867927 | lrm: 0.90 | dt: 1323.67ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 24 | total time: 862.89m | eta: 3557.4m
step 39100/200000 (19.55%) | loss: 2.886845 | lrm: 0.90 | dt: 1318.55ms | tok/sec: 12,425 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 26 | total time: 863.99m | eta: 3556.3m
step 39150/200000 (19.57%) | loss: 2.830893 | lrm: 0.90 | dt: 1324.12ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 28 | total time: 865.10m | eta: 3555.2m
step 39200/200000 (19.60%) | loss: 2.874036 | lrm: 0.90 | dt: 1324.09ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 30 | total time: 866.20m | eta: 3554.1m
step 39250/200000 (19.62%) | loss: 2.813238 | lrm: 0.90 | dt: 1324.56ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 32 | total time: 867.30m | eta: 3553.0m
step 39300/200000 (19.65%) | loss: 2.820463 | lrm: 0.90 | dt: 1323.74ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 34 | total time: 868.41m | eta: 3551.9m
step 39350/200000 (19.68%) | loss: 2.791882 | lrm: 0.90 | dt: 1323.57ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 36 | total time: 869.51m | eta: 3550.8m
step 39400/200000 (19.70%) | loss: 2.835715 | lrm: 0.90 | dt: 1325.55ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 38 | total time: 870.62m | eta: 3549.7m
step 39450/200000 (19.73%) | loss: 2.823061 | lrm: 0.90 | dt: 1325.93ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 40 | total time: 871.72m | eta: 3548.6m
step 39500/200000 (19.75%) | loss: 2.823516 | lrm: 0.90 | dt: 1323.81ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 42 | total time: 872.83m | eta: 3547.5m
step 39550/200000 (19.77%) | loss: 2.819636 | lrm: 0.90 | dt: 1324.38ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 44 | total time: 873.93m | eta: 3546.4m
step 39600/200000 (19.80%) | loss: 2.828763 | lrm: 0.90 | dt: 1327.23ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 46 | total time: 875.04m | eta: 3545.2m
step 39650/200000 (19.82%) | loss: 2.833473 | lrm: 0.90 | dt: 1328.54ms | tok/sec: 12,332 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 48 | total time: 876.14m | eta: 3544.1m
step 39700/200000 (19.85%) | loss: 2.916724 | lrm: 0.90 | dt: 1325.26ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 50 | total time: 877.25m | eta: 3543.0m
step 39750/200000 (19.88%) | loss: 2.812045 | lrm: 0.90 | dt: 1324.77ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 52 | total time: 878.35m | eta: 3541.9m
step 39800/200000 (19.90%) | loss: 2.812548 | lrm: 0.90 | dt: 1326.97ms | tok/sec: 12,346 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 54 | total time: 879.45m | eta: 3540.8m
step 39850/200000 (19.93%) | loss: 2.865533 | lrm: 0.90 | dt: 1323.97ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 56 | total time: 880.56m | eta: 3539.7m
step 39900/200000 (19.95%) | loss: 2.840647 | lrm: 0.89 | dt: 1322.64ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 59 | total time: 881.66m | eta: 3538.6m
step 39950/200000 (19.98%) | loss: 2.799055 | lrm: 0.89 | dt: 1322.95ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 61 | total time: 882.77m | eta: 3537.5m
Step 40000 | Validation bpb: 1.029889
<|bos|>The capital of France is the Crescent, the city of Marseille. It is the second-largest city, and the capital of
<|bos|>The chemical symbol of gold is AGB, and is represented by the letter AG in the following table.
Chemical symbol
CnH
<|bos|>If yesterday was Friday, then tomorrow will be the same? I know they do, but maybe there's something very different to them? One can be called "brick
<|bos|>The opposite of hot is the opposite of cold, so why? Here's a breakdown of the terms:
Summer
It's no secret
<|bos|>The planets of the solar system are: Mars, Jupiter, Saturn, Saturn, Venus, Jupiter, Navig
<|bos|>My favorite color is red. So I'm looking for a brand that I can use to buy from the company that they use to buy it.
<|bos|>If 5*x + 3 = 13, then x is 13
If x + 3 = 13, then x is 13
If x = 13, then
2026-03-17 13:32:24,125 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_040000.pt
2026-03-17 13:32:24,125 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_040000.json
2026-03-17 13:32:25,422 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_040000_rank0.pt
step 40000/200000 (20.00%) | loss: 2.822616 | lrm: 0.89 | dt: 1530.70ms | tok/sec: 10,703 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 63 | total time: 883.88m | eta: 3536.4m
step 40050/200000 (20.02%) | loss: 2.833763 | lrm: 0.89 | dt: 1322.36ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 65 | total time: 884.98m | eta: 3535.3m
step 40100/200000 (20.05%) | loss: 2.853485 | lrm: 0.89 | dt: 1324.19ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 67 | total time: 886.09m | eta: 3534.2m
step 40150/200000 (20.07%) | loss: 2.745671 | lrm: 0.89 | dt: 1322.02ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 69 | total time: 887.20m | eta: 3533.1m
step 40200/200000 (20.10%) | loss: 2.870025 | lrm: 0.89 | dt: 1326.23ms | tok/sec: 12,353 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 71 | total time: 888.30m | eta: 3532.0m
step 40250/200000 (20.12%) | loss: 2.765455 | lrm: 0.89 | dt: 1329.35ms | tok/sec: 12,324 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 73 | total time: 889.41m | eta: 3530.9m
step 40300/200000 (20.15%) | loss: 2.902990 | lrm: 0.89 | dt: 1322.11ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 75 | total time: 890.51m | eta: 3529.8m
step 40350/200000 (20.18%) | loss: 2.812950 | lrm: 0.89 | dt: 1326.66ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 77 | total time: 891.61m | eta: 3528.7m
step 40400/200000 (20.20%) | loss: 2.867986 | lrm: 0.89 | dt: 1322.25ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 79 | total time: 892.72m | eta: 3527.6m
step 40450/200000 (20.23%) | loss: 2.801383 | lrm: 0.89 | dt: 1328.21ms | tok/sec: 12,335 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 81 | total time: 893.82m | eta: 3526.4m
step 40500/200000 (20.25%) | loss: 2.800362 | lrm: 0.89 | dt: 1325.55ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 83 | total time: 894.93m | eta: 3525.3m
step 40550/200000 (20.27%) | loss: 2.821592 | lrm: 0.89 | dt: 1324.11ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 1 | total time: 896.03m | eta: 3524.2m
step 40600/200000 (20.30%) | loss: 2.866617 | lrm: 0.89 | dt: 1324.33ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 3 | total time: 897.14m | eta: 3523.1m
step 40650/200000 (20.32%) | loss: 2.754834 | lrm: 0.89 | dt: 1320.65ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 5 | total time: 898.24m | eta: 3522.0m
step 40700/200000 (20.35%) | loss: 2.796410 | lrm: 0.89 | dt: 1434.06ms | tok/sec: 11,424 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 7 | total time: 899.35m | eta: 3520.9m
step 40750/200000 (20.38%) | loss: 2.833636 | lrm: 0.89 | dt: 1321.35ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 9 | total time: 900.45m | eta: 3519.8m
step 40800/200000 (20.40%) | loss: 2.847770 | lrm: 0.89 | dt: 1347.18ms | tok/sec: 12,161 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 11 | total time: 901.56m | eta: 3518.7m
step 40850/200000 (20.43%) | loss: 2.824802 | lrm: 0.89 | dt: 1326.38ms | tok/sec: 12,352 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 14 | total time: 902.66m | eta: 3517.6m
step 40900/200000 (20.45%) | loss: 2.839961 | lrm: 0.89 | dt: 1328.57ms | tok/sec: 12,332 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 16 | total time: 903.77m | eta: 3516.5m
step 40950/200000 (20.48%) | loss: 2.809479 | lrm: 0.89 | dt: 1322.53ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 18 | total time: 904.87m | eta: 3515.4m
step 41000/200000 (20.50%) | loss: 2.866436 | lrm: 0.89 | dt: 1322.65ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 20 | total time: 905.97m | eta: 3514.3m
step 41050/200000 (20.52%) | loss: 2.863527 | lrm: 0.89 | dt: 1328.82ms | tok/sec: 12,329 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 22 | total time: 907.08m | eta: 3513.2m
step 41100/200000 (20.55%) | loss: 2.854521 | lrm: 0.89 | dt: 1322.65ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 24 | total time: 908.18m | eta: 3512.1m
step 41150/200000 (20.57%) | loss: 2.850509 | lrm: 0.89 | dt: 1321.54ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 26 | total time: 909.29m | eta: 3510.9m
step 41200/200000 (20.60%) | loss: 2.831847 | lrm: 0.89 | dt: 1325.86ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 28 | total time: 910.39m | eta: 3509.8m
step 41250/200000 (20.62%) | loss: 2.777041 | lrm: 0.89 | dt: 1340.01ms | tok/sec: 12,226 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 30 | total time: 911.50m | eta: 3508.7m
step 41300/200000 (20.65%) | loss: 2.842663 | lrm: 0.89 | dt: 1338.99ms | tok/sec: 12,236 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 32 | total time: 912.60m | eta: 3507.6m
step 41350/200000 (20.68%) | loss: 2.825411 | lrm: 0.89 | dt: 1324.96ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 34 | total time: 913.71m | eta: 3506.5m
step 41400/200000 (20.70%) | loss: 2.802206 | lrm: 0.89 | dt: 1328.83ms | tok/sec: 12,329 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 36 | total time: 914.81m | eta: 3505.4m
step 41450/200000 (20.73%) | loss: 2.805453 | lrm: 0.89 | dt: 1330.35ms | tok/sec: 12,315 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 38 | total time: 915.92m | eta: 3504.3m
step 41500/200000 (20.75%) | loss: 2.850745 | lrm: 0.89 | dt: 1325.43ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 40 | total time: 917.02m | eta: 3503.2m
step 41550/200000 (20.77%) | loss: 2.844029 | lrm: 0.89 | dt: 1327.90ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 42 | total time: 918.13m | eta: 3502.1m
step 41600/200000 (20.80%) | loss: 2.887872 | lrm: 0.89 | dt: 1325.36ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 44 | total time: 919.23m | eta: 3501.0m
step 41650/200000 (20.82%) | loss: 2.817279 | lrm: 0.89 | dt: 1329.47ms | tok/sec: 12,323 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 46 | total time: 920.34m | eta: 3499.9m
step 41700/200000 (20.85%) | loss: 2.842371 | lrm: 0.89 | dt: 1322.80ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 48 | total time: 921.44m | eta: 3498.8m
step 41750/200000 (20.88%) | loss: 2.734823 | lrm: 0.89 | dt: 1327.83ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 50 | total time: 922.54m | eta: 3497.7m
step 41800/200000 (20.90%) | loss: 2.883756 | lrm: 0.88 | dt: 1348.63ms | tok/sec: 12,148 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 52 | total time: 923.65m | eta: 3496.6m
step 41850/200000 (20.93%) | loss: 2.815013 | lrm: 0.88 | dt: 1328.57ms | tok/sec: 12,332 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 54 | total time: 924.76m | eta: 3495.5m
step 41900/200000 (20.95%) | loss: 2.818000 | lrm: 0.88 | dt: 1327.81ms | tok/sec: 12,339 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 56 | total time: 925.87m | eta: 3494.4m
step 41950/200000 (20.98%) | loss: 2.839892 | lrm: 0.88 | dt: 1325.89ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 58 | total time: 926.97m | eta: 3493.3m
Step 42000 | Validation bpb: 1.027968
step 42000/200000 (21.00%) | loss: 2.817698 | lrm: 0.88 | dt: 1422.83ms | tok/sec: 11,515 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 61 | total time: 928.08m | eta: 3492.2m
step 42050/200000 (21.02%) | loss: 2.823952 | lrm: 0.88 | dt: 1329.67ms | tok/sec: 12,321 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 63 | total time: 929.19m | eta: 3491.1m
step 42100/200000 (21.05%) | loss: 2.868271 | lrm: 0.88 | dt: 1329.61ms | tok/sec: 12,322 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 65 | total time: 930.30m | eta: 3490.0m
step 42150/200000 (21.07%) | loss: 2.866918 | lrm: 0.88 | dt: 1322.94ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 67 | total time: 931.40m | eta: 3488.9m
step 42200/200000 (21.10%) | loss: 2.825335 | lrm: 0.88 | dt: 1325.37ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 69 | total time: 932.51m | eta: 3487.8m
step 42250/200000 (21.12%) | loss: 2.877627 | lrm: 0.88 | dt: 1325.99ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 71 | total time: 933.61m | eta: 3486.7m
step 42300/200000 (21.15%) | loss: 2.742062 | lrm: 0.88 | dt: 1321.10ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 73 | total time: 934.71m | eta: 3485.6m
step 42350/200000 (21.18%) | loss: 2.884804 | lrm: 0.88 | dt: 1322.13ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 75 | total time: 935.82m | eta: 3484.5m
step 42400/200000 (21.20%) | loss: 2.897621 | lrm: 0.88 | dt: 1327.08ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 77 | total time: 936.92m | eta: 3483.3m
step 42450/200000 (21.23%) | loss: 2.843548 | lrm: 0.88 | dt: 1322.44ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 79 | total time: 938.03m | eta: 3482.2m
step 42500/200000 (21.25%) | loss: 2.853518 | lrm: 0.88 | dt: 1328.79ms | tok/sec: 12,330 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 81 | total time: 939.13m | eta: 3481.1m
step 42550/200000 (21.27%) | loss: 2.832216 | lrm: 0.88 | dt: 1327.56ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 0 | total time: 940.24m | eta: 3480.0m
step 42600/200000 (21.30%) | loss: 2.741796 | lrm: 0.88 | dt: 1327.08ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 2 | total time: 941.35m | eta: 3479.0m
step 42650/200000 (21.32%) | loss: 2.785307 | lrm: 0.88 | dt: 1326.96ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 4 | total time: 942.46m | eta: 3477.9m
step 42700/200000 (21.35%) | loss: 2.841254 | lrm: 0.88 | dt: 1326.11ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 6 | total time: 943.56m | eta: 3476.8m
step 42750/200000 (21.38%) | loss: 2.740892 | lrm: 0.88 | dt: 1323.72ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 8 | total time: 944.67m | eta: 3475.6m
step 42800/200000 (21.40%) | loss: 2.749357 | lrm: 0.88 | dt: 1325.85ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 10 | total time: 945.78m | eta: 3474.5m
step 42850/200000 (21.43%) | loss: 2.821182 | lrm: 0.88 | dt: 1327.68ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 12 | total time: 946.88m | eta: 3473.4m
step 42900/200000 (21.45%) | loss: 2.841504 | lrm: 0.88 | dt: 1327.77ms | tok/sec: 12,339 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 14 | total time: 947.98m | eta: 3472.3m
step 42950/200000 (21.48%) | loss: 2.816037 | lrm: 0.88 | dt: 1325.27ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 16 | total time: 949.09m | eta: 3471.2m
step 43000/200000 (21.50%) | loss: 2.821198 | lrm: 0.88 | dt: 1331.31ms | tok/sec: 12,306 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 18 | total time: 950.20m | eta: 3470.1m
step 43050/200000 (21.52%) | loss: 2.807775 | lrm: 0.88 | dt: 1324.36ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 21 | total time: 951.30m | eta: 3469.0m
step 43100/200000 (21.55%) | loss: 2.871158 | lrm: 0.88 | dt: 1325.54ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 23 | total time: 952.41m | eta: 3467.9m
step 43150/200000 (21.57%) | loss: 2.827643 | lrm: 0.88 | dt: 1332.29ms | tok/sec: 12,297 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 25 | total time: 953.51m | eta: 3466.8m
step 43200/200000 (21.60%) | loss: 2.807612 | lrm: 0.88 | dt: 1325.53ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 27 | total time: 954.62m | eta: 3465.7m
step 43250/200000 (21.62%) | loss: 2.839329 | lrm: 0.88 | dt: 1325.68ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 29 | total time: 955.72m | eta: 3464.6m
step 43300/200000 (21.65%) | loss: 2.824480 | lrm: 0.88 | dt: 1322.62ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 31 | total time: 956.83m | eta: 3463.5m
step 43350/200000 (21.68%) | loss: 2.801395 | lrm: 0.88 | dt: 1330.32ms | tok/sec: 12,315 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 33 | total time: 957.94m | eta: 3462.4m
step 43400/200000 (21.70%) | loss: 2.771914 | lrm: 0.88 | dt: 1325.45ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 35 | total time: 959.04m | eta: 3461.3m
step 43450/200000 (21.73%) | loss: 2.794884 | lrm: 0.88 | dt: 1322.19ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 37 | total time: 960.15m | eta: 3460.2m
step 43500/200000 (21.75%) | loss: 2.846759 | lrm: 0.88 | dt: 1327.52ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 39 | total time: 961.25m | eta: 3459.1m
step 43550/200000 (21.77%) | loss: 2.859175 | lrm: 0.88 | dt: 1328.25ms | tok/sec: 12,335 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 41 | total time: 962.35m | eta: 3458.0m
step 43600/200000 (21.80%) | loss: 2.799906 | lrm: 0.88 | dt: 1328.35ms | tok/sec: 12,334 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 43 | total time: 963.46m | eta: 3456.9m
step 43650/200000 (21.82%) | loss: 2.759987 | lrm: 0.88 | dt: 1325.12ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 45 | total time: 964.57m | eta: 3455.8m
step 43700/200000 (21.85%) | loss: 2.844917 | lrm: 0.87 | dt: 1327.24ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 47 | total time: 965.67m | eta: 3454.7m
step 43750/200000 (21.88%) | loss: 2.723926 | lrm: 0.87 | dt: 1322.85ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 49 | total time: 966.78m | eta: 3453.6m
step 43800/200000 (21.90%) | loss: 2.751665 | lrm: 0.87 | dt: 1327.19ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 51 | total time: 967.88m | eta: 3452.5m
step 43850/200000 (21.93%) | loss: 2.786266 | lrm: 0.87 | dt: 1321.24ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 53 | total time: 968.99m | eta: 3451.4m
step 43900/200000 (21.95%) | loss: 2.755647 | lrm: 0.87 | dt: 1321.90ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 55 | total time: 970.09m | eta: 3450.2m
step 43950/200000 (21.98%) | loss: 2.780352 | lrm: 0.87 | dt: 1322.15ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 57 | total time: 971.20m | eta: 3449.1m
Step 44000 | Validation bpb: 1.025903
<|bos|>The capital of France is the capital of the United States.
Potato is a widely used ingredient in cooking. It has been used to make
<|bos|>The chemical symbol of gold is Ku
The symbol Ku is given as Ku = G's, which means that when it comes to solvent
<|bos|>If yesterday was Friday, then tomorrow will be the day I will have to pay back my free time. I need to go to a movie, a book, and a
<|bos|>The opposite of hot is the hot air. Hot air is very strong and has a definite pressure. When the air temperature is high it will
<|bos|>The planets of the solar system are: Mars, Jupiter, Saturn, Uranus, Neptune, Uranus, Nept
<|bos|>My favorite color is red. You can find other colors on different screens, such as Cartesian, Lorentz, and
<|bos|>If 5*x + 3 = 13, then x is the number of times that the first half of the periodic table can be converted into a whole. Thus, if you
2026-03-17 15:01:18,563 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_044000.pt
2026-03-17 15:01:18,563 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_044000.json
2026-03-17 15:01:19,836 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_044000_rank0.pt
step 44000/200000 (22.00%) | loss: 2.806322 | lrm: 0.87 | dt: 1531.37ms | tok/sec: 10,698 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 59 | total time: 972.30m | eta: 3448.0m
step 44050/200000 (22.02%) | loss: 2.784354 | lrm: 0.87 | dt: 1327.66ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 61 | total time: 973.41m | eta: 3447.0m
step 44100/200000 (22.05%) | loss: 2.809115 | lrm: 0.87 | dt: 1321.96ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 64 | total time: 974.52m | eta: 3445.8m
step 44150/200000 (22.07%) | loss: 2.852035 | lrm: 0.87 | dt: 1326.42ms | tok/sec: 12,352 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 66 | total time: 975.62m | eta: 3444.7m
step 44200/200000 (22.10%) | loss: 2.861962 | lrm: 0.87 | dt: 1324.46ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 68 | total time: 976.73m | eta: 3443.6m
step 44250/200000 (22.12%) | loss: 2.840442 | lrm: 0.87 | dt: 1324.32ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 70 | total time: 977.83m | eta: 3442.5m
step 44300/200000 (22.15%) | loss: 2.776825 | lrm: 0.87 | dt: 1323.84ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 72 | total time: 978.94m | eta: 3441.4m
step 44350/200000 (22.18%) | loss: 2.700121 | lrm: 0.87 | dt: 1346.43ms | tok/sec: 12,168 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 74 | total time: 980.04m | eta: 3440.3m
step 44400/200000 (22.20%) | loss: 2.819576 | lrm: 0.87 | dt: 1322.99ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 76 | total time: 981.15m | eta: 3439.2m
step 44450/200000 (22.23%) | loss: 2.723962 | lrm: 0.87 | dt: 1326.30ms | tok/sec: 12,353 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 78 | total time: 982.25m | eta: 3438.1m
step 44500/200000 (22.25%) | loss: 2.856007 | lrm: 0.87 | dt: 1324.63ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 80 | total time: 983.36m | eta: 3437.0m
step 44550/200000 (22.27%) | loss: 2.854346 | lrm: 0.87 | dt: 1323.72ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 82 | total time: 984.46m | eta: 3435.9m
step 44600/200000 (22.30%) | loss: 2.870533 | lrm: 0.87 | dt: 1323.80ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 0 | total time: 985.56m | eta: 3434.8m
step 44650/200000 (22.32%) | loss: 2.782011 | lrm: 0.87 | dt: 1324.44ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 2 | total time: 986.67m | eta: 3433.7m
step 44700/200000 (22.35%) | loss: 2.822292 | lrm: 0.87 | dt: 1322.10ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 4 | total time: 987.77m | eta: 3432.6m
step 44750/200000 (22.38%) | loss: 2.803883 | lrm: 0.87 | dt: 1322.53ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 6 | total time: 988.88m | eta: 3431.5m
step 44800/200000 (22.40%) | loss: 2.754713 | lrm: 0.87 | dt: 1323.27ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 8 | total time: 989.98m | eta: 3430.4m
step 44850/200000 (22.43%) | loss: 2.902179 | lrm: 0.87 | dt: 1323.52ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 10 | total time: 991.09m | eta: 3429.3m
step 44900/200000 (22.45%) | loss: 2.791218 | lrm: 0.87 | dt: 1320.87ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 12 | total time: 992.19m | eta: 3428.1m
step 44950/200000 (22.48%) | loss: 2.797060 | lrm: 0.87 | dt: 1320.20ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 14 | total time: 993.30m | eta: 3427.0m
step 45000/200000 (22.50%) | loss: 2.741040 | lrm: 0.87 | dt: 1336.89ms | tok/sec: 12,255 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 16 | total time: 994.40m | eta: 3425.9m
step 45050/200000 (22.52%) | loss: 2.826204 | lrm: 0.87 | dt: 1325.85ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 18 | total time: 995.51m | eta: 3424.8m
step 45100/200000 (22.55%) | loss: 2.784473 | lrm: 0.87 | dt: 1330.29ms | tok/sec: 12,316 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 20 | total time: 996.62m | eta: 3423.7m
step 45150/200000 (22.57%) | loss: 2.821900 | lrm: 0.87 | dt: 1325.72ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 22 | total time: 997.72m | eta: 3422.6m
step 45200/200000 (22.60%) | loss: 2.861799 | lrm: 0.87 | dt: 1326.69ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 24 | total time: 998.83m | eta: 3421.5m
step 45250/200000 (22.62%) | loss: 2.846018 | lrm: 0.87 | dt: 1323.92ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 27 | total time: 999.93m | eta: 3420.4m
step 45300/200000 (22.65%) | loss: 2.808833 | lrm: 0.87 | dt: 1325.97ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 29 | total time: 1001.04m | eta: 3419.3m
step 45350/200000 (22.68%) | loss: 2.856761 | lrm: 0.87 | dt: 1327.99ms | tok/sec: 12,337 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 31 | total time: 1002.14m | eta: 3418.2m
step 45400/200000 (22.70%) | loss: 2.813868 | lrm: 0.87 | dt: 1326.62ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 33 | total time: 1003.25m | eta: 3417.1m
step 45450/200000 (22.73%) | loss: 2.822941 | lrm: 0.87 | dt: 1325.53ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 35 | total time: 1004.35m | eta: 3416.0m
step 45500/200000 (22.75%) | loss: 2.746690 | lrm: 0.87 | dt: 1322.30ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 37 | total time: 1005.46m | eta: 3414.9m
step 45550/200000 (22.77%) | loss: 2.816305 | lrm: 0.87 | dt: 1329.24ms | tok/sec: 12,325 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 39 | total time: 1006.56m | eta: 3413.8m
step 45600/200000 (22.80%) | loss: 2.853829 | lrm: 0.86 | dt: 1329.53ms | tok/sec: 12,323 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 41 | total time: 1007.67m | eta: 3412.7m
step 45650/200000 (22.82%) | loss: 2.815186 | lrm: 0.86 | dt: 1323.72ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 43 | total time: 1008.77m | eta: 3411.6m
step 45700/200000 (22.85%) | loss: 2.803662 | lrm: 0.86 | dt: 1323.77ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 45 | total time: 1009.88m | eta: 3410.5m
step 45750/200000 (22.88%) | loss: 2.841228 | lrm: 0.86 | dt: 1328.20ms | tok/sec: 12,335 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 47 | total time: 1010.98m | eta: 3409.4m
step 45800/200000 (22.90%) | loss: 2.816143 | lrm: 0.86 | dt: 1324.39ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 49 | total time: 1012.09m | eta: 3408.3m
step 45850/200000 (22.93%) | loss: 2.760636 | lrm: 0.86 | dt: 1324.73ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 51 | total time: 1013.20m | eta: 3407.2m
step 45900/200000 (22.95%) | loss: 2.773013 | lrm: 0.86 | dt: 1331.07ms | tok/sec: 12,308 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 53 | total time: 1014.30m | eta: 3406.1m
step 45950/200000 (22.98%) | loss: 2.776050 | lrm: 0.86 | dt: 1324.60ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 55 | total time: 1015.41m | eta: 3405.0m
Step 46000 | Validation bpb: 1.022556
step 46000/200000 (23.00%) | loss: 2.892681 | lrm: 0.86 | dt: 1423.17ms | tok/sec: 11,512 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 57 | total time: 1016.51m | eta: 3403.9m
step 46050/200000 (23.02%) | loss: 2.847950 | lrm: 0.86 | dt: 1324.11ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 59 | total time: 1017.62m | eta: 3402.8m
step 46100/200000 (23.05%) | loss: 2.829813 | lrm: 0.86 | dt: 1329.83ms | tok/sec: 12,320 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 61 | total time: 1018.73m | eta: 3401.7m
step 46150/200000 (23.07%) | loss: 2.818370 | lrm: 0.86 | dt: 1324.29ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 63 | total time: 1019.84m | eta: 3400.6m
step 46200/200000 (23.10%) | loss: 2.741146 | lrm: 0.86 | dt: 1324.27ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 65 | total time: 1020.94m | eta: 3399.5m
step 46250/200000 (23.12%) | loss: 2.802846 | lrm: 0.86 | dt: 1322.04ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 67 | total time: 1022.05m | eta: 3398.3m
step 46300/200000 (23.15%) | loss: 2.847510 | lrm: 0.86 | dt: 1326.08ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 69 | total time: 1023.15m | eta: 3397.2m
step 46350/200000 (23.18%) | loss: 2.857416 | lrm: 0.86 | dt: 1328.35ms | tok/sec: 12,334 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 71 | total time: 1024.26m | eta: 3396.1m
step 46400/200000 (23.20%) | loss: 2.808652 | lrm: 0.86 | dt: 1321.76ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 73 | total time: 1025.36m | eta: 3395.0m
step 46450/200000 (23.23%) | loss: 2.858354 | lrm: 0.86 | dt: 1323.93ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 75 | total time: 1026.47m | eta: 3393.9m
step 46500/200000 (23.25%) | loss: 2.833456 | lrm: 0.86 | dt: 1324.04ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 78 | total time: 1027.57m | eta: 3392.8m
step 46550/200000 (23.27%) | loss: 2.824655 | lrm: 0.86 | dt: 1320.76ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 80 | total time: 1028.68m | eta: 3391.7m
step 46600/200000 (23.30%) | loss: 2.799973 | lrm: 0.86 | dt: 1324.60ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 82 | total time: 1029.78m | eta: 3390.6m
step 46650/200000 (23.32%) | loss: 2.821023 | lrm: 0.86 | dt: 1324.53ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 1 | total time: 1030.89m | eta: 3389.5m
step 46700/200000 (23.35%) | loss: 2.845408 | lrm: 0.86 | dt: 1326.90ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 3 | total time: 1031.99m | eta: 3388.4m
step 46750/200000 (23.38%) | loss: 2.827185 | lrm: 0.86 | dt: 1319.92ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 5 | total time: 1033.09m | eta: 3387.3m
step 46800/200000 (23.40%) | loss: 2.931007 | lrm: 0.86 | dt: 1323.52ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 7 | total time: 1034.20m | eta: 3386.2m
step 46850/200000 (23.43%) | loss: 2.846051 | lrm: 0.86 | dt: 1324.70ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 9 | total time: 1035.30m | eta: 3385.1m
step 46900/200000 (23.45%) | loss: 2.805392 | lrm: 0.86 | dt: 1326.94ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 11 | total time: 1036.41m | eta: 3384.0m
step 46950/200000 (23.48%) | loss: 2.851305 | lrm: 0.86 | dt: 1323.99ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 13 | total time: 1037.52m | eta: 3382.9m
step 47000/200000 (23.50%) | loss: 2.798180 | lrm: 0.86 | dt: 1324.69ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 15 | total time: 1038.62m | eta: 3381.8m
step 47050/200000 (23.52%) | loss: 2.800546 | lrm: 0.86 | dt: 1325.26ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 17 | total time: 1039.73m | eta: 3380.7m
step 47100/200000 (23.55%) | loss: 2.794626 | lrm: 0.86 | dt: 1320.70ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 19 | total time: 1040.83m | eta: 3379.6m
step 47150/200000 (23.57%) | loss: 2.840131 | lrm: 0.86 | dt: 1325.92ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 21 | total time: 1041.94m | eta: 3378.4m
step 47200/200000 (23.60%) | loss: 2.846672 | lrm: 0.86 | dt: 1324.19ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 23 | total time: 1043.04m | eta: 3377.3m
step 47250/200000 (23.62%) | loss: 2.851251 | lrm: 0.86 | dt: 1326.46ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 25 | total time: 1044.14m | eta: 3376.2m
step 47300/200000 (23.65%) | loss: 2.817018 | lrm: 0.86 | dt: 1322.54ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 27 | total time: 1045.25m | eta: 3375.1m
step 47350/200000 (23.68%) | loss: 2.859364 | lrm: 0.86 | dt: 1325.65ms | tok/sec: 12,359 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 29 | total time: 1046.35m | eta: 3374.0m
step 47400/200000 (23.70%) | loss: 2.856131 | lrm: 0.86 | dt: 1327.70ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 31 | total time: 1047.46m | eta: 3372.9m
step 47450/200000 (23.73%) | loss: 2.790629 | lrm: 0.86 | dt: 1322.82ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 33 | total time: 1048.56m | eta: 3371.8m
step 47500/200000 (23.75%) | loss: 2.831708 | lrm: 0.85 | dt: 1319.15ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 35 | total time: 1049.67m | eta: 3370.7m
step 47550/200000 (23.77%) | loss: 2.743378 | lrm: 0.85 | dt: 1323.93ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 37 | total time: 1050.77m | eta: 3369.6m
step 47600/200000 (23.80%) | loss: 2.824817 | lrm: 0.85 | dt: 1319.93ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 39 | total time: 1051.87m | eta: 3368.5m
step 47650/200000 (23.82%) | loss: 2.834827 | lrm: 0.85 | dt: 1323.21ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 41 | total time: 1052.98m | eta: 3367.4m
step 47700/200000 (23.85%) | loss: 2.818504 | lrm: 0.85 | dt: 1326.61ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 43 | total time: 1054.08m | eta: 3366.3m
step 47750/200000 (23.88%) | loss: 2.814591 | lrm: 0.85 | dt: 1324.34ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 45 | total time: 1055.19m | eta: 3365.1m
step 47800/200000 (23.90%) | loss: 2.726204 | lrm: 0.85 | dt: 1322.25ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 48 | total time: 1056.29m | eta: 3364.0m
step 47850/200000 (23.93%) | loss: 2.817690 | lrm: 0.85 | dt: 1324.99ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 50 | total time: 1057.40m | eta: 3363.0m
step 47900/200000 (23.95%) | loss: 2.896827 | lrm: 0.85 | dt: 1327.01ms | tok/sec: 12,346 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 52 | total time: 1058.51m | eta: 3361.8m
step 47950/200000 (23.98%) | loss: 2.735527 | lrm: 0.85 | dt: 1321.86ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 54 | total time: 1059.61m | eta: 3360.7m
Step 48000 | Validation bpb: 1.020694
<|bos|>The capital of France is Toulouse, the capital of Rome. The capital is named after an Italian word for a small town.
R
<|bos|>The chemical symbol of gold is Ca.
The symbol of gold is Ca.
The chemical symbol of gold is Ca.
Acid and al
<|bos|>If yesterday was Friday, then tomorrow will be Friday. If Friday's are Saturday, then so will be Sunday. But for the first time
<|bos|>The opposite of hot is the hot part. Hot is the hot part. The hot part is not hot.
Down the heat of a boil
<|bos|>The planets of the solar system are: 1. The sun-2. The moon-3. The moon-4. The solar system's surface.
The
<|bos|>My favorite color is red. That's the color my husband and I use at Christmas Eve, and they have red on the outside
<|bos|>If 5*x + 3 = 13, then x is the sum of the sum of its sides. There are 13 coefficients (A). For every 13 co
2026-03-17 16:30:12,660 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_048000.pt
2026-03-17 16:30:12,660 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_048000.json
2026-03-17 16:30:13,921 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_048000_rank0.pt
step 48000/200000 (24.00%) | loss: 2.856830 | lrm: 0.85 | dt: 1531.28ms | tok/sec: 10,699 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 56 | total time: 1060.72m | eta: 3359.6m
step 48050/200000 (24.02%) | loss: 2.881570 | lrm: 0.85 | dt: 1322.17ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 58 | total time: 1061.83m | eta: 3358.5m
step 48100/200000 (24.05%) | loss: 2.718537 | lrm: 0.85 | dt: 1323.79ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 60 | total time: 1062.93m | eta: 3357.4m
step 48150/200000 (24.07%) | loss: 2.755620 | lrm: 0.85 | dt: 1323.39ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 62 | total time: 1064.04m | eta: 3356.3m
step 48200/200000 (24.10%) | loss: 2.834941 | lrm: 0.85 | dt: 1332.30ms | tok/sec: 12,297 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 64 | total time: 1065.14m | eta: 3355.2m
step 48250/200000 (24.12%) | loss: 2.717919 | lrm: 0.85 | dt: 1325.42ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 66 | total time: 1066.25m | eta: 3354.1m
step 48300/200000 (24.15%) | loss: 2.839746 | lrm: 0.85 | dt: 1327.25ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 68 | total time: 1067.35m | eta: 3353.0m
step 48350/200000 (24.18%) | loss: 2.808046 | lrm: 0.85 | dt: 1324.21ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 70 | total time: 1068.46m | eta: 3351.9m
step 48400/200000 (24.20%) | loss: 2.744862 | lrm: 0.85 | dt: 1331.19ms | tok/sec: 12,307 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 72 | total time: 1069.56m | eta: 3350.8m
step 48450/200000 (24.23%) | loss: 2.769850 | lrm: 0.85 | dt: 1321.62ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 74 | total time: 1070.67m | eta: 3349.7m
step 48500/200000 (24.25%) | loss: 2.726290 | lrm: 0.85 | dt: 1323.74ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 76 | total time: 1071.77m | eta: 3348.6m
step 48550/200000 (24.27%) | loss: 2.802393 | lrm: 0.85 | dt: 1321.06ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 78 | total time: 1072.87m | eta: 3347.5m
step 48600/200000 (24.30%) | loss: 2.774934 | lrm: 0.85 | dt: 1326.60ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 80 | total time: 1073.98m | eta: 3346.4m
step 48650/200000 (24.32%) | loss: 2.774131 | lrm: 0.85 | dt: 1327.97ms | tok/sec: 12,337 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 82 | total time: 1075.08m | eta: 3345.3m
step 48700/200000 (24.35%) | loss: 2.779821 | lrm: 0.85 | dt: 1323.38ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 1 | total time: 1076.19m | eta: 3344.2m
step 48750/200000 (24.38%) | loss: 2.804968 | lrm: 0.85 | dt: 1324.85ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 3 | total time: 1077.29m | eta: 3343.1m
step 48800/200000 (24.40%) | loss: 2.723940 | lrm: 0.85 | dt: 1320.80ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 5 | total time: 1078.40m | eta: 3342.0m
step 48850/200000 (24.43%) | loss: 2.750577 | lrm: 0.85 | dt: 1327.82ms | tok/sec: 12,339 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 7 | total time: 1079.50m | eta: 3340.8m
step 48900/200000 (24.45%) | loss: 2.751143 | lrm: 0.85 | dt: 1321.81ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 9 | total time: 1080.61m | eta: 3339.7m
step 48950/200000 (24.48%) | loss: 2.717650 | lrm: 0.85 | dt: 1322.94ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 11 | total time: 1081.71m | eta: 3338.6m
step 49000/200000 (24.50%) | loss: 2.819704 | lrm: 0.85 | dt: 1329.59ms | tok/sec: 12,322 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 13 | total time: 1082.82m | eta: 3337.5m
step 49050/200000 (24.52%) | loss: 2.832283 | lrm: 0.85 | dt: 1324.54ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 16 | total time: 1083.92m | eta: 3336.4m
step 49100/200000 (24.55%) | loss: 2.815051 | lrm: 0.85 | dt: 1332.14ms | tok/sec: 12,298 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 18 | total time: 1085.03m | eta: 3335.3m
step 49150/200000 (24.57%) | loss: 2.822025 | lrm: 0.85 | dt: 1325.97ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 20 | total time: 1086.13m | eta: 3334.2m
step 49200/200000 (24.60%) | loss: 2.874218 | lrm: 0.85 | dt: 2018.20ms | tok/sec: 8,118 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 22 | total time: 1087.25m | eta: 3333.1m
step 49250/200000 (24.62%) | loss: 2.755930 | lrm: 0.85 | dt: 1361.62ms | tok/sec: 12,032 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 24 | total time: 1088.38m | eta: 3332.1m
step 49300/200000 (24.65%) | loss: 2.742897 | lrm: 0.85 | dt: 1359.31ms | tok/sec: 12,053 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 26 | total time: 1089.53m | eta: 3331.1m
step 49350/200000 (24.68%) | loss: 2.789776 | lrm: 0.85 | dt: 1340.54ms | tok/sec: 12,221 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 28 | total time: 1090.66m | eta: 3330.1m
step 49400/200000 (24.70%) | loss: 2.763529 | lrm: 0.84 | dt: 1335.40ms | tok/sec: 12,268 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 30 | total time: 1091.78m | eta: 3329.0m
step 49450/200000 (24.73%) | loss: 2.750080 | lrm: 0.84 | dt: 1343.35ms | tok/sec: 12,196 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 32 | total time: 1092.89m | eta: 3328.0m
step 49500/200000 (24.75%) | loss: 2.820538 | lrm: 0.84 | dt: 1348.89ms | tok/sec: 12,146 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 34 | total time: 1094.01m | eta: 3326.9m
step 49550/200000 (24.77%) | loss: 2.788290 | lrm: 0.84 | dt: 1338.46ms | tok/sec: 12,240 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 36 | total time: 1095.13m | eta: 3325.8m
step 49600/200000 (24.80%) | loss: 2.823753 | lrm: 0.84 | dt: 1338.63ms | tok/sec: 12,239 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 38 | total time: 1096.25m | eta: 3324.8m
step 49650/200000 (24.82%) | loss: 2.787500 | lrm: 0.84 | dt: 1344.23ms | tok/sec: 12,188 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 40 | total time: 1097.37m | eta: 3323.7m
step 49700/200000 (24.85%) | loss: 2.841512 | lrm: 0.84 | dt: 1338.43ms | tok/sec: 12,241 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 42 | total time: 1098.48m | eta: 3322.6m
step 49750/200000 (24.88%) | loss: 2.841394 | lrm: 0.84 | dt: 1337.37ms | tok/sec: 12,250 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 44 | total time: 1099.60m | eta: 3321.6m
step 49800/200000 (24.90%) | loss: 2.799476 | lrm: 0.84 | dt: 1339.31ms | tok/sec: 12,233 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 46 | total time: 1100.72m | eta: 3320.5m
step 49850/200000 (24.93%) | loss: 2.771194 | lrm: 0.84 | dt: 1332.90ms | tok/sec: 12,291 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 48 | total time: 1101.84m | eta: 3319.4m
step 49900/200000 (24.95%) | loss: 2.775977 | lrm: 0.84 | dt: 1366.93ms | tok/sec: 11,985 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 50 | total time: 1102.95m | eta: 3318.4m
step 49950/200000 (24.98%) | loss: 2.756242 | lrm: 0.84 | dt: 1336.12ms | tok/sec: 12,262 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 52 | total time: 1104.07m | eta: 3317.3m
Step 50000 | Validation bpb: 1.018188
step 50000/200000 (25.00%) | loss: 2.837262 | lrm: 0.84 | dt: 1432.63ms | tok/sec: 11,436 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 54 | total time: 1105.19m | eta: 3316.2m
step 50050/200000 (25.02%) | loss: 2.864203 | lrm: 0.84 | dt: 1347.96ms | tok/sec: 12,154 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 56 | total time: 1106.32m | eta: 3315.2m
step 50100/200000 (25.05%) | loss: 2.815016 | lrm: 0.84 | dt: 1352.37ms | tok/sec: 12,114 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 58 | total time: 1107.44m | eta: 3314.1m
step 50150/200000 (25.07%) | loss: 2.779557 | lrm: 0.84 | dt: 1344.72ms | tok/sec: 12,183 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 60 | total time: 1108.56m | eta: 3313.1m
step 50200/200000 (25.10%) | loss: 2.874597 | lrm: 0.84 | dt: 1369.89ms | tok/sec: 11,960 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 62 | total time: 1109.69m | eta: 3312.0m
step 50250/200000 (25.12%) | loss: 2.770460 | lrm: 0.84 | dt: 1349.59ms | tok/sec: 12,140 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 64 | total time: 1110.81m | eta: 3311.0m
step 50300/200000 (25.15%) | loss: 2.777371 | lrm: 0.84 | dt: 1343.48ms | tok/sec: 12,195 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 66 | total time: 1111.93m | eta: 3309.9m
step 50350/200000 (25.18%) | loss: 2.826606 | lrm: 0.84 | dt: 1360.77ms | tok/sec: 12,040 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 68 | total time: 1113.05m | eta: 3308.9m
step 50400/200000 (25.20%) | loss: 2.815281 | lrm: 0.84 | dt: 1352.76ms | tok/sec: 12,111 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 71 | total time: 1114.18m | eta: 3307.8m
step 50450/200000 (25.23%) | loss: 2.792358 | lrm: 0.84 | dt: 1351.20ms | tok/sec: 12,125 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 73 | total time: 1115.31m | eta: 3306.8m
step 50500/200000 (25.25%) | loss: 2.838635 | lrm: 0.84 | dt: 1351.61ms | tok/sec: 12,121 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 75 | total time: 1116.44m | eta: 3305.8m
step 50550/200000 (25.27%) | loss: 2.850061 | lrm: 0.84 | dt: 1349.44ms | tok/sec: 12,141 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 77 | total time: 1117.57m | eta: 3304.7m
step 50600/200000 (25.30%) | loss: 2.852408 | lrm: 0.84 | dt: 1350.04ms | tok/sec: 12,135 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 79 | total time: 1118.70m | eta: 3303.7m
step 50650/200000 (25.32%) | loss: 2.799861 | lrm: 0.84 | dt: 1356.88ms | tok/sec: 12,074 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 81 | total time: 1119.82m | eta: 3302.6m
step 50700/200000 (25.35%) | loss: 2.858097 | lrm: 0.84 | dt: 1333.60ms | tok/sec: 12,285 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 0 | total time: 1120.94m | eta: 3301.6m
step 50750/200000 (25.38%) | loss: 2.843708 | lrm: 0.84 | dt: 1345.07ms | tok/sec: 12,180 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 2 | total time: 1122.06m | eta: 3300.5m
step 50800/200000 (25.40%) | loss: 2.762331 | lrm: 0.84 | dt: 1348.68ms | tok/sec: 12,148 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 4 | total time: 1123.18m | eta: 3299.4m
step 50850/200000 (25.43%) | loss: 2.801561 | lrm: 0.84 | dt: 1350.69ms | tok/sec: 12,130 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 6 | total time: 1124.30m | eta: 3298.4m
step 50900/200000 (25.45%) | loss: 2.785069 | lrm: 0.84 | dt: 1341.14ms | tok/sec: 12,216 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 8 | total time: 1125.43m | eta: 3297.3m
step 50950/200000 (25.48%) | loss: 2.838686 | lrm: 0.84 | dt: 1346.67ms | tok/sec: 12,166 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 10 | total time: 1126.56m | eta: 3296.3m
step 51000/200000 (25.50%) | loss: 2.793651 | lrm: 0.84 | dt: 1352.69ms | tok/sec: 12,112 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 12 | total time: 1127.68m | eta: 3295.2m
step 51050/200000 (25.52%) | loss: 2.769800 | lrm: 0.84 | dt: 1333.51ms | tok/sec: 12,286 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 14 | total time: 1128.81m | eta: 3294.2m
step 51100/200000 (25.55%) | loss: 2.807500 | lrm: 0.84 | dt: 1355.29ms | tok/sec: 12,088 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 16 | total time: 1129.93m | eta: 3293.2m
step 51150/200000 (25.57%) | loss: 2.689889 | lrm: 0.84 | dt: 1351.86ms | tok/sec: 12,119 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 18 | total time: 1131.06m | eta: 3292.1m
step 51200/200000 (25.60%) | loss: 2.831633 | lrm: 0.84 | dt: 1351.55ms | tok/sec: 12,122 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 20 | total time: 1132.19m | eta: 3291.1m
step 51250/200000 (25.62%) | loss: 2.812577 | lrm: 0.84 | dt: 1346.47ms | tok/sec: 12,168 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 22 | total time: 1133.31m | eta: 3290.0m
step 51300/200000 (25.65%) | loss: 2.769315 | lrm: 0.83 | dt: 1356.59ms | tok/sec: 12,077 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 24 | total time: 1134.44m | eta: 3289.0m
step 51350/200000 (25.68%) | loss: 2.763668 | lrm: 0.83 | dt: 1333.24ms | tok/sec: 12,288 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 26 | total time: 1135.57m | eta: 3287.9m
step 51400/200000 (25.70%) | loss: 2.738240 | lrm: 0.83 | dt: 1355.64ms | tok/sec: 12,085 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 28 | total time: 1136.69m | eta: 3286.9m
step 51450/200000 (25.73%) | loss: 2.692090 | lrm: 0.83 | dt: 1360.08ms | tok/sec: 12,046 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 30 | total time: 1137.82m | eta: 3285.8m
step 51500/200000 (25.75%) | loss: 2.768325 | lrm: 0.83 | dt: 1342.61ms | tok/sec: 12,203 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 32 | total time: 1138.94m | eta: 3284.8m
step 51550/200000 (25.77%) | loss: 2.724697 | lrm: 0.83 | dt: 1355.48ms | tok/sec: 12,087 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 35 | total time: 1140.06m | eta: 3283.7m
step 51600/200000 (25.80%) | loss: 2.803564 | lrm: 0.83 | dt: 1344.60ms | tok/sec: 12,185 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 37 | total time: 1141.18m | eta: 3282.6m
step 51650/200000 (25.82%) | loss: 2.779397 | lrm: 0.83 | dt: 1338.36ms | tok/sec: 12,241 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 39 | total time: 1142.29m | eta: 3281.6m
step 51700/200000 (25.85%) | loss: 2.791628 | lrm: 0.83 | dt: 1394.56ms | tok/sec: 11,748 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 41 | total time: 1143.41m | eta: 3280.5m
step 51750/200000 (25.88%) | loss: 2.692427 | lrm: 0.83 | dt: 1335.46ms | tok/sec: 12,268 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 43 | total time: 1144.53m | eta: 3279.4m
step 51800/200000 (25.90%) | loss: 2.840543 | lrm: 0.83 | dt: 1342.37ms | tok/sec: 12,205 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 45 | total time: 1145.65m | eta: 3278.3m
step 51850/200000 (25.93%) | loss: 2.781686 | lrm: 0.83 | dt: 1353.65ms | tok/sec: 12,103 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 47 | total time: 1146.77m | eta: 3277.3m
step 51900/200000 (25.95%) | loss: 2.793225 | lrm: 0.83 | dt: 1352.22ms | tok/sec: 12,116 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 49 | total time: 1147.89m | eta: 3276.2m
step 51950/200000 (25.98%) | loss: 2.819943 | lrm: 0.83 | dt: 1353.40ms | tok/sec: 12,105 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 51 | total time: 1149.02m | eta: 3275.2m
Step 52000 | Validation bpb: 1.016525
<|bos|>The capital of France is Fremen, a city that is close to the city of Jolene. In 1383, the capital of
<|bos|>The chemical symbol of gold is `Au` in the Greek alphabet. The symbol is in the letters A, B, AG, AH
<|bos|>If yesterday was Friday, then tomorrow will be Friday. Saturday is Friday, and Saturday is Sunday. If Saturday were F
<|bos|>`The opposite of hot is cold`. In the same way we think about the future, the new things can be made. So what does this mean for
<|bos|>`The planets of the solar system are: Mars, Jupiter, Saturn, Uranus, Neptune, Tyra, Merc`
<|bos|>My favorite color is red. In the last decade, humans have been making an enormous number of different colors that we have been able to
<|bos|>If 5*x + 3 = 13, then x is the number of times a triple (5*x) is greater than y. So, y=1/(2
2026-03-17 18:00:11,990 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_052000.pt
2026-03-17 18:00:11,994 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_052000.json
2026-03-17 18:00:13,643 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_052000_rank0.pt
step 52000/200000 (26.00%) | loss: 2.800354 | lrm: 0.83 | dt: 1595.93ms | tok/sec: 10,266 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 53 | total time: 1150.15m | eta: 3274.1m
step 52050/200000 (26.02%) | loss: 2.763841 | lrm: 0.83 | dt: 1343.71ms | tok/sec: 12,193 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 55 | total time: 1151.28m | eta: 3273.1m
step 52100/200000 (26.05%) | loss: 2.844473 | lrm: 0.83 | dt: 1342.79ms | tok/sec: 12,201 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 57 | total time: 1152.40m | eta: 3272.0m
step 52150/200000 (26.07%) | loss: 2.795162 | lrm: 0.83 | dt: 1336.92ms | tok/sec: 12,255 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 59 | total time: 1153.52m | eta: 3271.0m
step 52200/200000 (26.10%) | loss: 2.854593 | lrm: 0.83 | dt: 1334.22ms | tok/sec: 12,279 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 61 | total time: 1154.63m | eta: 3269.9m
step 52250/200000 (26.12%) | loss: 2.721819 | lrm: 0.83 | dt: 1342.94ms | tok/sec: 12,200 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 63 | total time: 1155.75m | eta: 3268.8m
step 52300/200000 (26.15%) | loss: 2.834313 | lrm: 0.83 | dt: 1351.71ms | tok/sec: 12,120 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 65 | total time: 1156.88m | eta: 3267.7m
step 52350/200000 (26.18%) | loss: 2.806286 | lrm: 0.83 | dt: 1354.46ms | tok/sec: 12,096 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 67 | total time: 1158.00m | eta: 3266.7m
step 52400/200000 (26.20%) | loss: 2.747749 | lrm: 0.83 | dt: 1359.10ms | tok/sec: 12,055 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 69 | total time: 1159.13m | eta: 3265.7m
step 52450/200000 (26.23%) | loss: 2.850943 | lrm: 0.83 | dt: 1348.90ms | tok/sec: 12,146 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 71 | total time: 1160.26m | eta: 3264.6m
step 52500/200000 (26.25%) | loss: 2.734795 | lrm: 0.83 | dt: 1355.47ms | tok/sec: 12,087 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 73 | total time: 1161.39m | eta: 3263.6m
step 52550/200000 (26.27%) | loss: 2.759863 | lrm: 0.83 | dt: 1345.63ms | tok/sec: 12,175 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 75 | total time: 1162.52m | eta: 3262.5m
step 52600/200000 (26.30%) | loss: 2.764984 | lrm: 0.83 | dt: 1348.54ms | tok/sec: 12,149 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 77 | total time: 1163.64m | eta: 3261.5m
step 52650/200000 (26.32%) | loss: 2.795015 | lrm: 0.83 | dt: 1345.01ms | tok/sec: 12,181 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 79 | total time: 1164.76m | eta: 3260.4m
step 52700/200000 (26.35%) | loss: 2.819700 | lrm: 0.83 | dt: 1349.51ms | tok/sec: 12,140 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 81 | total time: 1165.89m | eta: 3259.4m
step 52750/200000 (26.38%) | loss: 2.834795 | lrm: 0.83 | dt: 1341.32ms | tok/sec: 12,214 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 1 | total time: 1167.01m | eta: 3258.3m
step 52800/200000 (26.40%) | loss: 2.827854 | lrm: 0.83 | dt: 1351.86ms | tok/sec: 12,119 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 3 | total time: 1168.14m | eta: 3257.2m
step 52850/200000 (26.43%) | loss: 2.767607 | lrm: 0.83 | dt: 1349.76ms | tok/sec: 12,138 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 5 | total time: 1169.26m | eta: 3256.2m
step 52900/200000 (26.45%) | loss: 2.826654 | lrm: 0.83 | dt: 1354.09ms | tok/sec: 12,099 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 7 | total time: 1170.38m | eta: 3255.1m
step 52950/200000 (26.48%) | loss: 2.786675 | lrm: 0.83 | dt: 1352.83ms | tok/sec: 12,110 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 9 | total time: 1171.51m | eta: 3254.1m
step 53000/200000 (26.50%) | loss: 2.765461 | lrm: 0.83 | dt: 1359.63ms | tok/sec: 12,050 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 11 | total time: 1172.64m | eta: 3253.0m
step 53050/200000 (26.52%) | loss: 2.790843 | lrm: 0.83 | dt: 1362.57ms | tok/sec: 12,024 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 13 | total time: 1173.78m | eta: 3252.0m
step 53100/200000 (26.55%) | loss: 2.789564 | lrm: 0.83 | dt: 1357.48ms | tok/sec: 12,069 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 15 | total time: 1174.91m | eta: 3251.0m
step 53150/200000 (26.57%) | loss: 2.845729 | lrm: 0.83 | dt: 1354.08ms | tok/sec: 12,099 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 17 | total time: 1176.04m | eta: 3249.9m
step 53200/200000 (26.60%) | loss: 2.791471 | lrm: 0.82 | dt: 1350.20ms | tok/sec: 12,134 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 19 | total time: 1177.17m | eta: 3248.9m
step 53250/200000 (26.62%) | loss: 2.789998 | lrm: 0.82 | dt: 1331.60ms | tok/sec: 12,304 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 21 | total time: 1178.29m | eta: 3247.8m
step 53300/200000 (26.65%) | loss: 2.793448 | lrm: 0.82 | dt: 1335.06ms | tok/sec: 12,272 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 23 | total time: 1179.40m | eta: 3246.7m
step 53350/200000 (26.68%) | loss: 2.745032 | lrm: 0.82 | dt: 1341.46ms | tok/sec: 12,213 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 25 | total time: 1180.52m | eta: 3245.7m
step 53400/200000 (26.70%) | loss: 2.835474 | lrm: 0.82 | dt: 1344.02ms | tok/sec: 12,190 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 27 | total time: 1181.65m | eta: 3244.6m
step 53450/200000 (26.73%) | loss: 2.720747 | lrm: 0.82 | dt: 1339.93ms | tok/sec: 12,227 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 29 | total time: 1182.77m | eta: 3243.5m
step 53500/200000 (26.75%) | loss: 2.755362 | lrm: 0.82 | dt: 1334.68ms | tok/sec: 12,275 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 31 | total time: 1183.89m | eta: 3242.5m
step 53550/200000 (26.77%) | loss: 2.810730 | lrm: 0.82 | dt: 1348.41ms | tok/sec: 12,150 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 33 | total time: 1185.01m | eta: 3241.4m
step 53600/200000 (26.80%) | loss: 2.717470 | lrm: 0.82 | dt: 1350.03ms | tok/sec: 12,135 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 35 | total time: 1186.13m | eta: 3240.3m
step 53650/200000 (26.82%) | loss: 2.801139 | lrm: 0.82 | dt: 1359.47ms | tok/sec: 12,051 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 37 | total time: 1187.26m | eta: 3239.3m
step 53700/200000 (26.85%) | loss: 2.816592 | lrm: 0.82 | dt: 1338.60ms | tok/sec: 12,239 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 39 | total time: 1188.39m | eta: 3238.2m
step 53750/200000 (26.88%) | loss: 2.702654 | lrm: 0.82 | dt: 1342.36ms | tok/sec: 12,205 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 41 | total time: 1189.50m | eta: 3237.2m
step 53800/200000 (26.90%) | loss: 2.783203 | lrm: 0.82 | dt: 1367.54ms | tok/sec: 11,980 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 44 | total time: 1190.63m | eta: 3236.1m
step 53850/200000 (26.93%) | loss: 2.798102 | lrm: 0.82 | dt: 1346.49ms | tok/sec: 12,167 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 46 | total time: 1191.76m | eta: 3235.1m
step 53900/200000 (26.95%) | loss: 2.770176 | lrm: 0.82 | dt: 1364.98ms | tok/sec: 12,003 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 48 | total time: 1192.89m | eta: 3234.0m
step 53950/200000 (26.98%) | loss: 2.801951 | lrm: 0.82 | dt: 1350.69ms | tok/sec: 12,130 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 50 | total time: 1194.02m | eta: 3233.0m
Step 54000 | Validation bpb: 1.014704
step 54000/200000 (27.00%) | loss: 2.806146 | lrm: 0.82 | dt: 1438.95ms | tok/sec: 11,386 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 52 | total time: 1195.15m | eta: 3231.9m
step 54050/200000 (27.02%) | loss: 2.848514 | lrm: 0.82 | dt: 1349.82ms | tok/sec: 12,137 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 54 | total time: 1196.28m | eta: 3230.9m
step 54100/200000 (27.05%) | loss: 2.745824 | lrm: 0.82 | dt: 1340.01ms | tok/sec: 12,226 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 56 | total time: 1197.40m | eta: 3229.8m
step 54150/200000 (27.07%) | loss: 2.816792 | lrm: 0.82 | dt: 1353.54ms | tok/sec: 12,104 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 58 | total time: 1198.53m | eta: 3228.8m
step 54200/200000 (27.10%) | loss: 2.814236 | lrm: 0.82 | dt: 1356.55ms | tok/sec: 12,077 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 60 | total time: 1199.66m | eta: 3227.7m
step 54250/200000 (27.12%) | loss: 2.773720 | lrm: 0.82 | dt: 1353.22ms | tok/sec: 12,107 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 62 | total time: 1200.80m | eta: 3226.7m
step 54300/200000 (27.15%) | loss: 2.798271 | lrm: 0.82 | dt: 1357.40ms | tok/sec: 12,070 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 64 | total time: 1201.93m | eta: 3225.7m
step 54350/200000 (27.18%) | loss: 2.808817 | lrm: 0.82 | dt: 1357.59ms | tok/sec: 12,068 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 66 | total time: 1203.06m | eta: 3224.6m
step 54400/200000 (27.20%) | loss: 2.727580 | lrm: 0.82 | dt: 1366.98ms | tok/sec: 11,985 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 68 | total time: 1204.19m | eta: 3223.6m
step 54450/200000 (27.23%) | loss: 2.758664 | lrm: 0.82 | dt: 1389.83ms | tok/sec: 11,788 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 70 | total time: 1205.33m | eta: 3222.6m
step 54500/200000 (27.25%) | loss: 2.747309 | lrm: 0.82 | dt: 1342.19ms | tok/sec: 12,206 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 72 | total time: 1206.46m | eta: 3221.5m
step 54550/200000 (27.27%) | loss: 2.765688 | lrm: 0.82 | dt: 1337.56ms | tok/sec: 12,249 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 74 | total time: 1207.58m | eta: 3220.4m
step 54600/200000 (27.30%) | loss: 2.816489 | lrm: 0.82 | dt: 1337.82ms | tok/sec: 12,246 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 76 | total time: 1208.70m | eta: 3219.4m
step 54650/200000 (27.32%) | loss: 2.831715 | lrm: 0.82 | dt: 1344.54ms | tok/sec: 12,185 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 78 | total time: 1209.81m | eta: 3218.3m
step 54700/200000 (27.35%) | loss: 2.815520 | lrm: 0.82 | dt: 1344.87ms | tok/sec: 12,182 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 80 | total time: 1210.94m | eta: 3217.2m
step 54750/200000 (27.38%) | loss: 2.701296 | lrm: 0.82 | dt: 1359.43ms | tok/sec: 12,052 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 0 | total time: 1212.06m | eta: 3216.1m
step 54800/200000 (27.40%) | loss: 2.676099 | lrm: 0.82 | dt: 1350.87ms | tok/sec: 12,128 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 2 | total time: 1213.19m | eta: 3215.1m
step 54850/200000 (27.43%) | loss: 2.770069 | lrm: 0.82 | dt: 1354.05ms | tok/sec: 12,099 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 5 | total time: 1214.32m | eta: 3214.0m
step 54900/200000 (27.45%) | loss: 2.816357 | lrm: 0.82 | dt: 1357.36ms | tok/sec: 12,070 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 7 | total time: 1215.45m | eta: 3213.0m
step 54950/200000 (27.48%) | loss: 2.783430 | lrm: 0.82 | dt: 1355.29ms | tok/sec: 12,088 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 9 | total time: 1216.57m | eta: 3211.9m
step 55000/200000 (27.50%) | loss: 2.738543 | lrm: 0.82 | dt: 1736.78ms | tok/sec: 9,433 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 11 | total time: 1217.71m | eta: 3210.9m
step 55050/200000 (27.52%) | loss: 2.780940 | lrm: 0.82 | dt: 1351.33ms | tok/sec: 12,124 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 13 | total time: 1218.84m | eta: 3209.9m
step 55100/200000 (27.55%) | loss: 2.796897 | lrm: 0.81 | dt: 1340.15ms | tok/sec: 12,225 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 15 | total time: 1219.96m | eta: 3208.8m
step 55150/200000 (27.57%) | loss: 2.835966 | lrm: 0.81 | dt: 1348.76ms | tok/sec: 12,147 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 17 | total time: 1221.09m | eta: 3207.7m
step 55200/200000 (27.60%) | loss: 2.732747 | lrm: 0.81 | dt: 1343.94ms | tok/sec: 12,191 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 19 | total time: 1222.21m | eta: 3206.7m
step 55250/200000 (27.62%) | loss: 2.728777 | lrm: 0.81 | dt: 1696.83ms | tok/sec: 9,655 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 21 | total time: 1223.40m | eta: 3205.8m
step 55300/200000 (27.65%) | loss: 2.760091 | lrm: 0.81 | dt: 1615.61ms | tok/sec: 10,141 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 23 | total time: 1224.82m | eta: 3205.5m
step 55350/200000 (27.68%) | loss: 2.836638 | lrm: 0.81 | dt: 1550.42ms | tok/sec: 10,567 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 25 | total time: 1226.13m | eta: 3204.9m
step 55400/200000 (27.70%) | loss: 2.769427 | lrm: 0.81 | dt: 1530.20ms | tok/sec: 10,707 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 27 | total time: 1227.42m | eta: 3204.3m
step 55450/200000 (27.73%) | loss: 2.848359 | lrm: 0.81 | dt: 1553.66ms | tok/sec: 10,545 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 29 | total time: 1228.71m | eta: 3203.6m
step 55500/200000 (27.75%) | loss: 2.807188 | lrm: 0.81 | dt: 1511.14ms | tok/sec: 10,842 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 31 | total time: 1230.00m | eta: 3203.0m
step 55550/200000 (27.77%) | loss: 2.798050 | lrm: 0.81 | dt: 1524.32ms | tok/sec: 10,748 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 33 | total time: 1231.28m | eta: 3202.4m
step 55600/200000 (27.80%) | loss: 2.782473 | lrm: 0.81 | dt: 1418.21ms | tok/sec: 11,552 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 35 | total time: 1232.51m | eta: 3201.6m
step 55650/200000 (27.82%) | loss: 2.834770 | lrm: 0.81 | dt: 1431.83ms | tok/sec: 11,442 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 37 | total time: 1233.69m | eta: 3200.6m
step 55700/200000 (27.85%) | loss: 2.814723 | lrm: 0.81 | dt: 1467.43ms | tok/sec: 11,165 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 39 | total time: 1234.88m | eta: 3199.7m
step 55750/200000 (27.88%) | loss: 2.715843 | lrm: 0.81 | dt: 1442.81ms | tok/sec: 11,355 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 41 | total time: 1236.09m | eta: 3198.9m
step 55800/200000 (27.90%) | loss: 2.764819 | lrm: 0.81 | dt: 1496.44ms | tok/sec: 10,948 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 43 | total time: 1237.31m | eta: 3198.1m
step 55850/200000 (27.93%) | loss: 2.732549 | lrm: 0.81 | dt: 1497.82ms | tok/sec: 10,938 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 45 | total time: 1238.55m | eta: 3197.3m
step 55900/200000 (27.95%) | loss: 2.846988 | lrm: 0.81 | dt: 1500.67ms | tok/sec: 10,917 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 47 | total time: 1239.81m | eta: 3196.6m
step 55950/200000 (27.98%) | loss: 2.780794 | lrm: 0.81 | dt: 1510.82ms | tok/sec: 10,844 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 49 | total time: 1241.07m | eta: 3195.9m
Step 56000 | Validation bpb: 1.012533
<|bos|>The capital of France is the capital of France, the capital of France is the capital of France, the capital of France is the capital of France is
<|bos|>The chemical symbol of gold is gold, an allotrope of noble gas. It is discovered to be in the shape of a ball or a
<|bos|>If yesterday was Friday, then tomorrow will be Friday. I am so confused about the concept of the first quarter of a year (I guess I'll just
<|bos|>The opposite of hot is cold. You can get really cold with clothing and other clothing, the longer the temperature goes, the harder the clothing, so
<|bos|>The planets of the solar system are: Mars, Jupiter, Saturn, and Saturn.
The sun is at apex position.
<|bos|>My favorite color is red. There are a lot of them from celeste to bristles of red, blue and yellow to blue
<|bos|>If 5*x + 3 = 13, then x is 13 times 5* 13 is 13 times 13 = 13 13. This means that x is
2026-03-17 19:32:56,896 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_056000.pt
2026-03-17 19:32:56,899 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_056000.json
2026-03-17 19:32:58,386 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_056000_rank0.pt
step 56000/200000 (28.00%) | loss: 2.824663 | lrm: 0.81 | dt: 1538.11ms | tok/sec: 10,652 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 52 | total time: 1242.32m | eta: 3195.1m
step 56050/200000 (28.02%) | loss: 2.775627 | lrm: 0.81 | dt: 1504.51ms | tok/sec: 10,889 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 54 | total time: 1243.59m | eta: 3194.4m
step 56100/200000 (28.05%) | loss: 2.782825 | lrm: 0.81 | dt: 1522.22ms | tok/sec: 10,763 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 56 | total time: 1244.86m | eta: 3193.7m
step 56150/200000 (28.07%) | loss: 2.811083 | lrm: 0.81 | dt: 1534.51ms | tok/sec: 10,677 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 58 | total time: 1246.13m | eta: 3193.0m
step 56200/200000 (28.10%) | loss: 2.782226 | lrm: 0.81 | dt: 1557.60ms | tok/sec: 10,518 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 60 | total time: 1247.42m | eta: 3192.4m
step 56250/200000 (28.12%) | loss: 2.739564 | lrm: 0.81 | dt: 1551.24ms | tok/sec: 10,561 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 62 | total time: 1248.71m | eta: 3191.7m
step 56300/200000 (28.15%) | loss: 2.757533 | lrm: 0.81 | dt: 1538.39ms | tok/sec: 10,650 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 64 | total time: 1250.00m | eta: 3191.1m
step 56350/200000 (28.18%) | loss: 2.769636 | lrm: 0.81 | dt: 1539.92ms | tok/sec: 10,639 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 66 | total time: 1251.28m | eta: 3190.4m
step 56400/200000 (28.20%) | loss: 2.800351 | lrm: 0.81 | dt: 1513.61ms | tok/sec: 10,824 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 68 | total time: 1252.53m | eta: 3189.6m
step 56450/200000 (28.23%) | loss: 2.717393 | lrm: 0.81 | dt: 1546.14ms | tok/sec: 10,596 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 70 | total time: 1253.82m | eta: 3189.0m
step 56500/200000 (28.25%) | loss: 2.799198 | lrm: 0.81 | dt: 1469.59ms | tok/sec: 11,148 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 72 | total time: 1255.10m | eta: 3188.3m


##
step 56600/200000 (28.30%) | loss: 2.762164 | lrm: 0.81 | dt: 1480.63ms | tok/sec: 11,065 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 76 | total time: 1257.64m | eta: 3186.9m
step 56650/200000 (28.32%) | loss: 2.785100 | lrm: 0.81 | dt: 1487.20ms | tok/sec: 11,016 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 78 | total time: 1258.90m | eta: 3186.1m
step 56700/200000 (28.35%) | loss: 2.801669 | lrm: 0.81 | dt: 1506.81ms | tok/sec: 10,873 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 80 | total time: 1260.14m | eta: 3185.4m
step 56750/200000 (28.38%) | loss: 2.788847 | lrm: 0.81 | dt: 1460.07ms | tok/sec: 11,221 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 82 | total time: 1261.39m | eta: 3184.6m
step 56800/200000 (28.40%) | loss: 2.800790 | lrm: 0.81 | dt: 1536.41ms | tok/sec: 10,663 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 1 | total time: 1262.64m | eta: 3183.8m
step 56850/200000 (28.43%) | loss: 2.816814 | lrm: 0.81 | dt: 1496.26ms | tok/sec: 10,949 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 3 | total time: 1263.88m | eta: 3183.0m
step 56900/200000 (28.45%) | loss: 2.743062 | lrm: 0.81 | dt: 1501.06ms | tok/sec: 10,914 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 5 | total time: 1265.12m | eta: 3182.3m
step 56950/200000 (28.48%) | loss: 2.793841 | lrm: 0.80 | dt: 1486.22ms | tok/sec: 11,023 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 7 | total time: 1266.36m | eta: 3181.5m
step 57000/200000 (28.50%) | loss: 2.764053 | lrm: 0.80 | dt: 1451.47ms | tok/sec: 11,287 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 9 | total time: 1267.61m | eta: 3180.7m
step 57050/200000 (28.52%) | loss: 2.839799 | lrm: 0.80 | dt: 1578.43ms | tok/sec: 10,379 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 11 | total time: 1268.86m | eta: 3179.9m
step 57100/200000 (28.55%) | loss: 2.781594 | lrm: 0.80 | dt: 1544.76ms | tok/sec: 10,606 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 13 | total time: 1270.17m | eta: 3179.3m
step 57150/200000 (28.57%) | loss: 2.708535 | lrm: 0.80 | dt: 1454.17ms | tok/sec: 11,266 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 16 | total time: 1271.42m | eta: 3178.6m
step 57200/200000 (28.60%) | loss: 2.802748 | lrm: 0.80 | dt: 1488.28ms | tok/sec: 11,008 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 18 | total time: 1272.66m | eta: 3177.7m
step 57250/200000 (28.62%) | loss: 2.750566 | lrm: 0.80 | dt: 1471.40ms | tok/sec: 11,134 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 20 | total time: 1273.89m | eta: 3176.9m
step 57300/200000 (28.65%) | loss: 2.801023 | lrm: 0.80 | dt: 1453.22ms | tok/sec: 11,274 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 22 | total time: 1275.13m | eta: 3176.1m
step 57350/200000 (28.68%) | loss: 2.769862 | lrm: 0.80 | dt: 1469.80ms | tok/sec: 11,147 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 24 | total time: 1276.37m | eta: 3175.3m
step 57400/200000 (28.70%) | loss: 2.798743 | lrm: 0.80 | dt: 1475.43ms | tok/sec: 11,104 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 26 | total time: 1277.61m | eta: 3174.5m
step 57450/200000 (28.73%) | loss: 2.795037 | lrm: 0.80 | dt: 1516.49ms | tok/sec: 10,803 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 28 | total time: 1278.84m | eta: 3173.7m
step 57500/200000 (28.75%) | loss: 2.699075 | lrm: 0.80 | dt: 1567.32ms | tok/sec: 10,453 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 30 | total time: 1280.10m | eta: 3173.0m
step 57550/200000 (28.77%) | loss: 2.749320 | lrm: 0.80 | dt: 1529.82ms | tok/sec: 10,709 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 32 | total time: 1281.38m | eta: 3172.3m
step 57600/200000 (28.80%) | loss: 2.823636 | lrm: 0.80 | dt: 1519.10ms | tok/sec: 10,785 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 34 | total time: 1282.64m | eta: 3171.5m
step 57650/200000 (28.82%) | loss: 2.772593 | lrm: 0.80 | dt: 1528.36ms | tok/sec: 10,719 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 36 | total time: 1283.91m | eta: 3170.8m
step 57700/200000 (28.85%) | loss: 2.707893 | lrm: 0.80 | dt: 1507.78ms | tok/sec: 10,866 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 38 | total time: 1285.17m | eta: 3170.0m
step 57750/200000 (28.88%) | loss: 2.869722 | lrm: 0.80 | dt: 1502.42ms | tok/sec: 10,905 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 40 | total time: 1286.44m | eta: 3169.3m
step 57800/200000 (28.90%) | loss: 2.810697 | lrm: 0.80 | dt: 1512.41ms | tok/sec: 10,833 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 42 | total time: 1287.70m | eta: 3168.6m
step 57850/200000 (28.93%) | loss: 2.785142 | lrm: 0.80 | dt: 1512.87ms | tok/sec: 10,829 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 44 | total time: 1288.96m | eta: 3167.8m
step 57900/200000 (28.95%) | loss: 2.840259 | lrm: 0.80 | dt: 1525.82ms | tok/sec: 10,737 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 46 | total time: 1290.23m | eta: 3167.1m
step 57950/200000 (28.98%) | loss: 2.753118 | lrm: 0.80 | dt: 1529.14ms | tok/sec: 10,714 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 48 | total time: 1291.50m | eta: 3166.3m
Step 58000 | Validation bpb: 1.010831
step 58000/200000 (29.00%) | loss: 2.789882 | lrm: 0.80 | dt: 1574.61ms | tok/sec: 10,405 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 50 | total time: 1292.76m | eta: 3165.6m
step 58050/200000 (29.02%) | loss: 2.755584 | lrm: 0.80 | dt: 1516.54ms | tok/sec: 10,803 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 52 | total time: 1294.03m | eta: 3164.9m
step 58100/200000 (29.05%) | loss: 2.767976 | lrm: 0.80 | dt: 1537.36ms | tok/sec: 10,657 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 54 | total time: 1295.31m | eta: 3164.1m
step 58150/200000 (29.07%) | loss: 2.738427 | lrm: 0.80 | dt: 1526.77ms | tok/sec: 10,731 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 56 | total time: 1296.58m | eta: 3163.4m
step 58200/200000 (29.10%) | loss: 2.774726 | lrm: 0.80 | dt: 1494.64ms | tok/sec: 10,961 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 58 | total time: 1297.84m | eta: 3162.6m
step 58250/200000 (29.12%) | loss: 2.794487 | lrm: 0.80 | dt: 1496.18ms | tok/sec: 10,950 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 60 | total time: 1299.11m | eta: 3161.9m
step 58300/200000 (29.15%) | loss: 2.691153 | lrm: 0.80 | dt: 1530.42ms | tok/sec: 10,705 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 63 | total time: 1300.37m | eta: 3161.1m
step 58350/200000 (29.18%) | loss: 2.785114 | lrm: 0.80 | dt: 1494.99ms | tok/sec: 10,959 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 65 | total time: 1301.63m | eta: 3160.4m
step 58400/200000 (29.20%) | loss: 2.709738 | lrm: 0.80 | dt: 1482.49ms | tok/sec: 11,051 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 67 | total time: 1302.88m | eta: 3159.6m
step 58450/200000 (29.23%) | loss: 2.785260 | lrm: 0.80 | dt: 1486.74ms | tok/sec: 11,020 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 69 | total time: 1304.12m | eta: 3158.8m
step 58500/200000 (29.25%) | loss: 2.864814 | lrm: 0.80 | dt: 1483.98ms | tok/sec: 11,040 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 71 | total time: 1305.36m | eta: 3157.9m
step 58550/200000 (29.27%) | loss: 2.779760 | lrm: 0.80 | dt: 1483.01ms | tok/sec: 11,047 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 73 | total time: 1306.61m | eta: 3157.2m
step 58600/200000 (29.30%) | loss: 2.762579 | lrm: 0.80 | dt: 1477.43ms | tok/sec: 11,089 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 75 | total time: 1307.85m | eta: 3156.4m
step 58650/200000 (29.32%) | loss: 2.750138 | lrm: 0.80 | dt: 1460.29ms | tok/sec: 11,219 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 77 | total time: 1309.10m | eta: 3155.5m
step 58700/200000 (29.35%) | loss: 2.700908 | lrm: 0.80 | dt: 1513.70ms | tok/sec: 10,823 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 79 | total time: 1310.34m | eta: 3154.7m
step 58750/200000 (29.38%) | loss: 2.719610 | lrm: 0.80 | dt: 1495.32ms | tok/sec: 10,956 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 81 | total time: 1311.59m | eta: 3153.9m
step 58800/200000 (29.40%) | loss: 2.768550 | lrm: 0.80 | dt: 1499.18ms | tok/sec: 10,928 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 1 | total time: 1312.83m | eta: 3153.1m
step 58850/200000 (29.43%) | loss: 2.783612 | lrm: 0.79 | dt: 1580.34ms | tok/sec: 10,367 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 3 | total time: 1314.12m | eta: 3152.4m
step 58900/200000 (29.45%) | loss: 2.743367 | lrm: 0.79 | dt: 1568.14ms | tok/sec: 10,448 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 5 | total time: 1315.43m | eta: 3151.8m
step 58950/200000 (29.48%) | loss: 2.776437 | lrm: 0.79 | dt: 1331.89ms | tok/sec: 12,301 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 7 | total time: 1316.58m | eta: 3150.7m
step 59000/200000 (29.50%) | loss: 2.697411 | lrm: 0.79 | dt: 1327.48ms | tok/sec: 12,342 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 9 | total time: 1317.69m | eta: 3149.6m
step 59050/200000 (29.52%) | loss: 2.752478 | lrm: 0.79 | dt: 1335.28ms | tok/sec: 12,270 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 11 | total time: 1318.80m | eta: 3148.5m
step 59100/200000 (29.55%) | loss: 2.737538 | lrm: 0.79 | dt: 1338.75ms | tok/sec: 12,238 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 13 | total time: 1319.92m | eta: 3147.4m
step 59150/200000 (29.57%) | loss: 2.752752 | lrm: 0.79 | dt: 1349.23ms | tok/sec: 12,143 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 15 | total time: 1321.05m | eta: 3146.3m
step 59200/200000 (29.60%) | loss: 2.734387 | lrm: 0.79 | dt: 1356.62ms | tok/sec: 12,077 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 17 | total time: 1322.18m | eta: 3145.2m
step 59250/200000 (29.62%) | loss: 2.783865 | lrm: 0.79 | dt: 1354.84ms | tok/sec: 12,092 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 19 | total time: 1323.31m | eta: 3144.1m
step 59300/200000 (29.65%) | loss: 2.758178 | lrm: 0.79 | dt: 1336.02ms | tok/sec: 12,263 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 21 | total time: 1324.44m | eta: 3143.0m
step 59350/200000 (29.68%) | loss: 2.727892 | lrm: 0.79 | dt: 2252.06ms | tok/sec: 7,275 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 23 | total time: 1326.41m | eta: 3143.9m
step 59400/200000 (29.70%) | loss: 2.805598 | lrm: 0.79 | dt: 2254.45ms | tok/sec: 7,267 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 26 | total time: 1328.30m | eta: 3144.6m
step 59450/200000 (29.73%) | loss: 2.670060 | lrm: 0.79 | dt: 2199.04ms | tok/sec: 7,450 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 28 | total time: 1330.14m | eta: 3145.2m
step 59500/200000 (29.75%) | loss: 2.774233 | lrm: 0.79 | dt: 2435.77ms | tok/sec: 6,726 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 30 | total time: 1332.06m | eta: 3146.0m
step 59550/200000 (29.77%) | loss: 2.757405 | lrm: 0.79 | dt: 2279.15ms | tok/sec: 7,188 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 26 | total time: 1328.30m | eta: 3144.6m
step 59450/200000 (29.73%) | loss: 2.670060 | lrm: 0.79 | dt: 2199.04ms | tok/sec: 7,450 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 28 | total time: 1330.14m | eta: 3145.2m
step 59500/200000 (29.75%) | loss: 2.774233 | lrm: 0.79 | dt: 2435.77ms | tok/sec: 6,726 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 30 | total time: 1332.06m | eta: 3146.0m
step 59550/200000 (29.77%) | loss: 2.757405 | lrm: 0.79 | dt: 2279.15ms | tok/sec: 7,188 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 32 | total time: 1333.99m | eta: 3146.8m
step 59600/200000 (29.80%) | loss: 2.718143 | lrm: 0.79 | dt: 4935.92ms | tok/sec: 3,319 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 34 | total time: 1337.24m | eta: 3150.7m
step 59650/200000 (29.82%) | loss: 2.713780 | lrm: 0.79 | dt: 3805.15ms | tok/sec: 4,305 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 36 | total time: 1340.84m | eta: 3155.4m
step 59700/200000 (29.85%) | loss: 2.774958 | lrm: 0.79 | dt: 4648.74ms | tok/sec: 3,524 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 38 | total time: 1344.45m | eta: 3160.1m
step 59750/200000 (29.88%) | loss: 2.819700 | lrm: 0.79 | dt: 4735.61ms | tok/sec: 3,459 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 40 | total time: 1348.57m | eta: 3166.0m
step 59800/200000 (29.90%) | loss: 2.817733 | lrm: 0.79 | dt: 3136.23ms | tok/sec: 5,224 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 42 | total time: 1353.49m | eta: 3173.8m
step 59850/200000 (29.93%) | loss: 2.774232 | lrm: 0.79 | dt: 3135.28ms | tok/sec: 5,225 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 44 | total time: 1358.46m | eta: 3181.6m
step 59900/200000 (29.95%) | loss: 2.712112 | lrm: 0.79 | dt: 9591.92ms | tok/sec: 1,708 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 46 | total time: 1364.85m | eta: 3192.8m
step 59950/200000 (29.98%) | loss: 2.774290 | lrm: 0.79 | dt: 9702.91ms | tok/sec: 1,688 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 48 | total time: 1372.92m | eta: 3207.8m
Step 60000 | Validation bpb: 1.011765
<|bos|>The capital of France is the city of Francisco, which is also known as the Francisco Cities. This city
<|bos|>The chemical symbol of gold is 92.5 mol mol-1. The physical properties of gold are 95.2 °C,
<|bos|>If yesterday was Friday, then tomorrow will be Friday. But, since it's Friday, we're getting this one. The entire world will be in one place
<|bos|>The opposite of hot is the opposite of cold, so that your own biblical story begins and ends in the opposite side. The first person
<|bos|>The planets of the solar system are: Mars, Jupiter, Saturn, Uranus, Iran, and Iran. The sun is
<|bos|>My favorite color is red. If you are curious about the opposite of red, let's say you see a black color and you like it
<|bos|>If 5*x + 3 = 13, then x is the number of times that the system starts by calling the system back. 5*x * x * 0 is
2026-03-17 21:53:30,238 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_060000.pt
2026-03-17 21:53:30,249 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_060000.json
2026-03-17 21:53:32,869 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_060000_rank0.pt


#
step 60050/200000 (30.02%) | loss: 2.806895 | lrm: 0.79 | dt: 1335.33ms | tok/sec: 12,269 | bf16_mfu: 0.00 | step 60100/200000 (30.05%) | loss: 2.818034 | lrm: 0.79 | dt: 1339.14ms | tok/sec: 12,234 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 54 | total time: 1387.08m | eta: 3229.4m
step 60150/200000 (30.07%) | loss: 2.762886 | lrm: 0.79 | dt: 1340.96ms | tok/sec: 12,218 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 56 | total time: 1388.19m | eta: 3228.1m
step 60200/200000 (30.10%) | loss: 2.761557 | lrm: 0.79 | dt: 1330.09ms | tok/sec: 12,317 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 58 | total time: 1389.30m | eta: 3226.9m
step 60250/200000 (30.12%) | loss: 2.784050 | lrm: 0.79 | dt: 1333.45ms | tok/sec: 12,286 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 60 | total time: 1390.42m | eta: 3225.6m
step 60300/200000 (30.15%) | loss: 2.759183 | lrm: 0.79 | dt: 1332.44ms | tok/sec: 12,296 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 62 | total time: 1391.53m | eta: 3224.4m
step 60350/200000 (30.18%) | loss: 2.826754 | lrm: 0.79 | dt: 1332.27ms | tok/sec: 12,297 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 64 | total time: 1392.64m | eta: 3223.1m
step 60400/200000 (30.20%) | loss: 2.756037 | lrm: 0.79 | dt: 1331.20ms | tok/sec: 12,307 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 66 | total time: 1393.75m | eta: 3221.9m
step 60450/200000 (30.23%) | loss: 2.757351 | lrm: 0.79 | dt: 1325.38ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 68 | total time: 1394.86m | eta: 3220.6m
step 60500/200000 (30.25%) | loss: 2.771959 | lrm: 0.79 | dt: 1326.03ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 71 | total time: 1395.97m | eta: 3219.3m
step 60550/200000 (30.27%) | loss: 2.777720 | lrm: 0.79 | dt: 1351.91ms | tok/sec: 12,119 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 73 | total time: 1397.08m | eta: 3218.1m
step 60600/200000 (30.30%) | loss: 2.787497 | lrm: 0.79 | dt: 1330.80ms | tok/sec: 12,311 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 75 | total time: 1398.18m | eta: 3216.8m
step 60650/200000 (30.32%) | loss: 2.751423 | lrm: 0.79 | dt: 1331.61ms | tok/sec: 12,303 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 77 | total time: 1399.29m | eta: 3215.6m
step 60700/200000 (30.35%) | loss: 2.842066 | lrm: 0.79 | dt: 1344.94ms | tok/sec: 12,181 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 79 | total time: 1400.40m | eta: 3214.3m
step 60750/200000 (30.38%) | loss: 2.768029 | lrm: 0.78 | dt: 1333.13ms | tok/sec: 12,289 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 81 | total time: 1401.51m | eta: 3213.1m
step 60800/200000 (30.40%) | loss: 2.737683 | lrm: 0.78 | dt: 1326.94ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 0 | total time: 1402.63m | eta: 3211.8m
step 60850/200000 (30.43%) | loss: 2.741790 | lrm: 0.78 | dt: 1328.09ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 2 | total time: 1403.74m | eta: 3210.6m
step 60900/200000 (30.45%) | loss: 2.815187 | lrm: 0.78 | dt: 1328.06ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 4 | total time: 1404.85m | eta: 3209.3m
step 60950/200000 (30.48%) | loss: 2.758430 | lrm: 0.78 | dt: 1327.53ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 6 | total time: 1405.95m | eta: 3208.0m
step 61000/200000 (30.50%) | loss: 2.758664 | lrm: 0.78 | dt: 1332.53ms | tok/sec: 12,295 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 8 | total time: 1407.06m | eta: 3206.8m
step 61050/200000 (30.52%) | loss: 2.682593 | lrm: 0.78 | dt: 1324.74ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 10 | total time: 1408.17m | eta: 3205.5m
step 61100/200000 (30.55%) | loss: 2.813177 | lrm: 0.78 | dt: 1325.09ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 12 | total time: 1409.27m | eta: 3204.3m
step 61150/200000 (30.57%) | loss: 2.700994 | lrm: 0.78 | dt: 1326.45ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 14 | total time: 1410.38m | eta: 3203.0m
step 61200/200000 (30.60%) | loss: 2.727810 | lrm: 0.78 | dt: 1324.23ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 16 | total time: 1411.48m | eta: 3201.7m
step 61250/200000 (30.62%) | loss: 2.756542 | lrm: 0.78 | dt: 1327.82ms | tok/sec: 12,339 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 18 | total time: 1412.59m | eta: 3200.5m
step 61300/200000 (30.65%) | loss: 2.779609 | lrm: 0.78 | dt: 1329.53ms | tok/sec: 12,323 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 20 | total time: 1413.69m | eta: 3199.2m
step 61350/200000 (30.68%) | loss: 2.738747 | lrm: 0.78 | dt: 1328.66ms | tok/sec: 12,331 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 22 | total time: 1414.80m | eta: 3197.9m
step 61400/200000 (30.70%) | loss: 2.688631 | lrm: 0.78 | dt: 1330.03ms | tok/sec: 12,318 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 24 | total time: 1415.91m | eta: 3196.7m
step 61450/200000 (30.73%) | loss: 2.741348 | lrm: 0.78 | dt: 1327.63ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 26 | total time: 1417.01m | eta: 3195.4m
step 61500/200000 (30.75%) | loss: 2.675143 | lrm: 0.78 | dt: 1326.12ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 28 | total time: 1418.12m | eta: 3194.2m
step 61550/200000 (30.77%) | loss: 2.740425 | lrm: 0.78 | dt: 1328.32ms | tok/sec: 12,334 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 30 | total time: 1419.23m | eta: 3192.9m
step 61600/200000 (30.80%) | loss: 2.818231 | lrm: 0.78 | dt: 1327.69ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 33 | total time: 1420.33m | eta: 3191.7m
step 61650/200000 (30.82%) | loss: 2.677043 | lrm: 0.78 | dt: 1325.76ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 35 | total time: 1421.44m | eta: 3190.4m
step 61700/200000 (30.85%) | loss: 2.743028 | lrm: 0.78 | dt: 1326.57ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 37 | total time: 1422.54m | eta: 3189.1m
step 61750/200000 (30.88%) | loss: 2.714895 | lrm: 0.78 | dt: 1327.35ms | tok/sec: 12,343 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 39 | total time: 1423.65m | eta: 3187.9m
step 61800/200000 (30.90%) | loss: 2.755617 | lrm: 0.78 | dt: 1325.20ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 41 | total time: 1424.75m | eta: 3186.6m
step 61850/200000 (30.93%) | loss: 2.675932 | lrm: 0.78 | dt: 1323.81ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 43 | total time: 1425.86m | eta: 3185.4m
step 61900/200000 (30.95%) | loss: 2.796973 | lrm: 0.78 | dt: 1321.23ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 45 | total time: 1426.96m | eta: 3184.1m
step 61950/200000 (30.98%) | loss: 2.767464 | lrm: 0.78 | dt: 1324.18ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 47 | total time: 1428.06m | eta: 3182.8m
Step 62000 | Validation bpb: 1.010119
step 62000/200000 (31.00%) | loss: 2.828991 | lrm: 0.78 | dt: 1404.76ms | tok/sec: 11,663 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 49 | total time: 1429.17m | eta: 3181.6m
step 62050/200000 (31.02%) | loss: 2.747911 | lrm: 0.78 | dt: 1324.03ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 51 | total time: 1430.28m | eta: 3180.3m
step 62100/200000 (31.05%) | loss: 2.712402 | lrm: 0.78 | dt: 1324.04ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 53 | total time: 1431.38m | eta: 3179.1m
step 62150/200000 (31.07%) | loss: 2.829292 | lrm: 0.78 | dt: 1324.32ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 55 | total time: 1432.48m | eta: 3177.8m
step 62200/200000 (31.10%) | loss: 2.709320 | lrm: 0.78 | dt: 1324.48ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 57 | total time: 1433.59m | eta: 3176.5m
step 62250/200000 (31.12%) | loss: 2.763556 | lrm: 0.78 | dt: 1325.64ms | tok/sec: 12,359 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 59 | total time: 1434.69m | eta: 3175.3m
step 62300/200000 (31.15%) | loss: 2.723763 | lrm: 0.78 | dt: 1322.55ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 61 | total time: 1435.80m | eta: 3174.0m
step 62350/200000 (31.18%) | loss: 2.736305 | lrm: 0.78 | dt: 1329.42ms | tok/sec: 12,324 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 63 | total time: 1436.90m | eta: 3172.8m
step 62400/200000 (31.20%) | loss: 2.777455 | lrm: 0.78 | dt: 1323.37ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 65 | total time: 1438.00m | eta: 3171.5m
step 62450/200000 (31.23%) | loss: 2.712396 | lrm: 0.78 | dt: 1322.61ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 67 | total time: 1439.11m | eta: 3170.2m
step 62500/200000 (31.25%) | loss: 2.777735 | lrm: 0.78 | dt: 1325.46ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 69 | total time: 1440.21m | eta: 3169.0m
step 62550/200000 (31.27%) | loss: 2.779580 | lrm: 0.78 | dt: 1327.56ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 71 | total time: 1441.32m | eta: 3167.7m
step 62600/200000 (31.30%) | loss: 2.747903 | lrm: 0.78 | dt: 1321.82ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 73 | total time: 1442.42m | eta: 3166.5m
step 62650/200000 (31.32%) | loss: 2.760577 | lrm: 0.77 | dt: 1328.73ms | tok/sec: 12,330 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 75 | total time: 1443.52m | eta: 3165.2m
step 62700/200000 (31.35%) | loss: 2.790592 | lrm: 0.77 | dt: 1323.99ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 77 | total time: 1444.63m | eta: 3163.9m
step 62750/200000 (31.38%) | loss: 2.830395 | lrm: 0.77 | dt: 1324.20ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 80 | total time: 1445.73m | eta: 3162.7m
step 62800/200000 (31.40%) | loss: 2.785469 | lrm: 0.77 | dt: 1329.41ms | tok/sec: 12,324 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 82 | total time: 1446.84m | eta: 3161.4m
step 62850/200000 (31.43%) | loss: 2.729436 | lrm: 0.77 | dt: 1327.91ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 1 | total time: 1447.94m | eta: 3160.2m
step 62900/200000 (31.45%) | loss: 2.738862 | lrm: 0.77 | dt: 1328.56ms | tok/sec: 12,332 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 3 | total time: 1449.05m | eta: 3158.9m
step 62950/200000 (31.48%) | loss: 2.732461 | lrm: 0.77 | dt: 1324.69ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 5 | total time: 1450.15m | eta: 3157.7m
step 63000/200000 (31.50%) | loss: 2.683821 | lrm: 0.77 | dt: 1324.88ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 7 | total time: 1451.26m | eta: 3156.4m
step 63050/200000 (31.52%) | loss: 2.758278 | lrm: 0.77 | dt: 1327.66ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 9 | total time: 1452.36m | eta: 3155.2m
step 63100/200000 (31.55%) | loss: 2.745796 | lrm: 0.77 | dt: 1327.78ms | tok/sec: 12,339 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 11 | total time: 1453.47m | eta: 3153.9m
step 63150/200000 (31.57%) | loss: 2.725155 | lrm: 0.77 | dt: 1326.71ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 13 | total time: 1454.57m | eta: 3152.6m
step 63200/200000 (31.60%) | loss: 2.765471 | lrm: 0.77 | dt: 1322.15ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 15 | total time: 1455.67m | eta: 3151.4m
step 63250/200000 (31.62%) | loss: 2.716576 | lrm: 0.77 | dt: 1324.02ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 17 | total time: 1456.78m | eta: 3150.1m
step 63300/200000 (31.65%) | loss: 2.791854 | lrm: 0.77 | dt: 1326.16ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 19 | total time: 1457.88m | eta: 3148.9m
step 63350/200000 (31.68%) | loss: 2.809351 | lrm: 0.77 | dt: 1324.37ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 21 | total time: 1458.99m | eta: 3147.6m
step 63400/200000 (31.70%) | loss: 2.837939 | lrm: 0.77 | dt: 1321.44ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 23 | total time: 1460.09m | eta: 3146.4m
step 63450/200000 (31.73%) | loss: 2.716070 | lrm: 0.77 | dt: 1321.72ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 25 | total time: 1461.19m | eta: 3145.1m
step 63500/200000 (31.75%) | loss: 2.766501 | lrm: 0.77 | dt: 1323.83ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 27 | total time: 1462.30m | eta: 3143.9m
step 63550/200000 (31.77%) | loss: 2.758331 | lrm: 0.77 | dt: 1323.88ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 29 | total time: 1463.41m | eta: 3142.6m
step 63600/200000 (31.80%) | loss: 2.727866 | lrm: 0.77 | dt: 1324.35ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 31 | total time: 1464.51m | eta: 3141.4m
step 63650/200000 (31.82%) | loss: 2.725713 | lrm: 0.77 | dt: 1325.30ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 33 | total time: 1465.61m | eta: 3140.1m
step 63700/200000 (31.85%) | loss: 2.774299 | lrm: 0.77 | dt: 1322.69ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 35 | total time: 1466.72m | eta: 3138.9m
step 63750/200000 (31.88%) | loss: 2.703890 | lrm: 0.77 | dt: 1327.28ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 37 | total time: 1467.82m | eta: 3137.6m
step 63800/200000 (31.90%) | loss: 2.636142 | lrm: 0.77 | dt: 1323.67ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 40 | total time: 1468.92m | eta: 3136.3m
step 63850/200000 (31.93%) | loss: 2.755949 | lrm: 0.77 | dt: 1325.49ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 42 | total time: 1470.03m | eta: 3135.1m
step 63900/200000 (31.95%) | loss: 2.691700 | lrm: 0.77 | dt: 1325.27ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 44 | total time: 1471.13m | eta: 3133.8m
step 63950/200000 (31.98%) | loss: 2.715046 | lrm: 0.77 | dt: 1322.11ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 46 | total time: 1472.23m | eta: 3132.6m
Step 64000 | Validation bpb: 1.008460
<|bos|>The capital of France is the capital of the French State (Gouda), which is owned by the country's officially formed capital and the
<|bos|>The chemical symbol of gold is gold, gold is the element in the periodic table. It is a white, crystalline metal with a density of
<|bos|>If yesterday was Friday, then tomorrow will be Friday, and the last one will be Friday, then all of your friends have been talking for days. This is
<|bos|>The opposite of hot is cold. As the name suggests, the ambient temperature of a given gasoline car is a hot gas. A gas
<|bos|>The planets of the solar system are: Mars, Jupiter, Saturn, and Venus (the largest planet in the solar system); Earth
<|bos|>My favorite color is a white bulls eyebrow. I am not sure on the colors, but I was thinking red, black,
<|bos|>If 5*x + 3 = 13, then x is the number of times 5.9 million is needed to produce 5 million times the number of 5.9 million
2026-03-17 23:26:34,688 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_064000.pt
2026-03-17 23:26:34,693 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_064000.json
2026-03-17 23:26:36,967 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_064000_rank0.pt
step 64000/200000 (32.00%) | loss: 2.757550 | lrm: 0.77 | dt: 1637.13ms | tok/sec: 10,007 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 48 | total time: 1473.34m | eta: 3131.3m
step 64050/200000 (32.02%) | loss: 2.795563 | lrm: 0.77 | dt: 1319.24ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 50 | total time: 1474.45m | eta: 3130.1m
step 64100/200000 (32.05%) | loss: 2.820500 | lrm: 0.77 | dt: 1323.41ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 52 | total time: 1475.55m | eta: 3128.8m
step 64150/200000 (32.08%) | loss: 2.722103 | lrm: 0.77 | dt: 1321.99ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 54 | total time: 1476.65m | eta: 3127.6m
step 64200/200000 (32.10%) | loss: 2.767024 | lrm: 0.77 | dt: 1321.76ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 56 | total time: 1477.75m | eta: 3126.3m
step 64250/200000 (32.12%) | loss: 2.699282 | lrm: 0.77 | dt: 1321.45ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 58 | total time: 1478.86m | eta: 3125.1m
step 64300/200000 (32.15%) | loss: 2.747555 | lrm: 0.77 | dt: 1321.85ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 60 | total time: 1479.96m | eta: 3123.8m
step 64350/200000 (32.17%) | loss: 2.727761 | lrm: 0.77 | dt: 1323.93ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 62 | total time: 1481.06m | eta: 3122.6m
step 64400/200000 (32.20%) | loss: 2.800653 | lrm: 0.77 | dt: 1322.03ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 64 | total time: 1482.17m | eta: 3121.3m
step 64450/200000 (32.23%) | loss: 2.772965 | lrm: 0.77 | dt: 1324.15ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 66 | total time: 1483.27m | eta: 3120.1m
step 64500/200000 (32.25%) | loss: 2.745821 | lrm: 0.77 | dt: 1321.15ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 68 | total time: 1484.38m | eta: 3118.8m
step 64550/200000 (32.27%) | loss: 2.765478 | lrm: 0.76 | dt: 1321.29ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 70 | total time: 1485.48m | eta: 3117.6m
step 64600/200000 (32.30%) | loss: 2.723188 | lrm: 0.76 | dt: 1326.08ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 72 | total time: 1486.58m | eta: 3116.3m
step 64650/200000 (32.33%) | loss: 2.851347 | lrm: 0.76 | dt: 1324.72ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 74 | total time: 1487.68m | eta: 3115.1m
step 64700/200000 (32.35%) | loss: 2.779354 | lrm: 0.76 | dt: 1323.96ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 76 | total time: 1488.79m | eta: 3113.8m
step 64750/200000 (32.38%) | loss: 2.699112 | lrm: 0.76 | dt: 1328.64ms | tok/sec: 12,331 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 78 | total time: 1489.89m | eta: 3112.6m
step 64800/200000 (32.40%) | loss: 2.792243 | lrm: 0.76 | dt: 1324.62ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 80 | total time: 1491.00m | eta: 3111.3m
step 64850/200000 (32.42%) | loss: 2.730111 | lrm: 0.76 | dt: 1322.30ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 82 | total time: 1492.10m | eta: 3110.1m
step 64900/200000 (32.45%) | loss: 2.771704 | lrm: 0.76 | dt: 1324.49ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 1 | total time: 1493.20m | eta: 3108.8m
step 64950/200000 (32.48%) | loss: 2.799043 | lrm: 0.76 | dt: 1324.38ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 3 | total time: 1494.31m | eta: 3107.6m
step 65000/200000 (32.50%) | loss: 2.792427 | lrm: 0.76 | dt: 1653.30ms | tok/sec: 9,909 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 5 | total time: 1495.41m | eta: 3106.3m
step 65050/200000 (32.52%) | loss: 2.804051 | lrm: 0.76 | dt: 1324.48ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 7 | total time: 1496.52m | eta: 3105.1m
step 65100/200000 (32.55%) | loss: 2.762830 | lrm: 0.76 | dt: 1323.05ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 10 | total time: 1497.62m | eta: 3103.8m
step 65150/200000 (32.58%) | loss: 2.762206 | lrm: 0.76 | dt: 1324.53ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 12 | total time: 1498.73m | eta: 3102.6m
step 65200/200000 (32.60%) | loss: 2.728898 | lrm: 0.76 | dt: 1321.32ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 14 | total time: 1499.83m | eta: 3101.3m
step 65250/200000 (32.62%) | loss: 2.762596 | lrm: 0.76 | dt: 1323.09ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 16 | total time: 1500.93m | eta: 3100.1m
step 65300/200000 (32.65%) | loss: 2.768546 | lrm: 0.76 | dt: 1320.81ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 18 | total time: 1502.03m | eta: 3098.8m
step 65350/200000 (32.67%) | loss: 2.827581 | lrm: 0.76 | dt: 1329.26ms | tok/sec: 12,325 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 20 | total time: 1503.14m | eta: 3097.6m
step 65400/200000 (32.70%) | loss: 2.780963 | lrm: 0.76 | dt: 1427.04ms | tok/sec: 11,481 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 22 | total time: 1504.24m | eta: 3096.4m
step 65450/200000 (32.73%) | loss: 2.700707 | lrm: 0.76 | dt: 1321.64ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 24 | total time: 1505.35m | eta: 3095.1m
step 65500/200000 (32.75%) | loss: 2.774403 | lrm: 0.76 | dt: 1328.91ms | tok/sec: 12,328 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 26 | total time: 1506.45m | eta: 3093.9m
step 65550/200000 (32.77%) | loss: 2.772018 | lrm: 0.76 | dt: 1323.94ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 28 | total time: 1507.55m | eta: 3092.6m
step 65600/200000 (32.80%) | loss: 2.739542 | lrm: 0.76 | dt: 1329.82ms | tok/sec: 12,320 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 30 | total time: 1508.66m | eta: 3091.4m
step 65650/200000 (32.83%) | loss: 2.727124 | lrm: 0.76 | dt: 1321.42ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 32 | total time: 1509.76m | eta: 3090.1m
step 65700/200000 (32.85%) | loss: 2.738037 | lrm: 0.76 | dt: 1319.10ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 34 | total time: 1510.86m | eta: 3088.9m
step 65750/200000 (32.88%) | loss: 2.678154 | lrm: 0.76 | dt: 1323.29ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 36 | total time: 1511.96m | eta: 3087.6m
step 65800/200000 (32.90%) | loss: 2.780638 | lrm: 0.76 | dt: 1324.35ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 38 | total time: 1513.07m | eta: 3086.4m
step 65850/200000 (32.92%) | loss: 2.737413 | lrm: 0.76 | dt: 1323.86ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 40 | total time: 1514.17m | eta: 3085.1m
step 65900/200000 (32.95%) | loss: 2.719158 | lrm: 0.76 | dt: 1322.23ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 42 | total time: 1515.27m | eta: 3083.9m
step 65950/200000 (32.98%) | loss: 2.815941 | lrm: 0.76 | dt: 1321.08ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 44 | total time: 1516.38m | eta: 3082.7m
Step 66000 | Validation bpb: 1.006148
step 66000/200000 (33.00%) | loss: 2.701772 | lrm: 0.76 | dt: 1397.73ms | tok/sec: 11,721 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 46 | total time: 1517.48m | eta: 3081.4m
step 66050/200000 (33.02%) | loss: 2.799261 | lrm: 0.76 | dt: 1320.78ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 48 | total time: 1518.59m | eta: 3080.2m
step 66100/200000 (33.05%) | loss: 2.722323 | lrm: 0.76 | dt: 1328.01ms | tok/sec: 12,337 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 50 | total time: 1519.69m | eta: 3078.9m
step 66150/200000 (33.08%) | loss: 2.749499 | lrm: 0.76 | dt: 1326.61ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 52 | total time: 1520.79m | eta: 3077.7m
step 66200/200000 (33.10%) | loss: 2.754555 | lrm: 0.76 | dt: 1319.93ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 54 | total time: 1521.89m | eta: 3076.4m
step 66250/200000 (33.12%) | loss: 2.728960 | lrm: 0.76 | dt: 1324.93ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 57 | total time: 1523.00m | eta: 3075.2m
step 66300/200000 (33.15%) | loss: 2.733566 | lrm: 0.76 | dt: 1321.38ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 59 | total time: 1524.10m | eta: 3073.9m
step 66350/200000 (33.17%) | loss: 2.722701 | lrm: 0.76 | dt: 1328.63ms | tok/sec: 12,331 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 61 | total time: 1525.20m | eta: 3072.7m
step 66400/200000 (33.20%) | loss: 2.712960 | lrm: 0.76 | dt: 1328.73ms | tok/sec: 12,330 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 63 | total time: 1526.31m | eta: 3071.5m
step 66450/200000 (33.23%) | loss: 2.738641 | lrm: 0.75 | dt: 1322.97ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 65 | total time: 1527.42m | eta: 3070.2m
step 66500/200000 (33.25%) | loss: 2.769158 | lrm: 0.75 | dt: 1326.36ms | tok/sec: 12,352 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 67 | total time: 1528.52m | eta: 3069.0m
step 66550/200000 (33.27%) | loss: 2.776882 | lrm: 0.75 | dt: 1330.87ms | tok/sec: 12,310 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 69 | total time: 1529.63m | eta: 3067.8m
step 66600/200000 (33.30%) | loss: 2.790530 | lrm: 0.75 | dt: 1321.59ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 71 | total time: 1530.73m | eta: 3066.5m
step 66650/200000 (33.33%) | loss: 2.785755 | lrm: 0.75 | dt: 1325.06ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 73 | total time: 1531.84m | eta: 3065.3m
step 66700/200000 (33.35%) | loss: 2.758202 | lrm: 0.75 | dt: 1323.63ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 75 | total time: 1532.94m | eta: 3064.0m
step 66750/200000 (33.38%) | loss: 2.785612 | lrm: 0.75 | dt: 1323.94ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 77 | total time: 1534.05m | eta: 3062.8m
step 66800/200000 (33.40%) | loss: 2.636132 | lrm: 0.75 | dt: 1320.16ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 79 | total time: 1535.15m | eta: 3061.6m
step 66850/200000 (33.42%) | loss: 2.744933 | lrm: 0.75 | dt: 1325.53ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 81 | total time: 1536.25m | eta: 3060.3m
step 66900/200000 (33.45%) | loss: 2.772123 | lrm: 0.75 | dt: 1326.54ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 0 | total time: 1537.36m | eta: 3059.1m
step 66950/200000 (33.48%) | loss: 2.738844 | lrm: 0.75 | dt: 1326.74ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 2 | total time: 1538.46m | eta: 3057.8m
step 67000/200000 (33.50%) | loss: 2.783233 | lrm: 0.75 | dt: 1340.31ms | tok/sec: 12,224 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 4 | total time: 1539.57m | eta: 3056.6m


###
step 67050/200000 (33.52%) | loss: 2.690167 | lrm: 0.75 | dt: 1336.07ms | tok/sec: 12,262 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 6 | total time: 1540.68m | eta: 3055.4m
step 67100/200000 (33.55%) | loss: 2.734617 | lrm: 0.75 | dt: 1326.92ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 8 | total time: 1541.79m | eta: 3054.2m
step 67150/200000 (33.58%) | loss: 2.725270 | lrm: 0.75 | dt: 1325.45ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 10 | total time: 1542.89m | eta: 3052.9m
step 67200/200000 (33.60%) | loss: 2.756559 | lrm: 0.75 | dt: 1325.67ms | tok/sec: 12,359 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 12 | total time: 1544.00m | eta: 3051.7m
step 67250/200000 (33.62%) | loss: 2.724215 | lrm: 0.75 | dt: 1327.48ms | tok/sec: 12,342 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 14 | total time: 1545.10m | eta: 3050.5m
step 67300/200000 (33.65%) | loss: 2.797355 | lrm: 0.75 | dt: 1327.30ms | tok/sec: 12,343 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 16 | total time: 1546.21m | eta: 3049.2m
step 67350/200000 (33.67%) | loss: 2.650014 | lrm: 0.75 | dt: 1330.81ms | tok/sec: 12,311 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 18 | total time: 1547.32m | eta: 3048.0m
step 67400/200000 (33.70%) | loss: 2.740139 | lrm: 0.75 | dt: 1326.10ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 20 | total time: 1548.42m | eta: 3046.8m
step 67450/200000 (33.73%) | loss: 2.704603 | lrm: 0.75 | dt: 1327.37ms | tok/sec: 12,343 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 22 | total time: 1549.53m | eta: 3045.5m
step 67500/200000 (33.75%) | loss: 2.804528 | lrm: 0.75 | dt: 1327.70ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 25 | total time: 1550.63m | eta: 3044.3m
step 67550/200000 (33.77%) | loss: 2.750343 | lrm: 0.75 | dt: 1327.16ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 27 | total time: 1551.74m | eta: 3043.1m
step 67600/200000 (33.80%) | loss: 2.704744 | lrm: 0.75 | dt: 1326.27ms | tok/sec: 12,353 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 29 | total time: 1552.84m | eta: 3041.8m
step 67650/200000 (33.83%) | loss: 2.768089 | lrm: 0.75 | dt: 1325.28ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 31 | total time: 1553.95m | eta: 3040.6m
step 67700/200000 (33.85%) | loss: 2.772752 | lrm: 0.75 | dt: 1331.05ms | tok/sec: 12,309 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 33 | total time: 1555.06m | eta: 3039.4m
step 67750/200000 (33.88%) | loss: 2.712490 | lrm: 0.75 | dt: 1329.22ms | tok/sec: 12,326 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 35 | total time: 1556.16m | eta: 3038.1m
step 67800/200000 (33.90%) | loss: 2.699965 | lrm: 0.75 | dt: 1321.77ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 37 | total time: 1557.27m | eta: 3036.9m
step 67850/200000 (33.92%) | loss: 2.718756 | lrm: 0.75 | dt: 1326.73ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 39 | total time: 1558.37m | eta: 3035.6m
step 67900/200000 (33.95%) | loss: 2.728090 | lrm: 0.75 | dt: 1326.01ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 41 | total time: 1559.47m | eta: 3034.4m
step 67950/200000 (33.98%) | loss: 2.751552 | lrm: 0.75 | dt: 1326.45ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 43 | total time: 1560.57m | eta: 3033.2m
Step 68000 | Validation bpb: 1.004653
<|bos|>`The capital of France is Paris`, and it is home to the town of Douglas.
The city is divided by geographical boundaries
<|bos|>The chemical symbol of gold is gold, Kufahr Ťtzek in an alphabetical order.
This is one of
<|bos|>If yesterday was Friday, then tomorrow will be Friday. The first one would be Friday. The first one would be Friday. Friday would be Satur
<|bos|>`The opposite of hot is cold`, in which case the cold is only associated with the body's warm-blooded, heat-stimulated tissues
<|bos|>`The planets of the solar system are: Jupiter, Neptune, Neptunes, Neptunes Mercury, Venus,`
<|bos|>My favorite color is blue. If you are lucky, you can add a touch of green with a hint of purple to your wall.
<|bos|>If 5*x + 3 = 13, then x is 13/13, so that 5*x + 3 = 13/13 (3*x). If
2026-03-18 00:55:29,839 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_068000.pt
2026-03-18 00:55:29,840 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_068000.json
2026-03-18 00:55:31,283 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_068000_rank0.pt
step 68000/200000 (34.00%) | loss: 2.754652 | lrm: 0.75 | dt: 1565.45ms | tok/sec: 10,465 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 45 | total time: 1561.68m | eta: 3031.9m
step 68050/200000 (34.02%) | loss: 2.761060 | lrm: 0.75 | dt: 1322.16ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 47 | total time: 1562.78m | eta: 3030.7m
step 68100/200000 (34.05%) | loss: 2.742387 | lrm: 0.75 | dt: 1320.20ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 49 | total time: 1563.89m | eta: 3029.5m
step 68150/200000 (34.08%) | loss: 2.723660 | lrm: 0.75 | dt: 1321.30ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 51 | total time: 1564.99m | eta: 3028.2m
step 68200/200000 (34.10%) | loss: 2.760495 | lrm: 0.75 | dt: 1321.98ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 53 | total time: 1566.09m | eta: 3027.0m
step 68250/200000 (34.12%) | loss: 2.711361 | lrm: 0.75 | dt: 1318.19ms | tok/sec: 12,429 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 55 | total time: 1567.19m | eta: 3025.8m
step 68300/200000 (34.15%) | loss: 2.808272 | lrm: 0.75 | dt: 1320.92ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 57 | total time: 1568.30m | eta: 3024.5m
step 68350/200000 (34.17%) | loss: 2.809680 | lrm: 0.74 | dt: 1323.82ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 59 | total time: 1569.40m | eta: 3023.3m
step 68400/200000 (34.20%) | loss: 2.736301 | lrm: 0.74 | dt: 1323.50ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 61 | total time: 1570.50m | eta: 3022.0m
step 68450/200000 (34.23%) | loss: 2.776075 | lrm: 0.74 | dt: 1322.28ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 63 | total time: 1571.60m | eta: 3020.8m
step 68500/200000 (34.25%) | loss: 2.747896 | lrm: 0.74 | dt: 1321.53ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 65 | total time: 1572.71m | eta: 3019.6m
step 68550/200000 (34.27%) | loss: 2.720187 | lrm: 0.74 | dt: 1323.61ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 67 | total time: 1573.81m | eta: 3018.3m
step 68600/200000 (34.30%) | loss: 2.767085 | lrm: 0.74 | dt: 1314.85ms | tok/sec: 12,460 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 69 | total time: 1574.91m | eta: 3017.1m
step 68650/200000 (34.33%) | loss: 2.720361 | lrm: 0.74 | dt: 1321.83ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 72 | total time: 1576.01m | eta: 3015.9m
step 68700/200000 (34.35%) | loss: 2.711380 | lrm: 0.74 | dt: 1326.09ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 74 | total time: 1577.12m | eta: 3014.6m
step 68750/200000 (34.38%) | loss: 2.662262 | lrm: 0.74 | dt: 1321.71ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 76 | total time: 1578.22m | eta: 3013.4m
step 68800/200000 (34.40%) | loss: 2.753262 | lrm: 0.74 | dt: 1324.99ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 78 | total time: 1579.32m | eta: 3012.2m
step 68850/200000 (34.42%) | loss: 2.803230 | lrm: 0.74 | dt: 1323.61ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 80 | total time: 1580.42m | eta: 3010.9m
step 68900/200000 (34.45%) | loss: 2.735052 | lrm: 0.74 | dt: 1322.02ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 82 | total time: 1581.53m | eta: 3009.7m
step 68950/200000 (34.48%) | loss: 2.777775 | lrm: 0.74 | dt: 1323.10ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 0 | total time: 1582.63m | eta: 3008.5m
step 69000/200000 (34.50%) | loss: 2.761830 | lrm: 0.74 | dt: 1324.52ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 2 | total time: 1583.73m | eta: 3007.2m
step 69050/200000 (34.52%) | loss: 2.694956 | lrm: 0.74 | dt: 1321.02ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 4 | total time: 1584.84m | eta: 3006.0m
step 69100/200000 (34.55%) | loss: 2.747459 | lrm: 0.74 | dt: 1325.91ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 6 | total time: 1585.94m | eta: 3004.8m
step 69150/200000 (34.58%) | loss: 2.751928 | lrm: 0.74 | dt: 1327.80ms | tok/sec: 12,339 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 8 | total time: 1587.04m | eta: 3003.5m
step 69200/200000 (34.60%) | loss: 2.777978 | lrm: 0.74 | dt: 1324.65ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 10 | total time: 1588.15m | eta: 3002.3m
step 69250/200000 (34.62%) | loss: 2.705843 | lrm: 0.74 | dt: 1321.58ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 12 | total time: 1589.25m | eta: 3001.1m
step 69300/200000 (34.65%) | loss: 2.757313 | lrm: 0.74 | dt: 1318.36ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 14 | total time: 1590.35m | eta: 2999.8m
step 69350/200000 (34.67%) | loss: 2.782744 | lrm: 0.74 | dt: 1323.91ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 16 | total time: 1591.45m | eta: 2998.6m
step 69400/200000 (34.70%) | loss: 2.732091 | lrm: 0.74 | dt: 1324.61ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 18 | total time: 1592.56m | eta: 2997.4m
step 69450/200000 (34.73%) | loss: 2.735993 | lrm: 0.74 | dt: 1322.25ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 20 | total time: 1593.66m | eta: 2996.1m
step 69500/200000 (34.75%) | loss: 2.749949 | lrm: 0.74 | dt: 1324.12ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 22 | total time: 1594.76m | eta: 2994.9m
step 69550/200000 (34.77%) | loss: 2.828736 | lrm: 0.74 | dt: 1324.77ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 24 | total time: 1595.86m | eta: 2993.7m
step 69600/200000 (34.80%) | loss: 2.727376 | lrm: 0.74 | dt: 1320.92ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 26 | total time: 1596.96m | eta: 2992.4m
step 69650/200000 (34.83%) | loss: 2.752296 | lrm: 0.74 | dt: 1322.87ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 28 | total time: 1598.07m | eta: 2991.2m
step 69700/200000 (34.85%) | loss: 2.746135 | lrm: 0.74 | dt: 1327.78ms | tok/sec: 12,339 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 30 | total time: 1599.17m | eta: 2990.0m
step 69750/200000 (34.88%) | loss: 2.744327 | lrm: 0.74 | dt: 1323.42ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 32 | total time: 1600.27m | eta: 2988.7m
step 69800/200000 (34.90%) | loss: 2.800330 | lrm: 0.74 | dt: 1325.97ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 34 | total time: 1601.37m | eta: 2987.5m
step 69850/200000 (34.92%) | loss: 2.756199 | lrm: 0.74 | dt: 1324.49ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 36 | total time: 1602.48m | eta: 2986.3m
step 69900/200000 (34.95%) | loss: 2.730292 | lrm: 0.74 | dt: 1322.50ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 38 | total time: 1603.58m | eta: 2985.1m
step 69950/200000 (34.98%) | loss: 2.696723 | lrm: 0.74 | dt: 1320.03ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 41 | total time: 1604.68m | eta: 2983.8m
Step 70000 | Validation bpb: 1.002898
step 70000/200000 (35.00%) | loss: 2.707726 | lrm: 0.74 | dt: 1396.24ms | tok/sec: 11,734 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 43 | total time: 1605.78m | eta: 2982.6m
step 70050/200000 (35.02%) | loss: 2.749301 | lrm: 0.74 | dt: 1320.93ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 45 | total time: 1606.89m | eta: 2981.4m
step 70100/200000 (35.05%) | loss: 2.762221 | lrm: 0.74 | dt: 1324.42ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 47 | total time: 1607.99m | eta: 2980.1m
step 70150/200000 (35.08%) | loss: 2.743624 | lrm: 0.74 | dt: 1321.83ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 49 | total time: 1609.09m | eta: 2978.9m
step 70200/200000 (35.10%) | loss: 2.725137 | lrm: 0.74 | dt: 1325.83ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 51 | total time: 1610.20m | eta: 2977.7m
step 70250/200000 (35.12%) | loss: 2.754499 | lrm: 0.73 | dt: 1324.58ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 53 | total time: 1611.30m | eta: 2976.5m
step 70300/200000 (35.15%) | loss: 2.722427 | lrm: 0.73 | dt: 1322.16ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 55 | total time: 1612.40m | eta: 2975.2m
step 70350/200000 (35.17%) | loss: 2.685899 | lrm: 0.73 | dt: 1325.72ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 57 | total time: 1613.51m | eta: 2974.0m
step 70400/200000 (35.20%) | loss: 2.706516 | lrm: 0.73 | dt: 1318.79ms | tok/sec: 12,423 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 59 | total time: 1614.61m | eta: 2972.8m
step 70450/200000 (35.23%) | loss: 2.821360 | lrm: 0.73 | dt: 1325.42ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 61 | total time: 1615.71m | eta: 2971.5m
step 70500/200000 (35.25%) | loss: 2.719806 | lrm: 0.73 | dt: 1326.82ms | tok/sec: 12,348 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 63 | total time: 1616.81m | eta: 2970.3m
step 70550/200000 (35.27%) | loss: 2.707408 | lrm: 0.73 | dt: 1323.35ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 65 | total time: 1617.92m | eta: 2969.1m
step 70600/200000 (35.30%) | loss: 2.744350 | lrm: 0.73 | dt: 1319.78ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 67 | total time: 1619.02m | eta: 2967.9m
step 70650/200000 (35.33%) | loss: 2.764010 | lrm: 0.73 | dt: 1321.70ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 69 | total time: 1620.12m | eta: 2966.6m
step 70700/200000 (35.35%) | loss: 2.682363 | lrm: 0.73 | dt: 1322.79ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 71 | total time: 1621.22m | eta: 2965.4m
step 70750/200000 (35.38%) | loss: 2.767477 | lrm: 0.73 | dt: 1323.66ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 73 | total time: 1622.33m | eta: 2964.2m
step 70800/200000 (35.40%) | loss: 2.712667 | lrm: 0.73 | dt: 1320.70ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 75 | total time: 1623.43m | eta: 2962.9m
step 70850/200000 (35.42%) | loss: 2.698139 | lrm: 0.73 | dt: 1319.62ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 77 | total time: 1624.53m | eta: 2961.7m
step 70900/200000 (35.45%) | loss: 2.707459 | lrm: 0.73 | dt: 1358.59ms | tok/sec: 12,059 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 79 | total time: 1625.64m | eta: 2960.5m
step 70950/200000 (35.48%) | loss: 2.757818 | lrm: 0.73 | dt: 1326.19ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 81 | total time: 1626.74m | eta: 2959.3m
step 71000/200000 (35.50%) | loss: 2.708363 | lrm: 0.73 | dt: 1328.11ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 1 | total time: 1627.85m | eta: 2958.1m
step 71050/200000 (35.52%) | loss: 2.708972 | lrm: 0.73 | dt: 1327.05ms | tok/sec: 12,346 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 4 | total time: 1628.95m | eta: 2956.8m
step 71100/200000 (35.55%) | loss: 2.787832 | lrm: 0.73 | dt: 1328.49ms | tok/sec: 12,332 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 6 | total time: 1630.05m | eta: 2955.6m
step 71150/200000 (35.58%) | loss: 2.742154 | lrm: 0.73 | dt: 1323.09ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 8 | total time: 1631.16m | eta: 2954.4m
step 71200/200000 (35.60%) | loss: 2.697187 | lrm: 0.73 | dt: 1327.16ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 10 | total time: 1632.26m | eta: 2953.1m
step 71250/200000 (35.62%) | loss: 2.809340 | lrm: 0.73 | dt: 1320.84ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 12 | total time: 1633.36m | eta: 2951.9m
step 71300/200000 (35.65%) | loss: 2.733555 | lrm: 0.73 | dt: 1321.99ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 14 | total time: 1634.46m | eta: 2950.7m
step 71350/200000 (35.67%) | loss: 2.685628 | lrm: 0.73 | dt: 1326.90ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 16 | total time: 1635.57m | eta: 2949.5m
step 71400/200000 (35.70%) | loss: 2.741246 | lrm: 0.73 | dt: 1321.61ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 18 | total time: 1636.67m | eta: 2948.3m
step 71450/200000 (35.73%) | loss: 2.691238 | lrm: 0.73 | dt: 1325.05ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 20 | total time: 1637.77m | eta: 2947.0m
step 71500/200000 (35.75%) | loss: 2.782813 | lrm: 0.73 | dt: 1323.16ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 22 | total time: 1638.87m | eta: 2945.8m
step 71550/200000 (35.77%) | loss: 2.677558 | lrm: 0.73 | dt: 1325.53ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 24 | total time: 1639.98m | eta: 2944.6m
step 71600/200000 (35.80%) | loss: 2.726761 | lrm: 0.73 | dt: 1329.58ms | tok/sec: 12,322 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 26 | total time: 1641.08m | eta: 2943.4m
step 71650/200000 (35.83%) | loss: 2.707596 | lrm: 0.73 | dt: 1323.45ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 28 | total time: 1642.18m | eta: 2942.1m
step 71700/200000 (35.85%) | loss: 2.732998 | lrm: 0.73 | dt: 1320.43ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 30 | total time: 1643.29m | eta: 2940.9m
step 71750/200000 (35.88%) | loss: 2.702602 | lrm: 0.73 | dt: 1324.96ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 32 | total time: 1644.39m | eta: 2939.7m
step 71800/200000 (35.90%) | loss: 2.749905 | lrm: 0.73 | dt: 1318.68ms | tok/sec: 12,424 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 34 | total time: 1645.49m | eta: 2938.5m
step 71850/200000 (35.92%) | loss: 2.732601 | lrm: 0.73 | dt: 1327.28ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 36 | total time: 1646.60m | eta: 2937.2m
step 71900/200000 (35.95%) | loss: 2.784413 | lrm: 0.73 | dt: 1324.90ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 38 | total time: 1647.70m | eta: 2936.0m
step 71950/200000 (35.98%) | loss: 2.637645 | lrm: 0.73 | dt: 1321.49ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 40 | total time: 1648.80m | eta: 2934.8m
Step 72000 | Validation bpb: 1.001226
<|bos|>The capital of France is the capital of the United States, and the capital of the United Kingdom is the capital of Great Britain. They are
<|bos|>The chemical symbol of gold is Cu-8CuSb-CuSbCuSbCuSb. It is found in
<|bos|>If yesterday was Friday, then tomorrow will be Friday! I will be having a break yesterday and Friday tomorrow for a 2pm E
<|bos|>`The opposite of hot is cold`. You can feel hot. That's why you feel cold. Not all people get hot when they are in their cold
<|bos|>`The planets of the solar system are: Mars, Jupiter, Saturn, Uranus, Neptune, and Neutron Jup`
<|bos|>My favorite color is the blue-variant of the santa color, that is, the blue-variant of the s
<|bos|>If 5*x + 3 = 13, then x is the number of times that the same number of elements can be divided. Then there is a constant. This gives us a way
2026-03-18 02:24:15,821 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_072000.pt
2026-03-18 02:24:15,823 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_072000.json
2026-03-18 02:24:17,424 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_072000_rank0.pt
step 72000/200000 (36.00%) | loss: 2.766786 | lrm: 0.73 | dt: 1565.14ms | tok/sec: 10,468 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 42 | total time: 1649.91m | eta: 2933.6m
step 72050/200000 (36.02%) | loss: 2.730313 | lrm: 0.73 | dt: 1323.49ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 44 | total time: 1651.02m | eta: 2932.4m
step 72100/200000 (36.05%) | loss: 2.706654 | lrm: 0.73 | dt: 1327.56ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 46 | total time: 1652.12m | eta: 2931.1m
step 72150/200000 (36.08%) | loss: 2.713362 | lrm: 0.72 | dt: 1325.27ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 48 | total time: 1653.22m | eta: 2929.9m
step 72200/200000 (36.10%) | loss: 2.775449 | lrm: 0.72 | dt: 1323.91ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 50 | total time: 1654.33m | eta: 2928.7m
step 72250/200000 (36.12%) | loss: 2.656622 | lrm: 0.72 | dt: 1324.76ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 53 | total time: 1655.43m | eta: 2927.5m
step 72300/200000 (36.15%) | loss: 2.705963 | lrm: 0.72 | dt: 1323.83ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 55 | total time: 1656.54m | eta: 2926.3m
step 72350/200000 (36.17%) | loss: 2.693232 | lrm: 0.72 | dt: 1324.86ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 57 | total time: 1657.64m | eta: 2925.0m
step 72400/200000 (36.20%) | loss: 2.752046 | lrm: 0.72 | dt: 1321.62ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 59 | total time: 1658.74m | eta: 2923.8m
step 72450/200000 (36.23%) | loss: 2.778501 | lrm: 0.72 | dt: 1326.68ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 61 | total time: 1659.85m | eta: 2922.6m
step 72500/200000 (36.25%) | loss: 2.750829 | lrm: 0.72 | dt: 1323.42ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 63 | total time: 1660.95m | eta: 2921.4m
step 72550/200000 (36.27%) | loss: 2.708286 | lrm: 0.72 | dt: 1323.07ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 65 | total time: 1662.05m | eta: 2920.2m
step 72600/200000 (36.30%) | loss: 2.737347 | lrm: 0.72 | dt: 1327.84ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 67 | total time: 1663.15m | eta: 2918.9m
step 72650/200000 (36.33%) | loss: 2.669595 | lrm: 0.72 | dt: 1321.84ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 69 | total time: 1664.26m | eta: 2917.7m
step 72700/200000 (36.35%) | loss: 2.831039 | lrm: 0.72 | dt: 1319.50ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 71 | total time: 1665.36m | eta: 2916.5m
step 72750/200000 (36.38%) | loss: 2.807053 | lrm: 0.72 | dt: 1321.36ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 73 | total time: 1666.46m | eta: 2915.3m
step 72800/200000 (36.40%) | loss: 2.697600 | lrm: 0.72 | dt: 1321.64ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 75 | total time: 1667.57m | eta: 2914.1m
step 72850/200000 (36.42%) | loss: 2.715533 | lrm: 0.72 | dt: 1326.42ms | tok/sec: 12,352 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 77 | total time: 1668.67m | eta: 2912.8m
step 72900/200000 (36.45%) | loss: 2.735629 | lrm: 0.72 | dt: 1325.53ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 79 | total time: 1669.77m | eta: 2911.6m
step 72950/200000 (36.48%) | loss: 2.800888 | lrm: 0.72 | dt: 1322.35ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 81 | total time: 1670.88m | eta: 2910.4m
step 73000/200000 (36.50%) | loss: 2.687245 | lrm: 0.72 | dt: 1323.41ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 1 | total time: 1671.99m | eta: 2909.2m
step 73050/200000 (36.52%) | loss: 2.696129 | lrm: 0.72 | dt: 1321.44ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 3 | total time: 1673.09m | eta: 2908.0m
step 73100/200000 (36.55%) | loss: 2.740420 | lrm: 0.72 | dt: 1321.77ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 5 | total time: 1674.19m | eta: 2906.8m
step 73150/200000 (36.58%) | loss: 2.733155 | lrm: 0.72 | dt: 1325.23ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 7 | total time: 1675.30m | eta: 2905.5m
step 73200/200000 (36.60%) | loss: 2.746930 | lrm: 0.72 | dt: 1323.51ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 9 | total time: 1676.40m | eta: 2904.3m
step 73250/200000 (36.62%) | loss: 2.755858 | lrm: 0.72 | dt: 1323.19ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 11 | total time: 1677.51m | eta: 2903.1m
step 73300/200000 (36.65%) | loss: 2.776507 | lrm: 0.72 | dt: 1326.57ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 13 | total time: 1678.61m | eta: 2901.9m
step 73350/200000 (36.67%) | loss: 2.741386 | lrm: 0.72 | dt: 1325.19ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 15 | total time: 1679.71m | eta: 2900.7m
step 73400/200000 (36.70%) | loss: 2.786655 | lrm: 0.72 | dt: 1323.86ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 18 | total time: 1680.82m | eta: 2899.5m
step 73450/200000 (36.73%) | loss: 2.704785 | lrm: 0.72 | dt: 1325.55ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 20 | total time: 1681.92m | eta: 2898.2m
step 73500/200000 (36.75%) | loss: 2.762467 | lrm: 0.72 | dt: 1321.39ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 22 | total time: 1683.03m | eta: 2897.0m
step 73550/200000 (36.77%) | loss: 2.654706 | lrm: 0.72 | dt: 1326.34ms | tok/sec: 12,352 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 24 | total time: 1684.13m | eta: 2895.8m
step 73600/200000 (36.80%) | loss: 2.773250 | lrm: 0.72 | dt: 1320.65ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 26 | total time: 1685.23m | eta: 2894.6m
step 73650/200000 (36.83%) | loss: 2.722753 | lrm: 0.72 | dt: 1323.45ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 28 | total time: 1686.34m | eta: 2893.4m
step 73700/200000 (36.85%) | loss: 2.719301 | lrm: 0.72 | dt: 1327.73ms | tok/sec: 12,339 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 30 | total time: 1687.44m | eta: 2892.2m
step 73750/200000 (36.88%) | loss: 2.720797 | lrm: 0.72 | dt: 1329.44ms | tok/sec: 12,324 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 32 | total time: 1688.55m | eta: 2891.0m
step 73800/200000 (36.90%) | loss: 2.712771 | lrm: 0.72 | dt: 1324.15ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 34 | total time: 1689.65m | eta: 2889.7m
step 73850/200000 (36.92%) | loss: 2.753650 | lrm: 0.72 | dt: 1325.08ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 36 | total time: 1690.76m | eta: 2888.5m
step 73900/200000 (36.95%) | loss: 2.804625 | lrm: 0.72 | dt: 1324.27ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 38 | total time: 1691.86m | eta: 2887.3m
step 73950/200000 (36.98%) | loss: 2.753111 | lrm: 0.72 | dt: 1325.96ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 40 | total time: 1692.97m | eta: 2886.1m
Step 74000 | Validation bpb: 0.999618
step 74000/200000 (37.00%) | loss: 2.744744 | lrm: 0.71 | dt: 1400.89ms | tok/sec: 11,695 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 42 | total time: 1694.07m | eta: 2884.9m
step 74050/200000 (37.02%) | loss: 2.655549 | lrm: 0.71 | dt: 1324.51ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 44 | total time: 1695.18m | eta: 2883.7m
step 74100/200000 (37.05%) | loss: 2.722075 | lrm: 0.71 | dt: 1326.44ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 46 | total time: 1696.28m | eta: 2882.5m
step 74150/200000 (37.08%) | loss: 2.721217 | lrm: 0.71 | dt: 1327.36ms | tok/sec: 12,343 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 48 | total time: 1697.39m | eta: 2881.3m
step 74200/200000 (37.10%) | loss: 2.695279 | lrm: 0.71 | dt: 1324.22ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 50 | total time: 1698.49m | eta: 2880.0m
step 74250/200000 (37.12%) | loss: 2.758929 | lrm: 0.71 | dt: 1338.84ms | tok/sec: 12,237 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 52 | total time: 1699.60m | eta: 2878.8m
step 74300/200000 (37.15%) | loss: 2.701168 | lrm: 0.71 | dt: 1323.46ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 54 | total time: 1700.70m | eta: 2877.6m
step 74350/200000 (37.17%) | loss: 2.753872 | lrm: 0.71 | dt: 1325.08ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 56 | total time: 1701.80m | eta: 2876.4m
step 74400/200000 (37.20%) | loss: 2.788121 | lrm: 0.71 | dt: 1323.95ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 59 | total time: 1702.91m | eta: 2875.2m
step 74450/200000 (37.23%) | loss: 2.759998 | lrm: 0.71 | dt: 1327.58ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 61 | total time: 1704.01m | eta: 2874.0m
step 74500/200000 (37.25%) | loss: 2.755894 | lrm: 0.71 | dt: 1324.82ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 63 | total time: 1705.12m | eta: 2872.8m
step 74550/200000 (37.27%) | loss: 2.741599 | lrm: 0.71 | dt: 1324.55ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 65 | total time: 1706.23m | eta: 2871.6m
step 74600/200000 (37.30%) | loss: 2.688686 | lrm: 0.71 | dt: 1326.03ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 67 | total time: 1707.33m | eta: 2870.4m
step 74650/200000 (37.33%) | loss: 2.737562 | lrm: 0.71 | dt: 1328.89ms | tok/sec: 12,329 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 69 | total time: 1708.44m | eta: 2869.1m
step 74700/200000 (37.35%) | loss: 2.692607 | lrm: 0.71 | dt: 1328.08ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 71 | total time: 1709.54m | eta: 2867.9m
step 74750/200000 (37.38%) | loss: 2.708623 | lrm: 0.71 | dt: 1323.87ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 73 | total time: 1710.65m | eta: 2866.7m
step 74800/200000 (37.40%) | loss: 2.664732 | lrm: 0.71 | dt: 1326.20ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 75 | total time: 1711.75m | eta: 2865.5m
step 74850/200000 (37.42%) | loss: 2.772589 | lrm: 0.71 | dt: 1326.86ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 77 | total time: 1712.86m | eta: 2864.3m
step 74900/200000 (37.45%) | loss: 2.746282 | lrm: 0.71 | dt: 1324.32ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 79 | total time: 1713.96m | eta: 2863.1m
step 74950/200000 (37.48%) | loss: 2.721640 | lrm: 0.71 | dt: 1319.26ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 81 | total time: 1715.07m | eta: 2861.9m
step 75000/200000 (37.50%) | loss: 2.750701 | lrm: 0.71 | dt: 1337.13ms | tok/sec: 12,253 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 1 | total time: 1716.17m | eta: 2860.7m
step 75050/200000 (37.52%) | loss: 2.691983 | lrm: 0.71 | dt: 1323.01ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 3 | total time: 1717.28m | eta: 2859.5m
step 75100/200000 (37.55%) | loss: 2.647327 | lrm: 0.71 | dt: 1323.56ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 5 | total time: 1718.39m | eta: 2858.3m
step 75150/200000 (37.58%) | loss: 2.677345 | lrm: 0.71 | dt: 1323.71ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 7 | total time: 1719.49m | eta: 2857.0m
step 75200/200000 (37.60%) | loss: 2.671187 | lrm: 0.71 | dt: 1327.29ms | tok/sec: 12,343 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 9 | total time: 1720.60m | eta: 2855.8m
step 75250/200000 (37.62%) | loss: 2.781740 | lrm: 0.71 | dt: 1329.80ms | tok/sec: 12,320 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 11 | total time: 1721.70m | eta: 2854.6m
step 75300/200000 (37.65%) | loss: 2.713469 | lrm: 0.71 | dt: 1325.47ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 13 | total time: 1722.80m | eta: 2853.4m
step 75350/200000 (37.67%) | loss: 2.734007 | lrm: 0.71 | dt: 1327.71ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 15 | total time: 1723.91m | eta: 2852.2m
step 75400/200000 (37.70%) | loss: 2.735621 | lrm: 0.71 | dt: 1321.49ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 17 | total time: 1725.01m | eta: 2851.0m
step 75450/200000 (37.73%) | loss: 2.757655 | lrm: 0.71 | dt: 1326.74ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 19 | total time: 1726.12m | eta: 2849.8m
step 75500/200000 (37.75%) | loss: 2.707277 | lrm: 0.71 | dt: 1327.70ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 22 | total time: 1727.22m | eta: 2848.6m
step 75550/200000 (37.77%) | loss: 2.720033 | lrm: 0.71 | dt: 1330.81ms | tok/sec: 12,311 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 24 | total time: 1728.33m | eta: 2847.4m
step 75600/200000 (37.80%) | loss: 2.747299 | lrm: 0.71 | dt: 1328.00ms | tok/sec: 12,337 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 26 | total time: 1729.44m | eta: 2846.2m
step 75650/200000 (37.83%) | loss: 2.765434 | lrm: 0.71 | dt: 1328.45ms | tok/sec: 12,333 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 28 | total time: 1730.54m | eta: 2845.0m
step 75700/200000 (37.85%) | loss: 2.694878 | lrm: 0.71 | dt: 1326.25ms | tok/sec: 12,353 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 30 | total time: 1731.65m | eta: 2843.8m
step 75750/200000 (37.88%) | loss: 2.745766 | lrm: 0.71 | dt: 1326.73ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 32 | total time: 1732.76m | eta: 2842.6m
step 75800/200000 (37.90%) | loss: 2.680725 | lrm: 0.71 | dt: 1330.17ms | tok/sec: 12,317 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 34 | total time: 1733.86m | eta: 2841.3m
step 75850/200000 (37.92%) | loss: 2.666019 | lrm: 0.71 | dt: 1329.04ms | tok/sec: 12,327 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 36 | total time: 1734.97m | eta: 2840.1m
step 75900/200000 (37.95%) | loss: 2.781450 | lrm: 0.70 | dt: 1324.85ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 38 | total time: 1736.08m | eta: 2838.9m
step 75950/200000 (37.98%) | loss: 2.734460 | lrm: 0.70 | dt: 1325.43ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 40 | total time: 1737.18m | eta: 2837.7m
Step 76000 | Validation bpb: 0.998081
<|bos|>The capital of France is the Bâteau du Rise, which is a huge lapel that is built in 1897.
<|bos|>The chemical symbol of gold is a symbol of the element in the periodic table. Gold is in the same group of metals as other elements, with the
<|bos|>If yesterday was Friday, then tomorrow will be Friday. Yeah, this is about the time you've actually gone to school. This week we're back to
<|bos|>`The opposite of hot is cold.` If you are sitting down outside, you have to move about and take care of your lawn, your lawn grass,
<|bos|>`The planets of the solar system are: Mars, Jupiter, Saturn, Mars, Neptune, Uranus` and Nez Per
<|bos|>My favorite color is the color I love most. When I find myself in a situation where I don't like the color of my home, I
<|bos|>If 5*x + 3 = 13, `then x is the number of times 5`. Then 5*x = 13*13 + 3*x = 13
2026-03-18 03:53:09,023 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_076000.pt
2026-03-18 03:53:09,025 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_076000.json
2026-03-18 03:53:10,680 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_076000_rank0.pt
step 76000/200000 (38.00%) | loss: 2.727178 | lrm: 0.70 | dt: 1527.11ms | tok/sec: 10,728 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 42 | total time: 1738.29m | eta: 2836.5m
step 76050/200000 (38.02%) | loss: 2.688484 | lrm: 0.70 | dt: 1326.03ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 44 | total time: 1739.40m | eta: 2835.3m
step 76100/200000 (38.05%) | loss: 2.656130 | lrm: 0.70 | dt: 1323.42ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 46 | total time: 1740.50m | eta: 2834.1m
step 76150/200000 (38.08%) | loss: 2.731261 | lrm: 0.70 | dt: 1328.21ms | tok/sec: 12,335 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 48 | total time: 1741.60m | eta: 2832.9m
step 76200/200000 (38.10%) | loss: 2.811233 | lrm: 0.70 | dt: 1322.66ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 50 | total time: 1742.71m | eta: 2831.7m
step 76250/200000 (38.12%) | loss: 2.720419 | lrm: 0.70 | dt: 1327.15ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 52 | total time: 1743.81m | eta: 2830.5m
step 76300/200000 (38.15%) | loss: 2.685174 | lrm: 0.70 | dt: 1332.56ms | tok/sec: 12,295 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 54 | total time: 1744.92m | eta: 2829.3m
step 76350/200000 (38.17%) | loss: 2.776695 | lrm: 0.70 | dt: 1327.99ms | tok/sec: 12,337 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 56 | total time: 1746.02m | eta: 2828.1m
step 76400/200000 (38.20%) | loss: 2.748269 | lrm: 0.70 | dt: 1328.78ms | tok/sec: 12,330 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 58 | total time: 1747.13m | eta: 2826.9m
step 76450/200000 (38.23%) | loss: 2.725863 | lrm: 0.70 | dt: 1324.08ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 60 | total time: 1748.24m | eta: 2825.7m
step 76500/200000 (38.25%) | loss: 2.705495 | lrm: 0.70 | dt: 1326.53ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 62 | total time: 1749.34m | eta: 2824.5m
step 76550/200000 (38.27%) | loss: 2.748887 | lrm: 0.70 | dt: 1327.00ms | tok/sec: 12,346 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 65 | total time: 1750.45m | eta: 2823.3m
step 76600/200000 (38.30%) | loss: 2.716756 | lrm: 0.70 | dt: 1323.46ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 67 | total time: 1751.55m | eta: 2822.1m
step 76650/200000 (38.33%) | loss: 2.728828 | lrm: 0.70 | dt: 1337.38ms | tok/sec: 12,250 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 69 | total time: 1752.66m | eta: 2820.9m
step 76700/200000 (38.35%) | loss: 2.786048 | lrm: 0.70 | dt: 1326.04ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 71 | total time: 1753.77m | eta: 2819.7m
step 76750/200000 (38.38%) | loss: 2.715283 | lrm: 0.70 | dt: 1326.38ms | tok/sec: 12,352 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 73 | total time: 1754.87m | eta: 2818.5m
step 76800/200000 (38.40%) | loss: 2.698993 | lrm: 0.70 | dt: 1324.58ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 75 | total time: 1755.98m | eta: 2817.2m
step 76850/200000 (38.42%) | loss: 2.768222 | lrm: 0.70 | dt: 1327.61ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 77 | total time: 1757.08m | eta: 2816.0m
step 76900/200000 (38.45%) | loss: 2.778309 | lrm: 0.70 | dt: 1327.76ms | tok/sec: 12,339 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 79 | total time: 1758.19m | eta: 2814.8m
step 76950/200000 (38.48%) | loss: 2.763135 | lrm: 0.70 | dt: 1325.74ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 81 | total time: 1759.29m | eta: 2813.6m
step 77000/200000 (38.50%) | loss: 2.769769 | lrm: 0.70 | dt: 1324.85ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 0 | total time: 1760.40m | eta: 2812.4m
step 77050/200000 (38.52%) | loss: 2.748028 | lrm: 0.70 | dt: 1324.96ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 2 | total time: 1761.50m | eta: 2811.2m
step 77100/200000 (38.55%) | loss: 2.790473 | lrm: 0.70 | dt: 1324.21ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 4 | total time: 1762.61m | eta: 2810.0m
step 77150/200000 (38.58%) | loss: 2.782440 | lrm: 0.70 | dt: 1326.13ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 6 | total time: 1763.71m | eta: 2808.8m
step 77200/200000 (38.60%) | loss: 2.789192 | lrm: 0.70 | dt: 1324.24ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 8 | total time: 1764.82m | eta: 2807.6m
step 77250/200000 (38.62%) | loss: 2.683646 | lrm: 0.70 | dt: 1326.19ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 10 | total time: 1765.93m | eta: 2806.4m
step 77300/200000 (38.65%) | loss: 2.739142 | lrm: 0.70 | dt: 1327.17ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 12 | total time: 1767.03m | eta: 2805.2m
step 77350/200000 (38.67%) | loss: 2.649395 | lrm: 0.70 | dt: 1333.37ms | tok/sec: 12,287 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 14 | total time: 1768.14m | eta: 2804.0m
step 77400/200000 (38.70%) | loss: 2.782029 | lrm: 0.70 | dt: 1326.00ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 16 | total time: 1769.25m | eta: 2802.8m
step 77450/200000 (38.73%) | loss: 2.748709 | lrm: 0.70 | dt: 1330.83ms | tok/sec: 12,311 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 18 | total time: 1770.35m | eta: 2801.6m
step 77500/200000 (38.75%) | loss: 2.682582 | lrm: 0.70 | dt: 1323.78ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 20 | total time: 1771.46m | eta: 2800.4m
step 77550/200000 (38.77%) | loss: 2.777313 | lrm: 0.70 | dt: 1323.70ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 22 | total time: 1772.56m | eta: 2799.2m
step 77600/200000 (38.80%) | loss: 2.728142 | lrm: 0.70 | dt: 1325.89ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 24 | total time: 1773.67m | eta: 2798.0m
step 77650/200000 (38.83%) | loss: 2.689596 | lrm: 0.70 | dt: 1324.63ms | tok/sec: 12,368 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 26 | total time: 1774.77m | eta: 2796.8m
step 77700/200000 (38.85%) | loss: 2.717356 | lrm: 0.70 | dt: 1323.66ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 28 | total time: 1775.88m | eta: 2795.6m
step 77750/200000 (38.88%) | loss: 2.709767 | lrm: 0.70 | dt: 1326.19ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 30 | total time: 1776.98m | eta: 2794.4m
step 77800/200000 (38.90%) | loss: 2.740951 | lrm: 0.69 | dt: 1329.99ms | tok/sec: 12,318 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 32 | total time: 1778.09m | eta: 2793.2m
step 77850/200000 (38.92%) | loss: 2.729484 | lrm: 0.69 | dt: 1324.99ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 34 | total time: 1779.19m | eta: 2792.0m
step 77900/200000 (38.95%) | loss: 2.712733 | lrm: 0.69 | dt: 1325.01ms | tok/sec: 12,365 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 37 | total time: 1780.30m | eta: 2790.8m
step 77950/200000 (38.98%) | loss: 2.774045 | lrm: 0.69 | dt: 1325.18ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 39 | total time: 1781.40m | eta: 2789.6m
Step 78000 | Validation bpb: 0.996190
step 78000/200000 (39.00%) | loss: 2.701586 | lrm: 0.69 | dt: 1396.75ms | tok/sec: 11,730 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 41 | total time: 1782.51m | eta: 2788.4m
step 78050/200000 (39.02%) | loss: 2.800001 | lrm: 0.69 | dt: 1324.31ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 43 | total time: 1783.61m | eta: 2787.2m
step 78100/200000 (39.05%) | loss: 2.757912 | lrm: 0.69 | dt: 1322.14ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 45 | total time: 1784.72m | eta: 2786.0m
step 78150/200000 (39.08%) | loss: 2.742848 | lrm: 0.69 | dt: 1320.13ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 47 | total time: 1785.82m | eta: 2784.8m
step 78200/200000 (39.10%) | loss: 2.710767 | lrm: 0.69 | dt: 1327.84ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 49 | total time: 1786.93m | eta: 2783.6m
step 78250/200000 (39.12%) | loss: 2.763089 | lrm: 0.69 | dt: 1328.42ms | tok/sec: 12,333 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 51 | total time: 1788.03m | eta: 2782.4m
step 78300/200000 (39.15%) | loss: 2.736257 | lrm: 0.69 | dt: 1328.79ms | tok/sec: 12,330 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 53 | total time: 1789.14m | eta: 2781.2m
step 78350/200000 (39.17%) | loss: 2.724804 | lrm: 0.69 | dt: 1321.97ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 55 | total time: 1790.24m | eta: 2780.0m
step 78400/200000 (39.20%) | loss: 2.765804 | lrm: 0.69 | dt: 1324.86ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 57 | total time: 1791.35m | eta: 2778.8m
step 78450/200000 (39.23%) | loss: 2.647363 | lrm: 0.69 | dt: 1327.21ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 59 | total time: 1792.45m | eta: 2777.6m
step 78500/200000 (39.25%) | loss: 2.692549 | lrm: 0.69 | dt: 1322.03ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 61 | total time: 1793.56m | eta: 2776.4m
step 78550/200000 (39.27%) | loss: 2.711609 | lrm: 0.69 | dt: 1322.67ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 63 | total time: 1794.66m | eta: 2775.2m
step 78600/200000 (39.30%) | loss: 2.846509 | lrm: 0.69 | dt: 1330.32ms | tok/sec: 12,315 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 65 | total time: 1795.77m | eta: 2774.0m
step 78650/200000 (39.33%) | loss: 2.645433 | lrm: 0.69 | dt: 1329.53ms | tok/sec: 12,323 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 67 | total time: 1796.87m | eta: 2772.8m
step 78700/200000 (39.35%) | loss: 2.716275 | lrm: 0.69 | dt: 1328.96ms | tok/sec: 12,328 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 69 | total time: 1797.98m | eta: 2771.6m
step 78750/200000 (39.38%) | loss: 2.692807 | lrm: 0.69 | dt: 1327.07ms | tok/sec: 12,346 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 71 | total time: 1799.09m | eta: 2770.4m
step 78800/200000 (39.40%) | loss: 2.711850 | lrm: 0.69 | dt: 1328.15ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 73 | total time: 1800.19m | eta: 2769.2m
step 78850/200000 (39.42%) | loss: 2.722543 | lrm: 0.69 | dt: 1324.73ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 75 | total time: 1801.30m | eta: 2768.0m
step 78900/200000 (39.45%) | loss: 2.607781 | lrm: 0.69 | dt: 1321.62ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 78 | total time: 1802.40m | eta: 2766.8m
step 78950/200000 (39.48%) | loss: 2.715913 | lrm: 0.69 | dt: 1322.21ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 80 | total time: 1803.51m | eta: 2765.6m
step 79000/200000 (39.50%) | loss: 2.761197 | lrm: 0.69 | dt: 1326.56ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 82 | total time: 1804.61m | eta: 2764.4m
step 79050/200000 (39.52%) | loss: 2.772906 | lrm: 0.69 | dt: 1326.51ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 1 | total time: 1805.72m | eta: 2763.2m
step 79100/200000 (39.55%) | loss: 2.727429 | lrm: 0.69 | dt: 1326.58ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 3 | total time: 1806.82m | eta: 2762.0m
step 79150/200000 (39.58%) | loss: 2.742603 | lrm: 0.69 | dt: 1338.25ms | tok/sec: 12,242 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 5 | total time: 1807.93m | eta: 2760.8m
step 79200/200000 (39.60%) | loss: 2.700866 | lrm: 0.69 | dt: 1322.32ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 7 | total time: 1809.04m | eta: 2759.6m
step 79250/200000 (39.62%) | loss: 2.723627 | lrm: 0.69 | dt: 1325.92ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 9 | total time: 1810.14m | eta: 2758.4m
step 79300/200000 (39.65%) | loss: 2.647719 | lrm: 0.69 | dt: 1324.86ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 11 | total time: 1811.24m | eta: 2757.2m
step 79350/200000 (39.67%) | loss: 2.731248 | lrm: 0.69 | dt: 1322.04ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 13 | total time: 1812.35m | eta: 2756.0m
step 79400/200000 (39.70%) | loss: 2.696536 | lrm: 0.69 | dt: 1328.35ms | tok/sec: 12,334 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 15 | total time: 1813.45m | eta: 2754.8m
step 79450/200000 (39.73%) | loss: 2.695783 | lrm: 0.69 | dt: 1323.77ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 17 | total time: 1814.56m | eta: 2753.6m
step 79500/200000 (39.75%) | loss: 2.728476 | lrm: 0.69 | dt: 1326.07ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 19 | total time: 1815.66m | eta: 2752.4m
step 79550/200000 (39.77%) | loss: 2.687088 | lrm: 0.69 | dt: 1323.80ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 21 | total time: 1816.77m | eta: 2751.2m
step 79600/200000 (39.80%) | loss: 2.776919 | lrm: 0.69 | dt: 1321.56ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 23 | total time: 1817.87m | eta: 2750.0m
step 79650/200000 (39.83%) | loss: 2.734205 | lrm: 0.69 | dt: 1328.48ms | tok/sec: 12,332 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 25 | total time: 1818.98m | eta: 2748.8m
step 79700/200000 (39.85%) | loss: 2.656944 | lrm: 0.68 | dt: 1325.31ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 27 | total time: 1820.08m | eta: 2747.6m
step 79750/200000 (39.88%) | loss: 2.681190 | lrm: 0.68 | dt: 1321.85ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 29 | total time: 1821.18m | eta: 2746.4m
step 79800/200000 (39.90%) | loss: 2.689441 | lrm: 0.68 | dt: 1323.78ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 31 | total time: 1822.29m | eta: 2745.2m
step 79850/200000 (39.92%) | loss: 2.689237 | lrm: 0.68 | dt: 1322.61ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 33 | total time: 1823.39m | eta: 2744.0m
step 79900/200000 (39.95%) | loss: 2.783171 | lrm: 0.68 | dt: 1326.77ms | tok/sec: 12,348 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 35 | total time: 1824.50m | eta: 2742.8m
step 79950/200000 (39.98%) | loss: 2.625876 | lrm: 0.68 | dt: 1338.13ms | tok/sec: 12,243 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 38 | total time: 1825.61m | eta: 2741.6m
`Step 80000` | Validation bpb: 0.994588
<|bos|>The capital of France is the Riviera. In its historic city, it is the second largest country in the world, and is one
<|bos|>`The chemical symbol of gold is Au`. `Gold has a yellowish color, and is also one of the most valuable metals` on earth. Gold is found
<|bos|>If yesterday was Friday, then tomorrow will be Friday. I am not so sure (yes, I'm too sure) but I can see where the fest
<|bos|>`The opposite of hot is cold`. You can get warm when you work or when you're alone. But you need to cool off. You're not
<|bos|>`The planets of the solar system are: Mercury, Mars, Venus, and Earth`. How does the solar system have the potential to change our
<|bos|>`My favorite color is black`. You can find me here in Grove Ridge, Colorado, where I am an avid read
<|bos|>If 5*x + 3 = 13, then x is 13.

If `x + 3` = 10, then x is 10.

If 3*x +
2026-03-18 05:22:05,185 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_080000.pt
2026-03-18 05:22:05,185 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_080000.json
2026-03-18 05:22:06,830 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_080000_rank0.pt
step 80000/200000 (40.00%) | loss: 2.737269 | lrm: 0.68 | dt: 1569.23ms | tok/sec: 10,440 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 40 | total time: 1826.71m | eta: 2740.4m
step 80050/200000 (40.02%) | loss: 2.676653 | lrm: 0.68 | dt: 1322.53ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 42 | total time: 1827.82m | eta: 2739.2m
step 80100/200000 (40.05%) | loss: 2.741674 | lrm: 0.68 | dt: 1327.38ms | tok/sec: 12,343 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 44 | total time: 1828.92m | eta: 2738.0m
step 80150/200000 (40.08%) | loss: 2.665593 | lrm: 0.68 | dt: 1324.29ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 46 | total time: 1830.03m | eta: 2736.8m
step 80200/200000 (40.10%) | loss: 2.740099 | lrm: 0.68 | dt: 1324.79ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 48 | total time: 1831.13m | eta: 2735.6m
step 80250/200000 (40.12%) | loss: 2.736851 | lrm: 0.68 | dt: 1321.64ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 50 | total time: 1832.24m | eta: 2734.4m
step 80300/200000 (40.15%) | loss: 2.734309 | lrm: 0.68 | dt: 1325.26ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 52 | total time: 1833.34m | eta: 2733.2m
step 80350/200000 (40.17%) | loss: 2.759807 | lrm: 0.68 | dt: 1322.74ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 54 | total time: 1834.44m | eta: 2732.0m
step 80400/200000 (40.20%) | loss: 2.707501 | lrm: 0.68 | dt: 1323.44ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 56 | total time: 1835.54m | eta: 2730.8m
step 80450/200000 (40.23%) | loss: 2.771779 | lrm: 0.68 | dt: 1324.91ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 58 | total time: 1836.65m | eta: 2729.6m
step 80500/200000 (40.25%) | loss: 2.713597 | lrm: 0.68 | dt: 1322.44ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 60 | total time: 1837.75m | eta: 2728.4m
step 80550/200000 (40.27%) | loss: 2.753496 | lrm: 0.68 | dt: 1319.86ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 62 | total time: 1838.86m | eta: 2727.2m
step 80600/200000 (40.30%) | loss: 2.662844 | lrm: 0.68 | dt: 1324.07ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 64 | total time: 1839.96m | eta: 2726.0m
step 80650/200000 (40.33%) | loss: 2.719807 | lrm: 0.68 | dt: 1335.02ms | tok/sec: 12,272 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 66 | total time: 1841.07m | eta: 2724.9m


##
step 80700/200000 (40.35%) | loss: 2.733672 | lrm: 0.68 | dt: 1322.11ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 68 | total time: 1842.18m | eta: 2723.7m
step 80750/200000 (40.38%) | loss: 2.660396 | lrm: 0.68 | dt: 1329.17ms | tok/sec: 12,326 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 70 | total time: 1843.29m | eta: 2722.5m
step 80800/200000 (40.40%) | loss: 2.715448 | lrm: 0.68 | dt: 1325.86ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 72 | total time: 1844.40m | eta: 2721.3m
step 80850/200000 (40.42%) | loss: 2.715987 | lrm: 0.68 | dt: 1328.83ms | tok/sec: 12,329 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 74 | total time: 1845.51m | eta: 2720.1m
step 80900/200000 (40.45%) | loss: 2.669213 | lrm: 0.68 | dt: 1343.89ms | tok/sec: 12,191 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 76 | total time: 1846.62m | eta: 2718.9m
step 80950/200000 (40.48%) | loss: 2.673071 | lrm: 0.68 | dt: 1323.94ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 78 | total time: 1847.73m | eta: 2717.7m
step 81000/200000 (40.50%) | loss: 2.657725 | lrm: 0.68 | dt: 1322.64ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 80 | total time: 1848.84m | eta: 2716.5m
step 81050/200000 (40.52%) | loss: 2.763670 | lrm: 0.68 | dt: 1321.65ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 0 | total time: 1849.94m | eta: 2715.3m
step 81100/200000 (40.55%) | loss: 2.644427 | lrm: 0.68 | dt: 1323.24ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 2 | total time: 1851.04m | eta: 2714.1m
step 81150/200000 (40.58%) | loss: 2.685844 | lrm: 0.68 | dt: 1325.25ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 4 | total time: 1852.15m | eta: 2712.9m
step 81200/200000 (40.60%) | loss: 2.734172 | lrm: 0.68 | dt: 1325.32ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 6 | total time: 1853.26m | eta: 2711.7m
step 81250/200000 (40.62%) | loss: 2.744846 | lrm: 0.68 | dt: 1326.92ms | tok/sec: 12,347 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 8 | total time: 1854.36m | eta: 2710.6m
step 81300/200000 (40.65%) | loss: 2.726311 | lrm: 0.68 | dt: 1324.73ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 10 | total time: 1855.47m | eta: 2709.4m
step 81350/200000 (40.67%) | loss: 2.671111 | lrm: 0.68 | dt: 1328.81ms | tok/sec: 12,329 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 13 | total time: 1856.57m | eta: 2708.2m
step 81400/200000 (40.70%) | loss: 2.801621 | lrm: 0.68 | dt: 1323.44ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 15 | total time: 1857.68m | eta: 2707.0m
step 81450/200000 (40.73%) | loss: 2.738750 | lrm: 0.68 | dt: 1323.66ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 17 | total time: 1858.78m | eta: 2705.8m
step 81500/200000 (40.75%) | loss: 2.689810 | lrm: 0.68 | dt: 1323.61ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 19 | total time: 1859.88m | eta: 2704.6m
step 81550/200000 (40.77%) | loss: 2.724049 | lrm: 0.68 | dt: 1325.90ms | tok/sec: 12,356 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 21 | total time: 1860.99m | eta: 2703.4m
step 81600/200000 (40.80%) | loss: 2.678162 | lrm: 0.67 | dt: 1324.52ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 23 | total time: 1862.09m | eta: 2702.2m
step 81650/200000 (40.83%) | loss: 2.633001 | lrm: 0.67 | dt: 1324.52ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 25 | total time: 1863.20m | eta: 2701.0m
step 81700/200000 (40.85%) | loss: 2.709044 | lrm: 0.67 | dt: 1323.77ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 27 | total time: 1864.30m | eta: 2699.8m
step 81750/200000 (40.88%) | loss: 2.721547 | lrm: 0.67 | dt: 1324.83ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 29 | total time: 1865.40m | eta: 2698.6m
step 81800/200000 (40.90%) | loss: 2.743002 | lrm: 0.67 | dt: 1323.24ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 31 | total time: 1866.50m | eta: 2697.4m
step 81850/200000 (40.92%) | loss: 2.698586 | lrm: 0.67 | dt: 1326.51ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 33 | total time: 1867.61m | eta: 2696.2m
step 81900/200000 (40.95%) | loss: 2.786204 | lrm: 0.67 | dt: 1322.60ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 35 | total time: 1868.71m | eta: 2695.0m
step 81950/200000 (40.98%) | loss: 2.699982 | lrm: 0.67 | dt: 1320.62ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 37 | total time: 1869.81m | eta: 2693.8m
Step 82000 | Validation bpb: 0.993055
step 82000/200000 (41.00%) | loss: 2.730208 | lrm: 0.67 | dt: 1405.32ms | tok/sec: 11,658 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 39 | total time: 1870.91m | eta: 2692.6m
step 82050/200000 (41.02%) | loss: 2.724948 | lrm: 0.67 | dt: 1320.05ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 41 | total time: 1872.02m | eta: 2691.4m
step 82100/200000 (41.05%) | loss: 2.699139 | lrm: 0.67 | dt: 1320.61ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 43 | total time: 1873.12m | eta: 2690.2m
step 82150/200000 (41.08%) | loss: 2.746901 | lrm: 0.67 | dt: 1319.81ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 45 | total time: 1874.22m | eta: 2689.0m
step 82200/200000 (41.10%) | loss: 2.668177 | lrm: 0.67 | dt: 1322.13ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 47 | total time: 1875.32m | eta: 2687.8m
step 82250/200000 (41.12%) | loss: 2.612953 | lrm: 0.67 | dt: 1322.85ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 49 | total time: 1876.42m | eta: 2686.6m
step 82300/200000 (41.15%) | loss: 2.702670 | lrm: 0.67 | dt: 1322.04ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 51 | total time: 1877.52m | eta: 2685.4m
step 82350/200000 (41.17%) | loss: 2.743365 | lrm: 0.67 | dt: 1325.80ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 53 | total time: 1878.63m | eta: 2684.2m
step 82400/200000 (41.20%) | loss: 2.738270 | lrm: 0.67 | dt: 1321.18ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 55 | total time: 1879.73m | eta: 2683.0m
step 82450/200000 (41.23%) | loss: 2.746170 | lrm: 0.67 | dt: 1323.49ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 57 | total time: 1880.83m | eta: 2681.8m
step 82500/200000 (41.25%) | loss: 2.750759 | lrm: 0.67 | dt: 1322.14ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 59 | total time: 1881.93m | eta: 2680.7m
step 82550/200000 (41.27%) | loss: 2.642079 | lrm: 0.67 | dt: 1326.29ms | tok/sec: 12,353 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 62 | total time: 1883.03m | eta: 2679.5m
step 82600/200000 (41.30%) | loss: 2.770665 | lrm: 0.67 | dt: 1326.51ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 64 | total time: 1884.13m | eta: 2678.3m
step 82650/200000 (41.33%) | loss: 2.695830 | lrm: 0.67 | dt: 1339.65ms | tok/sec: 12,230 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 66 | total time: 1885.24m | eta: 2677.1m
step 82700/200000 (41.35%) | loss: 2.791889 | lrm: 0.67 | dt: 1321.15ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 68 | total time: 1886.34m | eta: 2675.9m
step 82750/200000 (41.38%) | loss: 2.647320 | lrm: 0.67 | dt: 1329.25ms | tok/sec: 12,325 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 70 | total time: 1887.44m | eta: 2674.7m
step 82800/200000 (41.40%) | loss: 2.737763 | lrm: 0.67 | dt: 1320.44ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 72 | total time: 1888.54m | eta: 2673.5m
step 82850/200000 (41.42%) | loss: 2.754608 | lrm: 0.67 | dt: 1320.97ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 74 | total time: 1889.64m | eta: 2672.3m
step 82900/200000 (41.45%) | loss: 2.737415 | lrm: 0.67 | dt: 1323.16ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 76 | total time: 1890.75m | eta: 2671.1m
step 82950/200000 (41.48%) | loss: 2.631347 | lrm: 0.67 | dt: 1326.58ms | tok/sec: 12,350 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 78 | total time: 1891.85m | eta: 2669.9m
step 83000/200000 (41.50%) | loss: 2.760287 | lrm: 0.67 | dt: 1323.53ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 80 | total time: 1892.95m | eta: 2668.7m
step 83050/200000 (41.52%) | loss: 2.806235 | lrm: 0.67 | dt: 1317.96ms | tok/sec: 12,431 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 82 | total time: 1894.05m | eta: 2667.5m
step 83100/200000 (41.55%) | loss: 2.668760 | lrm: 0.67 | dt: 1318.07ms | tok/sec: 12,430 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 1 | total time: 1895.15m | eta: 2666.3m
step 83150/200000 (41.58%) | loss: 2.828321 | lrm: 0.67 | dt: 1322.57ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 3 | total time: 1896.26m | eta: 2665.1m
step 83200/200000 (41.60%) | loss: 2.671973 | lrm: 0.67 | dt: 1322.69ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 5 | total time: 1897.36m | eta: 2663.9m
step 83250/200000 (41.62%) | loss: 2.760731 | lrm: 0.67 | dt: 1319.20ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 7 | total time: 1898.46m | eta: 2662.7m
step 83300/200000 (41.65%) | loss: 2.677026 | lrm: 0.67 | dt: 1322.17ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 9 | total time: 1899.56m | eta: 2661.5m
step 83350/200000 (41.67%) | loss: 2.632568 | lrm: 0.67 | dt: 1323.81ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 11 | total time: 1900.66m | eta: 2660.3m
step 83400/200000 (41.70%) | loss: 2.755694 | lrm: 0.67 | dt: 1323.36ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 13 | total time: 1901.76m | eta: 2659.1m
step 83450/200000 (41.73%) | loss: 2.762570 | lrm: 0.67 | dt: 1323.94ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 15 | total time: 1902.87m | eta: 2657.9m
step 83500/200000 (41.75%) | loss: 2.758202 | lrm: 0.66 | dt: 1320.48ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 17 | total time: 1903.97m | eta: 2656.8m
step 83550/200000 (41.77%) | loss: 2.681898 | lrm: 0.66 | dt: 1321.24ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 19 | total time: 1905.07m | eta: 2655.6m
step 83600/200000 (41.80%) | loss: 2.711836 | lrm: 0.66 | dt: 1323.37ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 21 | total time: 1906.17m | eta: 2654.4m
step 83650/200000 (41.83%) | loss: 2.666603 | lrm: 0.66 | dt: 1321.51ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 23 | total time: 1907.27m | eta: 2653.2m
step 83700/200000 (41.85%) | loss: 2.661920 | lrm: 0.66 | dt: 1322.42ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 26 | total time: 1908.38m | eta: 2652.0m
step 83750/200000 (41.88%) | loss: 2.738551 | lrm: 0.66 | dt: 1327.14ms | tok/sec: 12,345 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 28 | total time: 1909.48m | eta: 2650.8m
step 83800/200000 (41.90%) | loss: 2.788054 | lrm: 0.66 | dt: 1319.59ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 30 | total time: 1910.58m | eta: 2649.6m
step 83850/200000 (41.92%) | loss: 2.750824 | lrm: 0.66 | dt: 1323.73ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 32 | total time: 1911.68m | eta: 2648.4m
step 83900/200000 (41.95%) | loss: 2.719116 | lrm: 0.66 | dt: 1322.31ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 34 | total time: 1912.78m | eta: 2647.2m
step 83950/200000 (41.98%) | loss: 2.606839 | lrm: 0.66 | dt: 1324.24ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 36 | total time: 1913.89m | eta: 2646.0m
Step 84000 | Validation bpb: 0.991403
<|bos|>The capital of France is the Cousteau River Valley, which is the major of the region. It is situated 10 km from
<|bos|>The chemical symbol of gold is 24. The chemical formula of mercury is Hg, the element of which is composed of a metal atom with
<|bos|>If yesterday was Friday, then tomorrow will be Friday.

Then today was Friday. But I don't know, I guess I'd like to say.

O
<|bos|>The opposite of hot is cold. Cold can be the result of either a cold or a hot dog. Hot dogs can be hot when they
<|bos|>The planets of the solar system are: Mercury, Earth, Mars, Jupiter and Saturn. It is called a planetary system and
<|bos|>My favorite color is the one about the 4th room. It's the most populated space in the room (I know it's the
<|bos|>If 5*x + 3 = 13, then x is 13 times the 13 of the 13. So if 5*x + 6, then 13 times
2026-03-18 06:50:51,869 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_084000.pt
2026-03-18 06:50:51,870 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_084000.json
2026-03-18 06:50:53,441 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_084000_rank0.pt
step 84000/200000 (42.00%) | loss: 2.695709 | lrm: 0.66 | dt: 1484.39ms | tok/sec: 11,037 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 38 | total time: 1914.99m | eta: 2644.8m
step 84050/200000 (42.02%) | loss: 2.697273 | lrm: 0.66 | dt: 1326.29ms | tok/sec: 12,353 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 40 | total time: 1916.10m | eta: 2643.6m
step 84100/200000 (42.05%) | loss: 2.735626 | lrm: 0.66 | dt: 1325.04ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 42 | total time: 1917.20m | eta: 2642.4m
step 84150/200000 (42.08%) | loss: 2.701763 | lrm: 0.66 | dt: 1323.75ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 44 | total time: 1918.30m | eta: 2641.3m
step 84200/200000 (42.10%) | loss: 2.688480 | lrm: 0.66 | dt: 1319.15ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 46 | total time: 1919.41m | eta: 2640.1m
step 84250/200000 (42.12%) | loss: 2.788106 | lrm: 0.66 | dt: 1322.53ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 48 | total time: 1920.51m | eta: 2638.9m
step 84300/200000 (42.15%) | loss: 2.679290 | lrm: 0.66 | dt: 1321.67ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 50 | total time: 1921.62m | eta: 2637.7m
step 84350/200000 (42.17%) | loss: 2.700495 | lrm: 0.66 | dt: 1319.73ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 52 | total time: 1922.72m | eta: 2636.5m
step 84400/200000 (42.20%) | loss: 2.717944 | lrm: 0.66 | dt: 1323.78ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 54 | total time: 1923.82m | eta: 2635.3m
step 84450/200000 (42.23%) | loss: 2.779950 | lrm: 0.66 | dt: 1319.14ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 56 | total time: 1924.92m | eta: 2634.1m
step 84500/200000 (42.25%) | loss: 2.696219 | lrm: 0.66 | dt: 1325.63ms | tok/sec: 12,359 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 58 | total time: 1926.02m | eta: 2632.9m
step 84550/200000 (42.27%) | loss: 2.801140 | lrm: 0.66 | dt: 1325.88ms | tok/sec: 12,357 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 60 | total time: 1927.13m | eta: 2631.7m
step 84600/200000 (42.30%) | loss: 2.686970 | lrm: 0.66 | dt: 1324.44ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 62 | total time: 1928.23m | eta: 2630.5m
step 84650/200000 (42.33%) | loss: 2.761483 | lrm: 0.66 | dt: 1319.44ms | tok/sec: 12,417 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 64 | total time: 1929.33m | eta: 2629.4m
step 84700/200000 (42.35%) | loss: 2.790963 | lrm: 0.66 | dt: 1322.65ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 66 | total time: 1930.44m | eta: 2628.2m
step 84750/200000 (42.38%) | loss: 2.640437 | lrm: 0.66 | dt: 1324.09ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 68 | total time: 1931.54m | eta: 2627.0m
step 84800/200000 (42.40%) | loss: 2.742614 | lrm: 0.66 | dt: 1324.86ms | tok/sec: 12,366 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 70 | total time: 1932.64m | eta: 2625.8m
step 84850/200000 (42.42%) | loss: 2.716128 | lrm: 0.66 | dt: 1323.92ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 72 | total time: 1933.74m | eta: 2624.6m
step 84900/200000 (42.45%) | loss: 2.674894 | lrm: 0.66 | dt: 1318.16ms | tok/sec: 12,429 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 75 | total time: 1934.85m | eta: 2623.4m
step 84950/200000 (42.48%) | loss: 2.664024 | lrm: 0.66 | dt: 1322.68ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 77 | total time: 1935.95m | eta: 2622.2m
step 85000/200000 (42.50%) | loss: 2.695212 | lrm: 0.66 | dt: 1353.60ms | tok/sec: 12,104 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 79 | total time: 1937.05m | eta: 2621.0m
step 85050/200000 (42.52%) | loss: 2.748742 | lrm: 0.66 | dt: 1323.11ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 81 | total time: 1938.16m | eta: 2619.8m
step 85100/200000 (42.55%) | loss: 2.722085 | lrm: 0.66 | dt: 1321.95ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 1 | total time: 1939.26m | eta: 2618.7m
step 85150/200000 (42.58%) | loss: 2.750825 | lrm: 0.66 | dt: 1323.37ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 3 | total time: 1940.36m | eta: 2617.5m
step 85200/200000 (42.60%) | loss: 2.776326 | lrm: 0.66 | dt: 1322.02ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 5 | total time: 1941.47m | eta: 2616.3m
step 85250/200000 (42.62%) | loss: 2.791664 | lrm: 0.66 | dt: 1320.70ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 7 | total time: 1942.57m | eta: 2615.1m
step 85300/200000 (42.65%) | loss: 2.693125 | lrm: 0.66 | dt: 1323.38ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 9 | total time: 1943.67m | eta: 2613.9m
step 85350/200000 (42.67%) | loss: 2.788330 | lrm: 0.66 | dt: 1321.34ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 11 | total time: 1944.77m | eta: 2612.7m
step 85400/200000 (42.70%) | loss: 2.634411 | lrm: 0.65 | dt: 1324.09ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 13 | total time: 1945.88m | eta: 2611.5m
step 85450/200000 (42.73%) | loss: 2.727305 | lrm: 0.65 | dt: 1324.41ms | tok/sec: 12,370 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 15 | total time: 1946.98m | eta: 2610.3m
step 85500/200000 (42.75%) | loss: 2.697270 | lrm: 0.65 | dt: 1326.53ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 17 | total time: 1948.08m | eta: 2609.1m
step 85550/200000 (42.77%) | loss: 2.736867 | lrm: 0.65 | dt: 1321.46ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 19 | total time: 1949.19m | eta: 2608.0m
step 85600/200000 (42.80%) | loss: 2.694910 | lrm: 0.65 | dt: 1319.02ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 21 | total time: 1950.29m | eta: 2606.8m
step 85650/200000 (42.83%) | loss: 2.671168 | lrm: 0.65 | dt: 1323.36ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 23 | total time: 1951.39m | eta: 2605.6m
step 85700/200000 (42.85%) | loss: 2.733721 | lrm: 0.65 | dt: 1319.57ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 25 | total time: 1952.49m | eta: 2604.4m
step 85750/200000 (42.88%) | loss: 2.630112 | lrm: 0.65 | dt: 1322.20ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 27 | total time: 1953.59m | eta: 2603.2m
step 85800/200000 (42.90%) | loss: 2.722007 | lrm: 0.65 | dt: 1319.55ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 29 | total time: 1954.69m | eta: 2602.0m
step 85850/200000 (42.92%) | loss: 2.789186 | lrm: 0.65 | dt: 1320.15ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 31 | total time: 1955.80m | eta: 2600.8m
step 85900/200000 (42.95%) | loss: 2.672504 | lrm: 0.65 | dt: 1319.69ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 33 | total time: 1956.90m | eta: 2599.6m
step 85950/200000 (42.98%) | loss: 2.677262 | lrm: 0.65 | dt: 1322.01ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 35 | total time: 1958.00m | eta: 2598.4m
Step 86000 | Validation bpb: 0.989838
step 86000/200000 (43.00%) | loss: 2.794456 | lrm: 0.65 | dt: 1396.22ms | tok/sec: 11,734 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 37 | total time: 1959.10m | eta: 2597.3m
step 86050/200000 (43.02%) | loss: 2.634308 | lrm: 0.65 | dt: 1324.21ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 39 | total time: 1960.21m | eta: 2596.1m
step 86100/200000 (43.05%) | loss: 2.767378 | lrm: 0.65 | dt: 1319.78ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 41 | total time: 1961.31m | eta: 2594.9m
step 86150/200000 (43.08%) | loss: 2.688155 | lrm: 0.65 | dt: 1320.37ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 43 | total time: 1962.41m | eta: 2593.7m
step 86200/200000 (43.10%) | loss: 2.671577 | lrm: 0.65 | dt: 1320.14ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 46 | total time: 1963.51m | eta: 2592.5m
step 86250/200000 (43.12%) | loss: 2.624508 | lrm: 0.65 | dt: 1318.88ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 48 | total time: 1964.62m | eta: 2591.3m
step 86300/200000 (43.15%) | loss: 2.689457 | lrm: 0.65 | dt: 1322.94ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 50 | total time: 1965.72m | eta: 2590.1m
step 86350/200000 (43.17%) | loss: 2.735587 | lrm: 0.65 | dt: 1319.42ms | tok/sec: 12,417 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 52 | total time: 1966.82m | eta: 2588.9m
step 86400/200000 (43.20%) | loss: 2.628327 | lrm: 0.65 | dt: 1322.57ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 54 | total time: 1967.93m | eta: 2587.8m
step 86450/200000 (43.23%) | loss: 2.790800 | lrm: 0.65 | dt: 1321.25ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 56 | total time: 1969.03m | eta: 2586.6m
step 86500/200000 (43.25%) | loss: 2.617840 | lrm: 0.65 | dt: 1321.63ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 58 | total time: 1970.13m | eta: 2585.4m
step 86550/200000 (43.27%) | loss: 2.686922 | lrm: 0.65 | dt: 1322.49ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 60 | total time: 1971.23m | eta: 2584.2m
step 86600/200000 (43.30%) | loss: 2.635100 | lrm: 0.65 | dt: 1318.39ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 62 | total time: 1972.34m | eta: 2583.0m
step 86650/200000 (43.33%) | loss: 2.702023 | lrm: 0.65 | dt: 1321.94ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 64 | total time: 1973.44m | eta: 2581.8m
step 86700/200000 (43.35%) | loss: 2.709612 | lrm: 0.65 | dt: 1322.81ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 66 | total time: 1974.54m | eta: 2580.6m
step 86750/200000 (43.38%) | loss: 2.741412 | lrm: 0.65 | dt: 1326.53ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 68 | total time: 1975.64m | eta: 2579.5m
step 86800/200000 (43.40%) | loss: 2.701012 | lrm: 0.65 | dt: 1326.15ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 70 | total time: 1976.75m | eta: 2578.3m
step 86850/200000 (43.42%) | loss: 2.687013 | lrm: 0.65 | dt: 1321.27ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 72 | total time: 1977.85m | eta: 2577.1m
step 86900/200000 (43.45%) | loss: 2.787104 | lrm: 0.65 | dt: 1322.67ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 74 | total time: 1978.95m | eta: 2575.9m
step 86950/200000 (43.48%) | loss: 2.745084 | lrm: 0.65 | dt: 1321.56ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 76 | total time: 1980.05m | eta: 2574.7m
step 87000/200000 (43.50%) | loss: 2.708190 | lrm: 0.65 | dt: 1318.88ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 78 | total time: 1981.15m | eta: 2573.5m
step 87050/200000 (43.52%) | loss: 2.688004 | lrm: 0.65 | dt: 1319.81ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 42 rg: 80 | total time: 1982.25m | eta: 2572.3m
step 87100/200000 (43.55%) | loss: 2.748852 | lrm: 0.65 | dt: 1315.47ms | tok/sec: 12,454 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 0 | total time: 1983.35m | eta: 2571.1m
step 87150/200000 (43.58%) | loss: 2.733065 | lrm: 0.65 | dt: 1318.68ms | tok/sec: 12,424 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 2 | total time: 1984.45m | eta: 2570.0m
step 87200/200000 (43.60%) | loss: 2.657091 | lrm: 0.65 | dt: 1322.95ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 4 | total time: 1985.55m | eta: 2568.8m
step 87250/200000 (43.62%) | loss: 2.745754 | lrm: 0.65 | dt: 1319.37ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 7 | total time: 1986.66m | eta: 2567.6m
step 87300/200000 (43.65%) | loss: 2.723511 | lrm: 0.64 | dt: 1317.24ms | tok/sec: 12,438 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 9 | total time: 1987.76m | eta: 2566.4m
step 87350/200000 (43.67%) | loss: 2.685687 | lrm: 0.64 | dt: 1320.15ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 11 | total time: 1988.86m | eta: 2565.2m
step 87400/200000 (43.70%) | loss: 2.726983 | lrm: 0.64 | dt: 1322.49ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 13 | total time: 1989.96m | eta: 2564.0m
step 87450/200000 (43.73%) | loss: 2.748682 | lrm: 0.64 | dt: 1320.57ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 15 | total time: 1991.06m | eta: 2562.8m
step 87500/200000 (43.75%) | loss: 2.730412 | lrm: 0.64 | dt: 1317.80ms | tok/sec: 12,432 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 17 | total time: 1992.16m | eta: 2561.6m
step 87550/200000 (43.77%) | loss: 2.618996 | lrm: 0.64 | dt: 1319.80ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 19 | total time: 1993.26m | eta: 2560.4m
step 87600/200000 (43.80%) | loss: 2.715326 | lrm: 0.64 | dt: 1320.09ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 21 | total time: 1994.35m | eta: 2559.3m
step 87650/200000 (43.83%) | loss: 2.700547 | lrm: 0.64 | dt: 1323.37ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 23 | total time: 1995.46m | eta: 2558.1m
step 87700/200000 (43.85%) | loss: 2.726936 | lrm: 0.64 | dt: 1317.38ms | tok/sec: 12,436 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 25 | total time: 1996.56m | eta: 2556.9m
step 87750/200000 (43.88%) | loss: 2.744757 | lrm: 0.64 | dt: 1321.72ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 27 | total time: 1997.66m | eta: 2555.7m
step 87800/200000 (43.90%) | loss: 2.732589 | lrm: 0.64 | dt: 1318.03ms | tok/sec: 12,430 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 29 | total time: 1998.76m | eta: 2554.5m
step 87850/200000 (43.92%) | loss: 2.674847 | lrm: 0.64 | dt: 1319.16ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 31 | total time: 1999.85m | eta: 2553.3m
step 87900/200000 (43.95%) | loss: 2.724711 | lrm: 0.64 | dt: 1320.63ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 33 | total time: 2000.95m | eta: 2552.1m
step 87950/200000 (43.98%) | loss: 2.677431 | lrm: 0.64 | dt: 1318.65ms | tok/sec: 12,424 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 35 | total time: 2002.05m | eta: 2550.9m
Step 88000 | Validation bpb: 0.988263
<|bos|>The capital of France is the capital of the United States (Lucía, inaugurated in 1913), which is
<|bos|>The chemical symbol of gold is 24. The symbol of a substance is: 24. Male 24 is a 2,000k
<|bos|>If yesterday was Friday, then tomorrow will be Friday. This week, it's on 7th March. (It's Friday. I will be making this
<|bos|>The opposite of hot is cold. You can feel cold and can heat up quickly. The airways will be cold and they may be very uncom
<|bos|>The planets of the solar system are: Jupiter, Neptune, and Pluto. Weighted planets are objects orbiting our solar
<|bos|>My favorite color is the one on the right. If you really want to change your home's atmosphere, I would use it, but it can
<|bos|>If 5*x + 3 = 13, then x is the number of times a quarter
compares to the number of times a quarter is written. Thus, the
2026-03-18 08:19:32,307 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_088000.pt
2026-03-18 08:19:32,309 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_088000.json
2026-03-18 08:19:33,813 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_088000_rank0.pt
step 88000/200000 (44.00%) | loss: 2.720382 | lrm: 0.64 | dt: 1538.16ms | tok/sec: 10,651 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 37 | total time: 2003.16m | eta: 2549.8m
step 88050/200000 (44.02%) | loss: 2.647084 | lrm: 0.64 | dt: 1318.61ms | tok/sec: 12,425 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 39 | total time: 2004.26m | eta: 2548.6m
step 88100/200000 (44.05%) | loss: 2.720617 | lrm: 0.64 | dt: 1319.42ms | tok/sec: 12,417 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 41 | total time: 2005.36m | eta: 2547.4m
step 88150/200000 (44.08%) | loss: 2.659930 | lrm: 0.64 | dt: 1319.17ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 43 | total time: 2006.46m | eta: 2546.2m
step 88200/200000 (44.10%) | loss: 2.752832 | lrm: 0.64 | dt: 1317.39ms | tok/sec: 12,436 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 45 | total time: 2007.56m | eta: 2545.0m
step 88250/200000 (44.12%) | loss: 2.750503 | lrm: 0.64 | dt: 1321.33ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 47 | total time: 2008.66m | eta: 2543.8m
step 88300/200000 (44.15%) | loss: 2.716436 | lrm: 0.64 | dt: 1320.16ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 49 | total time: 2009.76m | eta: 2542.6m
step 88350/200000 (44.17%) | loss: 2.709789 | lrm: 0.64 | dt: 1323.25ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 45 | total time: 2007.56m | eta: 2545.0m
step 88250/200000 (44.12%) | loss: 2.750503 | lrm: 0.64 | dt: 1321.33ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 47 | total time: 2008.66m | eta: 2543.8m
step 88300/200000 (44.15%) | loss: 2.716436 | lrm: 0.64 | dt: 1320.16ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 49 | total time: 2009.76m | eta: 2542.6m
step 88350/200000 (44.17%) | loss: 2.709789 | lrm: 0.64 | dt: 1323.25ms | tok/sec: 12,381 | bf16_mfu: 0.00 | step 88350/200000 (44.17%) | loss: 2.709789 | lrm: 0.64 | dt: 1323.25ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 51 | total time: 2010.86m | eta: 2541.5m
step 88400/200000 (44.20%) | loss: 2.748386 | lrm: 0.64 | dt: 1320.78ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 53 | total time: 2011.96m | eta: 2540.3m
step 88450/200000 (44.23%) | loss: 2.660870 | lrm: 0.64 | dt: 1319.37ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 55 | total time: 2013.06m | eta: 2539.1m
step 88500/200000 (44.25%) | loss: 2.785162 | lrm: 0.64 | dt: 1319.65ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 58 | total time: 2014.16m | eta: 2537.9m
step 88550/200000 (44.27%) | loss: 2.746557 | lrm: 0.64 | dt: 1324.30ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 60 | total time: 2015.26m | eta: 2536.7m
step 88600/200000 (44.30%) | loss: 2.689259 | lrm: 0.64 | dt: 1318.60ms | tok/sec: 12,425 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 62 | total time: 2016.36m | eta: 2535.5m
step 88650/200000 (44.33%) | loss: 2.671057 | lrm: 0.64 | dt: 1319.91ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 64 | total time: 2017.46m | eta: 2534.3m
step 88700/200000 (44.35%) | loss: 2.803140 | lrm: 0.64 | dt: 1319.48ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 66 | total time: 2018.56m | eta: 2533.2m
step 88750/200000 (44.38%) | loss: 2.755059 | lrm: 0.64 | dt: 1322.88ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 68 | total time: 2019.66m | eta: 2532.0m
step 88800/200000 (44.40%) | loss: 2.728128 | lrm: 0.64 | dt: 1320.42ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 70 | total time: 2020.76m | eta: 2530.8m
step 88850/200000 (44.42%) | loss: 2.695744 | lrm: 0.64 | dt: 1317.99ms | tok/sec: 12,431 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 72 | total time: 2021.86m | eta: 2529.6m
step 88900/200000 (44.45%) | loss: 2.730676 | lrm: 0.64 | dt: 1325.08ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 74 | total time: 2022.97m | eta: 2528.4m
step 88950/200000 (44.48%) | loss: 2.737235 | lrm: 0.64 | dt: 1321.16ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 76 | total time: 2024.07m | eta: 2527.2m
step 89000/200000 (44.50%) | loss: 2.640257 | lrm: 0.64 | dt: 1322.31ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 78 | total time: 2025.18m | eta: 2526.1m
step 89050/200000 (44.52%) | loss: 2.663453 | lrm: 0.64 | dt: 1324.54ms | tok/sec: 12,369 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 80 | total time: 2026.28m | eta: 2524.9m
step 89100/200000 (44.55%) | loss: 2.682366 | lrm: 0.64 | dt: 1326.10ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 43 rg: 82 | total time: 2027.38m | eta: 2523.7m
step 89150/200000 (44.58%) | loss: 2.692306 | lrm: 0.64 | dt: 1322.89ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 44 rg: 1 | total time: 2028.49m | eta: 2522.5m
step 89200/200000 (44.60%) | loss: 2.758057 | lrm: 0.63 | dt: 1321.14ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 44 rg: 3 | total time: 2029.59m | eta: 2521.3m
step 89250/200000 (44.62%) | loss: 2.698590 | lrm: 0.63 | dt: 1323.32ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 44 rg: 5 | total time: 2030.69m | eta: 2520.2m
step 89300/200000 (44.65%) | loss: 2.672316 | lrm: 0.63 | dt: 1323.57ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 44 rg: 7 | total time: 2031.80m | eta: 2519.0m
step 89350/200000 (44.67%) | loss: 2.708341 | lrm: 0.63 | dt: 1328.29ms | tok/sec: 12,334 | bf16_mfu: 0.00 | epoch: 1 pq: 44 rg: 9 | total time: 2032.90m | eta: 2517.8m
step 89400/200000 (44.70%) | loss: 2.676540 | lrm: 0.63 | dt: 1321.07ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 44 rg: 11 | total time: 2034.00m | eta: 2516.6m
step 89450/200000 (44.73%) | loss: 2.695199 | lrm: 0.63 | dt: 1324.07ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 44 rg: 13 | total time: 2035.11m | eta: 2515.4m
step 89500/200000 (44.75%) | loss: 2.655397 | lrm: 0.63 | dt: 1325.60ms | tok/sec: 12,359 | bf16_mfu: 0.00 | epoch: 1 pq: 44 rg: 15 | total time: 2036.21m | eta: 2514.3m
step 89550/200000 (44.77%) | loss: 2.682230 | lrm: 0.63 | dt: 1331.20ms | tok/sec: 12,307 | bf16_mfu: 0.00 | epoch: 1 pq: 44 rg: 17 | total time: 2037.32m | eta: 2513.1m
step 89600/200000 (44.80%) | loss: 2.680814 | lrm: 0.63 | dt: 1332.18ms | tok/sec: 12,298 | bf16_mfu: 0.00 | epoch: 1 pq: 44 rg: 20 | total time: 2038.43m | eta: 2511.9m

#
epoch: 1 pq: 46 rg: 72 | total time: 2155.18m | eta: 2387.1m
step 94950/200000 (47.48%) | loss: 2.690022 | lrm: 0.60 | dt: 1322.16ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 46 rg: 74 | total time: 2156.29m | eta: 2385.9m
step 95000/200000 (47.50%) | loss: 2.701902 | lrm: 0.60 | dt: 1342.79ms | tok/sec: 12,201 | bf16_mfu: 0.00 | epoch: 1 pq: 46 rg: 76 | total time: 2157.39m | eta: 2384.7m
step 95050/200000 (47.52%) | loss: 2.737383 | lrm: 0.60 | dt: 1320.82ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 46 rg: 78 | total time: 2158.49m | eta: 2383.6m
step 95100/200000 (47.55%) | loss: 2.668731 | lrm: 0.60 | dt: 1319.56ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 46 rg: 80 | total time: 2159.59m | eta: 2382.4m
step 95150/200000 (47.58%) | loss: 2.694399 | lrm: 0.60 | dt: 1317.95ms | tok/sec: 12,431 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 0 | total time: 2160.69m | eta: 2381.2m
step 95200/200000 (47.60%) | loss: 2.715009 | lrm: 0.60 | dt: 1318.36ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 2 | total time: 2161.79m | eta: 2380.0m
step 95250/200000 (47.62%) | loss: 2.720851 | lrm: 0.60 | dt: 1321.37ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 4 | total time: 2162.89m | eta: 2378.9m
step 95300/200000 (47.65%) | loss: 2.694278 | lrm: 0.60 | dt: 1320.31ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 6 | total time: 2163.99m | eta: 2377.7m
step 95350/200000 (47.67%) | loss: 2.718189 | lrm: 0.60 | dt: 1315.08ms | tok/sec: 12,458 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 8 | total time: 2165.09m | eta: 2376.5m
step 95400/200000 (47.70%) | loss: 2.710029 | lrm: 0.60 | dt: 1318.06ms | tok/sec: 12,430 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 10 | total time: 2166.19m | eta: 2375.3m
step 95450/200000 (47.73%) | loss: 2.661109 | lrm: 0.60 | dt: 1319.92ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 12 | total time: 2167.30m | eta: 2374.2m
step 95500/200000 (47.75%) | loss: 2.718217 | lrm: 0.60 | dt: 1320.34ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 14 | total time: 2168.40m | eta: 2373.0m
step 95550/200000 (47.77%) | loss: 2.722032 | lrm: 0.60 | dt: 1321.25ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 16 | total time: 2169.50m | eta: 2371.8m
step 95600/200000 (47.80%) | loss: 2.661254 | lrm: 0.60 | dt: 1320.43ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 18 | total time: 2170.60m | eta: 2370.7m
step 95650/200000 (47.83%) | loss: 2.697715 | lrm: 0.60 | dt: 1322.94ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 20 | total time: 2171.70m | eta: 2369.5m
step 95700/200000 (47.85%) | loss: 2.661602 | lrm: 0.60 | dt: 1318.85ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 22 | total time: 2172.80m | eta: 2368.3m
step 95750/200000 (47.88%) | loss: 2.755520 | lrm: 0.60 | dt: 1320.58ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 24 | total time: 2173.90m | eta: 2367.1m
step 95800/200000 (47.90%) | loss: 2.743951 | lrm: 0.60 | dt: 1321.48ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 27 | total time: 2175.00m | eta: 2366.0m
step 95850/200000 (47.92%) | loss: 2.728775 | lrm: 0.60 | dt: 1320.06ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 29 | total time: 2176.10m | eta: 2364.8m
step 95900/200000 (47.95%) | loss: 2.639637 | lrm: 0.60 | dt: 1322.50ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 31 | total time: 2177.20m | eta: 2363.6m
step 95950/200000 (47.98%) | loss: 2.659958 | lrm: 0.60 | dt: 1322.07ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 33 | total time: 2178.30m | eta: 2362.4m
Step 96000 | Validation bpb: 0.982678
<|bos|>The capital of France is the city of Lisbon. It is divided into a cityscape which is characterized by open space and a lot of
<|bos|>The chemical symbol of gold is Au 2.9Cu 2.9H2O 2.9Ba 2.9
<|bos|>If yesterday was Friday, then tomorrow will be Friday Friday, Friday Friday, Friday Friday, Friday Friday, Sunday, S
<|bos|>The opposite of hot is cold. When you are travelling and trying to understand how to properly set up your vehicle, your body will be tempt
<|bos|>The planets of the solar system are: Mercury, Earth, Venus, Mars, Venus, Mars/Jupiter, and
<|bos|>My favorite color is black. A lot of people say it's light, but I also like a lot of it. The most common color is
<|bos|>If 5*x + 3 = 13, then x is the number of times that second-order expansion was applied to the graph. In other words, it was the number of times
2026-03-18 11:16:49,723 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_096000.pt
2026-03-18 11:16:49,724 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_096000.json
2026-03-18 11:16:51,260 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_096000_rank0.pt
step 96000/200000 (48.00%) | loss: 2.688050 | lrm: 0.60 | dt: 1578.52ms | tok/sec: 10,379 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 35 | total time: 2179.41m | eta: 2361.3m
step 96050/200000 (48.02%) | loss: 2.723200 | lrm: 0.60 | dt: 1318.47ms | tok/sec: 12,426 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 37 | total time: 2180.51m | eta: 2360.1m
step 96100/200000 (48.05%) | loss: 2.734635 | lrm: 0.60 | dt: 1322.17ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 39 | total time: 2181.61m | eta: 2358.9m
step 96150/200000 (48.08%) | loss: 2.689943 | lrm: 0.60 | dt: 1316.71ms | tok/sec: 12,443 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 41 | total time: 2182.71m | eta: 2357.8m
step 96200/200000 (48.10%) | loss: 2.637699 | lrm: 0.60 | dt: 1336.95ms | tok/sec: 12,254 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 43 | total time: 2183.81m | eta: 2356.6m
step 96250/200000 (48.12%) | loss: 2.680509 | lrm: 0.60 | dt: 1319.41ms | tok/sec: 12,417 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 45 | total time: 2184.91m | eta: 2355.4m
step 96300/200000 (48.15%) | loss: 2.672940 | lrm: 0.60 | dt: 1322.50ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 47 | total time: 2186.01m | eta: 2354.2m
step 96350/200000 (48.17%) | loss: 2.751723 | lrm: 0.60 | dt: 1319.34ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 49 | total time: 2187.12m | eta: 2353.1m
step 96400/200000 (48.20%) | loss: 2.590782 | lrm: 0.60 | dt: 1319.86ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 51 | total time: 2188.22m | eta: 2351.9m
step 96450/200000 (48.23%) | loss: 2.720526 | lrm: 0.60 | dt: 1318.75ms | tok/sec: 12,423 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 53 | total time: 2189.32m | eta: 2350.7m
step 96500/200000 (48.25%) | loss: 2.680940 | lrm: 0.60 | dt: 1322.37ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 55 | total time: 2190.42m | eta: 2349.6m
step 96550/200000 (48.27%) | loss: 2.591662 | lrm: 0.60 | dt: 1323.64ms | tok/sec: 12,378 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 57 | total time: 2191.52m | eta: 2348.4m
step 96600/200000 (48.30%) | loss: 2.705349 | lrm: 0.60 | dt: 1323.25ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 59 | total time: 2192.62m | eta: 2347.2m
step 96650/200000 (48.33%) | loss: 2.711278 | lrm: 0.60 | dt: 1322.43ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 61 | total time: 2193.72m | eta: 2346.0m
step 96700/200000 (48.35%) | loss: 2.710562 | lrm: 0.60 | dt: 1323.98ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 63 | total time: 2194.83m | eta: 2344.9m
step 96750/200000 (48.38%) | loss: 2.786051 | lrm: 0.59 | dt: 1320.07ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 65 | total time: 2195.93m | eta: 2343.7m
step 96800/200000 (48.40%) | loss: 2.640685 | lrm: 0.59 | dt: 1319.73ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 67 | total time: 2197.03m | eta: 2342.5m
step 96850/200000 (48.42%) | loss: 2.660568 | lrm: 0.59 | dt: 1320.09ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 69 | total time: 2198.13m | eta: 2341.4m
step 96900/200000 (48.45%) | loss: 2.702513 | lrm: 0.59 | dt: 1319.20ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 72 | total time: 2199.23m | eta: 2340.2m
step 96950/200000 (48.48%) | loss: 2.694457 | lrm: 0.59 | dt: 1318.26ms | tok/sec: 12,428 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 74 | total time: 2200.33m | eta: 2339.0m
step 97000/200000 (48.50%) | loss: 2.654883 | lrm: 0.59 | dt: 1319.81ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 76 | total time: 2201.43m | eta: 2337.8m
step 97050/200000 (48.52%) | loss: 2.653281 | lrm: 0.59 | dt: 1318.61ms | tok/sec: 12,425 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 78 | total time: 2202.53m | eta: 2336.7m
step 97100/200000 (48.55%) | loss: 2.695166 | lrm: 0.59 | dt: 1317.65ms | tok/sec: 12,434 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 80 | total time: 2203.63m | eta: 2335.5m
step 97150/200000 (48.58%) | loss: 2.630807 | lrm: 0.59 | dt: 1324.37ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 47 rg: 82 | total time: 2204.73m | eta: 2334.3m
step 97200/200000 (48.60%) | loss: 2.661602 | lrm: 0.59 | dt: 1320.97ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 1 | total time: 2205.83m | eta: 2333.2m
step 97250/200000 (48.62%) | loss: 2.707190 | lrm: 0.59 | dt: 1321.08ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 3 | total time: 2206.93m | eta: 2332.0m
step 97300/200000 (48.65%) | loss: 2.705135 | lrm: 0.59 | dt: 1320.62ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 5 | total time: 2208.03m | eta: 2330.8m
step 97350/200000 (48.67%) | loss: 2.689399 | lrm: 0.59 | dt: 1325.77ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 7 | total time: 2209.13m | eta: 2329.6m
step 97400/200000 (48.70%) | loss: 2.721000 | lrm: 0.59 | dt: 1320.94ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 9 | total time: 2210.23m | eta: 2328.5m
step 97450/200000 (48.73%) | loss: 2.675740 | lrm: 0.59 | dt: 1325.73ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 11 | total time: 2211.33m | eta: 2327.3m
step 97500/200000 (48.75%) | loss: 2.660138 | lrm: 0.59 | dt: 1317.38ms | tok/sec: 12,436 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 13 | total time: 2212.43m | eta: 2326.1m
step 97550/200000 (48.77%) | loss: 2.696141 | lrm: 0.59 | dt: 1319.70ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 15 | total time: 2213.53m | eta: 2325.0m
step 97600/200000 (48.80%) | loss: 2.647820 | lrm: 0.59 | dt: 1319.28ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 17 | total time: 2214.63m | eta: 2323.8m
step 97650/200000 (48.83%) | loss: 2.766643 | lrm: 0.59 | dt: 1321.30ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 19 | total time: 2215.73m | eta: 2322.6m
step 97700/200000 (48.85%) | loss: 2.668134 | lrm: 0.59 | dt: 1320.11ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 21 | total time: 2216.83m | eta: 2321.4m
step 97750/200000 (48.88%) | loss: 2.711450 | lrm: 0.59 | dt: 1323.93ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 23 | total time: 2217.93m | eta: 2320.3m
step 97800/200000 (48.90%) | loss: 2.657918 | lrm: 0.59 | dt: 1319.54ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 25 | total time: 2219.03m | eta: 2319.1m
step 97850/200000 (48.92%) | loss: 2.712108 | lrm: 0.59 | dt: 1320.44ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 27 | total time: 2220.14m | eta: 2317.9m
step 97900/200000 (48.95%) | loss: 2.672014 | lrm: 0.59 | dt: 1316.91ms | tok/sec: 12,441 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 29 | total time: 2221.24m | eta: 2316.8m
step 97950/200000 (48.98%) | loss: 2.692336 | lrm: 0.59 | dt: 1320.14ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 32 | total time: 2222.34m | eta: 2315.6m
Step 98000 | Validation bpb: 0.980714
step 98000/200000 (49.00%) | loss: 2.623237 | lrm: 0.59 | dt: 1395.89ms | tok/sec: 11,737 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 34 | total time: 2223.44m | eta: 2314.4m
step 98050/200000 (49.02%) | loss: 2.682570 | lrm: 0.59 | dt: 1321.24ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 36 | total time: 2224.54m | eta: 2313.3m
step 98100/200000 (49.05%) | loss: 2.658702 | lrm: 0.59 | dt: 1319.34ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 38 | total time: 2225.64m | eta: 2312.1m
step 98150/200000 (49.08%) | loss: 2.614166 | lrm: 0.59 | dt: 1328.08ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 40 | total time: 2226.74m | eta: 2310.9m
step 98200/200000 (49.10%) | loss: 2.661964 | lrm: 0.59 | dt: 1317.71ms | tok/sec: 12,433 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 42 | total time: 2227.84m | eta: 2309.8m
step 98250/200000 (49.12%) | loss: 2.654617 | lrm: 0.59 | dt: 1319.27ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 44 | total time: 2228.94m | eta: 2308.6m
step 98300/200000 (49.15%) | loss: 2.708731 | lrm: 0.59 | dt: 1318.65ms | tok/sec: 12,424 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 46 | total time: 2230.05m | eta: 2307.4m
step 98350/200000 (49.17%) | loss: 2.656864 | lrm: 0.59 | dt: 1322.13ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 48 | total time: 2231.15m | eta: 2306.2m
step 98400/200000 (49.20%) | loss: 2.666096 | lrm: 0.59 | dt: 1320.58ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 50 | total time: 2232.25m | eta: 2305.1m
step 98450/200000 (49.23%) | loss: 2.691760 | lrm: 0.59 | dt: 1323.90ms | tok/sec: 12,375 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 52 | total time: 2233.35m | eta: 2303.9m
step 98500/200000 (49.25%) | loss: 2.712693 | lrm: 0.59 | dt: 1319.02ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 54 | total time: 2234.45m | eta: 2302.7m
step 98550/200000 (49.27%) | loss: 2.707001 | lrm: 0.59 | dt: 1318.73ms | tok/sec: 12,424 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 56 | total time: 2235.55m | eta: 2301.6m
step 98600/200000 (49.30%) | loss: 2.618534 | lrm: 0.59 | dt: 1320.98ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 58 | total time: 2236.65m | eta: 2300.4m
step 98650/200000 (49.33%) | loss: 2.644121 | lrm: 0.58 | dt: 1319.48ms | tok/sec: 12,417 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 60 | total time: 2237.75m | eta: 2299.2m
step 98700/200000 (49.35%) | loss: 2.703335 | lrm: 0.58 | dt: 1318.35ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 62 | total time: 2238.85m | eta: 2298.1m
step 98750/200000 (49.38%) | loss: 2.663033 | lrm: 0.58 | dt: 1320.49ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 64 | total time: 2239.95m | eta: 2296.9m
step 98800/200000 (49.40%) | loss: 2.656475 | lrm: 0.58 | dt: 1318.98ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 66 | total time: 2241.05m | eta: 2295.7m
step 98850/200000 (49.42%) | loss: 2.664022 | lrm: 0.58 | dt: 1320.71ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 68 | total time: 2242.15m | eta: 2294.6m
step 98900/200000 (49.45%) | loss: 2.680080 | lrm: 0.58 | dt: 1319.88ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 70 | total time: 2243.25m | eta: 2293.4m
step 98950/200000 (49.48%) | loss: 2.592200 | lrm: 0.58 | dt: 1321.87ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 72 | total time: 2244.36m | eta: 2292.2m
step 99000/200000 (49.50%) | loss: 2.654505 | lrm: 0.58 | dt: 1319.52ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 74 | total time: 2245.46m | eta: 2291.0m
step 99050/200000 (49.52%) | loss: 2.623659 | lrm: 0.58 | dt: 1318.90ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 76 | total time: 2246.56m | eta: 2289.9m
step 99100/200000 (49.55%) | loss: 2.700104 | lrm: 0.58 | dt: 1319.97ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 78 | total time: 2247.66m | eta: 2288.7m
step 99150/200000 (49.58%) | loss: 2.731060 | lrm: 0.58 | dt: 1322.11ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 48 rg: 81 | total time: 2248.76m | eta: 2287.5m
step 99200/200000 (49.60%) | loss: 2.592692 | lrm: 0.58 | dt: 1319.75ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 0 | total time: 2249.86m | eta: 2286.4m
step 99250/200000 (49.62%) | loss: 2.656190 | lrm: 0.58 | dt: 1322.43ms | tok/sec: 12,389 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 2 | total time: 2250.96m | eta: 2285.2m
step 99300/200000 (49.65%) | loss: 2.616259 | lrm: 0.58 | dt: 1317.85ms | tok/sec: 12,432 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 4 | total time: 2252.06m | eta: 2284.0m
step 99350/200000 (49.67%) | loss: 2.653876 | lrm: 0.58 | dt: 1318.89ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 6 | total time: 2253.16m | eta: 2282.9m
step 99400/200000 (49.70%) | loss: 2.645021 | lrm: 0.58 | dt: 1323.98ms | tok/sec: 12,374 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 8 | total time: 2254.26m | eta: 2281.7m
step 99450/200000 (49.73%) | loss: 2.634250 | lrm: 0.58 | dt: 1326.04ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 10 | total time: 2255.36m | eta: 2280.5m
step 99500/200000 (49.75%) | loss: 2.670321 | lrm: 0.58 | dt: 1320.81ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 12 | total time: 2256.46m | eta: 2279.4m
step 99550/200000 (49.77%) | loss: 2.704648 | lrm: 0.58 | dt: 1321.82ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 14 | total time: 2257.56m | eta: 2278.2m
step 99600/200000 (49.80%) | loss: 2.671715 | lrm: 0.58 | dt: 1320.06ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 16 | total time: 2258.66m | eta: 2277.0m
step 99650/200000 (49.83%) | loss: 2.754731 | lrm: 0.58 | dt: 1321.97ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 18 | total time: 2259.76m | eta: 2275.9m
step 99700/200000 (49.85%) | loss: 2.638520 | lrm: 0.58 | dt: 1318.00ms | tok/sec: 12,430 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 20 | total time: 2260.86m | eta: 2274.7m
step 99750/200000 (49.88%) | loss: 2.671485 | lrm: 0.58 | dt: 1322.06ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 22 | total time: 2261.97m | eta: 2273.5m
step 99800/200000 (49.90%) | loss: 2.719321 | lrm: 0.58 | dt: 1316.00ms | tok/sec: 12,449 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 24 | total time: 2263.07m | eta: 2272.4m
step 99850/200000 (49.92%) | loss: 2.666615 | lrm: 0.58 | dt: 1321.57ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 26 | total time: 2264.17m | eta: 2271.2m
step 99900/200000 (49.95%) | loss: 2.646689 | lrm: 0.58 | dt: 1328.05ms | tok/sec: 12,336 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 28 | total time: 2265.27m | eta: 2270.0m
step 99950/200000 (49.98%) | loss: 2.632246 | lrm: 0.58 | dt: 1322.51ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 30 | total time: 2266.37m | eta: 2268.9m
Step 100000 | `Validation bpb: 0.979477`
<|bos|>The capital of France is the city of Merre de Necantes. The city is also known as the French city of Merre
<|bos|>The chemical symbol of gold is gold, symbolized as gold/gold; the symbol of silver is silver/gold; gold/gold is
<|bos|>If yesterday was Friday, then tomorrow will be Friday evening, Sunday, July 15. (Heavy snow will be there in the morning) You
<|bos|>`The opposite of hot is cold`. This is a natural state of the body that is often thought of as a state of heat and cold. Hot
<|bos|>`The planets of the solar system are: Mars, Jupiter, Saturn, Jupiter,` Pluto, Europa, and
<|bos|>`My favorite color is blue.` If you're trying to change the color of your blue pillow, you should try changing your blue pillow's
<|bos|>If 5*x + 3 = 13, then x is 13/5, because if 5*x + 3 = 13 then 5/5 is 13
2026-03-18 12:45:24,146 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_100000.pt
2026-03-18 12:45:24,147 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_100000.json
2026-03-18 12:45:25,706 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_100000_rank0.pt
step 100000/200000 (50.00%) | loss: 2.722186 | lrm: 0.58 | dt: 1540.21ms | tok/sec: 10,637 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 32 | total time: 2267.48m | eta: 2267.7m
step 100050/200000 (50.02%) | loss: 2.649500 | lrm: 0.58 | dt: 1319.68ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 34 | total time: 2268.58m | eta: 2266.5m
step 100100/200000 (50.05%) | loss: 2.617892 | lrm: 0.58 | dt: 1324.10ms | tok/sec: 12,373 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 36 | total time: 2269.68m | eta: 2265.4m
step 100150/200000 (50.08%) | loss: 2.710699 | lrm: 0.58 | dt: 1320.30ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 38 | total time: 2270.78m | eta: 2264.2m
step 100200/200000 (50.10%) | loss: 2.672865 | lrm: 0.58 | dt: 1323.27ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 40 | total time: 2271.88m | eta: 2263.0m
step 100250/200000 (50.12%) | loss: 2.666024 | lrm: 0.58 | dt: 1322.93ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 42 | total time: 2272.99m | eta: 2261.9m
step 100300/200000 (50.15%) | loss: 2.552261 | lrm: 0.58 | dt: 1321.10ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 44 | total time: 2274.09m | eta: 2260.7m
step 100350/200000 (50.17%) | loss: 2.693552 | lrm: 0.58 | dt: 1323.27ms | tok/sec: 12,381 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 46 | total time: 2275.19m | eta: 2259.5m
step 100400/200000 (50.20%) | loss: 2.650414 | lrm: 0.58 | dt: 1321.10ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 48 | total time: 2276.29m | eta: 2258.4m
step 100450/200000 (50.23%) | loss: 2.635786 | lrm: 0.58 | dt: 1320.58ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 51 | total time: 2277.39m | eta: 2257.2m
step 100500/200000 (50.25%) | loss: 2.626880 | lrm: 0.58 | dt: 1321.63ms | tok/sec: 12,396 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 53 | total time: 2278.49m | eta: 2256.0m
step 100550/200000 (50.27%) | loss: 2.746442 | lrm: 0.57 | dt: 1318.58ms | tok/sec: 12,425 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 55 | total time: 2279.59m | eta: 2254.9m
step 100600/200000 (50.30%) | loss: 2.682345 | lrm: 0.57 | dt: 1321.87ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 57 | total time: 2280.69m | eta: 2253.7m
step 100650/200000 (50.33%) | loss: 2.712415 | lrm: 0.57 | dt: 1320.49ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 59 | total time: 2281.79m | eta: 2252.5m
step 100700/200000 (50.35%) | loss: 2.669290 | lrm: 0.57 | dt: 1319.65ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 61 | total time: 2282.89m | eta: 2251.4m
step 100750/200000 (50.38%) | loss: 2.669175 | lrm: 0.57 | dt: 1319.64ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 63 | total time: 2283.99m | eta: 2250.2m
step 100800/200000 (50.40%) | loss: 2.642975 | lrm: 0.57 | dt: 1325.15ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 65 | total time: 2285.09m | eta: 2249.0m
step 100850/200000 (50.42%) | loss: 2.642365 | lrm: 0.57 | dt: 1321.99ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 67 | total time: 2286.19m | eta: 2247.9m
step 100900/200000 (50.45%) | loss: 2.661013 | lrm: 0.57 | dt: 1324.19ms | tok/sec: 12,372 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 69 | total time: 2287.29m | eta: 2246.7m
step 100950/200000 (50.48%) | loss: 2.676998 | lrm: 0.57 | dt: 1325.09ms | tok/sec: 12,364 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 71 | total time: 2288.39m | eta: 2245.5m
step 101000/200000 (50.50%) | loss: 2.733892 | lrm: 0.57 | dt: 1317.91ms | tok/sec: 12,431 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 73 | total time: 2289.49m | eta: 2244.4m
step 101050/200000 (50.52%) | loss: 2.708748 | lrm: 0.57 | dt: 1319.17ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 75 | total time: 2290.59m | eta: 2243.2m
step 101100/200000 (50.55%) | loss: 2.678375 | lrm: 0.57 | dt: 1318.15ms | tok/sec: 12,429 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 77 | total time: 2291.70m | eta: 2242.0m
step 101150/200000 (50.58%) | loss: 2.653501 | lrm: 0.57 | dt: 1324.33ms | tok/sec: 12,371 | bf16_mfu: 0.00 | epoch: 1 pq: 49 rg: 79 | total time: 2292.80m | eta: 2240.9m
step 101200/200000 (50.60%) | loss: 2.672319 | lrm: 0.57 | dt: 1319.74ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 0 | total time: 2293.90m | eta: 2239.7m
step 101250/200000 (50.62%) | loss: 2.671291 | lrm: 0.57 | dt: 1321.28ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 2 | total time: 2295.00m | eta: 2238.6m
step 101300/200000 (50.65%) | loss: 2.762534 | lrm: 0.57 | dt: 1322.20ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 4 | total time: 2296.10m | eta: 2237.4m
step 101350/200000 (50.67%) | loss: 2.613413 | lrm: 0.57 | dt: 1326.14ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 6 | total time: 2297.20m | eta: 2236.2m
step 101400/200000 (50.70%) | loss: 2.718783 | lrm: 0.57 | dt: 1322.15ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 8 | total time: 2298.30m | eta: 2235.1m
step 101450/200000 (50.73%) | loss: 2.680731 | lrm: 0.57 | dt: 1321.45ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 10 | total time: 2299.40m | eta: 2233.9m
step 101500/200000 (50.75%) | loss: 2.741212 | lrm: 0.57 | dt: 1319.74ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 12 | total time: 2300.50m | eta: 2232.7m
step 101550/200000 (50.77%) | loss: 2.709682 | lrm: 0.57 | dt: 1320.53ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 14 | total time: 2301.60m | eta: 2231.6m
step 101600/200000 (50.80%) | loss: 2.649801 | lrm: 0.57 | dt: 1320.71ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 17 | total time: 2302.70m | eta: 2230.4m
step 101650/200000 (50.83%) | loss: 2.704166 | lrm: 0.57 | dt: 1318.99ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 19 | total time: 2303.80m | eta: 2229.2m
step 101700/200000 (50.85%) | loss: 2.700607 | lrm: 0.57 | dt: 1348.13ms | tok/sec: 12,153 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 21 | total time: 2304.91m | eta: 2228.1m
step 101750/200000 (50.88%) | loss: 2.661403 | lrm: 0.57 | dt: 1321.35ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 23 | total time: 2306.01m | eta: 2226.9m
step 101800/200000 (50.90%) | loss: 2.649048 | lrm: 0.57 | dt: 1323.19ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 25 | total time: 2307.11m | eta: 2225.7m
step 101850/200000 (50.92%) | loss: 2.625185 | lrm: 0.57 | dt: 1320.12ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 27 | total time: 2308.21m | eta: 2224.6m
step 101900/200000 (50.95%) | loss: 2.742452 | lrm: 0.57 | dt: 1320.87ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 29 | total time: 2309.31m | eta: 2223.4m
step 101950/200000 (50.98%) | loss: 2.688484 | lrm: 0.57 | dt: 1322.72ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 31 | total time: 2310.41m | eta: 2222.2m
Step 102000 | Validation bpb: 0.977421
step 102000/200000 (51.00%) | loss: 2.646568 | lrm: 0.57 | dt: 1392.04ms | tok/sec: 11,769 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 33 | total time: 2311.51m | eta: 2221.1m
step 102050/200000 (51.02%) | loss: 2.629371 | lrm: 0.57 | dt: 1320.98ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 35 | total time: 2312.61m | eta: 2219.9m
step 102100/200000 (51.05%) | loss: 2.678480 | lrm: 0.57 | dt: 1320.69ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 37 | total time: 2313.71m | eta: 2218.8m
step 102150/200000 (51.08%) | loss: 2.630000 | lrm: 0.57 | dt: 1323.01ms | tok/sec: 12,383 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 39 | total time: 2314.81m | eta: 2217.6m
step 102200/200000 (51.10%) | loss: 2.672404 | lrm: 0.57 | dt: 1317.05ms | tok/sec: 12,439 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 41 | total time: 2315.92m | eta: 2216.4m
step 102250/200000 (51.12%) | loss: 2.667024 | lrm: 0.57 | dt: 1321.34ms | tok/sec: 12,399 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 43 | total time: 2317.02m | eta: 2215.3m
step 102300/200000 (51.15%) | loss: 2.625436 | lrm: 0.57 | dt: 1320.37ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 45 | total time: 2318.12m | eta: 2214.1m
step 102350/200000 (51.17%) | loss: 2.608799 | lrm: 0.57 | dt: 1319.15ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 47 | total time: 2319.22m | eta: 2212.9m
step 102400/200000 (51.20%) | loss: 2.701892 | lrm: 0.57 | dt: 1318.95ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 49 | total time: 2320.32m | eta: 2211.8m
step 102450/200000 (51.23%) | loss: 2.670526 | lrm: 0.56 | dt: 1321.42ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 51 | total time: 2321.42m | eta: 2210.6m
step 102500/200000 (51.25%) | loss: 2.681698 | lrm: 0.56 | dt: 1318.62ms | tok/sec: 12,425 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 53 | total time: 2322.52m | eta: 2209.4m
step 102550/200000 (51.27%) | loss: 2.629730 | lrm: 0.56 | dt: 1319.85ms | tok/sec: 12,413 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 55 | total time: 2323.62m | eta: 2208.3m
step 102600/200000 (51.30%) | loss: 2.708279 | lrm: 0.56 | dt: 1318.24ms | tok/sec: 12,428 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 57 | total time: 2324.72m | eta: 2207.1m
step 102650/200000 (51.33%) | loss: 2.700082 | lrm: 0.56 | dt: 1322.49ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 59 | total time: 2325.82m | eta: 2205.9m
step 102700/200000 (51.35%) | loss: 2.701001 | lrm: 0.56 | dt: 1322.24ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 61 | total time: 2326.92m | eta: 2204.8m
step 102750/200000 (51.38%) | loss: 2.689077 | lrm: 0.56 | dt: 1321.59ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 63 | total time: 2328.02m | eta: 2203.6m
step 102800/200000 (51.40%) | loss: 2.716202 | lrm: 0.56 | dt: 1321.17ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 65 | total time: 2329.12m | eta: 2202.5m
step 102850/200000 (51.42%) | loss: 2.643968 | lrm: 0.56 | dt: 1326.19ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 67 | total time: 2330.22m | eta: 2201.3m
step 102900/200000 (51.45%) | loss: 2.655334 | lrm: 0.56 | dt: 1337.87ms | tok/sec: 12,246 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 70 | total time: 2331.32m | eta: 2200.1m
step 102950/200000 (51.48%) | loss: 2.638548 | lrm: 0.56 | dt: 1319.63ms | tok/sec: 12,415 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 72 | total time: 2332.42m | eta: 2199.0m
step 103000/200000 (51.50%) | loss: 2.695345 | lrm: 0.56 | dt: 1325.44ms | tok/sec: 12,361 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 74 | total time: 2333.52m | eta: 2197.8m
step 103050/200000 (51.52%) | loss: 2.647903 | lrm: 0.56 | dt: 1319.98ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 76 | total time: 2334.62m | eta: 2196.6m
step 103100/200000 (51.55%) | loss: 2.641477 | lrm: 0.56 | dt: 1319.37ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 78 | total time: 2335.72m | eta: 2195.5m
step 103150/200000 (51.58%) | loss: 2.638097 | lrm: 0.56 | dt: 1320.13ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 50 rg: 80 | total time: 2336.82m | eta: 2194.3m
step 103200/200000 (51.60%) | loss: 2.557949 | lrm: 0.56 | dt: 1321.13ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 0 | total time: 2337.93m | eta: 2193.2m
step 103250/200000 (51.62%) | loss: 2.681385 | lrm: 0.56 | dt: 1320.93ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 2 | total time: 2339.03m | eta: 2192.0m
step 103300/200000 (51.65%) | loss: 2.631632 | lrm: 0.56 | dt: 1317.94ms | tok/sec: 12,431 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 4 | total time: 2340.13m | eta: 2190.8m
step 103350/200000 (51.67%) | loss: 2.631811 | lrm: 0.56 | dt: 1321.09ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 6 | total time: 2341.23m | eta: 2189.7m
step 103400/200000 (51.70%) | loss: 2.725666 | lrm: 0.56 | dt: 1318.47ms | tok/sec: 12,426 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 8 | total time: 2342.33m | eta: 2188.5m
step 103450/200000 (51.73%) | loss: 2.717354 | lrm: 0.56 | dt: 1317.44ms | tok/sec: 12,436 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 10 | total time: 2343.43m | eta: 2187.3m
step 103500/200000 (51.75%) | loss: 2.585522 | lrm: 0.56 | dt: 1316.00ms | tok/sec: 12,449 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 12 | total time: 2344.53m | eta: 2186.2m
step 103550/200000 (51.77%) | loss: 2.632027 | lrm: 0.56 | dt: 1319.29ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 14 | total time: 2345.63m | eta: 2185.0m
step 103600/200000 (51.80%) | loss: 2.657585 | lrm: 0.56 | dt: 1319.03ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 16 | total time: 2346.73m | eta: 2183.8m
step 103650/200000 (51.83%) | loss: 2.654464 | lrm: 0.56 | dt: 1319.48ms | tok/sec: 12,417 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 18 | total time: 2347.83m | eta: 2182.7m
step 103700/200000 (51.85%) | loss: 2.736541 | lrm: 0.56 | dt: 1320.84ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 20 | total time: 2348.93m | eta: 2181.5m
step 103750/200000 (51.88%) | loss: 2.633654 | lrm: 0.56 | dt: 1316.18ms | tok/sec: 12,448 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 22 | total time: 2350.03m | eta: 2180.4m
step 103800/200000 (51.90%) | loss: 2.658818 | lrm: 0.56 | dt: 1317.94ms | tok/sec: 12,431 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 24 | total time: 2351.13m | eta: 2179.2m
step 103850/200000 (51.92%) | loss: 2.750917 | lrm: 0.56 | dt: 1318.87ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 26 | total time: 2352.23m | eta: 2178.0m
step 103900/200000 (51.95%) | loss: 2.669822 | lrm: 0.56 | dt: 1318.58ms | tok/sec: 12,425 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 28 | total time: 2353.33m | eta: 2176.9m
step 103950/200000 (51.98%) | loss: 2.655338 | lrm: 0.56 | dt: 1322.48ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 30 | total time: 2354.43m | eta: 2175.7m
Step 104000 | Validation bpb: 0.976156
<|bos|>T`he capital of France is Paris,` which has the world's largest pine forest, at 12,300 square miles (30,00
<|bos|>The chemical symbol of gold is Kg.

The atomic number of gold is 92, so it has 92 protons, 92 electrons,
<|bos|>If yesterday was Friday, then tomorrow will be Friday. Fridays can be so chaotic, in some ways, that it seems that a lot of people
<|bos|>`The opposite of hot is cold,` so the heat will only be 70% of the radiation in the room, and therefore will be 20%
<|bos|>T`he planets of the solar system are: Jupiter, Neptune, Saturn, and Uranus`. It is important for the astronomers
<|bos|>`My favorite color is blue.` There are a lot of blue toys around, and they aren't always the perfect color. It's a bit difficult
<|bos|>If 5*x + 3 = 13, then x is 13*x. (I would multiply that number by 13 to get x*. So, X +
2026-03-18 14:13:57,581 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_104000.pt
2026-03-18 14:13:57,582 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_104000.json
2026-03-18 14:13:59,099 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_104000_rank0.pt
step 104000/200000 (52.00%) | loss: 2.617082 | lrm: 0.56 | dt: 1523.97ms | tok/sec: 10,750 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 32 | total time: 2355.54m | eta: 2174.6m
step 104050/200000 (52.02%) | loss: 2.680017 | lrm: 0.56 | dt: 1323.77ms | tok/sec: 12,376 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 34 | total time: 2356.64m | eta: 2173.4m
step 104100/200000 (52.05%) | loss: 2.695480 | lrm: 0.56 | dt: 1320.05ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 36 | total time: 2357.74m | eta: 2172.2m
step 104150/200000 (52.08%) | loss: 2.629905 | lrm: 0.56 | dt: 1322.29ms | tok/sec: 12,390 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 38 | total time: 2358.84m | eta: 2171.1m
step 104200/200000 (52.10%) | loss: 2.705942 | lrm: 0.56 | dt: 1320.75ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 41 | total time: 2359.94m | eta: 2169.9m
step 104250/200000 (52.12%) | loss: 2.655193 | lrm: 0.56 | dt: 1319.10ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 43 | total time: 2361.04m | eta: 2168.7m
step 104300/200000 (52.15%) | loss: 2.596784 | lrm: 0.56 | dt: 1318.13ms | tok/sec: 12,429 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 45 | total time: 2362.14m | eta: 2167.6m
step 104350/200000 (52.17%) | loss: 2.693733 | lrm: 0.55 | dt: 1317.19ms | tok/sec: 12,438 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 47 | total time: 2363.24m | eta: 2166.4m
step 104400/200000 (52.20%) | loss: 2.689865 | lrm: 0.55 | dt: 1319.97ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 49 | total time: 2364.34m | eta: 2165.3m
step 104450/200000 (52.23%) | loss: 2.725174 | lrm: 0.55 | dt: 1320.38ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 51 | total time: 2365.44m | eta: 2164.1m
step 104500/200000 (52.25%) | loss: 2.604546 | lrm: 0.55 | dt: 1320.65ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 53 | total time: 2366.54m | eta: 2162.9m
step 104550/200000 (52.27%) | loss: 2.653740 | lrm: 0.55 | dt: 1320.23ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 55 | total time: 2367.65m | eta: 2161.8m
step 104600/200000 (52.30%) | loss: 2.638478 | lrm: 0.55 | dt: 1320.28ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 57 | total time: 2368.75m | eta: 2160.6m
step 104650/200000 (52.33%) | loss: 2.703743 | lrm: 0.55 | dt: 1317.52ms | tok/sec: 12,435 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 59 | total time: 2369.85m | eta: 2159.5m
step 104700/200000 (52.35%) | loss: 2.656243 | lrm: 0.55 | dt: 1317.66ms | tok/sec: 12,434 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 61 | total time: 2370.95m | eta: 2158.3m
step 104750/200000 (52.38%) | loss: 2.642705 | lrm: 0.55 | dt: 1319.94ms | tok/sec: 12,412 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 63 | total time: 2372.05m | eta: 2157.1m
step 104800/200000 (52.40%) | loss: 2.641423 | lrm: 0.55 | dt: 1319.73ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 65 | total time: 2373.15m | eta: 2156.0m
step 104850/200000 (52.42%) | loss: 2.669853 | lrm: 0.55 | dt: 1319.10ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 67 | total time: 2374.25m | eta: 2154.8m
step 104900/200000 (52.45%) | loss: 2.628774 | lrm: 0.55 | dt: 1320.30ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 69 | total time: 2375.35m | eta: 2153.6m
step 104950/200000 (52.48%) | loss: 2.677592 | lrm: 0.55 | dt: 1319.74ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 71 | total time: 2376.45m | eta: 2152.5m
step 105000/200000 (52.50%) | loss: 2.724854 | lrm: 0.55 | dt: 1339.67ms | tok/sec: 12,229 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 73 | total time: 2377.55m | eta: 2151.3m
step 105050/200000 (52.52%) | loss: 2.679073 | lrm: 0.55 | dt: 1326.74ms | tok/sec: 12,349 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 75 | total time: 2378.65m | eta: 2150.2m
step 105100/200000 (52.55%) | loss: 2.632809 | lrm: 0.55 | dt: 1321.94ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 77 | total time: 2379.75m | eta: 2149.0m
step 105150/200000 (52.58%) | loss: 2.698156 | lrm: 0.55 | dt: 1320.14ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 79 | total time: 2380.85m | eta: 2147.8m
step 105200/200000 (52.60%) | loss: 2.624181 | lrm: 0.55 | dt: 1318.23ms | tok/sec: 12,428 | bf16_mfu: 0.00 | epoch: 1 pq: 51 rg: 81 | total time: 2381.95m | eta: 2146.7m
step 105250/200000 (52.62%) | loss: 2.635643 | lrm: 0.55 | dt: 1319.02ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 2 | total time: 2383.06m | eta: 2145.5m
step 105300/200000 (52.65%) | loss: 2.723589 | lrm: 0.55 | dt: 1320.19ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 4 | total time: 2384.16m | eta: 2144.4m
step 105350/200000 (52.67%) | loss: 2.706450 | lrm: 0.55 | dt: 1321.93ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 6 | total time: 2385.26m | eta: 2143.2m
step 105400/200000 (52.70%) | loss: 2.704130 | lrm: 0.55 | dt: 1315.69ms | tok/sec: 12,452 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 8 | total time: 2386.36m | eta: 2142.0m
step 105450/200000 (52.73%) | loss: 2.602376 | lrm: 0.55 | dt: 1335.41ms | tok/sec: 12,268 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 10 | total time: 2387.46m | eta: 2140.9m
step 105500/200000 (52.75%) | loss: 2.585192 | lrm: 0.55 | dt: 1323.12ms | tok/sec: 12,382 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 12 | total time: 2388.56m | eta: 2139.7m
step 105550/200000 (52.77%) | loss: 2.670757 | lrm: 0.55 | dt: 1325.33ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 14 | total time: 2389.68m | eta: 2138.6m
step 105600/200000 (52.80%) | loss: 2.636386 | lrm: 0.55 | dt: 1326.00ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 16 | total time: 2390.78m | eta: 2137.4m
step 105650/200000 (52.83%) | loss: 2.616928 | lrm: 0.55 | dt: 1326.37ms | tok/sec: 12,352 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 18 | total time: 2391.89m | eta: 2136.3m
step 105700/200000 (52.85%) | loss: 2.627573 | lrm: 0.55 | dt: 1317.37ms | tok/sec: 12,436 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 20 | total time: 2392.99m | eta: 2135.1m
step 105750/200000 (52.88%) | loss: 2.663290 | lrm: 0.55 | dt: 1321.11ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 22 | total time: 2394.09m | eta: 2133.9m
step 105800/200000 (52.90%) | loss: 2.591221 | lrm: 0.55 | dt: 1320.25ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 24 | total time: 2395.19m | eta: 2132.8m
step 105850/200000 (52.92%) | loss: 2.678720 | lrm: 0.55 | dt: 1318.17ms | tok/sec: 12,429 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 26 | total time: 2396.29m | eta: 2131.6m
step 105900/200000 (52.95%) | loss: 2.613076 | lrm: 0.55 | dt: 1322.22ms | tok/sec: 12,391 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 28 | total time: 2397.40m | eta: 2130.5m
step 105950/200000 (52.98%) | loss: 2.639534 | lrm: 0.55 | dt: 1323.42ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 30 | total time: 2398.50m | eta: 2129.3m
Step 106000 | Validation bpb: 0.974751
step 106000/200000 (53.00%) | loss: 2.683584 | lrm: 0.55 | dt: 1399.18ms | tok/sec: 11,709 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 32 | total time: 2399.60m | eta: 2128.1m
step 106050/200000 (53.02%) | loss: 2.652667 | lrm: 0.55 | dt: 1318.07ms | tok/sec: 12,430 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 34 | total time: 2400.70m | eta: 2127.0m
step 106100/200000 (53.05%) | loss: 2.685527 | lrm: 0.55 | dt: 1316.00ms | tok/sec: 12,449 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 36 | total time: 2401.80m | eta: 2125.8m
step 106150/200000 (53.08%) | loss: 2.687738 | lrm: 0.55 | dt: 1321.79ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 38 | total time: 2402.90m | eta: 2124.7m
step 106200/200000 (53.10%) | loss: 2.681034 | lrm: 0.55 | dt: 1318.85ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 40 | total time: 2404.00m | eta: 2123.5m
step 106250/200000 (53.12%) | loss: 2.674991 | lrm: 0.54 | dt: 1321.92ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 42 | total time: 2405.10m | eta: 2122.4m
step 106300/200000 (53.15%) | loss: 2.624359 | lrm: 0.54 | dt: 1320.04ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 44 | total time: 2406.20m | eta: 2121.2m
step 106350/200000 (53.17%) | loss: 2.594769 | lrm: 0.54 | dt: 1321.10ms | tok/sec: 12,401 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 46 | total time: 2407.30m | eta: 2120.0m
step 106400/200000 (53.20%) | loss: 2.615454 | lrm: 0.54 | dt: 1322.74ms | tok/sec: 12,386 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 48 | total time: 2408.40m | eta: 2118.9m
step 106450/200000 (53.23%) | loss: 2.632035 | lrm: 0.54 | dt: 1320.03ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 50 | total time: 2409.51m | eta: 2117.7m
step 106500/200000 (53.25%) | loss: 2.720057 | lrm: 0.54 | dt: 1323.32ms | tok/sec: 12,380 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 53 | total time: 2410.61m | eta: 2116.6m
step 106550/200000 (53.27%) | loss: 2.716675 | lrm: 0.54 | dt: 1319.12ms | tok/sec: 12,420 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 55 | total time: 2411.71m | eta: 2115.4m
step 106600/200000 (53.30%) | loss: 2.610546 | lrm: 0.54 | dt: 1320.51ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 57 | total time: 2412.80m | eta: 2114.2m
step 106650/200000 (53.33%) | loss: 2.705486 | lrm: 0.54 | dt: 1317.95ms | tok/sec: 12,431 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 59 | total time: 2413.90m | eta: 2113.1m
step 106700/200000 (53.35%) | loss: 2.653458 | lrm: 0.54 | dt: 1320.43ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 61 | total time: 2415.01m | eta: 2111.9m
step 106750/200000 (53.38%) | loss: 2.664237 | lrm: 0.54 | dt: 1318.18ms | tok/sec: 12,429 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 63 | total time: 2416.11m | eta: 2110.8m
step 106800/200000 (53.40%) | loss: 2.702749 | lrm: 0.54 | dt: 1320.89ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 65 | total time: 2417.21m | eta: 2109.6m
step 106850/200000 (53.42%) | loss: 2.693693 | lrm: 0.54 | dt: 1318.77ms | tok/sec: 12,423 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 67 | total time: 2418.31m | eta: 2108.4m
step 106900/200000 (53.45%) | loss: 2.652222 | lrm: 0.54 | dt: 1321.55ms | tok/sec: 12,397 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 69 | total time: 2419.41m | eta: 2107.3m
step 106950/200000 (53.48%) | loss: 2.737395 | lrm: 0.54 | dt: 1319.27ms | tok/sec: 12,418 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 71 | total time: 2420.51m | eta: 2106.1m
step 107000/200000 (53.50%) | loss: 2.670499 | lrm: 0.54 | dt: 1320.72ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 73 | total time: 2421.62m | eta: 2105.0m
step 107050/200000 (53.52%) | loss: 2.733811 | lrm: 0.54 | dt: 1320.28ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 75 | total time: 2422.72m | eta: 2103.8m
step 107100/200000 (53.55%) | loss: 2.613682 | lrm: 0.54 | dt: 1320.77ms | tok/sec: 12,404 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 77 | total time: 2423.82m | eta: 2102.7m
step 107150/200000 (53.58%) | loss: 2.672737 | lrm: 0.54 | dt: 1320.51ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 52 rg: 79 | total time: 2424.92m | eta: 2101.5m
step 107200/200000 (53.60%) | loss: 2.721889 | lrm: 0.54 | dt: 1317.13ms | tok/sec: 12,439 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 0 | total time: 2426.02m | eta: 2100.3m
step 107250/200000 (53.62%) | loss: 2.669206 | lrm: 0.54 | dt: 1316.85ms | tok/sec: 12,441 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 2 | total time: 2427.12m | eta: 2099.2m
step 107300/200000 (53.65%) | loss: 2.616314 | lrm: 0.54 | dt: 1316.34ms | tok/sec: 12,446 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 4 | total time: 2428.22m | eta: 2098.0m
step 107350/200000 (53.67%) | loss: 2.643846 | lrm: 0.54 | dt: 1318.32ms | tok/sec: 12,427 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 6 | total time: 2429.32m | eta: 2096.9m
step 107400/200000 (53.70%) | loss: 2.642778 | lrm: 0.54 | dt: 1317.57ms | tok/sec: 12,434 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 8 | total time: 2430.42m | eta: 2095.7m
step 107450/200000 (53.73%) | loss: 2.677397 | lrm: 0.54 | dt: 1317.00ms | tok/sec: 12,440 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 10 | total time: 2431.52m | eta: 2094.5m
step 107500/200000 (53.75%) | loss: 2.700081 | lrm: 0.54 | dt: 1319.22ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 12 | total time: 2432.62m | eta: 2093.4m
step 107550/200000 (53.77%) | loss: 2.661670 | lrm: 0.54 | dt: 1321.98ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 14 | total time: 2433.72m | eta: 2092.2m
step 107600/200000 (53.80%) | loss: 2.628536 | lrm: 0.54 | dt: 1318.98ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 16 | total time: 2434.82m | eta: 2091.1m
step 107650/200000 (53.83%) | loss: 2.674387 | lrm: 0.54 | dt: 1321.42ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 19 | total time: 2435.92m | eta: 2089.9m
step 107700/200000 (53.85%) | loss: 2.651070 | lrm: 0.54 | dt: 1323.45ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 21 | total time: 2437.02m | eta: 2088.7m
step 107750/200000 (53.88%) | loss: 2.601946 | lrm: 0.54 | dt: 1319.40ms | tok/sec: 12,417 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 23 | total time: 2438.12m | eta: 2087.6m
step 107800/200000 (53.90%) | loss: 2.700768 | lrm: 0.54 | dt: 1322.03ms | tok/sec: 12,393 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 25 | total time: 2439.22m | eta: 2086.4m
step 107850/200000 (53.92%) | loss: 2.659229 | lrm: 0.54 | dt: 1321.80ms | tok/sec: 12,395 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 27 | total time: 2440.32m | eta: 2085.3m
step 107900/200000 (53.95%) | loss: 2.631604 | lrm: 0.54 | dt: 1321.45ms | tok/sec: 12,398 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 29 | total time: 2441.42m | eta: 2084.1m
step 107950/200000 (53.98%) | loss: 2.643393 | lrm: 0.54 | dt: 1318.79ms | tok/sec: 12,423 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 31 | total time: 2442.52m | eta: 2083.0m
Step 108000 | Validation bpb: 0.972945
<|bos|>The capital of France is the city of Ermendes, now known as the Netherlands. Its capital is Dartmouth,
<|bos|>The chemical symbol of gold is K, so it's a gold or very dark gold. Gold is usually very hard and, for that reason, very m
<|bos|>`If yesterday was Friday, then tomorrow will be Saturday.` This is when all the plants will be gone and the earth will be planted and we'll be done with
<|bos|>`The opposite of hot is cold`. You can feel hot or cold. Warm is hot, cold is cold, and it feels hot. If you
<|bos|>`The planets of the solar system are: Mars, Jupiter, Saturn, and Uranus`
The solar system is made of planets of
<|bos|>`My favorite color is blue`. blue is a blue pigment that is a component of my RAW color palette and my favorite color is
<|bos|>If 5*x + 3 = 13, then x is the number of times that the person should reach the end of the list. Therefore, we can find 13 times that person
2026-03-18 15:42:32,918 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_108000.pt
2026-03-18 15:42:32,919 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_108000.json
2026-03-18 15:42:34,455 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_108000_rank0.pt
step 108000/200000 (54.00%) | loss: 2.680305 | lrm: 0.54 | dt: 1536.63ms | tok/sec: 10,662 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 33 | total time: 2443.62m | eta: 2081.8m
step 108050/200000 (54.02%) | loss: 2.678974 | lrm: 0.54 | dt: 1319.49ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 35 | total time: 2444.72m | eta: 2080.6m
step 108100/200000 (54.05%) | loss: 2.708640 | lrm: 0.54 | dt: 1322.60ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 37 | total time: 2445.82m | eta: 2079.5m
step 108150/200000 (54.08%) | loss: 2.691541 | lrm: 0.53 | dt: 1320.34ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 39 | total time: 2446.92m | eta: 2078.3m
step 108200/200000 (54.10%) | loss: 2.672532 | lrm: 0.53 | dt: 1318.79ms | tok/sec: 12,423 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 41 | total time: 2448.02m | eta: 2077.2m
step 108250/200000 (54.12%) | loss: 2.610811 | lrm: 0.53 | dt: 1317.29ms | tok/sec: 12,437 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 43 | total time: 2449.12m | eta: 2076.0m
step 108300/200000 (54.15%) | loss: 2.624194 | lrm: 0.53 | dt: 1317.72ms | tok/sec: 12,433 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 45 | total time: 2450.22m | eta: 2074.8m
step 108350/200000 (54.17%) | loss: 2.589072 | lrm: 0.53 | dt: 1320.49ms | tok/sec: 12,407 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 47 | total time: 2451.32m | eta: 2073.7m
step 108400/200000 (54.20%) | loss: 2.591630 | lrm: 0.53 | dt: 1318.82ms | tok/sec: 12,423 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 49 | total time: 2452.42m | eta: 2072.5m
step 108450/200000 (54.23%) | loss: 2.743006 | lrm: 0.53 | dt: 1322.58ms | tok/sec: 12,387 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 51 | total time: 2453.52m | eta: 2071.4m
step 108500/200000 (54.25%) | loss: 2.637719 | lrm: 0.53 | dt: 1320.38ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 53 | total time: 2454.62m | eta: 2070.2m
step 108550/200000 (54.27%) | loss: 2.690307 | lrm: 0.53 | dt: 1320.62ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 55 | total time: 2455.72m | eta: 2069.1m
step 108600/200000 (54.30%) | loss: 2.699466 | lrm: 0.53 | dt: 1321.26ms | tok/sec: 12,400 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 57 | total time: 2456.82m | eta: 2067.9m
step 108650/200000 (54.33%) | loss: 2.685663 | lrm: 0.53 | dt: 1323.71ms | tok/sec: 12,377 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 59 | total time: 2457.92m | eta: 2066.7m
step 108700/200000 (54.35%) | loss: 2.629329 | lrm: 0.53 | dt: 1320.34ms | tok/sec: 12,408 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 61 | total time: 2459.02m | eta: 2065.6m
step 108750/200000 (54.38%) | loss: 2.687617 | lrm: 0.53 | dt: 1320.25ms | tok/sec: 12,409 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 64 | total time: 2460.13m | eta: 2064.4m
step 108800/200000 (54.40%) | loss: 2.614564 | lrm: 0.53 | dt: 1318.88ms | tok/sec: 12,422 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 66 | total time: 2461.23m | eta: 2063.3m
step 108850/200000 (54.42%) | loss: 2.716835 | lrm: 0.53 | dt: 1319.59ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 68 | total time: 2462.33m | eta: 2062.1m
step 108900/200000 (54.45%) | loss: 2.731834 | lrm: 0.53 | dt: 1320.60ms | tok/sec: 12,406 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 70 | total time: 2463.43m | eta: 2061.0m
step 108950/200000 (54.48%) | loss: 2.658882 | lrm: 0.53 | dt: 1319.51ms | tok/sec: 12,416 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 72 | total time: 2464.53m | eta: 2059.8m
step 109000/200000 (54.50%) | loss: 2.654428 | lrm: 0.53 | dt: 1319.70ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 74 | total time: 2465.63m | eta: 2058.7m
step 109050/200000 (54.52%) | loss: 2.582220 | lrm: 0.53 | dt: 1320.09ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 76 | total time: 2466.73m | eta: 2057.5m
step 109100/200000 (54.55%) | loss: 2.686946 | lrm: 0.53 | dt: 1319.77ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 78 | total time: 2467.83m | eta: 2056.3m
step 109150/200000 (54.58%) | loss: 2.711593 | lrm: 0.53 | dt: 1316.02ms | tok/sec: 12,449 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 80 | total time: 2468.93m | eta: 2055.2m
step 109200/200000 (54.60%) | loss: 2.681664 | lrm: 0.53 | dt: 1319.24ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 53 rg: 82 | total time: 2470.03m | eta: 2054.0m
step 109250/200000 (54.62%) | loss: 2.707993 | lrm: 0.53 | dt: 1317.06ms | tok/sec: 12,439 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 1 | total time: 2471.13m | eta: 2052.9m
step 109300/200000 (54.65%) | loss: 2.660298 | lrm: 0.53 | dt: 1319.04ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 3 | total time: 2472.23m | eta: 2051.7m
step 109350/200000 (54.67%) | loss: 2.564842 | lrm: 0.53 | dt: 1319.21ms | tok/sec: 12,419 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 5 | total time: 2473.33m | eta: 2050.6m
step 109400/200000 (54.70%) | loss: 2.681629 | lrm: 0.53 | dt: 1319.00ms | tok/sec: 12,421 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 7 | total time: 2474.43m | eta: 2049.4m
step 109450/200000 (54.73%) | loss: 2.694021 | lrm: 0.53 | dt: 1318.78ms | tok/sec: 12,423 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 9 | total time: 2475.53m | eta: 2048.2m
step 109500/200000 (54.75%) | loss: 2.678443 | lrm: 0.53 | dt: 1322.81ms | tok/sec: 12,385 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 11 | total time: 2476.63m | eta: 2047.1m
step 109550/200000 (54.77%) | loss: 2.682394 | lrm: 0.53 | dt: 1322.96ms | tok/sec: 12,384 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 13 | total time: 2477.74m | eta: 2045.9m
step 109600/200000 (54.80%) | loss: 2.679764 | lrm: 0.53 | dt: 1320.70ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 15 | total time: 2478.84m | eta: 2044.8m
step 109650/200000 (54.83%) | loss: 2.653390 | lrm: 0.53 | dt: 1320.92ms | tok/sec: 12,403 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 17 | total time: 2479.94m | eta: 2043.6m
step 109700/200000 (54.85%) | loss: 2.642818 | lrm: 0.53 | dt: 1315.95ms | tok/sec: 12,450 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 19 | total time: 2481.04m | eta: 2042.5m
step 109750/200000 (54.88%) | loss: 2.656377 | lrm: 0.53 | dt: 1322.55ms | tok/sec: 12,388 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 21 | total time: 2482.14m | eta: 2041.3m
step 109800/200000 (54.90%) | loss: 2.666621 | lrm: 0.53 | dt: 1320.03ms | tok/sec: 12,411 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 23 | total time: 2483.24m | eta: 2040.2m
step 109850/200000 (54.92%) | loss: 2.584996 | lrm: 0.53 | dt: 1325.15ms | tok/sec: 12,363 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 26 | total time: 2484.34m | eta: 2039.0m
step 109900/200000 (54.95%) | loss: 2.661732 | lrm: 0.53 | dt: 1319.77ms | tok/sec: 12,414 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 28 | total time: 2485.44m | eta: 2037.8m
step 109950/200000 (54.98%) | loss: 2.646012 | lrm: 0.53 | dt: 1322.08ms | tok/sec: 12,392 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 30 | total time: 2486.54m | eta: 2036.7m
Step 110000 | Validation bpb: 0.971396
step 110000/200000 (55.00%) | loss: 2.686504 | lrm: 0.53 | dt: 1395.78ms | tok/sec: 11,738 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 32 | total time: 2487.64m | eta: 2035.5m
step 110050/200000 (55.02%) | loss: 2.714190 | lrm: 0.52 | dt: 1320.72ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 34 | total time: 2488.75m | eta: 2034.4m
step 110100/200000 (55.05%) | loss: 2.658407 | lrm: 0.52 | dt: 1317.55ms | tok/sec: 12,435 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 36 | total time: 2489.84m | eta: 2033.2m
step 110150/200000 (55.08%) | loss: 2.654971 | lrm: 0.52 | dt: 1320.72ms | tok/sec: 12,405 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 38 | total time: 2490.94m | eta: 2032.1m
step 110200/200000 (55.10%) | loss: 2.599117 | lrm: 0.52 | dt: 1321.02ms | tok/sec: 12,402 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 40 | total time: 2492.05m | eta: 2030.9m
step 110250/200000 (55.12%) | loss: 2.704090 | lrm: 0.52 | dt: 1316.61ms | tok/sec: 12,444 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 42 | total time: 2493.15m | eta: 2029.8m
step 110300/200000 (55.15%) | loss: 2.648553 | lrm: 0.52 | dt: 1317.46ms | tok/sec: 12,436 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 44 | total time: 2494.25m | eta: 2028.6m
step 110350/200000 (55.17%) | loss: 2.641955 | lrm: 0.52 | dt: 1317.36ms | tok/sec: 12,437 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 46 | total time: 2495.34m | eta: 2027.4m
step 110400/200000 (55.20%) | loss: 2.631132 | lrm: 0.52 | dt: 1320.14ms | tok/sec: 12,410 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 48 | total time: 2496.44m | eta: 2026.3m
step 110450/200000 (55.23%) | loss: 2.679278 | lrm: 0.52 | dt: 1326.45ms | tok/sec: 12,351 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 50 | total time: 2497.55m | eta: 2025.1m
step 110500/200000 (55.25%) | loss: 2.624218 | lrm: 0.52 | dt: 1331.21ms | tok/sec: 12,307 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 52 | total time: 2498.66m | eta: 2024.0m
step 110550/200000 (55.27%) | loss: 2.732700 | lrm: 0.52 | dt: 1374.77ms | tok/sec: 11,917 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 54 | total time: 2499.80m | eta: 2022.9m
step 110600/200000 (55.30%) | loss: 2.645187 | lrm: 0.52 | dt: 1377.61ms | tok/sec: 11,893 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 56 | total time: 2500.95m | eta: 2021.7m
step 110650/200000 (55.33%) | loss: 2.667929 | lrm: 0.52 | dt: 1411.56ms | tok/sec: 11,607 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 58 | total time: 2502.10m | eta: 2020.6m
step 110700/200000 (55.35%) | loss: 2.656804 | lrm: 0.52 | dt: 1367.68ms | tok/sec: 11,979 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 60 | total time: 2503.24m | eta: 2019.5m
step 110750/200000 (55.38%) | loss: 2.640370 | lrm: 0.52 | dt: 1369.63ms | tok/sec: 11,962 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 62 | total time: 2504.38m | eta: 2018.4m
step 110800/200000 (55.40%) | loss: 2.708138 | lrm: 0.52 | dt: 1372.10ms | tok/sec: 11,940 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 64 | total time: 2505.53m | eta: 2017.3m
step 110850/200000 (55.42%) | loss: 2.727553 | lrm: 0.52 | dt: 1371.56ms | tok/sec: 11,945 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 66 | total time: 2506.67m | eta: 2016.1m
step 110900/200000 (55.45%) | loss: 2.640253 | lrm: 0.52 | dt: 1361.96ms | tok/sec: 12,029 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 69 | total time: 2507.81m | eta: 2015.0m
step 110950/200000 (55.48%) | loss: 2.669448 | lrm: 0.52 | dt: 1366.62ms | tok/sec: 11,988 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 71 | total time: 2508.95m | eta: 2013.9m
step 111000/200000 (55.50%) | loss: 2.607525 | lrm: 0.52 | dt: 1370.51ms | tok/sec: 11,954 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 73 | total time: 2510.08m | eta: 2012.8m
step 111050/200000 (55.52%) | loss: 2.722624 | lrm: 0.52 | dt: 1373.87ms | tok/sec: 11,925 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 75 | total time: 2511.23m | eta: 2011.7m
step 111100/200000 (55.55%) | loss: 2.649660 | lrm: 0.52 | dt: 1369.74ms | tok/sec: 11,961 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 77 | total time: 2512.38m | eta: 2010.5m
step 111150/200000 (55.58%) | loss: 2.658615 | lrm: 0.52 | dt: 1372.18ms | tok/sec: 11,940 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 79 | total time: 2513.52m | eta: 2009.4m
step 111200/200000 (55.60%) | loss: 2.627154 | lrm: 0.52 | dt: 1363.16ms | tok/sec: 12,019 | bf16_mfu: 0.00 | epoch: 1 pq: 54 rg: 81 | total time: 2514.65m | eta: 2008.3m
step 111250/200000 (55.62%) | loss: 2.676918 | lrm: 0.52 | dt: 1346.11ms | tok/sec: 12,171 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 0 | total time: 2515.77m | eta: 2007.1m
step 111300/200000 (55.65%) | loss: 2.655859 | lrm: 0.52 | dt: 1334.88ms | tok/sec: 12,273 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 2 | total time: 2516.90m | eta: 2006.0m
step 111350/200000 (55.67%) | loss: 2.728940 | lrm: 0.52 | dt: 1338.41ms | tok/sec: 12,241 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 4 | total time: 2518.02m | eta: 2004.9m
step 111400/200000 (55.70%) | loss: 2.657700 | lrm: 0.52 | dt: 1363.35ms | tok/sec: 12,017 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 6 | total time: 2519.16m | eta: 2003.7m
step 111450/200000 (55.73%) | loss: 2.610811 | lrm: 0.52 | dt: 1364.33ms | tok/sec: 12,008 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 8 | total time: 2520.30m | eta: 2002.6m
step 111500/200000 (55.75%) | loss: 2.665734 | lrm: 0.52 | dt: 1367.09ms | tok/sec: 11,984 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 10 | total time: 2521.44m | eta: 2001.5m
step 111550/200000 (55.77%) | loss: 2.647709 | lrm: 0.52 | dt: 1374.59ms | tok/sec: 11,919 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 12 | total time: 2522.57m | eta: 2000.4m
step 111600/200000 (55.80%) | loss: 2.634941 | lrm: 0.52 | dt: 1368.61ms | tok/sec: 11,971 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 14 | total time: 2523.72m | eta: 1999.3m
step 111650/200000 (55.83%) | loss: 2.651868 | lrm: 0.52 | dt: 1367.50ms | tok/sec: 11,980 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 16 | total time: 2524.87m | eta: 1998.1m
step 111700/200000 (55.85%) | loss: 2.597669 | lrm: 0.52 | dt: 1370.91ms | tok/sec: 11,951 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 18 | total time: 2526.01m | eta: 1997.0m
step 111750/200000 (55.88%) | loss: 2.695463 | lrm: 0.52 | dt: 1372.16ms | tok/sec: 11,940 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 20 | total time: 2527.15m | eta: 1995.9m
step 111800/200000 (55.90%) | loss: 2.743174 | lrm: 0.52 | dt: 1372.59ms | tok/sec: 11,936 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 22 | total time: 2528.30m | eta: 1994.8m
step 111850/200000 (55.92%) | loss: 2.717274 | lrm: 0.52 | dt: 1364.67ms | tok/sec: 12,005 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 24 | total time: 2529.44m | eta: 1993.7m
step 111900/200000 (55.95%) | loss: 2.666804 | lrm: 0.51 | dt: 1359.00ms | tok/sec: 12,055 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 26 | total time: 2530.57m | eta: 1992.5m
step 111950/200000 (55.98%) | loss: 2.598382 | lrm: 0.51 | dt: 1364.25ms | tok/sec: 12,009 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 28 | total time: 2531.70m | eta: 1991.4m
Step 112000 | Validation bpb: 0.970304
<|bos|>The capital of France is the city of L'Anais, in the north of the equator. This city is the only one that is
<|bos|>The chemical symbol of gold is Au
`The symbol of gold is Au`
The chemical symbol of gold is Au
The chemical symbol of gold
<|bos|>If yesterday was Friday, then tomorrow will be Friday afternoons. And we get to that. I'd been thinking about a very good time with my grandp
<|bos|>`The opposite of hot is cold,` because the heat energy can be retained in the system or put on the circuit. A cold circuit is not a
<|bos|>`The planets of the solar system are: Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, Mars`
<|bos|>`My favorite color is blue.` And I'm surprised how I use it in my day-to-day life.
We're always looking for color in
<|bos|>If 5*x + 3 = 13, then x is the number of times 5 can be represented as 13+ 10. For example, x is 10 *
2026-03-18 17:12:25,625 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_112000.pt
2026-03-18 17:12:25,628 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_112000.json
2026-03-18 17:12:28,573 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_112000_rank0.pt
step 112000/200000 (56.00%) | loss: 2.577331 | lrm: 0.51 | dt: 1763.76ms | tok/sec: 9,289 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 30 | total time: 2532.86m | eta: 1990.3m
step 112050/200000 (56.02%) | loss: 2.672358 | lrm: 0.51 | dt: 1352.70ms | tok/sec: 12,112 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 32 | total time: 2534.00m | eta: 1989.2m
step 112100/200000 (56.05%) | loss: 2.652602 | lrm: 0.51 | dt: 1353.99ms | tok/sec: 12,100 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 35 | total time: 2535.12m | eta: 1988.0m
step 112150/200000 (56.08%) | loss: 2.666614 | lrm: 0.51 | dt: 1361.37ms | tok/sec: 12,034 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 37 | total time: 2536.24m | eta: 1986.9m
step 112200/200000 (56.10%) | loss: 2.573696 | lrm: 0.51 | dt: 1334.08ms | tok/sec: 12,281 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 39 | total time: 2537.36m | eta: 1985.7m
step 112250/200000 (56.12%) | loss: 2.703131 | lrm: 0.51 | dt: 1341.76ms | tok/sec: 12,210 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 41 | total time: 2538.49m | eta: 1984.6m
step 112300/200000 (56.15%) | loss: 2.643530 | lrm: 0.51 | dt: 1369.01ms | tok/sec: 11,967 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 43 | total time: 2539.63m | eta: 1983.5m
step 112350/200000 (56.17%) | loss: 2.676032 | lrm: 0.51 | dt: 1371.64ms | tok/sec: 11,944 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 45 | total time: 2540.78m | eta: 1982.4m
step 112400/200000 (56.20%) | loss: 2.672598 | lrm: 0.51 | dt: 1375.17ms | tok/sec: 11,914 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 47 | total time: 2541.93m | eta: 1981.2m
step 112450/200000 (56.23%) | loss: 2.640583 | lrm: 0.51 | dt: 1372.12ms | tok/sec: 11,940 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 49 | total time: 2543.07m | eta: 1980.1m
step 112500/200000 (56.25%) | loss: 2.701819 | lrm: 0.51 | dt: 1379.98ms | tok/sec: 11,872 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 51 | total time: 2544.22m | eta: 1979.0m
step 112550/200000 (56.27%) | loss: 2.635083 | lrm: 0.51 | dt: 1383.06ms | tok/sec: 11,846 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 53 | total time: 2545.36m | eta: 1977.9m
step 112600/200000 (56.30%) | loss: 2.630472 | lrm: 0.51 | dt: 1372.87ms | tok/sec: 11,934 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 55 | total time: 2546.51m | eta: 1976.8m
step 112650/200000 (56.33%) | loss: 2.639915 | lrm: 0.51 | dt: 1370.47ms | tok/sec: 11,954 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 57 | total time: 2547.66m | eta: 1975.7m
step 112700/200000 (56.35%) | loss: 2.691775 | lrm: 0.51 | dt: 1376.30ms | tok/sec: 11,904 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 59 | total time: 2548.80m | eta: 1974.5m
step 112750/200000 (56.38%) | loss: 2.661350 | lrm: 0.51 | dt: 1371.57ms | tok/sec: 11,945 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 61 | total time: 2549.94m | eta: 1973.4m
step 112800/200000 (56.40%) | loss: 2.703238 | lrm: 0.51 | dt: 1369.77ms | tok/sec: 11,961 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 63 | total time: 2551.08m | eta: 1972.3m
step 112850/200000 (56.42%) | loss: 2.615791 | lrm: 0.51 | dt: 1368.31ms | tok/sec: 11,973 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 65 | total time: 2552.23m | eta: 1971.2m
step 112900/200000 (56.45%) | loss: 2.613172 | lrm: 0.51 | dt: 1354.30ms | tok/sec: 12,097 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 67 | total time: 2553.36m | eta: 1970.0m
step 112950/200000 (56.48%) | loss: 2.686799 | lrm: 0.51 | dt: 1338.77ms | tok/sec: 12,238 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 69 | total time: 2554.49m | eta: 1968.9m
step 113000/200000 (56.50%) | loss: 2.731849 | lrm: 0.51 | dt: 1340.40ms | tok/sec: 12,223 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 71 | total time: 2555.61m | eta: 1967.8m
step 113050/200000 (56.52%) | loss: 2.620911 | lrm: 0.51 | dt: 1373.27ms | tok/sec: 11,930 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 73 | total time: 2556.74m | eta: 1966.6m
step 113100/200000 (56.55%) | loss: 2.632720 | lrm: 0.51 | dt: 1389.58ms | tok/sec: 11,790 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 75 | total time: 2557.87m | eta: 1965.5m
step 113150/200000 (56.58%) | loss: 2.654525 | lrm: 0.51 | dt: 1379.63ms | tok/sec: 11,875 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 77 | total time: 2559.02m | eta: 1964.4m
step 113200/200000 (56.60%) | loss: 2.603773 | lrm: 0.51 | dt: 1380.03ms | tok/sec: 11,872 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 79 | total time: 2560.17m | eta: 1963.3m
step 113250/200000 (56.62%) | loss: 2.632619 | lrm: 0.51 | dt: 1377.60ms | tok/sec: 11,893 | bf16_mfu: 0.00 | epoch: 1 pq: 55 rg: 82 | total time: 2561.32m | eta: 1962.2m
step 113300/200000 (56.65%) | loss: 2.663528 | lrm: 0.51 | dt: 1377.71ms | tok/sec: 11,892 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 1 | total time: 2562.46m | eta: 1961.0m
step 113350/200000 (56.67%) | loss: 2.627666 | lrm: 0.51 | dt: 1374.69ms | tok/sec: 11,918 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 3 | total time: 2563.61m | eta: 1959.9m
step 113400/200000 (56.70%) | loss: 2.630239 | lrm: 0.51 | dt: 1363.99ms | tok/sec: 12,011 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 5 | total time: 2564.75m | eta: 1958.8m
step 113450/200000 (56.73%) | loss: 2.695065 | lrm: 0.51 | dt: 1381.97ms | tok/sec: 11,855 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 7 | total time: 2565.90m | eta: 1957.7m
step 113500/200000 (56.75%) | loss: 2.696041 | lrm: 0.51 | dt: 1360.43ms | tok/sec: 12,043 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 9 | total time: 2567.05m | eta: 1956.6m
step 113550/200000 (56.77%) | loss: 2.653502 | lrm: 0.51 | dt: 1384.60ms | tok/sec: 11,833 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 11 | total time: 2568.20m | eta: 1955.4m
step 113600/200000 (56.80%) | loss: 2.601410 | lrm: 0.51 | dt: 1382.29ms | tok/sec: 11,852 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 13 | total time: 2569.35m | eta: 1954.3m
step 113650/200000 (56.83%) | loss: 2.643945 | lrm: 0.51 | dt: 1375.90ms | tok/sec: 11,907 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 15 | total time: 2570.50m | eta: 1953.2m
step 113700/200000 (56.85%) | loss: 2.593887 | lrm: 0.51 | dt: 1381.04ms | tok/sec: 11,863 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 17 | total time: 2571.63m | eta: 1952.1m
step 113750/200000 (56.88%) | loss: 2.673684 | lrm: 0.51 | dt: 1345.38ms | tok/sec: 12,177 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 19 | total time: 2572.76m | eta: 1950.9m
step 113800/200000 (56.90%) | loss: 2.577393 | lrm: 0.50 | dt: 1366.77ms | tok/sec: 11,987 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 21 | total time: 2573.88m | eta: 1949.8m
step 113850/200000 (56.92%) | loss: 2.655425 | lrm: 0.50 | dt: 1337.46ms | tok/sec: 12,250 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 23 | total time: 2575.00m | eta: 1948.7m
step 113900/200000 (56.95%) | loss: 2.529730 | lrm: 0.50 | dt: 1368.90ms | tok/sec: 11,968 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 25 | total time: 2576.13m | eta: 1947.5m
step 113950/200000 (56.98%) | loss: 2.586965 | lrm: 0.50 | dt: 1359.13ms | tok/sec: 12,054 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 27 | total time: 2577.26m | eta: 1946.4m
Step 114000 | Validation bpb: 0.969237
step 114000/200000 (57.00%) | loss: 2.613321 | lrm: 0.50 | dt: 1406.68ms | tok/sec: 11,647 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 29 | total time: 2578.40m | eta: 1945.3m
step 114050/200000 (57.02%) | loss: 2.740208 | lrm: 0.50 | dt: 1378.48ms | tok/sec: 11,885 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 31 | total time: 2579.54m | eta: 1944.2m
step 114100/200000 (57.05%) | loss: 2.686807 | lrm: 0.50 | dt: 1379.95ms | tok/sec: 11,872 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 33 | total time: 2580.68m | eta: 1943.0m
step 114150/200000 (57.08%) | loss: 2.634976 | lrm: 0.50 | dt: 1354.36ms | tok/sec: 12,097 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 35 | total time: 2581.82m | eta: 1941.9m
step 114200/200000 (57.10%) | loss: 2.708731 | lrm: 0.50 | dt: 1348.58ms | tok/sec: 12,149 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 37 | total time: 2582.95m | eta: 1940.8m
step 114250/200000 (57.12%) | loss: 2.661827 | lrm: 0.50 | dt: 1354.32ms | tok/sec: 12,097 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 39 | total time: 2584.08m | eta: 1939.6m
step 114300/200000 (57.15%) | loss: 2.594199 | lrm: 0.50 | dt: 1346.64ms | tok/sec: 12,166 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 41 | total time: 2585.20m | eta: 1938.5m
step 114350/200000 (57.17%) | loss: 2.683365 | lrm: 0.50 | dt: 1378.82ms | tok/sec: 11,882 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 43 | total time: 2586.33m | eta: 1937.4m
step 114400/200000 (57.20%) | loss: 2.647407 | lrm: 0.50 | dt: 1374.65ms | tok/sec: 11,918 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 45 | total time: 2587.48m | eta: 1936.3m
step 114450/200000 (57.23%) | loss: 2.641326 | lrm: 0.50 | dt: 1369.17ms | tok/sec: 11,966 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 48 | total time: 2588.62m | eta: 1935.1m
step 114500/200000 (57.25%) | loss: 2.608485 | lrm: 0.50 | dt: 1356.56ms | tok/sec: 12,077 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 50 | total time: 2589.74m | eta: 1934.0m
step 114550/200000 (57.27%) | loss: 2.722480 | lrm: 0.50 | dt: 1382.00ms | tok/sec: 11,855 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 52 | total time: 2590.88m | eta: 1932.9m
step 114600/200000 (57.30%) | loss: 2.633389 | lrm: 0.50 | dt: 1347.69ms | tok/sec: 12,157 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 54 | total time: 2592.02m | eta: 1931.7m
step 114650/200000 (57.33%) | loss: 2.643631 | lrm: 0.50 | dt: 1337.74ms | tok/sec: 12,247 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 56 | total time: 2593.14m | eta: 1930.6m
step 114700/200000 (57.35%) | loss: 2.645278 | lrm: 0.50 | dt: 4453.69ms | tok/sec: 3,678 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 58 | total time: 2595.65m | eta: 1930.5m
step 114750/200000 (57.38%) | loss: 2.717832 | lrm: 0.50 | dt: 3149.37ms | tok/sec: 5,202 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 60 | total time: 2598.61m | eta: 1930.7m
step 114800/200000 (57.40%) | loss: 2.575412 | lrm: 0.50 | dt: 14228.47ms | tok/sec: 1,151 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 62 | total time: 2602.62m | eta: 1931.7m
step 114850/200000 (57.42%) | loss: 2.661590 | lrm: 0.50 | dt: 2593.50ms | tok/sec: 6,317 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 64 | total time: 2607.34m | eta: 1933.3m
step 114900/200000 (57.45%) | loss: 2.589656 | lrm: 0.50 | dt: 4410.82ms | tok/sec: 3,714 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 66 | total time: 2610.89m | eta: 1933.9m
step 114950/200000 (57.48%) | loss: 2.649565 | lrm: 0.50 | dt: 2589.79ms | tok/sec: 6,326 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 68 | total time: 2613.12m | eta: 1933.6m
step 115000/200000 (57.50%) | loss: 2.640975 | lrm: 0.50 | dt: 5188.28ms | tok/sec: 3,157 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 70 | total time: 2615.39m | eta: 1933.3m
step 115050/200000 (57.52%) | loss: 2.685028 | lrm: 0.50 | dt: 2633.91ms | tok/sec: 6,220 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 72 | total time: 2617.62m | eta: 1933.0m
step 115100/200000 (57.55%) | loss: 2.669518 | lrm: 0.50 | dt: 2969.36ms | tok/sec: 5,517 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 74 | total time: 2619.89m | eta: 1932.6m
step 115150/200000 (57.58%) | loss: 2.697401 | lrm: 0.50 | dt: 2518.62ms | tok/sec: 6,505 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 76 | total time: 2622.16m | eta: 1932.3m
step 115200/200000 (57.60%) | loss: 2.640925 | lrm: 0.50 | dt: 2846.28ms | tok/sec: 5,756 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 78 | total time: 2624.36m | eta: 1932.0m
step 115250/200000 (57.62%) | loss: 2.634680 | lrm: 0.50 | dt: 2733.03ms | tok/sec: 5,994 | bf16_mfu: 0.00 | epoch: 1 pq: 56 rg: 80 | total time: 2626.80m | eta: 1931.8m
step 115300/200000 (57.65%) | loss: 2.673060 | lrm: 0.50 | dt: 2988.76ms | tok/sec: 5,481 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 0 | total time: 2629.17m | eta: 1931.6m
step 115350/200000 (57.67%) | loss: 2.654223 | lrm: 0.50 | dt: 9040.01ms | tok/sec: 1,812 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 2 | total time: 2633.49m | eta: 1932.8m
step 115400/200000 (57.70%) | loss: 2.584741 | lrm: 0.50 | dt: 1338.19ms | tok/sec: 12,243 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 4 | total time: 2641.12m | eta: 1936.4m
step 115450/200000 (57.73%) | loss: 2.614804 | lrm: 0.50 | dt: 1328.17ms | tok/sec: 12,335 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 6 | total time: 2642.22m | eta: 1935.2m
step 115500/200000 (57.75%) | loss: 2.668842 | lrm: 0.50 | dt: 1337.19ms | tok/sec: 12,252 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 8 | total time: 2643.34m | eta: 1934.0m
step 115550/200000 (57.77%) | loss: 2.625275 | lrm: 0.50 | dt: 1335.08ms | tok/sec: 12,271 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 10 | total time: 2644.46m | eta: 1932.9m
step 115600/200000 (57.80%) | loss: 2.628082 | lrm: 0.50 | dt: 1346.96ms | tok/sec: 12,163 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 13 | total time: 2645.58m | eta: 1931.7m
step 115650/200000 (57.83%) | loss: 2.633187 | lrm: 0.50 | dt: 1379.07ms | tok/sec: 11,880 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 15 | total time: 2646.70m | eta: 1930.6m
step 115700/200000 (57.85%) | loss: 2.659744 | lrm: 0.49 | dt: 1372.90ms | tok/sec: 11,933 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 17 | total time: 2647.85m | eta: 1929.4m
step 115750/200000 (57.88%) | loss: 2.660633 | lrm: 0.49 | dt: 1443.61ms | tok/sec: 11,349 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 19 | total time: 2649.01m | eta: 1928.3m
step 115800/200000 (57.90%) | loss: 2.600851 | lrm: 0.49 | dt: 1376.52ms | tok/sec: 11,902 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 21 | total time: 2650.16m | eta: 1927.1m
step 115850/200000 (57.92%) | loss: 2.659058 | lrm: 0.49 | dt: 1390.13ms | tok/sec: 11,785 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 23 | total time: 2651.31m | eta: 1926.0m
step 115900/200000 (57.95%) | loss: 2.633704 | lrm: 0.49 | dt: 1378.66ms | tok/sec: 11,884 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 25 | total time: 2652.46m | eta: 1924.9m
step 115950/200000 (57.98%) | loss: 2.644130 | lrm: 0.49 | dt: 1369.39ms | tok/sec: 11,964 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 27 | total time: 2653.62m | eta: 1923.7m
Step 116000 | Validation bpb: 0.967017
<|bos|>The capital of France is the city of Chardon.
Lawrence's 2006 "Supersewer", which was built to
<|bos|>`The chemical symbol of gold is Au.` Gold has a very high melting point, and only a little bit (about 4.3%
<|bos|>`If yesterday was Friday, then tomorrow will be Saturday.` If there isn't a ton of time for work, we'll be up Sunday. It's
<|bos|>`The opposite of hot is cold.` We are not inherently cold. We are not necessarily hot in the same way, or even in any way
<|bos|>`The planets of the solar system are: Jupiter, Venus, Mercury, and Venus. Mars, Jupiter,` and
<|bos|>`My favorite color is blue.` And I'm not the only one. I'm the coolest. I like to look up colors. My favorite
<|bos|>If 5*x + 3 = 13, then x is the number of times that the two numbers are counted. 36.3 is the number of times that the two numbers
2026-03-18 19:15:16,139 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_116000.pt
2026-03-18 19:15:16,160 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_116000.json
2026-03-18 19:15:19,747 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_116000_rank0.pt
step 116000/200000 (58.00%) | loss: 2.699806 | lrm: 0.49 | dt: 1386.92ms | tok/sec: 11,813 | bf16_mfu: 0.00 |he\nanochat\base_checkpoints\d16\model_116000.pt
2026-03-18 19:15:16,160 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_116000.json
2026-03-18 19:15:19,747 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_116000_rank0.pt
step 116000/200000 (58.00%) | loss: 2.699806 | lrm: 0.49 | dt: 1386.92ms | tok/sec: 11,813 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 29 | total time: 2654.77m | eta: 1922.6m
step 116050/200000 (58.02%) | loss: 2.694660 | lrm: 0.49 | dt: 1483.76ms | tok/sec: 11,042 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 31 | total time: 2655.98m | eta: 1921.5m
step 116100/200000 (58.05%) | loss: 2.619813 | lrm: 0.49 | dt: 1443.13ms | tok/sec: 11,353 | bf16_mfu: 0.00 |he\nanochat\base_checkpoints\d16\model_116000.pt
2026-03-18 19:15:16,160 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_116000.json
2026-03-18 19:15:19,747 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_116000_rank0.pt
step 116000/200000 (58.00%) | loss: 2.699806 | lrm: 0.49 | dt: 1386.92ms | tok/sec: 11,813 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 29 | total time: 2654.77m | eta: 1922.6m
step 116050/200000 (58.02%) | loss: 2.694660 | lrm: 0.49 | dt: 1483.76ms | tok/sec: 11,042 | bf16_mfu: 0.00 |he\nanochat\base_checkpoints\d16\model_116000.pt
2026-03-18 19:15:16,160 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_116000.json
2026-03-18 19:15:19,747 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_116000_rank0.pt
step 116000/200000 (58.00%) | loss: 2.699806 | lrm: 0.49 | dt: 1386.92ms | tok/sec: 11,813 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 29 | total time: 2654.77m | eta: 1922.6m
2026-03-18 19:15:19,747 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_116000_rank0.pt
step 116000/200000 (58.00%) | loss: 2.699806 | lrm: 0.49 | dt: 1386.92ms | tok/sec: 11,813 | bf16_mfu: 0.00 |e\nanochat\base_checkpoints\d16\optim_116000_rank0.pt
step 116000/200000 (58.00%) | loss: 2.699806 | lrm: 0.49 | dt: 1386.92ms | tok/sec: 11,813 | bf16_mfu: 0.00 |step 116000/200000 (58.00%) | loss: 2.699806 | lrm: 0.49 | dt: 1386.92ms | tok/sec: 11,813 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 29 | total time: 2654.77m | eta: 1922.6m
step 116050/200000 (58.02%) | loss: 2.694660 | lrm: 0.49 | dt: 1483.76ms | tok/sec: 11,042 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 31 | total time: 2655.98m | eta: 1921.5m
step 116100/200000 (58.05%) | loss: 2.619813 | lrm: 0.49 | dt: 1443.13ms | tok/sec: 11,353 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 33 | total time: 2657.14m | eta: 1920.4m
step 116150/200000 (58.08%) | loss: 2.606652 | lrm: 0.49 | dt: 1385.50ms | tok/sec: 11,825 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 35 | total time: 2658.29m | eta: 1919.2m
step 116200/200000 (58.10%) | loss: 2.633792 | lrm: 0.49 | dt: 1390.05ms | tok/sec: 11,786 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 37 | total time: 2659.45m | eta: 1918.1m
step 116250/200000 (58.12%) | loss: 2.701164 | lrm: 0.49 | dt: 1384.27ms | tok/sec: 11,835 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 39 | total time: 2660.61m | eta: 1916.9m
step 116300/200000 (58.15%) | loss: 2.600456 | lrm: 0.49 | dt: 1379.91ms | tok/sec: 11,873 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 41 | total time: 2661.77m | eta: 1915.8m
step 116350/200000 (58.17%) | loss: 2.677112 | lrm: 0.49 | dt: 1414.98ms | tok/sec: 11,578 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 43 | total time: 2662.97m | eta: 1914.7m
step 116400/200000 (58.20%) | loss: 2.634903 | lrm: 0.49 | dt: 1332.97ms | tok/sec: 12,291 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 45 | total time: 2664.09m | eta: 1913.5m
step 116450/200000 (58.23%) | loss: 2.609461 | lrm: 0.49 | dt: 1335.75ms | tok/sec: 12,265 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 47 | total time: 2665.21m | eta: 1912.4m
step 116450/200000 (58.23%) | loss: 2.609461 | lrm: 0.49 | dt: 1335.75ms | tok/sec: 12,265 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 47 | total time: 2665.21m | eta: 1912.4m
step 116500/200000 (58.25%) | loss: 2.633364 | lrm: 0.49 | dt: 1340.86ms | tok/sec: 12,219 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 49 | total time: 2666.33m | eta: 1911.2m
step 116550/200000 (58.27%) | loss: 2.651451 | lrm: 0.49 | dt: 1337.81ms | tok/sec: 12,246 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 51 | total time: 2667.44m | eta: 1910.1m
step 116600/200000 (58.30%) | loss: 2.670089 | lrm: 0.49 | dt: 1362.05ms | tok/sec: 12,028 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 53 | total time: 2668.57m | eta: 1908.9m
step 116650/200000 (58.33%) | loss: 2.653267 | lrm: 0.49 | dt: 1365.70ms | tok/sec: 11,996 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 55 | total time: 2669.70m | eta: 1907.7m
step 116700/200000 (58.35%) | loss: 2.608388 | lrm: 0.49 | dt: 1363.06ms | tok/sec: 12,019 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 58 | total time: 2670.83m | eta: 1906.6m
step 116750/200000 (58.38%) | loss: 2.683716 | lrm: 0.49 | dt: 1346.78ms | tok/sec: 12,165 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 60 | total time: 2671.96m | eta: 1905.4m
step 116800/200000 (58.40%) | loss: 2.664471 | lrm: 0.49 | dt: 1333.53ms | tok/sec: 12,286 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 62 | total time: 2673.07m | eta: 1904.3m
step 116850/200000 (58.42%) | loss: 2.670490 | lrm: 0.49 | dt: 1327.29ms | tok/sec: 12,343 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 64 | total time: 2674.18m | eta: 1903.1m
step 116900/200000 (58.45%) | loss: 2.681820 | lrm: 0.49 | dt: 1327.36ms | tok/sec: 12,343 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 66 | total time: 2675.29m | eta: 1901.9m
step 116950/200000 (58.48%) | loss: 2.677586 | lrm: 0.49 | dt: 1327.58ms | tok/sec: 12,341 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 68 | total time: 2676.40m | eta: 1900.8m
step 117000/200000 (58.50%) | loss: 2.637723 | lrm: 0.49 | dt: 1375.08ms | tok/sec: 11,914 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 70 | total time: 2677.53m | eta: 1899.6m
step 117050/200000 (58.52%) | loss: 2.608628 | lrm: 0.49 | dt: 1364.80ms | tok/sec: 12,004 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 72 | total time: 2678.67m | eta: 1898.5m
step 117100/200000 (58.55%) | loss: 2.648225 | lrm: 0.49 | dt: 1369.57ms | tok/sec: 11,962 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 74 | total time: 2679.81m | eta: 1897.3m
step 117150/200000 (58.58%) | loss: 2.606904 | lrm: 0.49 | dt: 1358.61ms | tok/sec: 12,059 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 76 | total time: 2680.95m | eta: 1896.2m
step 117200/200000 (58.60%) | loss: 2.706685 | lrm: 0.49 | dt: 1375.31ms | tok/sec: 11,912 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 78 | total time: 2682.09m | eta: 1895.0m
step 117250/200000 (58.62%) | loss: 2.668017 | lrm: 0.49 | dt: 1368.66ms | tok/sec: 11,970 | bf16_mfu: 0.00 | epoch: 1 pq: 57 rg: 80 | total time: 2683.23m | eta: 1893.9m
step 117300/200000 (58.65%) | loss: 2.589070 | lrm: 0.49 | dt: 1358.72ms | tok/sec: 12,058 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 0 | total time: 2684.37m | eta: 1892.7m
step 117350/200000 (58.67%) | loss: 2.661412 | lrm: 0.49 | dt: 1359.46ms | tok/sec: 12,051 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 2 | total time: 2685.50m | eta: 1891.6m
step 117400/200000 (58.70%) | loss: 2.562903 | lrm: 0.49 | dt: 1373.14ms | tok/sec: 11,931 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 4 | total time: 2686.65m | eta: 1890.4m
step 117450/200000 (58.73%) | loss: 2.656368 | lrm: 0.49 | dt: 1367.08ms | tok/sec: 11,984 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 6 | total time: 2687.79m | eta: 1889.3m
step 117500/200000 (58.75%) | loss: 2.651290 | lrm: 0.49 | dt: 1359.24ms | tok/sec: 12,053 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 8 | total time: 2688.93m | eta: 1888.1m
step 117550/200000 (58.77%) | loss: 2.605790 | lrm: 0.49 | dt: 1363.69ms | tok/sec: 12,014 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 10 | total time: 2690.07m | eta: 1887.0m
step 117600/200000 (58.80%) | loss: 2.626179 | lrm: 0.48 | dt: 1356.83ms | tok/sec: 12,075 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 12 | total time: 2691.21m | eta: 1885.8m
step 117650/200000 (58.83%) | loss: 2.627203 | lrm: 0.48 | dt: 1367.39ms | tok/sec: 11,981 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 14 | total time: 2692.34m | eta: 1884.7m
step 117700/200000 (58.85%) | loss: 2.658671 | lrm: 0.48 | dt: 1379.39ms | tok/sec: 11,877 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 16 | total time: 2693.48m | eta: 1883.5m
step 117750/200000 (58.88%) | loss: 2.622214 | lrm: 0.48 | dt: 1364.20ms | tok/sec: 12,009 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 18 | total time: 2694.62m | eta: 1882.4m
step 117800/200000 (58.90%) | loss: 2.668705 | lrm: 0.48 | dt: 1370.41ms | tok/sec: 11,955 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 20 | total time: 2695.77m | eta: 1881.2m
step 117850/200000 (58.92%) | loss: 2.726721 | lrm: 0.48 | dt: 1366.02ms | tok/sec: 11,993 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 23 | total time: 2696.90m | eta: 1880.1m
step 117900/200000 (58.95%) | loss: 2.647642 | lrm: 0.48 | dt: 1362.18ms | tok/sec: 12,027 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 25 | total time: 2698.05m | eta: 1879.0m
step 117950/200000 (58.98%) | loss: 2.663872 | lrm: 0.48 | dt: 1362.80ms | tok/sec: 12,022 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 27 | total time: 2699.19m | eta: 1877.8m
Step 118000 | Validation bpb: 0.966229
step 118000/200000 (59.00%) | loss: 2.679140 | lrm: 0.48 | dt: 1386.79ms | tok/sec: 11,814 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 29 | total time: 2700.33m | eta: 1876.7m
step 118050/200000 (59.02%) | loss: 2.642083 | lrm: 0.48 | dt: 1361.34ms | tok/sec: 12,035 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 31 | total time: 2701.47m | eta: 1875.5m
step 118100/200000 (59.05%) | loss: 2.669547 | lrm: 0.48 | dt: 1377.45ms | tok/sec: 11,894 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 33 | total time: 2702.61m | eta: 1874.4m
step 118150/200000 (59.08%) | loss: 2.589987 | lrm: 0.48 | dt: 1363.16ms | tok/sec: 12,019 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 35 | total time: 2703.75m | eta: 1873.2m
 epoch: 1 pq: 58 rg: 33 | total time: 2702.61m | eta: 1874.4m
step 118150/200000 (59.08%) | loss: 2.589987 | lrm: 0.48 | dt: 1363.16ms | tok/sec: 12,019 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 33 | total time: 2702.61m | eta: 1874.4m
step 118150/200000 (59.08%) | loss: 2.589987 | lrm: 0.48 | dt: 1363.16ms | tok/sec: 12,019 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 35 | total time: 2703.75m | eta: 1873.2m
step 118200/200000 (59.10%) | loss: 2.692495 | lrm: 0.48 | dt: 1349.15ms | tok/sec: 12,143 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 37 | total time: 2704.88m | eta: 1872.1m
 epoch: 1 pq: 58 rg: 37 | total time: 2704.88m | eta: 1872.1m
step 118250/200000 (59.12%) | loss: 2.696706 | lrm: 0.48 | dt: 1325.73ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 39 | total time: 2705.99m | eta: 1870.9m
step 118300/200000 (59.15%) | loss: 2.658708 | lrm: 0.48 | dt: 1325.32ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 41 | total time: 2707.10m | eta: 1869.7m
step 118350/200000 (59.17%) | loss: 2.686618 | lrm: 0.48 | dt: 1324.79ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 37 | total time: 2704.88m | eta: 1872.1m
step 118250/200000 (59.12%) | loss: 2.696706 | lrm: 0.48 | dt: 1325.73ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 39 | total time: 2705.99m | eta: 1870.9m
 epoch: 1 pq: 58 rg: 37 | total time: 2704.88m | eta: 1872.1m
step 118250/200000 (59.12%) | loss: 2.696706 | lrm: 0.48 | dt: 1325.73ms | tok/sec: 12,358 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 39 | total time: 2705.99m | eta: 1870.9m
step 118300/200000 (59.15%) | loss: 2.658708 | lrm: 0.48 | dt: 1325.32ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 41 | total time: 2707.10m | eta: 1869.7m
step 118350/200000 (59.17%) | loss: 2.686618 | lrm: 0.48 | dt: 1324.79ms | tok/sec: 12,367 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 43 | total time: 2708.21m | eta: 1868.6m
step 118400/200000 (59.20%) | loss: 2.616474 | lrm: 0.48 | dt: 1337.00ms | tok/sec: 12,254 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 45 | total time: 2709.32m | eta: 1867.4m
step 118450/200000 (59.23%) | loss: 2.671464 | lrm: 0.48 | dt: 1335.17ms | tok/sec: 12,271 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 47 | total time: 2710.43m | eta: 1866.2m
step 118500/200000 (59.25%) | loss: 2.632606 | lrm: 0.48 | dt: 1327.84ms | tok/sec: 12,338 | bf16_mfu: 0.00 |step 118450/200000 (59.23%) | loss: 2.671464 | lrm: 0.48 | dt: 1335.17ms | tok/sec: 12,271 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 47 | total time: 2710.43m | eta: 1866.2m
step 118450/200000 (59.23%) | loss: 2.671464 | lrm: 0.48 | dt: 1335.17ms | tok/sec: 12,271 | bf16_mfu: 0.00 |step 118450/200000 (59.23%) | loss: 2.671464 | lrm: 0.48 | dt: 1335.17ms | tok/sec: 12,271 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 47 | total time: 2710.43m | eta: 1866.2m
step 118500/200000 (59.25%) | loss: 2.632606 | lrm: 0.48 | dt: 1327.84ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 49 | total time: 2711.53m | eta: 1865.1m
step 118550/200000 (59.27%) | loss: 2.736219 | lrm: 0.48 | dt: 1334.47ms | tok/sec: 12,277 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 47 | total time: 2710.43m | eta: 1866.2m
step 118500/200000 (59.25%) | loss: 2.632606 | lrm: 0.48 | dt: 1327.84ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 49 | total time: 2711.53m | eta: 1865.1m
step 118550/200000 (59.27%) | loss: 2.736219 | lrm: 0.48 | dt: 1334.47ms | tok/sec: 12,277 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 49 | total time: 2711.53m | eta: 1865.1m
step 118550/200000 (59.27%) | loss: 2.736219 | lrm: 0.48 | dt: 1334.47ms | tok/sec: 12,277 | bf16_mfu: 0.00 |step 118550/200000 (59.27%) | loss: 2.736219 | lrm: 0.48 | dt: 1334.47ms | tok/sec: 12,277 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 51 | total time: 2712.65m | eta: 1863.9m
step 118600/200000 (59.30%) | loss: 2.656266 | lrm: 0.48 | dt: 1328.96ms | tok/sec: 12,328 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 53 | total time: 2713.76m | eta: 1862.7m
step 118650/200000 (59.33%) | loss: 2.635278 | lrm: 0.48 | dt: 1323.52ms | tok/sec: 12,379 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 55 | total time: 2714.87m | eta: 1861.6m
step 118700/200000 (59.35%) | loss: 2.507833 | lrm: 0.48 | dt: 1329.44ms | tok/sec: 12,323 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 57 | total time: 2715.98m | eta: 1860.4m
step 118750/200000 (59.38%) | loss: 2.582781 | lrm: 0.48 | dt: 1333.44ms | tok/sec: 12,287 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 59 | total time: 2717.09m | eta: 1859.2m
step 118800/200000 (59.40%) | loss: 2.688544 | lrm: 0.48 | dt: 1329.96ms | tok/sec: 12,319 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 61 | total time: 2718.19m | eta: 1858.0m
step 118850/200000 (59.42%) | loss: 2.605590 | lrm: 0.48 | dt: 1333.26ms | tok/sec: 12,288 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 63 | total time: 2719.30m | eta: 1856.9m
step 118900/200000 (59.45%) | loss: 2.650878 | lrm: 0.48 | dt: 1327.73ms | tok/sec: 12,339 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 65 | total time: 2720.41m | eta: 1855.7m
step 118950/200000 (59.48%) | loss: 2.651906 | lrm: 0.48 | dt: 1328.51ms | tok/sec: 12,332 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 68 | total time: 2721.52m | eta: 1854.5m
step 119000/200000 (59.50%) | loss: 2.618880 | lrm: 0.48 | dt: 1328.46ms | tok/sec: 12,333 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 70 | total time: 2722.63m | eta: 1853.4m
step 119050/200000 (59.52%) | loss: 2.643317 | lrm: 0.48 | dt: 1335.79ms | tok/sec: 12,265 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 72 | total time: 2723.74m | eta: 1852.2m
step 119100/200000 (59.55%) | loss: 2.590576 | lrm: 0.48 | dt: 1327.37ms | tok/sec: 12,343 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 74 | total time: 2724.85m | eta: 1851.0m
step 119150/200000 (59.58%) | loss: 2.640811 | lrm: 0.48 | dt: 1327.21ms | tok/sec: 12,344 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 76 | total time: 2725.96m | eta: 1849.9m
step 119200/200000 (59.60%) | loss: 2.642303 | lrm: 0.48 | dt: 1325.51ms | tok/sec: 12,360 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 78 | total time: 2727.07m | eta: 1848.7m
step 119250/200000 (59.62%) | loss: 2.606391 | lrm: 0.48 | dt: 1330.43ms | tok/sec: 12,314 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 80 | total time: 2728.18m | eta: 1847.5m
step 119300/200000 (59.65%) | loss: 2.609925 | lrm: 0.48 | dt: 1326.05ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 58 rg: 82 | total time: 2729.28m | eta: 1846.4m
step 119350/200000 (59.67%) | loss: 2.624258 | lrm: 0.48 | dt: 1328.28ms | tok/sec: 12,334 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 1 | total time: 2730.39m | eta: 1845.2m
step 119400/200000 (59.70%) | loss: 2.594838 | lrm: 0.48 | dt: 1327.06ms | tok/sec: 12,346 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 3 | total time: 2731.50m | eta: 1844.0m
step 119450/200000 (59.73%) | loss: 2.638919 | lrm: 0.48 | dt: 1327.70ms | tok/sec: 12,340 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 5 | total time: 2732.61m | eta: 1842.9m
step 119500/200000 (59.75%) | loss: 2.559611 | lrm: 0.47 | dt: 1326.39ms | tok/sec: 12,352 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 7 | total time: 2733.71m | eta: 1841.7m
step 119550/200000 (59.77%) | loss: 2.701004 | lrm: 0.47 | dt: 1329.64ms | tok/sec: 12,322 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 9 | total time: 2734.82m | eta: 1840.5m
step 119600/200000 (59.80%) | loss: 2.685871 | lrm: 0.47 | dt: 1328.69ms | tok/sec: 12,330 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 11 | total time: 2735.93m | eta: 1839.4m
step 119650/200000 (59.83%) | loss: 2.670528 | lrm: 0.47 | dt: 1330.40ms | tok/sec: 12,315 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 13 | total time: 2737.03m | eta: 1838.2m
step 119700/200000 (59.85%) | loss: 2.669780 | lrm: 0.47 | dt: 1330.37ms | tok/sec: 12,315 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 15 | total time: 2738.14m | eta: 1837.0m
step 119750/200000 (59.88%) | loss: 2.643356 | lrm: 0.47 | dt: 1326.15ms | tok/sec: 12,354 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 17 | total time: 2739.25m | eta: 1835.8m
step 119800/200000 (59.90%) | loss: 2.530425 | lrm: 0.47 | dt: 1325.29ms | tok/sec: 12,362 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 19 | total time: 2740.35m | eta: 1834.7m
step 119850/200000 (59.92%) | loss: 2.548207 | lrm: 0.47 | dt: 1326.05ms | tok/sec: 12,355 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 21 | total time: 2741.45m | eta: 1833.5m
step 119900/200000 (59.95%) | loss: 2.616765 | lrm: 0.47 | dt: 1327.85ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 23 | total time: 2742.56m | eta: 1832.3m
step 119950/200000 (59.98%) | loss: 2.589670 | lrm: 0.47 | dt: 1330.60ms | tok/sec: 12,313 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 25 | total time: 2743.67m | eta: 1831.2m
Step 120000 | Validation bpb: 0.964415
<|bos|>The capital of France is the city of L'Ambrosia, which is located in the heart of the Ibidracua
<|bos|>`The chemical symbol of gold is Au`. Gold has a wide range of applications in various fields. Its unique properties and versatility make it a valuable addition
<|bos|>`If yesterday was Friday, then tomorrow will be Saturday.` That's what this winter might be like for Solar and wind energy. And, for the most part
<|bos|>`The opposite of hot is cold`. Cold means that there is a different temperature of the hot fluid (the hestep 119900/200000 (59.95%) | loss: 2.616765 | lrm: 0.47 | dt: 1327.85ms | tok/sec: 12,338 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 23 | total time: 2742.56m | eta: 1832.3m
step 119950/200000 (59.98%) | loss: 2.589670 | lrm: 0.47 | dt: 1330.60ms | tok/sec: 12,313 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 25 | total time: 2743.67m | eta: 1831.2m
Step 120000 | Validation bpb: 0.964415
<|bos|>The planets of the solar system are: Mars
The solar system was made up of three planets
Planets, according to astronomers, are
<|bos|>`My favorite color is blue`. When I first started out, I was so excited about colors that I hadn't had much luck with them before
<|bos|>If 5*x + 3 = 13, then x is 13 because 13 is the base of the second derivative. Since 13 is the first derivative,
2026-03-18 20:45:57,006 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_120000.pt
2026-03-18 20:45:57,013 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_120000.json
2026-03-18 20:45:58,939 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_120000_rank0.pt
step `120000/200000 (60.00%) | loss: 2.582003 | `lrm: 0.47 `| dt: 1626.42ms | tok/sec: 10,073 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 27 | total time: 2744.78m | eta: 1830.0m
step 120050/200000 (60.02%) | loss: 2.667263 | lrm: 0.47 | dt: 1337.85ms | tok/sec: 12,246 | bf16_mfu: 0.00 | epoch: 1 pq: 59 rg: 30 | total time: 2745.89m | eta: 1828.8m

#
