
(my_project_env) C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat>python -m scripts.base_train --depth=16 --save-every=4000 --num-iterations=200000 --run=dummy --head-dim=64 --window-pattern=L --max-seq-len=512 --device-batch-size=8 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=4000 --log-every=50 --eval-every=2000 --max-seq-len=512 --warmup-steps=2000 --warmdown-ratio=0.9 --aspect-ratio=64
C:\Users\hongf\miniconda3.1\envs\my_project_env\Lib\site-packages\torch\cuda\__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]

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
step 02100/200000 (1.05%) | loss: 3.601834 | lrm: 1.00 | dt: 1335.76ms | tok/sec: 12,265 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 3 | total time: 46.73m | eta: 4425.1m
step 02150/200000 (1.07%) | loss: 3.685702 | lrm: 1.00 | dt: 1332.79ms | tok/sec: 12,293 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 5 | total time: 47.85m | eta: 4423.4m
step 02200/200000 (1.10%) | loss: 3.616113 | lrm: 1.00 | dt: 1335.95ms | tok/sec: 12,263 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 7 | total time: 48.96m | eta: 4421.7m
step 02250/200000 (1.12%) | loss: 3.577833 | lrm: 1.00 | dt: 1334.04ms | tok/sec: 12,281 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 9 | total time: 50.07m | eta: 4420.0m
step 02300/200000 (1.15%) | loss: 3.591042 | lrm: 1.00 | dt: 1333.78ms | tok/sec: 12,283 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 11 | total time: 51.18m | eta: 4418.4m