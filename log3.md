
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

