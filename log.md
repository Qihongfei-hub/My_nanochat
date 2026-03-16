(my_project_env) C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat>python -m scripts.base_train --depth=12 --save-every=4000 --num-iterations=22000 --run=dummy --head-dim=64 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=16 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=800 --log-every=100 --eval-every=800 --max-seq-len=512

                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    
Autodetected device type: cuda
2026-03-14 21:26:25,973 - nanochat.common - INFO - Distributed world size: 1
2026-03-14 21:26:25,973 - nanochat.common - WARNING - Peak flops undefined for: NVIDIA GeForce RTX 4070 Laptop GPU, MFU will show as 0%
GPU: NVIDIA GeForce RTX 4070 Laptop GPU | Peak FLOPS (BF16): inf
COMPUTE_DTYPE: torch.bfloat16 (auto-detected: CUDA SM 89 (bf16 supported))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
WARNING: SDPA has no support for sliding window attention (window_pattern='SSSL'). Your GPU utilization will be terrible.
WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Vocab size: 32,768
Model config:
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
Estimated FLOPs per token: 2.205968e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.049497 for depth 12
Scaling the LR for the AdamW parameters ∝1/√(384/768) = 1.414214
Scaling weight decay from 0.280000 to 0.049497 for depth 12
Scaling the LR for the AdamW parameters ∝1/√(384/768) = 1.414214
Using user-provided number of iterations: 22,000
Scaling the LR for the AdamW parameters ∝1/√(384/768) = 1.414214
Using user-provided number of iterations: 22,000
Total number of training tokens: 360,448,000
Using user-provided number of iterations: 22,000
Total number of training tokens: 360,448,000
Total number of training tokens: 360,448,000
Tokens : Scaling params ratio: 10.66
Total training FLOPs estimate: 7.951366e+16
Tokens / micro-batch / rank: 16 x 512 = 8,192
Tokens / micro-batch: 8,192
Total batch size 16,384 => gradient accumulation steps: 2
Step 00000 | Validation bpb: 3.194519
step 00000/22000 (0.00%) | loss: 10.396973 | lrm: 0.00 | dt: 567.83ms | tok/sec: 28,853 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00100/22000 (0.45%) | loss: 7.874814 | lrm: 0.14 | dt: 452.42ms | tok/sec: 36,214 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 5 | total time: 0.68m | eta: 165.8m
step 00200/22000 (0.91%) | loss: 6.588317 | lrm: 0.29 | dt: 452.39ms | tok/sec: 36,216 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 10 | total time: 1.44m | eta: 165.3m
step 00300/22000 (1.36%) | loss: 6.089131 | lrm: 0.43 | dt: 453.10ms | tok/sec: 36,159 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 14 | total time: 2.20m | eta: 164.6m
step 00400/22000 (1.82%) | loss: 5.696850 | lrm: 0.57 | dt: 451.24ms | tok/sec: 36,308 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 18 | total time: 2.96m | eta: 163.8m
step 00500/22000 (2.27%) | loss: 5.383866 | lrm: 0.72 | dt: 447.06ms | tok/sec: 36,648 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 22 | total time: 3.71m | eta: 162.8m
step 00600/22000 (2.73%) | loss: 5.219279 | lrm: 0.86 | dt: 452.86ms | tok/sec: 36,178 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 27 | total time: 4.46m | eta: 161.9m
step 00700/22000 (3.18%) | loss: 4.981095 | lrm: 1.00 | dt: 451.93ms | tok/sec: 36,253 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 31 | total time: 5.22m | eta: 161.1m
Step 00800 | Validation bpb: 1.476396
<|bos|>The capital of France is a great, and it is just a fact that the capital of the power chain is about 8000 tons. There is
<|bos|>The chemical symbol of gold is a common, and it is commonly used in traditional methods. It is also found in copper and mineral is a common method that
<|bos|>The capital of France is a great, and it is just a fact that the capital of the power chain is about 8000 tons. There is
<|bos|>The chemical symbol of gold is a common, and it is commonly used in traditional methods. It is also found <|bos|>The capital of France is a great, and it is just a fact that the capital of the power chain is about 8000 tons. There is
<|bos|>The capital of France is a great, and it is just a fact that the capital of the power chain is about 8000 <|bos|>The capital of France is a great, and it is just a fact that the capital of the power chain is about 8000 tons. There is
<|bos|>The chemical symbol of gold is a common, and it is commonly used in traditional methods. It is also found in copper and mineral is a common method that
<|bos|>The capital of France is a great, and it is just a fact that the capital of the power chain is about 8000 tons. There is
<|bos|>The capital of France is a great, and it is just a fact that the capital of the power chain is about 8000 <|bos|>The capital of France is a great, and it is just a fact that the capital of the power chain is about 8000 tons. There is
<|bos|>The chemical symbol of gold is a common, and it is commonly used in traditional methods. It is also found in copper and mineral is a common method that
<|bos|>If yesterday was Friday, then tomorrow will be a few years ago. Now you were exploring the world of new "over" world and from what they did to learn,
<|bos|>The opposite of hot is the most common type, but in my part, the first thing I noticed. I've stopped at 300-20
<|bos|>The planets of the solar system are: the Moon will be the last one.
Why should you take a lot of energy?
There are no more than 2.
<|bos|>My favorite color is a great taste. It is an excellent form of color. The first color color is the original color of the color and color
<|bos|>If 5*x + 3 = 13, then x is 5 - 3 = 7 is 3 = 6 = 6, 2 - 3 =
step 00800/22000 (3.64%) | loss: 4.834725 | lrm: 1.00 | dt: 472.01ms | tok/sec: 34,711 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 35 | total time: 5.97m | eta: 160.3m
step 00900/22000 (4.09%) | loss: 4.715682 | lrm: 1.00 | dt: 449.91ms | tok/sec: 36,416 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 40 | total time: 6.73m | eta: 159.5m
step 01000/22000 (4.55%) | loss: 4.545008 | lrm: 1.00 | dt: 453.30ms | tok/sec: 36,144 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 44 | total time: 7.48m | eta: 158.8m
step 01100/22000 (5.00%) | loss: 4.512958 | lrm: 1.00 | dt: 452.22ms | tok/sec: 36,230 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 48 | total time: 8.24m | eta: 158.0m
step 01200/22000 (5.45%) | loss: 4.422898 | lrm: 1.00 | dt: 450.55ms | tok/sec: 36,364 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 52 | total time: 8.99m | eta: 157.2m
step 01300/22000 (5.91%) | loss: 4.384027 | lrm: 1.00 | dt: 452.32ms | tok/sec: 36,221 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 57 | total time: 9.75m | eta: 156.4m
step 01400/22000 (6.36%) | loss: 4.397570 | lrm: 1.00 | dt: 453.47ms | tok/sec: 36,130 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 61 | total time: 10.50m | eta: 155.6m
step 01500/22000 (6.82%) | loss: 4.284790 | lrm: 1.00 | dt: 454.51ms | tok/sec: 36,047 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 65 | total time: 11.26m | eta: 154.9m
Step 01600 | Validation bpb: 1.316800
<|bos|>The capital of France is the largest in the country, which is what is the largest in the world.
The capital of Spain was 2017 and
<|bos|>The chemical symbol of gold is the symbol of gold, gold, and gold. The gold is the symbol of gold and gold, but the gold has been
<|bos|>If yesterday was Friday, then tomorrow will be able to bring up a new homepage game with 240 and 480 members to bring home this game with $
<|bos|>The opposite of hot is the same kind of effect as hot is the same. The same effect is in the air, as it does not matter,
<|bos|>The planets of the solar system are: the solar sun, solar power, sunlight, and the sun's brightness. It is<|bos|>The chemical symbol of gold is the symbol of gold, gold, and gold. The gold is the symbol of gold and gold, but the gold has been
<|bos|>If yesterday was Friday, then tomorrow will be able to bring up a new homepage game with 240 and 480 members to bring home this game with $
<|bos|>The opposite of hot is the same kind of effect as hot is the same. The same effect is in the air, as it does not matter,
<|bos|>The planets of the solar system are: the solar sun, solar power, sunlight, and the sun's brightness. It isrs to bring home this game with $
<|bos|>The opposite of hot is the same kind of effect as hot is the same. The same effect is in the air, as it does not matter,
<|bos|>The planets of the solar system are: the solar sun, solar power, sunlight, and the sun's brightness. It is<|bos|>The opposite of hot is the same kind of effect as hot is the same. The same effect is in the air, as it does not matter,
<|bos|>The planets of the solar system are: the solar sun, solar power, sunlight, and the sun's brightness. It ises not matter,
<|bos|>The planets of the solar system are: the solar sun, solar power, sunlight, and the sun's brightness. It is<|bos|>The planets of the solar system are: the solar sun, solar power, sunlight, and the sun's brightness. It is a dark brown light that is black with
 a dark brown light that is black with
<|bos|>My favorite color is the color you want. My colors are similar to the color you want to look like blue. You want to look for color
u want to look for color
<|bos|>If 5*x + 3 = 13, then x is 5 (s 0 + 0 = 0 + 0, then x is 1) x 0
<|bos|>If 5*x + 3 = 13, then x is 5 (s 0 + 0 = 0 + 0, then x is 1) x 0
step 01600/22000 (7.27%) | loss: 4.285885 | lrm: 1.00 | dt: 494.96ms | tok/sec: 33,101 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 70 | total time: 12.01m | eta: 154.1m
step 01700/22000 (7.73%) | loss: 4.204241 | lrm: 1.00 | dt: 454.15ms | tok/sec: 36,075 | bf16_mfu: 0.00 | epoch: step 01600/22000 (7.27%) | loss: 4.285885 | lrm: 1.00 | dt: 494.96ms | tok/sec: 33,101 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 70 | total time: 12.01m | eta: 154.1m
step 01700/22000 (7.73%) | loss: 4.204241 | lrm: 1.00 | dt: 454.15ms | tok/sec: 36,075 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 74 | total time: 12.77m | eta: 153.4m
1 pq: 0 rg: 70 | total time: 12.01m | eta: 154.1m
step 01700/22000 (7.73%) | loss: 4.204241 | lrm: 1.00 | dt: 454.15ms | tok/sec: 36,075 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 74 | total time: 12.77m | eta: 153.4m
1 pq: 0 rg: 74 | total time: 12.77m | eta: 153.4m
step 01800/22000 (8.18%) | loss: 4.250202 | lrm: 1.00 | dt: 454.75ms | tok/sec: 36,028 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 78 | total time: 13.52m | eta: 152.6m
step 01900/22000 (8.64%) | loss: 4.197877 | lrm: 1.00 | dt: 451.72ms | tok/sec: 36,270 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 83 | total time: 14.28m | eta: 151.9m
step 02000/22000 (9.09%) | loss: 4.149134 | lrm: 1.00 | dt: 448.21ms | tok/sec: 36,554 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 3 | total time: 15.03m | eta: 151.1m
step 02100/22000 (9.55%) | loss: 4.142405 | lrm: 1.00 | dt: 450.33ms | tok/sec: 36,382 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 7 | total time: 15.78m | eta: 150.3m
step 02200/22000 (10.00%) | loss: 4.090506 | lrm: 1.00 | dt: 452.02ms | tok/sec: 36,246 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 11 | total time: 16.54m | eta: 149.5m
step 02300/22000 (10.45%) | loss: 4.148544 | lrm: 1.00 | dt: 452.17ms | tok/sec: 36,234 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 16 | total time: 17.29m | eta: 148.7m
Step 02400 | Validation bpb: 1.262453
<|bos|>The capital of France is the largest international organization of the land known to be the largest international and international country of international origin. This country is known for
<|bos|>The chemical symbol of gold is the symbol of gold and silver is the symbol of gold. Gold is the symbol of gold and silver, and it is the
<|bos|>If yesterday was Friday, then tomorrow will be the first time the first public has been opened. The first public has been the first time the next 15 years in the
<|bos|>The opposite of hot is the heat pump. A pump has a compressor pump, and the evaporator pump operates on the compressor pump.
I have no idea
<|bos|>The planets of the solar system are: the planets of the planets of the planets, the planets of the planets, the planets of the planets and the planets of the
<|bos|>My favorite color is a bit of a bit of color using a piece of paper, using a sharp knife and a scissors, to cut down the
<|bos|>If 5*x + 3 = 13, then x is 3 *x = 2.
Lith
For a two day x is 3 = 14, 1
step 02400/22000 (10.91%) | loss: 4.132757 | lrm: 1.00 | dt: 478.48ms | tok/sec: 34,241 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 20 | total time: 18.04m | eta: 148.0m
step 02500/22000 (11.36%) | loss: 4.099538 | lrm: 1.00 | dt: 449.66ms | tok/sec: 36,436 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 24 | total time: 18.79m | eta: 147.2m
step 02600/22000 (11.82%) | loss: 4.046348 | lrm: 1.00 | dt: 453.25ms | tok/sec: 36,148 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 29 | total time: 19.55m | eta: 146.4m
step 02700/22000 (12.27%) | loss: 4.030886 | lrm: 1.00 | dt: 449.80ms | tok/sec: 36,424 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 33 | total time: 20.30m | eta: 145.6m
step 02800/22000 (12.73%) | loss: 3.967545 | lrm: 1.00 | dt: 453.81ms | tok/sec: 36,103 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 37 | total time: 21.05m | eta: 144.8m
step 02900/22000 (13.18%) | loss: 4.038313 | lrm: 1.00 | dt: 462.23ms | tok/sec: 35,445 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 42 | total time: 21.80m | eta: 144.1m
step 03000/22000 (13.64%) | loss: 4.047466 | lrm: 1.00 | dt: 446.19ms | tok/sec: 36,719 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 46 | total time: 22.55m | eta: 143.3m
step 03100/22000 (14.09%) | loss: 4.037325 | lrm: 1.00 | dt: 448.21ms | tok/sec: 36,554 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 50 | total time: 23.30m | eta: 142.5m
Step 03200 | Validation bpb: 1.231248
<|bos|>The capital of France is the capital of France, where people, their minds, and their imagination are met. But what's in the capital of France
<|bos|>The chemical symbol of gold is a symbol of the "global warming" of the earth, in which there is no mass and a mass of other species
<|bos|>If yesterday was Friday, then tomorrow will be the first time that we're all being encouraged by the idea of going to another place of the last week of the first month
<|bos|>The opposite of hot is the same as the cold. This is actually the same as the cold. If you use cold as your heat, the temperature
<|bos|>The planets of the solar system are: the sun of the solar system and the sun of the solar system. Our Sun is located in the Sun. The Sun is
<|bos|>My favorite color is the white or white, while the white is the white. It is brown with a brown or red and is a black and
<|bos|>If 5*x + 3 = 13, then x is 3 and 5 = 6.2. It is often noted that 10*x + 3.2
step 03200/22000 (14.55%) | loss: 3.934252 | lrm: 1.00 | dt: 470.70ms | tok/sec: 34,807 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 55 | total time: 24.05m | eta: 141.8m
step 03300/22000 (15.00%) | loss: 3.985535 | lrm: 1.00 | dt: 450.32ms | tok/sec: 36,383 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 59 | total time: 24.81m | eta: 141.0m
step 03400/22000 (15.45%) | loss: 3.940081 | lrm: 1.00 | dt: 450.03ms | tok/sec: 36,406 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 63 | total time: 25.56m | eta: 140.2m
step 03500/22000 (15.91%) | loss: 3.948497 | lrm: 1.00 | dt: 449.12ms | tok/sec: 36,480 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 67 | total time: 26.31m | eta: 139.4m


### 22:03
step 03700/22000 (16.82%) | loss: 3.906464 | lrm: 1.00 | dt: 446.39ms | tok/sec: 36,703 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 76 | total time: 27.81m | eta: 137.9m
step 03800/22000 (17.27%) | loss: 3.891526 | lrm: 1.00 | dt: 450.96ms | tok/sec: 36,331 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 80 | total time: 28.57m | eta: 137.2m
step 03900/22000 (17.73%) | loss: 3.939152 | lrm: 1.00 | dt: 448.36ms | tok/sec: 36,542 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 1 | total time: 29.32m | eta: 136.4m
Step 04000 | Validation bpb: 1.209994
<|bos|>The capital of France is the capital of the world. From the present era of the capital of the country to the capital of the country, the capital
<|bos|>The chemical symbol of gold is a symbol of gold, often with a small yellow or brown. There are 10 elements of gold in gold, including a
<|bos|>If yesterday was Friday, then tomorrow will be the next night. But in this week's Christmas, the first one of you will find that will come along with this amazing
<|bos|>The opposite of hot is the hot glue. It is very similar to hot glue, but when hot glue is applied, it can be used as a
<|bos|>The planets of the solar system are: the solar system is a simple device that displays the solar system's motion.

Solar system can be described by its name as the
<|bos|>My favorite color is red. What is the color of pink??
I love to create colors with pink and red color in the same way,
<|bos|>If 5*x + 3 = 13, then x is 5*x = 12, then x is 13*x = 5*x. For example, the
2026-03-14 21:57:19,643 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\model_004000.pt
2026-03-14 21:57:19,643 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\meta_004000.json
2026-03-14 21:57:20,140 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\optim_004000_rank0.pt
step 04000/22000 (18.18%) | loss: 3.919168 | lrm: 1.00 | dt: 514.81ms | tok/sec: 31,825 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 5 | total time: 30.07m | eta: 135.7m
step 04100/22000 (18.64%) | loss: 3.920458 | lrm: 1.00 | dt: 447.29ms | tok/sec: 36,629 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 9 | total time: 30.82m | eta: 134.9m
step 04200/22000 (19.09%) | loss: 3.896531 | lrm: 1.00 | dt: 448.59ms | tok/sec: 36,523 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 14 | total time: 31.57m | eta: 134.1m
step 04300/22000 (19.55%) | loss: 3.886005 | lrm: 1.00 | dt: 451.16ms | tok/sec: 36,315 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 18 | total time: 32.32m | eta: 133.4m
step 04400/22000 (20.00%) | loss: 3.934749 | lrm: 1.00 | dt: 447.51ms | tok/sec: 36,611 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 22 | total time: 33.07m | eta: 132.6m
step 04500/22000 (20.45%) | loss: 3.887890 | lrm: 1.00 | dt: 448.08ms | tok/sec: 36,565 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 26 | total time: 33.83m | eta: 131.8m
step 04600/22000 (20.91%) | loss: 3.902588 | lrm: 1.00 | dt: 452.68ms | tok/sec: 36,193 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 31 | total time: 34.58m | eta: 131.1m


## 
step 04700/22000 (21.36%) | loss: 3.921921 | lrm: 1.00 | dt: 451.85ms | tok/sec: 36,259 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 35 | total time: 35.33m | eta: 130.3m
Step 04800 | Validation bpb: 1.190126
<|bos|>The capital of France is the capital of the capital of the Kingdom of France. The capital of France is the capital of the Kingdom of France.
The
<|bos|>The chemical symbol of gold is a symbol of gold. It is one of the most significant symbol of gold's gold's gold color, its gold's gold
<|bos|>If yesterday was Friday, then tomorrow will be Friday. A few days ago, if he was the first to pass it and he had to stop. This is one of
<|bos|>The opposite of hot is the sun at the equator. In the cold water, the sun also starts to rise in the opposite direction of the sun.
<|bos|>The planets of the solar system are: the planets have the solar system that gives them the ability to convert waste into something that they can’t imagine, and this is
step 04700/22000 (21.36%) | loss: 3.921921 | lrm: 1.00 | dt: 451.85ms | tok/sec: 36,259 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 35 | total time: 35.33m | eta: 130.3m
Step 04800 | Validation bpb: 1.190126
<|bos|>The capital of France is the capital of the capital of the Kingdom of France. The capital of France is the capital of the Kingdom of France.
The
<|bos|>The chemical symbol of gold is a symbol of gold. It is one of the most significant symbol of gold's gold's gold color, its gold's gold
<|bos|>If yesterday was Friday, then tomorrow will be Friday. A few days ago, if he was the first to pass it and he had to stop. This is one of
<|bos|>The opposite of hot is the sun at the equator. In the cold water, the sun also starts to rise in the opposite direction of the sun.
<|bos|>The planets of the solar system are: the planets have the solar system that gives them the ability to convert waste into something that they can’t imagine, and this is
Step 04800 | Validation bpb: 1.190126
<|bos|>The capital of France is the capital of the capital of the Kingdom of France. The capital of France is the capital of the Kingdom of France.
The
<|bos|>The chemical symbol of gold is a symbol of gold. It is one of the most significant symbol of gold's gold's gold color, its gold's gold
<|bos|>If yesterday was Friday, then tomorrow will be Friday. A few days ago, if he was the first to pass it and he had to stop. This is one of
<|bos|>The opposite of hot is the sun at the equator. In the cold water, the sun also starts to rise in the opposite direction of the sun.
<|bos|>The planets of the solar system are: the planets have the solar system that gives them the ability to convert waste into something that they can’t imagine, and this is
The
<|bos|>The chemical symbol of gold is a symbol of gold. It is one of the most significant symbol of gold's gold's gold color, its gold's gold
<|bos|>If yesterday was Friday, then tomorrow will be Friday. A few days ago, if he was the first to pass it and he had to stop. This is one of
<|bos|>The opposite of hot is the sun at the equator. In the cold water, the sun also starts to rise in the opposite direction of the sun.
<|bos|>The planets of the solar system are: the planets have the solar system that gives them the ability to convert waste into something that they can’t imagine, and this is
he had to stop. This is one of
<|bos|>The opposite of hot is the sun at the equator. In the cold water, the sun also starts to rise in the opposite direction of the sun.
<|bos|>The planets of the solar system are: the planets have the solar system that gives them the ability to convert waste into something that they can’t imagine, and this is
ite direction of the sun.
<|bos|>The planets of the solar system are: the planets have the solar system that gives them the ability to convert waste into something that they can’t imagine, and this is
ert waste into something that they can’t imagine, and this is
<|bos|>My favorite color is the color of the flower. When you go to the flower, it's color is dark and green.    
When you go to
<|bos|>If 5*x + 3 = 13, then x is 5 to 5.0, then x is 5 to 8.0. So if x is 5
step 04800/22000 (21.82%) | loss: 3.913952 | lrm: 1.00 | dt: 473.52ms | tok/sec: 34,600 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 39 | total time: 36.08m | eta: 129.6m
step 04900/22000 (22.27%) | loss: 3.871506 | lrm: 1.00 | dt: 448.37ms | tok/sec: 36,541 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 44 | total time: 36.84m | eta: 128.8m
step 05000/22000 (22.73%) | loss: 3.821950 | lrm: 1.00 | dt: 449.66ms | tok/sec: 36,436 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 48 | total time: 37.58m | eta: 128.0m
step 05100/22000 (23.18%) | loss: 3.833410 | lrm: 1.00 | dt: 448.90ms | tok/sec: 36,497 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 52 | total time: 38.33m | eta: 127.3m
step 05200/22000 (23.64%) | loss: 3.897758 | lrm: 1.00 | dt: 446.29ms | tok/sec: 36,711 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 56 | total time: 39.08m | eta: 126.5m

<|bos|>My favorite color is the color of the flower. When you go to the flower, it's color is dark and green.    
When you go to
<|bos|>If 5*x + 3 = 13, then x is 5 to 5.0, then x is 5 to 8.0. So if x is 5
step 04800/22000 (21.82%) | loss: 3.913952 | lrm: 1.00 | dt: 473.52ms | tok/sec: 34,600 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 39 | total time: 36.08m | eta: 129.6m
step 04900/22000 (22.27%) | loss: 3.871506 | lrm: 1.00 | dt: 448.37ms | tok/sec: 36,541 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 44 | total time: 36.84m | eta: 128.8m
step 05000/22000 (22.73%) | loss: 3.821950 | lrm: 1.00 | dt: 449.66ms | tok/sec: 36,436 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 48 | total time: 37.58m | eta: 128.0m
step 05100/22000 (23.18%) | loss: 3.833410 | lrm: 1.00 | dt: 448.90ms | tok/sec: 36,497 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 52 | total time: 38.33m | eta: 127.3m
step 05200/22000 (23.64%) | loss: 3.897758 | lrm: 1.00 | dt: 446.29ms | tok/sec: 36,711 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 56 | total time: 39.08m | eta: 126.5m
<|bos|>My favorite color is the color of the flower. When you go to the flower, it's color is dark and green.    
When you go to
<|bos|>If 5*x + 3 = 13, then x is 5 to 5.0, then x is 5 to 8.0. So if x is 5
step 04800/22000 (21.82%) | loss: 3.913952 | lrm: 1.00 | dt: 473.52ms | tok/sec: 34,600 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 39 | total time: 36.08m | eta: 129.6m
step 04900/22000 (22.27%) | loss: 3.871506 | lrm: 1.00 | dt: 448.37ms | tok/sec: 36,541 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 44 | total time: 36.84m | eta: 128.8m
step 05000/22000 (22.73%) | loss: 3.821950 | lrm: 1.00 | dt: 449.66ms | tok/sec: 36,436 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 48 | total time: 37.58m | eta: 128.0m
step 05100/22000 (23.18%) | loss: 3.833410 | lrm: 1.00 | dt: 448.90ms | tok/sec: 36,497 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 52 | total time: 38.33m | eta: 127.3m
step 05200/22000 (23.64%) | loss: 3.897758 | lrm: 1.00 | dt: 446.29ms | tok/sec: 36,711 | bf16_mfu: 0.00 | epoch:<|bos|>My favorite color is the color of the flower. When you go to the flower, it's color is dark and green.    
When you go to
<|bos|>If 5*x + 3 = 13, then x is 5 to 5.0, then x is 5 to 8.0. So if x is 5
step 04800/22000 (21.82%) | loss: 3.913952 | lrm: 1.00 | dt: 473.52ms | tok/sec: 34,600 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 39 | total time: 36.08m | eta: 129.6m
step 04900/22000 (22.27%) | loss: 3.871506 | lrm: 1.00 | dt: 448.37ms | tok/sec: 36,541 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 44 | total time: 36.84m | eta: 128.8m
step 05000/22000 (22.73%) | loss: 3.821950 | lrm: 1.00 | dt: 449.66ms | tok/sec: 36,436 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 48 | total time: 37.58m | eta: 128.0m
<|bos|>My favorite color is the color of the flower. When you go to the flower, it's color is dark and green.    
When you go to
<|bos|>If 5*x + 3 = 13, then x is 5 to 5.0, then x is 5 to 8.0. So if x is 5
step 04800/22000 (21.82%) | loss: 3.913952 | lrm: 1.00 | dt: 473.52ms | tok/sec: 34,600 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 39 | total time: 36.08m | eta: 129.6m
step 04900/22000 (22.27%) | loss: 3.871506 | lrm: 1.00 | dt: 448.37ms | tok/sec: 36,541 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 44 | total time: 36.84m | eta: 128.8m
<|bos|>My favorite color is the color of the flower. When you go to the flower, it's color is dark and green.    
When you go to
<|bos|>If 5*x + 3 = 13, then x is 5 to 5.0, then x is 5 to 8.0. So if x is 5
<|bos|>My favorite color is the color of the flower. When you go to the flower, it's color is dark and green.    
<|bos|>My favorite color is the color of the flower. When you go to the flower, it's color is dark and green.    
<|bos|>My favorite color is the color of the flower. When you go to the flower, it's color is dark and green.    
<|bos|>My favorite color is the color of the flower. When you go to the flower, it's color is dark and green.    
When you go to
<|bos|>If 5*x + 3 = 13, then x is 5 to 5.0, then x is 5 to 8.0. So if x is 5
step 04800/22000 (21.82%) | loss: 3.913952 | lrm: 1.00 | dt: 473.52ms | tok/sec: 34,600 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 39 | total time: 36.08m | eta: 129.6m
step 04900/22000 (22.27%) | loss: 3.871506 | lrm: 1.00 | dt: 448.37ms | tok/sec: 36,541 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 44 | total time: 36.84m | eta: 128.8m
step 05000/22000 (22.73%) | loss: 3.821950 | lrm: 1.00 | dt: 449.66ms | tok/sec: 36,436 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 48 | total time: 37.58m | eta: 128.0m
step 05100/22000 (23.18%) | loss: 3.833410 | lrm: 1.00 | dt: 448.90ms | tok/sec: 36,497 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 52 | total time: 38.33m | eta: 127.3m
step 05200/22000 (23.64%) | loss: 3.897758 | lrm: 1.00 | dt: 446.29ms | tok/sec: 36,711 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 56 | total time: 39.08m | eta: 126.5m
step 05300/22000 (24.09%) | loss: 3.814256 | lrm: 1.00 | dt: 453.12ms | tok/sec: 36,158 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 61 | total time: 39.84m | eta: 125.8m
step 05400/22000 (24.55%) | loss: 3.846051 | lrm: 1.00 | dt: 448.38ms | tok/sec: 36,540 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 65 | total time: 40.59m | eta: 125.0m
step 05500/22000 (25.00%) | loss: 3.820654 | lrm: 1.00 | dt: 451.76ms | tok/sec: 36,267 | bf16_mfu: 0.00 | epoch:step 05500/22000 (25.00%) | loss: 3.820654 | lrm: 1.00 | dt: 451.76ms | tok/sec: 36,267 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 69 | total time: 41.34m | eta: 124.3m
Step 05600 | Validation bpb: 1.178916
Step 05600 | Validation bpb: 1.178916
<|bos|>The capital of France is the capital of France, the city of France. The capital of France is the capital of France that represents a significant economic burden
<|bos|>The chemical symbol of gold is a symbol of gold and gold. For example, gold is a white gold color, which is more commonly known as gold color
<|bos|>If yesterday was Friday, then tomorrow will be the first one. If I want to try to get the best news about tomorrow, I would just want to try something out
<|bos|>The opposite of hot is the opposite of hot, when heated at high temperature. The heat to which is the heat of the day, is usually the
<|bos|>The planets of the solar system are: the planets of the planets of the solar system are: the planets of the planets of the planets of the planet of the planets
<|bos|>My favorite color is a dark blue color, it makes out of wood, and is much more expensive than other colors. The dark color makes out
<|bos|>If 5*x + 3 = 13, then x is 5x + x 13+m
How does 5*x + 3 = 17, then x
step 05600/22000 (25.45%) | loss: 3.871602 | lrm: 1.00 | dt: 481.06ms | tok/sec: 34,058 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 74 | total time: 42.09m | eta: 123.5m
step 05700/22000 (25.91%) | loss: 3.814878 | lrm: 1.00 | dt: 452.61ms | tok/sec: 36,198 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 78 | total time: 42.85m | eta: 122.7m
step 05800/22000 (26.36%) | loss: 3.856816 | lrm: 1.00 | dt: 448.93ms | tok/sec: 36,495 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 82 | total time: 43.60m | eta: 122.0m
step 05900/22000 (26.82%) | loss: 3.881757 | lrm: 1.00 | dt: 447.10ms | tok/sec: 36,645 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 3 | total time: 44.35m | eta: 121.2m
step 06000/22000 (27.27%) | loss: 3.847271 | lrm: 1.00 | dt: 449.81ms | tok/sec: 36,424 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 8 | total time: 45.10m | eta: 120.5m
step 06100/22000 (27.73%) | loss: 3.788100 | lrm: 1.00 | dt: 455.31ms | tok/sec: 35,984 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 12 | total time: 45.85m | eta: 119.7m


## 22:20
##
step 06200/22000 (28.18%) | loss: 3.817738 | lrm: 1.00 | dt: 448.51ms | tok/sec: 36,529 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 16 | total time: 46.60m | eta: 119.0m
step 06300/22000 (28.64%) | loss: 3.806128 | lrm: 1.00 | dt: 449.82ms | tok/sec: 36,423 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 21 | total time: 47.35m | eta: 118.2m
Step 06400 | Validation bpb: 1.169679
<|bos|>The capital of France is a country which is a country, located in the capital of the Cicin region, and in Spain. The capital of
<|bos|>The chemical symbol of gold is a symbol of gold's symbol for gold. gold is a symbol of gold in gold. gold also has a symbol of gold
<|bos|>If yesterday was Friday, then tomorrow will be a week in the morning. If Friday was Friday, then the school next night will be going into its morning. After
<|bos|>The opposite of hot is the hot in the universe. This is only the beginning of the universe. We have only one or the same thing and it
<|bos|>The planets of the solar system are: the solar system is a new wave of gravity, the solar system is a new wave of gravity, it is a new wave
<|bos|>My favorite color is a red hue. It's like a rose blue, and it's a purple color. It's also a red shade.

<|bos|>If 5*x + 3 = 13, then x is the same in the 5*x + 3 = 6. 1/2 x + 3 = 
step 06400/22000 (29.09%) | loss: 3.866969 | lrm: 1.00 | dt: 467.47ms | tok/sec: 35,048 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 25 | total time: 48.11m | eta: 117.4m
step 06500/22000 (29.55%) | loss: 3.789935 | lrm: 1.00 | dt: 449.51ms | tok/sec: 36,448 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 29 | total time: 48.86m | eta: 116.7m
step 06600/22000 (30.00%) | loss: 3.829403 | lrm: 1.00 | dt: 449.44ms | tok/sec: 36,453 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 34 | total time: 49.61m | eta: 115.9m
step 06700/22000 (30.45%) | loss: 3.848134 | lrm: 1.00 | dt: 449.22ms | tok/sec: 36,472 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 38 | total time: 50.35m | eta: 115.2m
step 06800/22000 (30.91%) | loss: 3.860619 | lrm: 1.00 | dt: 448.61ms | tok/sec: 36,522 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 42 | total time: 51.10m | eta: 114.4m
step 06900/22000 (31.36%) | loss: 3.811013 | lrm: 1.00 | dt: 450.84ms | tok/sec: 36,341 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 47 | total time: 51.85m | eta: 113.6m
step 07000/22000 (31.82%) | loss: 3.775299 | lrm: 1.00 | dt: 450.17ms | tok/sec: 36,395 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 51 | total time: 52.60m | eta: 112.9m


#
step 07100/22000 (32.27%) | loss: 3.741754 | lrm: 1.00 | dt: 451.67ms | tok/sec: 36,273 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 55 | total time: 53.36m | eta: 112.1m
Step 07200 | Validation bpb: 1.161402
<|bos|>The capital of France is the capital of France, the capital of France. The capital of France is the capital of France. This capital of France is
<|bos|>The chemical symbol of gold is the gold (the symbol of gold) has a symbol of gold, called gold (the symbol of gold), and its symbol
<|bos|>If yesterday was Friday, then tomorrow will be the first time that we could do what we expect. The time I see today is now three to six months. So if
<|bos|>The opposite of hot is the fact that hot water is made from warm water. The heat causes the hot water to boil out, causing the hot water
<|bos|>The planets of the solar system are: the sun rays, sun rays, and UV rays. The sun rays are not the same as the rays of the sun rays
<|bos|>My favorite color is the red-blue. I think of bluegillers as a very close match. If you want to have a dar<|bos|>The planets of the solar system are: the sun rays, sun rays, and UV rays. The sun rays are not the same as the rays of the sun rays
<|bos|>My favorite color is the red-blue. I think of bluegillers as a very close match. If you want to have a darker red
<|bos|>The planets of the solar system are: the sun rays, sun rays, and UV rays. The sun rays are not the same as the rays of the sun rays
<|bos|>My favorite color is the red-blue. I think of bluegillers as a very close match. If you want to have a dar<|bos|>The planets of the solar system are: the sun rays, sun rays, and UV rays. The sun rays are not the same as<|bos|>The planets of the solar system are: the sun rays, sun rays, and UV rays. The sun rays are not the same as<|bos|>The planets of the solar system are: the sun rays, sun rays, and UV rays. The sun rays are not the same as the rays of the sun rays
<|bos|>My favorite color is the red-blue. I think of bluegillers as a very close match. If you want to have a dar<|bos|>The planets of the solar system are: the sun rays, sun rays, and UV rays. The sun rays are not the same as the rays of the sun rays
<|bos|>The planets of the solar system are: the sun rays, sun rays, and UV rays. The sun rays are not the same as the rays of the sun rays
<|bos|>My favorite color is the red-blue. I think of bluegillers as a very close match. If you want to have a darker red
<|bos|>If 5*x + 3 = 13, then x is the same as 3*x + 3 = 13, then x is the same as 3*x +
 the rays of the sun rays
<|bos|>My favorite color is the red-blue. I think of bluegillers as a very close match. If you want to have a darker red
<|bos|>If 5*x + 3 = 13, then x is the same as 3*x + 3 = 13, then x is the same as 3*x +
step 07200/22000 (32.73%) | loss: 3.854763 | lrm: 1.00 | dt: 457.06ms | tok/sec: 35,846 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 59 | total time: 54.11m | eta: 111.4m
step 07300/22000 (33.18%) | loss: 3.809674 | lrm: 1.00 | dt: 449.75ms | tok/sec: 36,428 | bf16_mfu: 0.00 | epoch: the rays of the sun rays
<|bos|>My favorite color is the red-blue. I think of bluegillers as a very close match. If you want to have a darker red
<|bos|>If 5*x + 3 = 13, then x is the same as 3*x + 3 = 13, then x is the same as 3*x +
ker red
<|bos|>If 5*x + 3 = 13, then x is the same as 3*x + 3 = 13, then x is the same as 3*x +
step 07200/22000 (32.73%) | loss: 3.854763 | lrm: 1.00 | dt: 457.06ms | tok/sec: 35,846 | bf16_mfu: 0.00 | epoch:<|bos|>If 5*x + 3 = 13, then x is the same as 3*x + 3 = 13, then x is the same as 3*x +
step 07200/22000 (32.73%) | loss: 3.854763 | lrm: 1.00 | dt: 457.06ms | tok/sec: 35,846 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 59 | total time: 54.11m | eta: 111.4m
step 07300/22000 (33.18%) | loss: 3.809674 | lrm: 1.00 | dt: 449.75ms | tok/sec: 36,428 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 64 | total time: 54.86m | eta: 110.6m
step 07400/22000 (33.64%) | loss: 3.832210 | lrm: 1.00 | dt: 449.29ms | tok/sec: 36,466 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 68 | total time: 55.61m | eta: 109.9m
step 07500/22000 (34.09%) | loss: 3.778653 | lrm: 1.00 | dt: 448.60ms | tok/sec: 36,522 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 72 | total time: 56.36m | eta: 109.1m
step 07600/22000 (34.55%) | loss: 3.739879 | lrm: 1.00 | dt: 448.06ms | tok/sec: 36,566 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 77 | total time: 57.11m | eta: 108.3m
step 07700/22000 (35.00%) | loss: 3.771260 | lrm: 1.00 | dt: 451.04ms | tok/sec: 36,324 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 81 | total time: 57.86m | eta: 107.6m
step 07800/22000 (35.45%) | loss: 3.764075 | lrm: 0.99 | dt: 448.15ms | tok/sec: 36,559 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 3 | total time: 58.61m | eta: 106.8m
step 07900/22000 (35.91%) | loss: 3.750274 | lrm: 0.99 | dt: 452.28ms | tok/sec: 36,225 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 7 | total time: 59.36m | eta: 106.1m
Step 08000 | Validation bpb: 1.154141
<|bos|>The capital of France is the capital of the region of France. One of the capital's oldest examples of the region is the Cairo capital,
<|bos|>The chemical symbol of gold is a symbol of gold's purity and purity. It is a symbol of gold's purity and purity to the point where it can
<|bos|>If yesterday was Friday, then tomorrow will be the next one. The second of the series, the next one for the 2014-16 season, and the first
<|bos|>The opposite of hot is the heat transfer between hot and cold. Some heat transfer between hot and cold may be due to the temperature, but heat transfer
<|bos|>The planets of the solar system are: the sun, the stars, the sky, the stars, and the stars in the Earth's natural system. The Earth's
<|bos|>My favorite color is the yellow yellow. It's also a big bloom that is all it takes.
I think it's not too bad, but
<|bos|>If 5*x + 3 = 13, then x is the same as 3*, as they are twox. We multiply by 4* 5* 13.
2026-03-14 22:28:00,460 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\model_008000.pt
2026-03-14 22:28:00,460 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\meta_008000.json
2026-03-14 22:28:00,958 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\optim_008000_rank0.pt
step 08000/22000 (36.36%) | loss: 3.731017 | lrm: 0.98 | dt: 510.68ms | tok/sec: 32,082 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 12 | total time: 60.11m | eta: 105.3m




step 08100/22000 (36.82%) | loss: 3.766318 | lrm: 0.97 | dt: 449.65ms | tok/sec: 36,437 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 16 | total time: 60.87m | eta: 104.6m
step 08200/22000 (37.27%) | loss: 3.730379 | lrm: 0.97 | dt: 451.70ms | tok/sec: 36,271 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 20 | total time: 61.62m | eta: 103.8m
step 08300/22000 (37.73%) | loss: 3.711678 | lrm: 0.96 | dt: 451.44ms | tok/sec: 36,292 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 25 | total time: 62.37m | eta: 103.1m
step 08400/22000 (38.18%) | loss: 3.756954 | lrm: 0.95 | dt: 448.15ms | tok/sec: 36,558 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 29 | total time: 63.12m | eta: 102.3m
step 08500/22000 (38.64%) | loss: 3.760604 | lrm: 0.95 | dt: 449.42ms | tok/sec: 36,456 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 33 | total time: 63.87m | eta: 101.6m
step 08600/22000 (39.09%) | loss: 3.696898 | lrm: 0.94 | dt: 451.03ms | tok/sec: 36,325 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 38 | total time: 64.61m | eta: 100.8m
step 08700/22000 (39.55%) | loss: 3.721053 | lrm: 0.93 | dt: 451.99ms | tok/sec: 36,248 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 42 | total time: 65.36m | eta: 100.0m
Step 08800 | Validation bpb: 1.144464
<|bos|>The capital of France is the capital of the capital of the capital of the capital of the capital of the capital of the capital of the capital of the
<|bos|>The chemical symbol of gold is a symbol of gold. It symbolizes the healing of the body, such as skin, eye, skin, and hair. It
<|bos|>If yesterday was Friday, then tomorrow will be Friday. If this is going to be Friday, then it's ok to leave the church and do it. If there are
<|bos|>The opposite of hot is the fact that the heat is more common than the heat is. But then there is the fact that heat is more common than
<|bos|>The planets of the solar system are: the planets of the sun with no rotation by the sun, the planets of the sun with no rotation by the sun, the
<|bos|>My favorite color is the red variety of red plants you grow in your garden. You just need to know about the species that are native to your
<|bos|>If 5*x + 3 = 13, then x is the number of times 12 and 13^2 = 18.5*, then x is the number of times
step 08800/22000 (40.00%) | loss: 3.764869 | lrm: 0.93 | dt: 457.89ms | tok/sec: 35,781 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 46 | total time: 66.11m | eta: 99.3m
step 08900/22000 (40.45%) | loss: 3.794378 | lrm: 0.92 | dt: 449.79ms | tok/sec: 36,426 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 50 | total time: 66.86m | eta: 98.5m
step 09000/22000 (40.91%) | loss: 3.737776 | lrm: 0.91 | dt: 449.45ms | tok/sec: 36,453 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 55 | total time: 67.61m | eta: 97.8m
step 09100/22000 (41.36%) | loss: 3.738596 | lrm: 0.91 | dt: 450.47ms | tok/sec: 36,370 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 59 | total time: 68.36m | eta: 97.0m
step 09200/22000 (41.82%) | loss: 3.725576 | lrm: 0.90 | dt: 450.01ms | tok/sec: 36,408 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 63 | total time: 69.11m | eta: 96.3m

step 09300/22000 (42.27%) | loss: 3.680403 | lrm: 0.89 | dt: 451.63ms | tok/sec: 36,277 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 68 | total time: 69.87m | eta: 95.5m
step 09400/22000 (42.73%) | loss: 3.681826 | lrm: 0.89 | dt: 449.61ms | tok/sec: 36,440 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 72 | total time: 70.62m | eta: 94.8m
step 09500/22000 (43.18%) | loss: 3.779902 | lrm: 0.88 | dt: 450.43ms | tok/sec: 36,373 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 76 | total time: 71.37m | eta: 94.0m
Step 09600 | Validation bpb: 1.136739
<|bos|>The capital of France is the capital of the United States, and the capital of the United States is the capital of the United States. The capital of
<|bos|>The chemical symbol of gold is a symbol of gold. It was believed that gold was the symbol of gold that was believed to be immortalized. In
<|bos|>If yesterday was Friday, then tomorrow will be the day that the 8th was there but not the 14th.

The second day will be the day that the
<|bos|>The opposite of hot is the heat transfer. Heat is used to heat one or more of the medium or medium, creating a warm and dry surface.
<|bos|>The planets of the solar system are: the sun rays, solar wind, solar dust, solar radiation, solar glass, solar glass, solar panels, solar glass,
<|bos|>My favorite color is the red-green color, which can be slightly creamier than the silver, or the white-white or red green color, which
<|bos|>If 5*x + 3 = 13, then x is the number of times 5*x + 3*x + 4*x + 4*x + 
step 09600/22000 (43.64%) | loss: 3.699670 | lrm: 0.87 | dt: 463.28ms | tok/sec: 35,365 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 81 | total time: 72.12m | eta: 93.2m
step 09700/22000 (44.09%) | loss: 3.760752 | lrm: 0.87 | dt: 449.87ms | tok/sec: 36,419 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 3 | total time: 72.87m | eta: 92.5m
step 09800/22000 (44.55%) | loss: 3.752346 | lrm: 0.86 | dt: 453.74ms | tok/sec: 36,109 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 7 | total time: 73.62m | eta: 91.7m
step 09900/22000 (45.00%) | loss: 3.671261 | lrm: 0.85 | dt: 449.93ms | tok/sec: 36,414 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 11 | total time: 74.37m | eta: 91.0m
step 10000/22000 (45.45%) | loss: 3.723935 | lrm: 0.85 | dt: 454.72ms | tok/sec: 36,031 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 16 | total time: 75.12m | eta: 90.2m
step 10100/22000 (45.91%) | loss: 3.664166 | lrm: 0.84 | dt: 450.08ms | tok/sec: 36,402 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 20 | total time: 75.87m | eta: 89.5m
step 10200/22000 (46.36%) | loss: 3.667012 | lrm: 0.83 | dt: 445.80ms | tok/sec: 36,751 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 24 | total time: 76.62m | eta: 88.7m
step 10300/22000 (46.82%) | loss: 3.684263 | lrm: 0.83 | dt: 448.60ms | tok/sec: 36,522 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 29 | total time: 77.37m | eta: 88.0m
Step 10400 | Validation bpb: 1.128989
<|bos|>The capital of France is the capital of France, with a focus on the capital of the world. There are three main cities that have the largest capital
<|bos|>The chemical symbol of gold is a symbol of gold. The most common means of gold is gold and silver being used to make coins, jewelry, jewellery,
<|bos|>If yesterday was Friday, then tomorrow will be the day, and we can still look like no other in the world. If we don't do this, then that would
<|bos|>The opposite of hot is the most interesting thing about the sun. So far, the sun has been found to have been very cold. The sun has
<|bos|>The planets of the solar system are: the planets have a radius of around 7.5 m and we have about 7.2 km of the moon.
<|bos|>My favorite color is blue. Blue is a white shade. For example, blue is used for shade. Red is a color of blue, but
<|bos|>If 5*x + 3 = 13, then x is the same as x. That doesn't mean x is the same as x if x is the same as x. You don
step 10400/22000 (47.27%) | loss: 3.679818 | lrm: 0.82 | dt: 465.60ms | tok/sec: 35,189 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 33 | total time: 78.12m | eta: 87.2m
step 10500/22000 (47.73%) | loss: 3.679326 | lrm: 0.81 | dt: 450.19ms | tok/sec: 36,393 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 37 | total time: 78.87m | eta: 86.5m
step 10600/22000 (48.18%) | loss: 3.692173 | lrm: 0.81 | dt: 446.64ms | tok/sec: 36,683 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 42 | total time: 79.62m | eta: 85.7m
step 10700/22000 (48.64%) | loss: 3.659667 | lrm: 0.80 | dt: 449.60ms | tok/sec: 36,440 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 46 | total time: 80.37m | eta: 85.0m
step 10800/22000 (49.09%) | loss: 3.655725 | lrm: 0.79 | dt: 450.38ms | tok/sec: 36,378 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 50 | total time: 81.12m | eta: 84.2m
step 10900/22000 (49.55%) | loss: 3.707320 | lrm: 0.79 | dt: 451.54ms | tok/sec: 36,284 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 54 | total time: 81.87m | eta: 83.4m
step 11000/22000 (50.00%) | loss: 3.698066 | lrm: 0.78 | dt: 450.84ms | tok/sec: 36,340 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 59 | total time: 82.62m | eta: 82.7m
step 11100/22000 (50.45%) | loss: 3.691860 | lrm: 0.77 | dt: 447.65ms | tok/sec: 36,600 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 63 | total time: 83.37m | eta: 81.9m
Step 11200 | Validation bpb: 1.121732
<|bos|>The capital of France is the capital of the country known as the capital of the country. The capital of the country is named in honor of the capital
<|bos|>The chemical symbol of gold is 1.5.8 and 1.5.5. An element of the element element of gold is 1
<|bos|>If yesterday was Friday, then tomorrow will be Friday. Today will have to wait until February. But then it is this month. Today will be Monday. Today will have
<|bos|>The opposite of hot is the heat produced by heating the metal around the container. The heat emitted by a metal can be absorbed in a way that means
<|bos|>The planets of the solar system are: 1. The planets of the solar system are: 1. The planets of the solar system are: 2.
<|bos|>My favorite color is red. What is it? A great colored tire has a color similar to those of the red. I can't help but
<|bos|>If 5*x + 3 = 13, then x is the number of times the number of times x is the number of times x is the number of times y is the number of
step 11200/22000 (50.91%) | loss: 3.645343 | lrm: 0.77 | dt: 459.05ms | tok/sec: 35,690 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 67 | total time: 84.12m | eta: 81.2m
step 11300/22000 (51.36%) | loss: 3.633755 | lrm: 0.76 | dt: 450.78ms | tok/sec: 36,346 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 72 | total time: 84.87m | eta: 80.4m
step 11400/22000 (51.82%) | loss: 3.599902 | lrm: 0.75 | dt: 449.21ms | tok/sec: 36,472 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 76 | total time: 85.62m | eta: 79.7m
step 11500/22000 (52.27%) | loss: 3.631308 | lrm: 0.75 | dt: 447.70ms | tok/sec: 36,595 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 80 | total time: 86.37m | eta: 78.9m
step 11600/22000 (52.73%) | loss: 3.617379 | lrm: 0.74 | dt: 448.23ms | tok/sec: 36,552 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 2 | total time: 87.11m | eta: 78.2m
step 11700/22000 (53.18%) | loss: 3.633921 | lrm: 0.73 | dt: 448.83ms | tok/sec: 36,503 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 6 | total time: 87.86m | eta: 77.4m
step 11800/22000 (53.64%) | loss: 3.671100 | lrm: 0.73 | dt: 449.65ms | tok/sec: 36,437 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 10 | total time: 88.61m | eta: 76.7m
step 11900/22000 (54.09%) | loss: 3.665442 | lrm: 0.72 | dt: 452.26ms | tok/sec: 36,226 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 14 | total time: 89.36m | eta: 75.9m
Step 12000 | Validation bpb: 1.115631
<|bos|>The capital of France is the capital of the city of Leibniz, the capital of the town of Chenogun. The capital of
<|bos|>The chemical symbol of gold is gold, symbol of gold is 13
Northern gold is yellow green in color with a red tint
The blue-green
<|bos|>If yesterday was Friday, then tomorrow will be the day in which the kids learned. She made a good friend so she was able to hold them for the day, then
<|bos|>The opposite of hot is the hot part of a hot hot dish. What is hot? Well, there is hot hot dish that you can soak in
<|bos|>The planets of the solar system are: the sun in the sky, the clouds in the sky, and the surface of the planet. This planet is the most stable
<|bos|>My favorite color is gray. In the case of black, some areas of the color color appears more dark than others. This is because they are
<|bos|>If 5*x + 3 = 13, then x is the number of stars. (2) Let 5*x + 3 = 13, and x is the number
2026-03-14 22:58:38,255 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\model_012000.pt
2026-03-14 22:58:38,255 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\meta_012000.json
2026-03-14 22:58:38,748 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\optim_012000_rank0.pt
step 12000/22000 (54.55%) | loss: 3.648049 | lrm: 0.71 | dt: 498.87ms | tok/sec: 32,842 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 19 | total time: 90.11m | eta: 75.2m
step 12100/22000 (55.00%) | loss: 3.624381 | lrm: 0.71 | dt: 450.27ms | tok/sec: 36,386 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 23 | total time: 90.86m | eta: 74.4m
step 12200/22000 (55.45%) | loss: 3.646500 | lrm: 0.70 | dt: 448.78ms | tok/sec: 36,508 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 27 | total time: 91.61m | eta: 73.6m
step 12300/22000 (55.91%) | loss: 3.611384 | lrm: 0.69 | dt: 447.27ms | tok/sec: 36,631 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 32 | total time: 92.35m | eta: 72.9m
step 12400/22000 (56.36%) | loss: 3.614138 | lrm: 0.69 | dt: 450.14ms | tok/sec: 36,397 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 36 | total time: 93.10m | eta: 72.1m
step 12500/22000 (56.82%) | loss: 3.656091 | lrm: 0.68 | dt: 447.34ms | tok/sec: 36,625 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 40 | total time: 93.85m | eta: 71.4m
step 12600/22000 (57.27%) | loss: 3.621444 | lrm: 0.67 | dt: 449.61ms | tok/sec: 36,440 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 44 | total time: 94.60m | eta: 70.6m
step 12700/22000 (57.73%) | loss: 3.605517 | lrm: 0.67 | dt: 446.03ms | tok/sec: 36,733 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 49 | total time: 95.35m | eta: 69.9m
Step 12800 | Validation bpb: 1.109549
<|bos|>The capital of France is the capital of the United States in the area of the United States, and has been declared as the nation's capital for the
<|bos|>The chemical symbol of gold is gold. gold is the colour of gold. It is the most colourless element in the universe. The earth is black.
<|bos|>If yesterday was Friday, then tomorrow will be Wednesday. We'll have to look out for Wednesday's face, where we could see some of the world's most magnificent birds
<|bos|>The opposite of hot is the heat which is absorbed by the skin after birth. This heat is absorbed and absorbed. The amount of heat absorbed varies by
<|bos|>The planets of the solar system are: the planets can be classified as solar radiation according to their position. Most of the solar radiation is caused by the suns radiation
<|bos|>My favorite color is red, white, or grey. These are usually red, but most people are not aware of it.
How to make a
<|bos|>If 5*x + 3 = 13, then x is 13 + 13*, if 6*x = 13*x + 4 = 13*x
step 12800/22000 (58.18%) | loss: 3.593183 | lrm: 0.66 | dt: 483.68ms | tok/sec: 33,873 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 53 | total time: 96.10m | eta: 69.1m
step 12900/22000 (58.64%) | loss: 3.655027 | lrm: 0.65 | dt: 450.15ms | tok/sec: 36,396 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 57 | total time: 96.85m | eta: 68.4m
step 13000/22000 (59.09%) | loss: 3.578857 | lrm: 0.65 | dt: 448.20ms | tok/sec: 36,555 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 62 | total time: 97.60m | eta: 67.6m
step 13100/22000 (59.55%) | loss: 3.624997 | lrm: 0.64 | dt: 449.22ms | tok/sec: 36,472 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 66 | total time: 98.34m | eta: 66.9m
step 13200/22000 (60.00%) | loss: 3.652022 | lrm: 0.63 | dt: 454.73ms | tok/sec: 36,029 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 70 | total time: 99.09m | eta: 66.1m
step 13300/22000 (60.45%) | loss: 3.570192 | lrm: 0.63 | dt: 449.63ms | tok/sec: 36,439 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 75 | total time: 99.84m | eta: 65.4m
step 13400/22000 (60.91%) | loss: 3.637350 | lrm: 0.62 | dt: 448.81ms | tok/sec: 36,505 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 79 | total time: 100.59m | eta: 64.6m
step 13500/22000 (61.36%) | loss: 3.569070 | lrm: 0.61 | dt: 450.08ms | tok/sec: 36,402 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 1 | total time: 101.34m | eta: 63.9m
Step 13600 | Validation bpb: 1.103299
<|bos|>The capital of France is the capital of the world of Europe. It is the largest capital of the world, with the total amount of capital spent is
<|bos|>The chemical symbol of gold is the symbol of gold, also known as gold with a symbol of gold. Silver is also a natural symbol of wealth, wealth
<|bos|>If yesterday was Friday, then tomorrow will be Friday. That's not right to say that a few times I just needed to go and run for the week. If I
<|bos|>The opposite of hot is the heat and the pressure exerted on the wire between the wire and the conductor (the pressure) on the wire. When the
<|bos|>The planets of the solar system are: the planets of the sun, which are invisible to the naked eye, and the planets of the Milky Way, which are invisible
<|bos|>My favorite color is the red of the red color (black. (A)). Why is my red red red?
I've been thinking about
<|bos|>If 5*x + 3 = 13, then x is the number of times x is the number of times x is the number of times x is the number of times x is the
step 13600/22000 (61.82%) | loss: 3.580750 | lrm: 0.61 | dt: 475.10ms | tok/sec: 34,485 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 5 | total time: 102.09m | eta: 63.1m
step 13700/22000 (62.27%) | loss: 3.607803 | lrm: 0.60 | dt: 448.66ms | tok/sec: 36,517 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 10 | total time: 102.84m | eta: 62.3m
step 13800/22000 (62.73%) | loss: 3.622764 | lrm: 0.59 | dt: 450.84ms | tok/sec: 36,341 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 14 | total time: 103.59m | eta: 61.6m
step 13900/22000 (63.18%) | loss: 3.581026 | lrm: 0.59 | dt: 448.88ms | tok/sec: 36,499 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 18 | total time: 104.34m | eta: 60.8m
step 14000/22000 (63.64%) | loss: 3.616742 | lrm: 0.58 | dt: 449.90ms | tok/sec: 36,416 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 23 | total time: 105.09m | eta: 60.1m
step 14100/22000 (64.09%) | loss: 3.626124 | lrm: 0.57 | dt: 449.94ms | tok/sec: 36,414 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 27 | total time: 105.83m | eta: 59.3m
step 14200/22000 (64.55%) | loss: 3.540096 | lrm: 0.57 | dt: 446.81ms | tok/sec: 36,668 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 31 | total time: 106.58m | eta: 58.6m
step 14300/22000 (65.00%) | loss: 3.591684 | lrm: 0.56 | dt: 446.85ms | tok/sec: 36,665 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 36 | total time: 107.33m | eta: 57.8m
Step 14400 | Validation bpb: 1.097582
<|bos|>The capital of France is the capital of the capital of Switzerland. The capital of the country is the capital of the capital of France, the capital of
<|bos|>The chemical symbol of gold is the symbol of the earth's atmosphere. Many of the chemical symbol of gold, also known as the symbol of the ocean,
<|bos|>If yesterday was Friday, then tomorrow will be the day before the new record for the past 12 months, then the previous record for the current month was the same month
<|bos|>The opposite of hot is the hot temperature. It is also called hot hot. The temperature rises on top of a hot water column, and rises on
<|bos|>The planets of the solar system are: Venus, Venus, and Venus together
These planets are the most significant planets. Venus is the only one in the Solar System
<|bos|>My favorite color is the red-orange. I think most of the colors I see on the page are white. That means most of the colors
<|bos|>If 5*x + 3 = 13, then x is the same as x = 2.
0
0
0
0
0
0
0
0
0
step 14400/22000 (65.45%) | loss: 3.591543 | lrm: 0.55 | dt: 472.84ms | tok/sec: 34,649 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 40 | total time: 108.08m | eta: 57.1m
step 14500/22000 (65.91%) | loss: 3.571266 | lrm: 0.55 | dt: 449.45ms | tok/sec: 36,453 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 44 | total time: 108.83m | eta: 56.3m
step 14600/22000 (66.36%) | loss: 3.615125 | lrm: 0.54 | dt: 447.65ms | tok/sec: 36,599 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 48 | total time: 109.57m | eta: 55.6m
step 14700/22000 (66.82%) | loss: 3.545578 | lrm: 0.53 | dt: 445.22ms | tok/sec: 36,799 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 53 | total time: 110.32m | eta: 54.8m
step 14800/22000 (67.27%) | loss: 3.591862 | lrm: 0.53 | dt: 449.18ms | tok/sec: 36,475 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 57 | total time: 111.07m | eta: 54.1m
step 14900/22000 (67.73%) | loss: 3.551454 | lrm: 0.52 | dt: 448.92ms | tok/sec: 36,496 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 61 | total time: 111.82m | eta: 53.3m
step 15000/22000 (68.18%) | loss: 3.580127 | lrm: 0.52 | dt: 449.64ms | tok/sec: 36,437 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 66 | total time: 112.57m | eta: 52.6m
step 15100/22000 (68.64%) | loss: 3.584812 | lrm: 0.51 | dt: 447.97ms | tok/sec: 36,573 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 70 | total time: 113.32m | eta: 51.8m
Step 15200 | Validation bpb: 1.092152
<|bos|>The capital of France is the capital of the world. There are so many different countries that use the capital of the world for the purpose of providing a
<|bos|>The chemical symbol of gold is the symbol of gold, or symbol of gold. It is also related to the silver symbol of the Earth. The term '
<|bos|>If yesterday was Friday, then tomorrow will be the day after the last few days of the week. You will now be a little late for that weekend. You have to
<|bos|>The opposite of hot is the hot one. It is more efficient than the cold one, only as far as possible. You will see that there is
<|bos|>The planets of the solar system are: the solar system, which is made up of the solar system, and the solar system is made up of the solar system,
<|bos|>My favorite color is red. But I'm not sure where to start. I've only heard a few words about blue and green. What color
<|bos|>If 5*x + 3 = 13, then x is the same as 3*x + 3 = 13;
Note to me that the current equation for x +
step 15200/22000 (69.09%) | loss: 3.589984 | lrm: 0.50 | dt: 476.58ms | tok/sec: 34,377 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 74 | total time: 114.07m | eta: 51.1m
step 15300/22000 (69.55%) | loss: 3.594351 | lrm: 0.50 | dt: 448.42ms | tok/sec: 36,537 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 78 | total time: 114.82m | eta: 50.3m
step 15400/22000 (70.00%) | loss: 3.581946 | lrm: 0.49 | dt: 450.09ms | tok/sec: 36,401 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 1 | total time: 115.56m | eta: 49.6m
step 15500/22000 (70.45%) | loss: 3.509755 | lrm: 0.48 | dt: 450.80ms | tok/sec: 36,344 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 5 | total time: 116.31m | eta: 48.8m
step 15600/22000 (70.91%) | loss: 3.531168 | lrm: 0.48 | dt: 450.34ms | tok/sec: 36,381 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 9 | total time: 117.06m | eta: 48.1m
step 15700/22000 (71.36%) | loss: 3.552929 | lrm: 0.47 | dt: 451.39ms | tok/sec: 36,296 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 14 | total time: 117.81m | eta: 47.3m
step 15800/22000 (71.82%) | loss: 3.522578 | lrm: 0.46 | dt: 447.67ms | tok/sec: 36,598 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 18 | total time: 118.56m | eta: 46.6m
step 15900/22000 (72.27%) | loss: 3.563003 | lrm: 0.46 | dt: 445.95ms | tok/sec: 36,739 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 22 | total time: 119.31m | eta: 45.8m
Step 16000 | Validation bpb: 1.086949
<|bos|>The capital of France is France, in which it is one of the most famous and famous landmarks in French history. France, which was founded in
<|bos|>The chemical symbol of gold is gold. gold is the first pure form of gold. Gold is more in gold than any other element in the world.
Gold
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. However, this is an event that won't be a one-horned event in the game.
There are plenty of
<|bos|>The opposite of hot is the hot heat. It is also heat
between 3.08 and 4.35.
Heat
of a hot
<|bos|>The planets of the solar system are: Jupiter, Saturn, Uranus, Uranus, Uranus and Uranus (in their place they can be found as the
<|bos|>My favorite color is black. And I think they do so quite well. They're in the color of their cars. And I think you could
<|bos|>If 5*x + 3 = 13, then x is the number of degrees. That means y = 12, so x = 13.2.
I am not surprised at
2026-03-14 23:29:13,070 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\model_016000.pt
2026-03-14 23:29:13,072 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\meta_016000.json
2026-03-14 23:29:13,516 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\optim_016000_rank0.pt
step 16000/22000 (72.73%) | loss: 3.514890 | lrm: 0.45 | dt: 476.45ms | tok/sec: 34,387 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 26 | total time: 120.05m | eta: 45.0m
step 16100/22000 (73.18%) | loss: 3.547534 | lrm: 0.44 | dt: 449.02ms | tok/sec: 36,488 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 31 | total time: 120.80m | eta: 44.3m
step 16200/22000 (73.64%) | loss: 3.590976 | lrm: 0.44 | dt: 447.59ms | tok/sec: 36,605 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 35 | total time: 121.55m | eta: 43.5m
step 16300/22000 (74.09%) | loss: 3.472949 | lrm: 0.43 | dt: 447.79ms | tok/sec: 36,588 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 39 | total time: 122.30m | eta: 42.8m
step 16400/22000 (74.55%) | loss: 3.571291 | lrm: 0.42 | dt: 451.18ms | tok/sec: 36,313 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 44 | total time: 123.05m | eta: 42.0m
step 16500/22000 (75.00%) | loss: 3.534511 | lrm: 0.42 | dt: 449.16ms | tok/sec: 36,477 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 48 | total time: 123.80m | eta: 41.3m
step 16600/22000 (75.45%) | loss: 3.524775 | lrm: 0.41 | dt: 447.86ms | tok/sec: 36,582 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 52 | total time: 124.54m | eta: 40.5m
step 16700/22000 (75.91%) | loss: 3.520017 | lrm: 0.40 | dt: 445.75ms | tok/sec: 36,756 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 56 | total time: 125.29m | eta: 39.8m
Step 16800 | Validation bpb: 1.082135
<|bos|>The capital of France is the French and German capital, but this capital is also the capital of the world. This capital is located in the country of
<|bos|>The chemical symbol of gold is the symbol of gold, which stands for gold standard. It is composed of three different chemical elements: copper, zinc, and
<|bos|>If yesterday was Friday, then tomorrow will be Friday. That's a good place to start. There are plenty of reasons why this was happening.
I am not the only
<|bos|>The opposite of hot is the heat source. In the hot air, the heat is not released in the air. In the cool air, the heat
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Mars, Jupiter, Saturn, Jupiter, Saturn, Saturn, Saturn, Mars, Saturn, Saturn
<|bos|>My favorite color is gray. But I don't really know what color I would like to put to it, so I can't really see it
<|bos|>If 5*x + 3 = 13, then x is the number of times that's going to be connected to the 20x of the 20x, and x is the
step 16800/22000 (76.36%) | loss: 3.542244 | lrm: 0.40 | dt: 456.04ms | tok/sec: 35,926 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 61 | total time: 126.04m | eta: 39.0m
step 16900/22000 (76.82%) | loss: 3.565379 | lrm: 0.39 | dt: 446.66ms | tok/sec: 36,681 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 65 | total time: 126.79m | eta: 38.3m
step 17000/22000 (77.27%) | loss: 3.512799 | lrm: 0.38 | dt: 449.92ms | tok/sec: 36,415 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 69 | total time: 127.54m | eta: 37.5m
step 17100/22000 (77.73%) | loss: 3.520754 | lrm: 0.38 | dt: 449.95ms | tok/sec: 36,413 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 74 | total time: 128.29m | eta: 36.8m
step 17200/22000 (78.18%) | loss: 3.517404 | lrm: 0.37 | dt: 447.06ms | tok/sec: 36,648 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 78 | total time: 129.04m | eta: 36.0m
step 17300/22000 (78.64%) | loss: 3.475357 | lrm: 0.36 | dt: 447.52ms | tok/sec: 36,610 | bf16_mfu: 0.00 | epoch: 1 pq: 8 rg: 82 | total time: 129.78m | eta: 35.3m
step 17400/22000 (79.09%) | loss: 3.493913 | lrm: 0.36 | dt: 450.34ms | tok/sec: 36,381 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 4 | total time: 130.53m | eta: 34.5m
step 17500/22000 (79.55%) | loss: 3.494166 | lrm: 0.35 | dt: 449.29ms | tok/sec: 36,466 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 8 | total time: 131.28m | eta: 33.8m
Step 17600 | Validation bpb: 1.077591
<|bos|>The capital of France is the capital of France, and in a relatively short time, the capital of Germany is the capital of Italy, and in a
<|bos|>The chemical symbol of gold is the symbol of gold, which can be found in the metal's official name and is the symbol of silver's purity.
In
<|bos|>If yesterday was Friday, then tomorrow will be Friday. Now, I don't really know the answer, but then I get a bit confused.
If tomorrow was Friday,
<|bos|>The opposite of hot is the heat source. Hot is heat of the ocean. The heat of the water is heat of the sun. The heat of
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Neptune, Saturn, Jupiter, Saturn, Saturn, Saturn, Jupiter, Saturn, Saturn
<|bos|>My favorite color is the red-green color. I like green colored greens. I've learned that they are good for making color for the environment.
<|bos|>If 5*x + 3 = 13, then x is the number of times you want to calculate the weight of the object with the x value of x + 3. In this
step 17600/22000 (80.00%) | loss: 3.613618 | lrm: 0.34 | dt: 487.92ms | tok/sec: 33,579 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 12 | total time: 132.03m | eta: 33.0m
step 17700/22000 (80.45%) | loss: 3.516048 | lrm: 0.34 | dt: 447.20ms | tok/sec: 36,637 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 16 | total time: 132.78m | eta: 32.3m
step 17800/22000 (80.91%) | loss: 3.531373 | lrm: 0.33 | dt: 449.62ms | tok/sec: 36,439 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 21 | total time: 133.53m | eta: 31.5m
step 17900/22000 (81.36%) | loss: 3.496967 | lrm: 0.32 | dt: 450.77ms | tok/sec: 36,346 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 25 | total time: 134.27m | eta: 30.8m
step 18000/22000 (81.82%) | loss: 3.491405 | lrm: 0.32 | dt: 446.84ms | tok/sec: 36,666 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 29 | total time: 135.02m | eta: 30.0m
step 18100/22000 (82.27%) | loss: 3.481445 | lrm: 0.31 | dt: 449.12ms | tok/sec: 36,479 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 34 | total time: 135.77m | eta: 29.3m
step 18200/22000 (82.73%) | loss: 3.539366 | lrm: 0.30 | dt: 449.23ms | tok/sec: 36,471 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 38 | total time: 136.52m | eta: 28.5m
step 18300/22000 (83.18%) | loss: 3.529438 | lrm: 0.30 | dt: 447.74ms | tok/sec: 36,592 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 42 | total time: 137.27m | eta: 27.8m
Step 18400 | Validation bpb: 1.072719
<|bos|>The capital of France is the capital of the country, where the capital of the country is very valuable for its capital. This means that the capital of
<|bos|>The chemical symbol of gold is the symbol of the earth's earth. One of the most common metal used today is silver. This symbol is used for the
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. Then again tomorrow will be Saturday by Saturday. But again on a more positive note, again today will be Saturday by
<|bos|>The opposite of hot is the heat wave. Heat waves cause heat waves to occur in the upper atmosphere in the middle of the Atlantic Ocean. When the
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Uranus, Jupiter, Jupiter, and Saturn. The solar system is composed of two layers:
<|bos|>My favorite color is the red wine. It's very hard to buy a wine, so I used to go with my own red wine and find
<|bos|>If 5*x + 3 = 13, then x is the number of times 5*x + 3 = 8. You can use the x-axis, or 2
step 18400/22000 (83.64%) | loss: 3.545751 | lrm: 0.29 | dt: 474.02ms | tok/sec: 34,563 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 47 | total time: 138.01m | eta: 27.0m
step 18500/22000 (84.09%) | loss: 3.546023 | lrm: 0.28 | dt: 446.05ms | tok/sec: 36,730 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 51 | total time: 138.76m | eta: 26.3m
step 18600/22000 (84.55%) | loss: 3.492068 | lrm: 0.28 | dt: 446.81ms | tok/sec: 36,668 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 55 | total time: 139.51m | eta: 25.5m
step 18700/22000 (85.00%) | loss: 3.541515 | lrm: 0.27 | dt: 447.17ms | tok/sec: 36,638 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 59 | total time: 140.26m | eta: 24.8m
step 18800/22000 (85.45%) | loss: 3.484701 | lrm: 0.26 | dt: 447.02ms | tok/sec: 36,651 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 64 | total time: 141.01m | eta: 24.0m
step 18900/22000 (85.91%) | loss: 3.483731 | lrm: 0.26 | dt: 449.13ms | tok/sec: 36,479 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 68 | total time: 141.75m | eta: 23.3m
step 19000/22000 (86.36%) | loss: 3.543488 | lrm: 0.25 | dt: 451.32ms | tok/sec: 36,302 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 72 | total time: 142.50m | eta: 22.5m
step 19100/22000 (86.82%) | loss: 3.485375 | lrm: 0.24 | dt: 449.73ms | tok/sec: 36,430 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 77 | total time: 143.25m | eta: 21.8m
Step 19200 | Validation bpb: 1.068270
<|bos|>The capital of France is the capital of the world. There are seven districts of France. Here are a few of the best places to visit. The
<|bos|>The chemical symbol of gold is the gold that is the strongest. However, even if the chemical symbol is "Gold", it will still have a negative impact
<|bos|>If yesterday was Friday, then tomorrow will be Friday. However, the end of the week is Monday, so here's an example: For example, a 5.
<|bos|>The opposite of hot is the heat transfer. Heat transfer happens when the two fluids are cool, warm or cool. The difference in heat transfer happens when
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Neptune, Saturn, Jupiter, Saturn, Saturn, Saturn, Neptune, Saturn
<|bos|>My favorite color is red. If I was working with white, I would probably have mixed it with red or yellow in a color that would be
<|bos|>If 5*x + 3 = 13, then x is the number of times 5*x + 3 = 6. 5*x + 3 = 7
step 19200/22000 (87.27%) | loss: 3.473068 | lrm: 0.24 | dt: 476.05ms | tok/sec: 34,416 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 81 | total time: 144.00m | eta: 21.0m
step 19300/22000 (87.73%) | loss: 3.515905 | lrm: 0.23 | dt: 450.96ms | tok/sec: 36,331 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 2 | total time: 144.75m | eta: 20.3m
step 19400/22000 (88.18%) | loss: 3.486256 | lrm: 0.22 | dt: 448.09ms | tok/sec: 36,564 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 6 | total time: 145.50m | eta: 19.5m
step 19500/22000 (88.64%) | loss: 3.443777 | lrm: 0.22 | dt: 450.83ms | tok/sec: 36,341 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 11 | total time: 146.25m | eta: 18.8m
step 19600/22000 (89.09%) | loss: 3.470482 | lrm: 0.21 | dt: 450.26ms | tok/sec: 36,387 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 15 | total time: 147.00m | eta: 18.0m
step 19700/22000 (89.55%) | loss: 3.487283 | lrm: 0.20 | dt: 448.80ms | tok/sec: 36,506 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 19 | total time: 147.75m | eta: 17.3m
step 19800/22000 (90.00%) | loss: 3.399562 | lrm: 0.20 | dt: 451.32ms | tok/sec: 36,302 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 24 | total time: 148.49m | eta: 16.5m
step 19900/22000 (90.45%) | loss: 3.472393 | lrm: 0.19 | dt: 447.99ms | tok/sec: 36,572 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 28 | total time: 149.24m | eta: 15.8m
Step 20000 | Validation bpb: 1.064363
<|bos|>The capital of France is the French capital of the world, so long as it is the least expensive of the other countries.

It is the least expensive
<|bos|>The chemical symbol of gold is the gold that is found in every other crystal. The symbol is located in the center of the gold. It is used for
<|bos|>If yesterday was Friday, then tomorrow will be Friday. However, this is just a sign that the future will be open, and I'm just wondering if it should be
<|bos|>The opposite of hot is the heat sink. Heat sink has a heated head and a hot head. A hot head is hot. The hot head is
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Earth, Saturn, Saturn, Saturn, Saturn, Saturn, Saturn, and Saturn. They are
<|bos|>My favorite color is the blue (or red) on the moon. I don't usually light my moon in the evening because it's so intense
<|bos|>If 5*x + 3 = 13, then x is the number of times 7x x + 3 = 0. (The difference between 3 and 4*
2026-03-14 23:59:47,682 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\model_020000.pt
2026-03-14 23:59:47,682 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\meta_020000.json
2026-03-14 23:59:48,157 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\optim_020000_rank0.pt
step 20000/22000 (90.91%) | loss: 3.398126 | lrm: 0.18 | dt: 493.04ms | tok/sec: 33,230 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 32 | total time: 149.99m | eta: 15.0m
step 20100/22000 (91.36%) | loss: 3.451240 | lrm: 0.18 | dt: 449.81ms | tok/sec: 36,424 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 37 | total time: 150.74m | eta: 14.3m
step 20200/22000 (91.82%) | loss: 3.513795 | lrm: 0.17 | dt: 449.34ms | tok/sec: 36,462 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 41 | total time: 151.49m | eta: 13.5m
step 20300/22000 (92.27%) | loss: 3.455459 | lrm: 0.16 | dt: 446.60ms | tok/sec: 36,686 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 45 | total time: 152.24m | eta: 12.8m
step 20400/22000 (92.73%) | loss: 3.376796 | lrm: 0.16 | dt: 451.28ms | tok/sec: 36,305 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 49 | total time: 152.99m | eta: 12.0m
step 20500/22000 (93.18%) | loss: 3.451589 | lrm: 0.15 | dt: 450.92ms | tok/sec: 36,334 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 54 | total time: 153.74m | eta: 11.3m
step 20600/22000 (93.64%) | loss: 3.468914 | lrm: 0.14 | dt: 445.55ms | tok/sec: 36,772 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 58 | total time: 154.48m | eta: 10.5m
step 20700/22000 (94.09%) | loss: 3.454218 | lrm: 0.14 | dt: 453.20ms | tok/sec: 36,151 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 62 | total time: 155.23m | eta: 9.8m
Step 20800 | Validation bpb: 1.061297
<|bos|>The capital of France is the capital of the world. There are eight markets in the country, however, the market is in a market that is known
<|bos|>The chemical symbol of gold is the gold-petaled colour (yellow) of the colour of diamonds. A diamond's colour is the chemical symbol of the
<|bos|>If yesterday was Friday, then tomorrow will be Friday. There is a little too much "thank you" to your children. I should make it clear that they are
<|bos|>The opposite of hot is the heat.
The hot is hotter than the cold.
The heat is only heat.
A hot gas can cool a home,
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Neptune, Saturn, Jupiter, Saturn, Saturn, Saturn, Neptune, Saturn
<|bos|>My favorite color is red. And I'm not so sure... I'm not really sure what colors are on the white line and what the color
<|bos|>If 5*x + 3 = 13, then x is the number of times 7 will be represented by 4*x + 3 = 10, but 13*
step 20800/22000 (94.55%) | loss: 3.490562 | lrm: 0.13 | dt: 470.51ms | tok/sec: 34,821 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 67 | total time: 155.98m | eta: 9.0m
step 20900/22000 (95.00%) | loss: 3.513712 | lrm: 0.12 | dt: 446.81ms | tok/sec: 36,669 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 71 | total time: 156.73m | eta: 8.3m
step 21000/22000 (95.45%) | loss: 3.442431 | lrm: 0.12 | dt: 450.22ms | tok/sec: 36,390 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 75 | total time: 157.48m | eta: 7.5m
step 21100/22000 (95.91%) | loss: 3.437524 | lrm: 0.11 | dt: 449.57ms | tok/sec: 36,443 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 80 | total time: 158.23m | eta: 6.8m
step 21200/22000 (96.36%) | loss: 3.474599 | lrm: 0.10 | dt: 451.94ms | tok/sec: 36,252 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 2 | total time: 158.98m | eta: 6.0m
step 21300/22000 (96.82%) | loss: 3.428605 | lrm: 0.10 | dt: 448.44ms | tok/sec: 36,535 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 6 | total time: 159.73m | eta: 5.3m
step 21400/22000 (97.27%) | loss: 3.444232 | lrm: 0.09 | dt: 450.00ms | tok/sec: 36,408 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 10 | total time: 160.47m | eta: 4.5m
step 21500/22000 (97.73%) | loss: 3.458054 | lrm: 0.08 | dt: 449.42ms | tok/sec: 36,455 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 15 | total time: 161.22m | eta: 3.8m
Step 21600 | Validation bpb: 1.058223
<|bos|>The capital of France is the capital of the world. There are almost 600 million countries in the world. The capital of the world is the
<|bos|>The chemical symbol of gold is the gold in the family of copper. Each atomic number has a positive, negative, or negative element that makes up the atomic
<|bos|>If yesterday was Friday, then tomorrow will be Friday. There is no "promise" or "send" on this site. This site has a 20
<|bos|>The opposite of hot is the heat wave. Heat waves travel more than electricity. That means if you were to take a long distance hike, and you
<|bos|>The planets of the solar system are: the planets orbiting the Sun, the planets hanging together in the sun, the planets orbiting the Sun, the planets orbiting the Sun
<|bos|>My favorite color is red, yellow, or yellow. These colors are all from the beginning of the color spectrum. Some colorants are just black
<|bos|>If 5*x + 3 = 13, then x is the number of times 7x (6X) is the number of times 3x 5 x 2.
step 21600/22000 (98.18%) | loss: 3.416885 | lrm: 0.08 | dt: 472.35ms | tok/sec: 34,685 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 19 | total time: 161.97m | eta: 3.0m
step 21700/22000 (98.64%) | loss: 3.481673 | lrm: 0.07 | dt: 449.37ms | tok/sec: 36,460 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 23 | total time: 162.72m | eta: 2.3m
step 21800/22000 (99.09%) | loss: 3.455044 | lrm: 0.06 | dt: 449.69ms | tok/sec: 36,434 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 28 | total time: 163.47m | eta: 1.5m
step 21900/22000 (99.55%) | loss: 3.477977 | lrm: 0.06 | dt: 448.95ms | tok/sec: 36,493 | bf16_mfu: 0.00 | epoch: 1 Step 22000 | Validation bpb: 1.057228
<|bos|>The capital of France is the capital of the world. There are almost 600 million members of the world's largest land-grant, but it
<|bos|>The chemical symbol of gold is the symbol of gold, which was coined in 1871 by Robert Burdard (1837) in the book The
<|bos|>If yesterday was Friday, then tomorrow will be Friday. There is no limit to what days today will be, it is 10am-11:00am, the
<|bos|>The opposite of hot is the heat transfer. The heat transfer depends on the temperature of the surface of the hot air. When hot air is cool,
<|bos|>The planets of the solar system are: Jupiter, Saturn, Uranus, Uranus, Uranus, Uranus, Uranus, Uranus, Uranus,
<|bos|>My favorite color is red, yellow, orange, yellow, and even a few shades of pink! I like the way I look at this color
<|bos|>If 5*x + 3 = 13, then x is the number of times 7x3 is 2. The sum of this number is 0x2. For example
2026-03-15 00:15:09,950 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\model_022000.pt
2026-03-15 00:15:09,950 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\meta_022000.json
2026-03-15 00:15:10,434 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d12\optim_022000_rank0.pt
Peak memory usage: 6735.17MiB
Total training time: 164.96m
Minimum validation bpb: 1.057228

(my_project_env) C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat>
