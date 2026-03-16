(my_project_env) C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat>python -m scripts.base_train --depth=16 --save-every=4000 --num-iterations=80000 --run=dummy --head-dim=128 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=8 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=800 --log-every=100 --eval-every=800 --max-seq-len=512

                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░

Autodetected device type: cuda
2026-03-15 03:57:55,686 - nanochat.common - INFO - Distributed world size: 1
2026-03-15 03:57:55,686 - nanochat.common - WARNING - Peak flops undefined for: NVIDIA GeForce RTX 4070 Laptop GPU, MFU will show as 0%
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
  "n_layer": 16,
  "n_head": 6,
  "n_kv_head": 6,
  "vocab_size": 32768,
  "n_layer": 16,
  "n_head": 6,
  "n_kv_head": 6,
  "n_embd": 768,
  "n_head": 6,
  "n_kv_head": 6,
  "n_embd": 768,
  "n_kv_head": 6,
  "n_embd": 768,
  "window_pattern": "SSSL"
}
  "n_embd": 768,
  "window_pattern": "SSSL"
}
  "window_pattern": "SSSL"
}
}
Parameter counts:
wte                     : 25,165,824
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
`step 00100/80000 (0.12%) | loss: 7.604645 | lrm: 0.14 | dt: 1503.10ms | tok/sec: 10,900 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 5 | total time: 2.18m | eta: 1939.5m`
step 00200/80000 (0.25%) | loss: 6.464067 | lrm: 0.29 | dt: 1447.66ms | tok/sec: 11,317 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 10 | total time: 4.62m | eta: 1939.6m
step 00300/80000 (0.38%) | loss: 5.919112 | `lrm: 0.43` | dt: 1448.27ms | tok/sec: 11,312 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 14 | total time: 7.04m | eta: 1935.1m
step 00400/80000 (0.50%) | loss: 5.578771 | `lrm: 0.57` | dt: 1451.43ms | tok/sec: 11,288 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 18 | total time: 9.47m | eta: 1932.6m
step 00500/80000 (0.62%) | loss: 5.238528 | `lrm: 0.72` | dt: 1454.44ms | tok/sec: 11,264 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 22 | total time: 11.89m | eta: 1929.3m
step 00600/80000 (0.75%) | loss: 5.032299 | `lrm: 0.86` | dt: 1450.40ms | tok/sec: 11,296 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 27 | total time: 14.32m | eta: 1926.5m
step 00700/80000 (0.88%) | loss: 4.780718 | `lrm: 1.00 `| dt: 1460.67ms | tok/sec: 11,216 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 31 | total time: 16.74m | eta: 1923.5m
Step 00800 | Validation bpb: 1.420637
<|bos|>The capital of France is a new city of New Zealand in the north of the east of France. He is one of the world's most famous cities
<|bos|>The chemical symbol of gold is a type of chemical element that is composed of material. It is usually defined as the concentration of chemical reactions (the atomic molecules
<|bos|>If yesterday was Friday, then tomorrow will be a few decades ago, she said there is a way to get your interest.
The end of the year is a little guy
<|bos|>The opposite of hot is the same as cold. 4.14 degrees (7.10 degrees in the ground) is 7.10 degrees
<|bos|>The planets of the solar system are: the solar system is the main energy industry. From the solar system is the energy industry in the field. It is used as
<|bos|>My favorite color is a color pattern. The color and color of the color is color shape. In color and color. It color is light and
<|bos|>If 5*x + 3 = 13, then x is 3 = 10, x is 3 = 1,, or 3 = 3. 5 =
step 00800/80000 (1.00%) | loss: 4.659812 | lrm: 1.00 | dt: 1464.24ms | tok/sec: 11,189 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 35 | total time: 19.15m | eta: 1920.1m
step 00900/80000 (1.12%) | loss: 4.547120 | lrm: 1.00 | dt: 1450.07ms | tok/sec: 11,298 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 40 | total time: 21.56m | eta: 1916.6m
step 01000/80000 (1.25%) | loss: 4.418716 | lrm: 1.00 | dt: 1450.96ms | tok/sec: 11,291 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 44 | total time: 23.98m | eta: 1913.5m
step 01100/80000 (1.38%) | loss: 4.355005 | lrm: 1.00 | dt: 1448.01ms | tok/sec: 11,314 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 48 | total time: 26.40m | eta: 1910.8m
step 01200/80000 (1.50%) | loss: 4.253076 | lrm: 1.00 | dt: 1457.94ms | tok/sec: 11,237 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 52 | total time: 28.81m | eta: 1908.0m
step 01300/80000 (1.62%) | loss: 4.258739 | lrm: 1.00 | dt: 1443.95ms | tok/sec: 11,346 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 57 | total time: 31.22m | eta: 1904.7m
step 01400/80000 (1.75%) | loss: 4.243877 | lrm: 1.00 | dt: 1444.01ms | tok/sec: 11,346 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 61 | total time: 33.63m | eta: 1901.7m
step 01500/80000 (1.88%) | loss: 4.161630 | lrm: 1.00 | dt: 1447.14ms | tok/sec: 11,321 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 65 | total time: 36.03m | eta: 1898.4m
Step 01600 | Validation bpb: 1.268534
<|bos|>The capital of France is the capital of France. The capital of France is the capital of France, which is the capital of French French. France is
<|bos|>The chemical symbol of gold is the symbol of the symbol of gold. You can also use the symbol of gold. This symbol may also symbolize gold.

The
<|bos|>If yesterday was Friday, then tomorrow will be the next day. It's going to be for all the other students to attend to, and I am sure you'll be
<|bos|>The opposite of hot is the opposite of the hot. If you eat hot, then it will form your body.
First of all, hot is the
<|bos|>The planets of the solar system are: the sun's energy. The sun's energy is converted to electricity. The heat is then converted to electricity through the system's
<|bos|>My favorite color is the color color. The color will make most of the colors color with an blue color and the color color is blue with white
<|bos|>If 5*x + 3 = 13, then x is 3 = 12, but x is 3 = 0, x is 3 = 10, x is
step 01600/80000 (2.00%) | loss: 4.201417 | lrm: 1.00 | dt: 1463.18ms | tok/sec: 11,197 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 70 | total time: 38.44m | eta: 1895.3m
step 01700/80000 (2.12%) | loss: 4.072316 | lrm: 1.00 | dt: 1448.71ms | tok/sec: 11,309 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 74 | total time: 40.84m | eta: 1892.2m
step 01800/80000 (2.25%) | loss: 4.076183 | lrm: 1.00 | dt: 1441.59ms | tok/sec: 11,365 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 78 | total time: 43.25m | eta: 1889.3m
step 01900/80000 (2.38%) | loss: 4.111737 | lrm: 1.00 | dt: 1439.42ms | tok/sec: 11,382 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 83 | total time: 45.65m | eta: 1886.4m
step 02000/80000 (2.50%) | loss: 4.035882 | lrm: 1.00 | dt: 1442.84ms | tok/sec: 11,355 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 3 | total time: 48.05m | eta: 1883.5m
step 02100/80000 (2.62%) | loss: 3.997901 | lrm: 1.00 | dt: 1440.97ms | tok/sec: 11,370 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 7 | total time: 50.46m | eta: 1880.7m
step 02200/80000 (2.75%) | loss: 3.934102 | lrm: 1.00 | dt: 1440.34ms | tok/sec: 11,375 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 11 | total time: 52.86m | eta: 1877.9m
step 02300/80000 (2.88%) | loss: 4.001818 | lrm: 1.00 | dt: 1446.46ms | tok/sec: 11,326 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 16 | total time: 55.26m | eta: 1875.1m
och: 1 pq: 1 rg: 16 | total time: 55.26m | eta: 1875.1m
Step 02400 | Validation bpb: 1.214327
Step 02400 | Validation bpb: 1.214327
<|bos|>The capital of France is the largest in the country, which is more than 150 years old in the country. Since its completion in September
Since its completion in September
<|bos|>The chemical symbol of gold is the symbol of gold, the symbol of gold, the symbol of gold, the symbol of gold, the symbol of gold,
<|bos|>If yesterday was Friday, then tomorrow will be the first, but it was so hot that the price of a piece of wood was a bit different than it was made of
<|bos|>The opposite of hot is the heat which is a good sign of extreme heat. This heat is a good sign of heat loss and a good sign of
<|bos|>The planets of the solar system are: the planets of the planets of the planets of the planets of the planets of the planets of the planets. the planets of the
<|bos|>My favorite color is red, green, blue, black, and even purple. The color is yellow, blue, and yellow, and I've
<|bos|>If 5*x + 3 = 13, then x is 3.
If you are going to play with 3*x + 3. Now it will be 4*
step 02400/80000 (3.00%) | loss: 3.947846 | lrm: 1.00 | dt: 1464.54ms | tok/sec: 11,187 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 20 | total time: 57.67m | eta: 1872.4m
step 02500/80000 (3.12%) | loss: 3.961550 | lrm: 1.00 | dt: 1437.13ms | tok/sec: 11,400 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 24 | total time: 60.07m | eta: 1869.6m
step 02600/80000 (3.25%) | loss: 3.907709 | lrm: 1.00 | dt: 1445.49ms | tok/sec: 11,334 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 29 | total time: 62.48m | eta: 1867.2m
step 02700/80000 (3.38%) | loss: 3.937029 | lrm: 1.00 | dt: 1447.46ms | tok/sec: 11,319 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 33 | total time: 64.91m | eta: 1865.2m
step 02800/80000 (3.50%) | loss: 3.808748 | lrm: 1.00 | dt: 1447.08ms | tok/sec: 11,322 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 37 | total time: 67.32m | eta: 1862.8m
step 02900/80000 (3.62%) | loss: 3.869679 | lrm: 1.00 | dt: 1447.94ms | tok/sec: 11,315 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 42 | total time: 69.73m | eta: 1860.4m
step 03000/80000 (3.75%) | loss: 3.896093 | lrm: 1.00 | dt: 1447.82ms | tok/sec: 11,316 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 46 | total time: 72.15m | eta: 1857.9m
step 03100/80000 (3.88%) | loss: 3.894404 | lrm: 1.00 | dt: 1444.53ms | tok/sec: 11,342 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 50 | total time: 74.56m | eta: 1855.5m
Step 03200 | Validation bpb: 1.182488
<|bos|>The capital of France is the capital of the United States. The capital of the United States is the capital of the United States and is the capital of
<|bos|>The chemical symbol of gold is the symbol of gold, and silver is the symbol of gold, and gold is the symbol of gold.
What is a gold
<|bos|>If yesterday was Friday, then tomorrow will be the day when the first weather was so long. I would be all down. I have been working with a lot of different
<|bos|>The opposite of hot is the temperature of the air. When the air is cold, it expands as air. If the air is in the middle of
<|bos|>The planets of the solar system are: the sun, the moon, the sun, the sun, and the sun. The solar system is located on the south side
<|bos|>My favorite color is white, black, or purple color. To start, I use colored colors for color. I usually mix white with green with
<|bos|>If 5*x + 3 = 13, then x is the number of times that can be counted so that 0 is the number of times that can be counted so that 2
step 03200/80000 (4.00%) | loss: 3.762668 | lrm: 1.00 | dt: 1456.55ms | tok/sec: 11,248 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 55 | total time: 76.97m | eta: 1853.0m
step 03300/80000 (4.12%) | loss: 3.819349 | lrm: 1.00 | dt: 1445.96ms | tok/sec: 11,330 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 59 | total time: 79.38m | eta: 1850.5m
step 03400/80000 (4.25%) | loss: 3.819523 | lrm: 1.00 | dt: 1445.48ms | tok/sec: 11,334 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 63 | total time: 81.78m | eta: 1847.9m
step 03500/80000 (4.38%) | loss: 3.763354 | lrm: 1.00 | dt: 1440.02ms | tok/sec: 11,377 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 67 | total time: 84.18m | eta: 1845.3m
step 03600/80000 (4.50%) | loss: 3.850353 | lrm: 1.00 | dt: 1443.58ms | tok/sec: 11,349 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 72 | total time: 86.59m | eta: 1842.7m
step 03700/80000 (4.62%) | loss: 3.754589 | lrm: 1.00 | dt: 1447.83ms | tok/sec: 11,316 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 76 | total time: 88.99m | eta: 1840.1m
step 03800/80000 (4.75%) | loss: 3.734163 | lrm: 1.00 | dt: 1443.32ms | tok/sec: 11,351 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 80 | total time: 91.39m | eta: 1837.5m
step 03900/80000 (4.88%) | loss: 3.775414 | lrm: 1.00 | dt: 1459.43ms | tok/sec: 11,226 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 1 | total time: 93.80m | eta: 1834.9m
Step 04000 | Validation bpb: 1.161550
<|bos|>The capital of France is the capital of the capital of the capital of the capital of the capital of the capital of the capital of the capital of the
<|bos|>The chemical symbol of gold is a symbol of the gold's beauty and energy. It is the culmination of the gold's ability to resist the flow of
<|bos|>If yesterday was Friday, then tomorrow will be the next weekend. It is very hot Friday afternoon. It's about the time of the day.
This morning, a huge
<|bos|>The opposite of hot is the temperature of the air. If the air is cold, it starts to boil. In this case, it is called boiling
<|bos|>The planets of the solar system are: the solar system, which is composed of gases and liquids, is found in the solar system. They contain both gases and liquids
<|bos|>My favorite color is red, white, or orange
A vibrant color is a combination of two or more different colors
I can't think of
<|bos|>If 5*x + 3 = 13, then x is 5*x = 3*x + 3 = 6*x = 7*x = 1
2026-03-15 05:35:39,702 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_004000.pt
2026-03-15 05:35:39,702 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_004000.json
2026-03-15 05:35:41,572 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_004000_rank0.pt
step 04000/80000 (5.00%) | loss: 3.750855 | lrm: 1.00 | dt: 1645.87ms | tok/sec: 9,954 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 5 | total time: 96.20m | eta: 1832.4m
step 04100/80000 (5.12%) | loss: 3.750283 | lrm: 1.00 | dt: 1437.57ms | tok/sec: 11,397 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 9 | total time: 98.61m | eta: 1829.9m
step 04200/80000 (5.25%) | loss: 3.744307 | lrm: 1.00 | dt: 1444.92ms | tok/sec: 11,339 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 14 | total time: 101.01m | eta: 1827.3m
step 04300/80000 (5.38%) | loss: 3.726005 | lrm: 1.00 | dt: 1436.46ms | tok/sec: 11,405 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 18 | total time: 103.41m | eta: 1824.7m
step 04400/80000 (5.50%) | loss: 3.802638 | lrm: 1.00 | dt: 1443.95ms | tok/sec: 11,346 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 22 | total time: 105.81m | eta: 1822.2m
step 04500/80000 (5.62%) | loss: 3.719362 | lrm: 1.00 | dt: 1444.33ms | tok/sec: 11,343 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 26 | total time: 108.22m | eta: 1819.7m
step 04600/80000 (5.75%) | loss: 3.780278 | lrm: 1.00 | dt: 1440.24ms | tok/sec: 11,375 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 31 | total time: 110.62m | eta: 1817.1m
step 04700/80000 (5.88%) | loss: 3.802707 | lrm: 1.00 | dt: 1441.27ms | tok/sec: 11,367 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 35 | total time: 113.02m | eta: 1814.6m
Step 04800 | Validation bpb: 1.144974
<|bos|>The capital of France is the capital of the United States, as well as the capital of the world. The capital of France has a large and powerful
<|bos|>The chemical symbol of gold is a symbol of gold. It can be found in gold's gold coins, beads, and other beads.
What is gold's
<|bos|>If yesterday was Friday, then tomorrow will be Friday, then tomorrow, then Monday, then tomorrow, then tomorrow, then Sunday, then tomorrow, tomorrow, then Tuesday,
<|bos|>The opposite of hot is the opposite of hot is the opposite of hot. When the hot is hot, the cold is the opposite of hot and is
<|bos|>The planets of the solar system are: the planets of the system: the planets of the system: the planets of the system: the planets of the system: the
<|bos|>My favorite color is the color of the eyes. When I try to get my eyes blue, there is not much light or any color in the
<|bos|>If 5*x + 3 = 13, then x is the number of times you should buy a unit that is 3 6.
If 5*x = 6,
step 04800/80000 (6.00%) | loss: 3.801063 | lrm: 1.00 | dt: 1464.47ms | tok/sec: 11,187 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 39 | total time: 115.42m | eta: 1812.0m
step 04900/80000 (6.12%) | loss: 3.736050 | lrm: 1.00 | dt: 1438.87ms | tok/sec: 11,386 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 44 | total time: 117.83m | eta: 1809.6m
step 05000/80000 (6.25%) | loss: 3.648293 | lrm: 1.00 | dt: 1442.38ms | tok/sec: 11,358 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 48 | total time: 120.23m | eta: 1807.0m
step 05100/80000 (6.38%) | loss: 3.659778 | lrm: 1.00 | dt: 1440.39ms | tok/sec: 11,374 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 52 | total time: 122.63m | eta: 1804.5m
step 05200/80000 (6.50%) | loss: 3.736786 | lrm: 1.00 | dt: 1436.41ms | tok/sec: 11,406 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 56 | total time: 125.03m | eta: 1802.0m
step 05300/80000 (6.62%) | loss: 3.632764 | lrm: 1.00 | dt: 1443.28ms | tok/sec: 11,351 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 61 | total time: 127.43m | eta: 1799.5m
step 05400/80000 (6.75%) | loss: 3.716644 | lrm: 1.00 | dt: 1443.14ms | tok/sec: 11,353 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 65 | total time: 129.83m | eta: 1797.0m
step 05500/80000 (6.88%) | loss: 3.717792 | lrm: 1.00 | dt: 1443.05ms | tok/sec: 11,353 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 69 | total time: 132.24m | eta: 1794.5m
Step 05600 | Validation bpb: 1.132623
<|bos|>The capital of France is the capital of the United States of the Far East. The capital of France is the capital of the Far East. The capital
<|bos|>The chemical symbol of gold is a gold mineral. The chemical symbol of gold is a gold mineral. The name of the mineral is often called gold.
The
<|bos|>If yesterday was Friday, then tomorrow will be Friday, then tomorrow is Friday tomorrow. When this happened, the two leaders of the 341(c)(3) have
<|bos|>The opposite of hot is the heat is the most difficult and easy to keep up. Heat is the simplest and best way to get to the most quickly
<|bos|>The planets of the solar system are: Jupiter, Jupiter, Jupiter, and Saturn. Jupiter is a gas planet, with a star-like disk, which is known as
<|bos|>My favorite color is the white part of my head which is one of the most popular
traces of my life. The color is very different
<|bos|>If 5*x + 3 = 13, then x is the number of the two nodes (3 * 13) and the number of nodes (3 * 13) will be
step 05600/80000 (7.00%) | loss: 3.667028 | lrm: 1.00 | dt: 1461.89ms | tok/sec: 11,207 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 74 | total time: 134.64m | eta: 1792.0m
step 05700/80000 (7.12%) | loss: 3.725415 | lrm: 1.00 | dt: 1431.31ms | tok/sec: 11,446 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 78 | total time: 137.04m | eta: 1789.5m
step 05800/80000 (7.25%) | loss: 3.668369 | lrm: 1.00 | dt: 1439.95ms | tok/sec: 11,378 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 82 | total time: 139.44m | eta: 1787.0m
step 05900/80000 (7.38%) | loss: 3.709079 | lrm: 1.00 | dt: 1454.63ms | tok/sec: 11,263 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 3 | total time: 141.84m | eta: 1784.5m
step 06000/80000 (7.50%) | loss: 3.674867 | lrm: 1.00 | dt: 1440.71ms | tok/sec: 11,372 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 8 | total time: 144.24m | eta: 1782.0m
step 06100/80000 (7.62%) | loss: 3.654957 | lrm: 1.00 | dt: 1450.14ms | tok/sec: 11,298 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 12 | total time: 146.65m | eta: 1779.5m
step 06200/80000 (7.75%) | loss: 3.636731 | lrm: 1.00 | dt: 1441.58ms | tok/sec: 11,365 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 16 | total time: 149.05m | eta: 1777.0m
step 06300/80000 (7.88%) | loss: 3.632092 | lrm: 1.00 | dt: 1439.19ms | tok/sec: 11,384 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 21 | total time: 151.45m | eta: 1774.5m
Step 06400 | Validation bpb: 1.122389
<|bos|>The capital of France is a city which is located in 1,458 km2 of the town of Gonide de Bourg de la
<|bos|>The chemical symbol of gold is the gold ring. The gold ring is the type of ring that is used on the Earth. The symbol is the color of
<|bos|>If yesterday was Friday, then tomorrow will be a day. It will be 11 weeks back. It will be 1.1 days now, 1.3
<|bos|>The opposite of hot is a hot day. The cold air is around 2,000 degrees and the cold air is about 1,00
<|bos|>The planets of the solar system are: 1. 2. 3. 4. 5. 6. 4. 5. 
<|bos|>My favorite color is red, orange, red, red, and red. I have some super sweet red that I use as a color and I
<|bos|>If 5*x + 3 = 13, then x is the number of times that the total number of times that the total number of times that the total number of times that the total
step 06400/80000 (8.00%) | loss: 3.700888 | lrm: 1.00 | dt: 1464.45ms | tok/sec: 11,187 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 25 | total time: 153.85m | eta: 1772.0m
step 06500/80000 (8.12%) | loss: 3.677020 | lrm: 1.00 | dt: 1444.30ms | tok/sec: 11,343 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 29 | total time: 156.25m | eta: 1769.5m
step 06600/80000 (8.25%) | loss: 3.656864 | lrm: 1.00 | dt: 1445.71ms | tok/sec: 11,332 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 34 | total time: 158.65m | eta: 1767.1m
step 06700/80000 (8.38%) | loss: 3.753624 | lrm: 1.00 | dt: 1435.36ms | tok/sec: 11,414 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 38 | total time: 161.05m | eta: 1764.6m
step 06800/80000 (8.50%) | loss: 3.724127 | lrm: 1.00 | dt: 1441.89ms | tok/sec: 11,362 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 42 | total time: 163.46m | eta: 1762.1m
step 06900/80000 (8.62%) | loss: 3.638087 | lrm: 1.00 | dt: 1440.18ms | tok/sec: 11,376 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 47 | total time: 165.86m | eta: 1759.7m
step 07000/80000 (8.75%) | loss: 3.603623 | lrm: 1.00 | dt: 1441.94ms | tok/sec: 11,362 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 51 | total time: 168.26m | eta: 1757.2m
step 07100/80000 (8.88%) | loss: 3.587487 | lrm: 1.00 | dt: 1440.88ms | tok/sec: 11,370 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 55 | total time: 170.66m | eta: 1754.8m
Step 07200 | Validation bpb: 1.113604
<|bos|>The capital of France is the capital of France, so there are countless things to come in for freeing your free time to free your free time.
<|bos|>The chemical symbol of gold is gold. A gold standard is gold gold. The gold standard is gold gold.

Gold is a pure element with a metal plating
<|bos|>If yesterday was Friday, then tomorrow will be the end of the week. So here's the final picture:
So, tomorrow, we're gonna go on to a whole
<|bos|>The opposite of hot is that the skin is a layer of organic matter. The skin is more layers of organic matter than is present in the air.
<|bos|>The planets of the solar system are: Jupiter, Jupiter, Saturn, Neptune, Jupiter, Saturn, Jupiter, Jupiter, Jupiter, Saturn, Jupiter, Jupiter
<|bos|>My favorite color is blue. And I've just discovered that many of the color variations (especially red and white) are very similar. For example
<|bos|>If 5*x + 3 = 13, then x is 5/5, y is 5/5, y is 1/6, etc, if x 5
step 07200/80000 (9.00%) | loss: 3.690058 | lrm: 1.00 | dt: 1466.43ms | tok/sec: 11,172 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 59 | total time: 173.07m | eta: 1752.3m
step 07300/80000 (9.12%) | loss: 3.668949 | lrm: 1.00 | dt: 1441.46ms | tok/sec: 11,366 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 64 | total time: 175.47m | eta: 1749.9m
step 07400/80000 (9.25%) | loss: 3.647507 | lrm: 1.00 | dt: 1440.99ms | tok/sec: 11,369 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 68 | total time: 177.87m | eta: 1747.4m
step 07500/80000 (9.38%) | loss: 3.606930 | lrm: 1.00 | dt: 1446.58ms | tok/sec: 11,326 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 72 | total time: 180.27m | eta: 1744.9m
step 07600/80000 (9.50%) | loss: 3.538247 | lrm: 1.00 | dt: 1442.07ms | tok/sec: 11,361 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 77 | total time: 182.67m | eta: 1742.5m
step 07700/80000 (9.62%) | loss: 3.596306 | lrm: 1.00 | dt: 1439.21ms | tok/sec: 11,384 | bf16_mfu: 0.00 | epoch: 1 pq: 3 rg: 81 | total time: 185.07m | eta: 1740.0m
step 07800/80000 (9.75%) | loss: 3.579794 | lrm: 1.00 | dt: 1437.27ms | tok/sec: 11,399 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 3 | total time: 187.47m | eta: 1737.5m
step 07900/80000 (9.88%) | loss: 3.602850 | lrm: 1.00 | dt: 1442.00ms | tok/sec: 11,361 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 7 | total time: 189.87m | eta: 1735.1m
Step 08000 | Validation bpb: 1.107124
<|bos|>The capital of France is the capital of the United States in the nation's capital, the capital of France, but the capital's capital is still the
<|bos|>The chemical symbol of gold is the symbol of gold, and its symbol is the symbol of gold. These symbolize the importance of gold, the importance of gold
<|bos|>If yesterday was Friday, then tomorrow will be the next time. The time to start collecting everything that you need to know for your project is, but this is one of
<|bos|>The opposite of hot is the hot. The heat is being reduced by the amount of hot it absorbs (the amount of heat absorbed by the water in
<|bos|>The planets of the solar system are: Jupiter, Jupiter, Saturn, the Jupiter, Jupiter, Jupiter, Saturn, Uranus, Uranus, and Uranus.
<|bos|>My favorite color is red, white, or brown and that's my favorite color. Not only will I bring it home with me, but I
<|bos|>If 5*x + 3 = 13, then x is the number of the digits for this combination. Therefore, the sum of the digits for a 13th of a number is
2026-03-15 07:12:52,951 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_008000.pt
2026-03-15 07:12:52,953 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_008000.json
2026-03-15 07:12:56,347 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_008000_rank0.pt
step 08000/80000 (10.00%) | loss: 3.536129 | lrm: 1.00 | dt: 1784.52ms | tok/sec: 9,181 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 12 | total time: 192.28m | eta: 1732.7m
step 08100/80000 (10.12%) | loss: 3.604156 | lrm: 1.00 | dt: 1449.12ms | tok/sec: 11,306 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 16 | total time: 194.68m | eta: 1730.2m
step 08200/80000 (10.25%) | loss: 3.543938 | lrm: 1.00 | dt: 1442.93ms | tok/sec: 11,354 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 20 | total time: 197.09m | eta: 1727.8m
step 08300/80000 (10.38%) | loss: 3.597667 | lrm: 1.00 | dt: 1435.29ms | tok/sec: 11,415 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 25 | total time: 199.48m | eta: 1725.3m
step 08400/80000 (10.50%) | loss: 3.594599 | lrm: 1.00 | dt: 1437.26ms | tok/sec: 11,399 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 29 | total time: 201.88m | eta: 1722.9m
step 08500/80000 (10.62%) | loss: 3.593795 | lrm: 1.00 | dt: 1440.15ms | tok/sec: 11,376 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 20 | total time: 197.09m | eta: 1727.8m
step 08300/80000 (10.38%) | loss: 3.597667 | lrm: 1.00 | dt: 1435.29ms | tok/sec: 11,415 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 25 | total time: 199.48m | eta: 1725.3m
step 08400/80000 (10.50%) | loss: 3.594599 | lrm: 1.00 | dt: 1437.26ms | tok/sec: 11,399 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 29 | total time: 201.88m | eta: 1722.9m
step 08500/80000 (10.62%) | loss: 3.593795 | lrm: 1.00 | dt: 1440.15ms | tok/sec: 11,376 | bf16_mfu: 0.00 | estep 08300/80000 (10.38%) | loss: 3.597667 | lrm: 1.00 | dt: 1435.29ms | tok/sec: 11,415 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 25 | total time: 199.48m | eta: 1725.3m
step 08400/80000 (10.50%) | loss: 3.594599 | lrm: 1.00 | dt: 1437.26ms | tok/sec: 11,399 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 29 | total time: 201.88m | eta: 1722.9m
step 08500/80000 (10.62%) | loss: 3.593795 | lrm: 1.00 | dt: 1440.15ms | tok/sec: 11,376 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 33 | total time: 204.29m | eta: 1720.4m
step 08600/80000 (10.75%) | loss: 3.452550 | lrm: 1.00 | dt: 1442.44ms | tok/sec: 11,358 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 25 | total time: 199.48m | eta: 1725.3m
step 08400/80000 (10.50%) | loss: 3.594599 | lrm: 1.00 | dt: 1437.26ms | tok/sec: 11,399 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 29 | total time: 201.88m | eta: 1722.9m
step 08500/80000 (10.62%) | loss: 3.593795 | lrm: 1.00 | dt: 1440.15ms | tok/sec: 11,376 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 33 | total time: 204.29m | eta: 1720.4m
step 08600/80000 (10.75%) | loss: 3.452550 | lrm: 1.00 | dt: 1442.44ms | tok/sec: 11,358 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 29 | total time: 201.88m | eta: 1722.9m
step 08500/80000 (10.62%) | loss: 3.593795 | lrm: 1.00 | dt: 1440.15ms | tok/sec: 11,376 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 33 | total time: 204.29m | eta: 1720.4m
step 08600/80000 (10.75%) | loss: 3.452550 | lrm: 1.00 | dt: 1442.44ms | tok/sec: 11,358 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 33 | total time: 204.29m | eta: 1720.4m
step 08600/80000 (10.75%) | loss: 3.452550 | lrm: 1.00 | dt: 1442.44ms | tok/sec: 11,358 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 37 | total time: 206.69m | eta: 1718.0m
step 08700/80000 (10.88%) | loss: 3.615289 | lrm: 1.00 | dt: 1444.77ms | tok/sec: 11,340 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 42 | total time: 209.09m | eta: 1715.5m
Step 08800 | Validation bpb: 1.099441
<|bos|>The capital of France is the capital of the French. French is the capital of France, of the French. Thpoch: 1 pq: 4 rg: 37 | total time: 206.69m | eta: 1718.0m
step 08700/80000 (10.88%) | loss: 3.615289 | lrm: 1.00 | dt: 1444.77ms | tok/sec: 11,340 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 42 | total time: 209.09m | eta: 1715.5m
Step 08800 | Validation bpb: 1.099441
<|bos|>The capital of France is the capital of the French. French is the capital of France, of the French. The capital of the world is called the
<|bos|>The chemical symbol of gold is a symbol of gold, and a symbol of gold is a symbol of gold. The symbol of gold is a symbol of gold
poch: 1 pq: 4 rg: 42 | total time: 209.09m | eta: 1715.5m
Step 08800 | Validation bpb: 1.099441
<|bos|>The capital of France is the capital of the French. French is the capital of France, of the French. The capital of the world is called the
<|bos|>The chemical symbol of gold is a symbol of gold, and a symbol of gold is a symbol of gold. The symbol of gold is a symbol of gold
<|bos|>If yesterday was Friday, then tomorrow will be Friday, but tomorrow is Friday tomorrow. tomorrow is Friday, so is Friday. Friday is Friday is Monday. Friday is Friday
<|bos|>The capital of France is the capital of the French. French is the capital of France, of the French. The capital of the world is called the
<|bos|>The chemical symbol of gold is a symbol of gold, and a symbol of gold is a symbol of gold. The symbol of gold is a symbol of gold
<|bos|>If yesterday was Friday, then tomorrow will be Friday, but tomorrow is Friday tomorrow. tomorrow is Friday, so is Friday. Friday is Friday is Monday. Friday is Friday
e capital of the world is called the
<|bos|>The chemical symbol of gold is a symbol of gold, and a symbol of gold is a symbol of gold. The symbol of gold is a symbol of gold
<|bos|>If yesterday was Friday, then tomorrow will be Friday, but tomorrow is Friday tomorrow. tomorrow is Friday, so is Friday. Friday is Friday is Monday. Friday is Friday
<|bos|>The opposite of hot is the hot. Hot can be found in warm rocks, rocks, or on sand. A hot rocks rock can be found in
<|bos|>The planets of the solar system are: 1) the planets of the solar system 2) the planets of the solar system 3) the planets of the
<|bos|>My favorite color is the red/white color of my eyes. I don't know anyone like to think about it. But t<|bos|>If yesterday was Friday, then tomorrow will be Friday, but tomorrow is Friday tomorrow. tomorrow is Friday, so is Friday. Friday is Friday is Monday. Friday is Friday
<|bos|>The opposite of hot is the hot. Hot can be found in warm rocks, rocks, or on sand. A hot rocks rock can be found in
<|bos|>The planets of the solar system are: 1) the planets of the solar system 2) the planets of the solar system 3) the planets of the
<|bos|>My favorite color is the red/white color of my eyes. I don't know anyone like to think about it. But there are plenty of
<|bos|>The opposite of hot is the hot. Hot can be found in warm rocks, rocks, or on sand. A hot rocks rock can be found in
<|bos|>The planets of the solar system are: 1) the planets of the solar system 2) the planets of the solar system 3) the planets of the
<|bos|>My favorite color is the red/white color of my eyes. I don't know anyone like to think about it. But there are plenty of
<|bos|>The planets of the solar system are: 1) the planets of the solar system 2) the planets of the solar system 3) the planets of the
<|bos|>My favorite color is the red/white color of my eyes. I don't know anyone like to think about it. But there are plenty of
<|bos|>If 5*x + 3 = 13, then x is the value of 5. So a 5*x + 3 = 13 is equivalent to a 1*
here are plenty of
<|bos|>If 5*x + 3 = 13, then x is the value of 5. So a 5*x + 3 = 13 is equivalent to a 1*
<|bos|>If 5*x + 3 = 13, then x is the value of 5. So a 5*x + 3 = 13 is equivalent to a 1*
step 08800/80000 (11.00%) | loss: 3.571429 | lrm: 1.00 | dt: 1461.86ms | tok/sec: 11,207 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 46 | total time: 211.49m | eta: 1713.1m
step 08900/80000 (11.12%) | loss: 3.656530 | lrm: 1.00 | dt: 1443.79ms | tok/sec: 11,347 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 50 | total time: 213.89m | eta: 1710.6m
step 09000/80000 (11.25%) | loss: 3.628409 | lrm: 1.00 | dt: 1440.49ms | tok/sec: 11,373 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 55 | total time: 216.29m | eta: 1708.2m
step 09100/80000 (11.38%) | loss: 3.589666 | lrm: 1.00 | dt: 1439.31ms | tok/sec: 11,383 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 59 | total time: 218.69m | eta: 1705.8m
step 09200/80000 (11.50%) | loss: 3.577196 | lrm: 1.00 | dt: 1449.54ms | tok/sec: 11,302 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 63 | total time: 221.11m | eta: 1703.4m
step 09300/80000 (11.62%) | loss: 3.500625 | lrm: 1.00 | dt: 1446.80ms | tok/sec: 11,324 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 68 | total time: 223.52m | eta: 1701.1m
step 09400/80000 (11.75%) | loss: 3.525615 | lrm: 1.00 | dt: 1443.43ms | tok/sec: 11,350 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 72 | total time: 225.93m | eta: 1698.7m
step 09500/80000 (11.88%) | loss: 3.658079 | lrm: 1.00 | dt: 1448.96ms | tok/sec: 11,307 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 76 | total time: 228.35m | eta: 1696.4m
Step 09600 | Validation bpb: 1.094657
<|bos|>The capital of France is the capital of the United States. The capital is the capital of the United States. `It is known in French as Paris`,
<|bos|>The chemical symbol of gold is a chemical symbol of the gold chemical symbol of gold. It signifies the gold molecule's ability to undergo an oxygen reaction and is
<|bos|>If yesterday was Friday, then tomorrow will be Friday, today will be Friday, today will be Friday, today will be Friday, tomorrow will be Monday, today will be
<|bos|>The opposite of hot is the hot sun. Hot sun has a golden red color, and hot sun has a dark red and warm red color.
Is
<|bos|>The planets of the solar system are: the sun, the stars, the star, the stars, the Sun, the stars. The planets, as the main planets
<|bos|>My favorite color is red, which is the color for the darkest. I use brown, black, gray, brown, black, blue,
<|bos|>If 5*x + 3 = 13, then x is the number of points in the equation which does not contain the number of points of the equation. This equation may be useful when
step 09600/80000 (12.00%) | loss: 3.525275 | lrm: 1.00 | dt: 1445.04ms | tok/sec: 11,338 | bf16_mfu: 0.00 | epoch: 1 pq: 4 rg: 81 | total time: 230.76m | eta: 1694.0m
step 09700/80000 (12.12%) | loss: 3.610864 | lrm: 1.00 | dt: 1455.20ms | tok/sec: 11,258 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 3 | total time: 233.17m | eta: 1691.7m
step 09800/80000 (12.25%) | loss: 3.602508 | lrm: 1.00 | dt: 1449.26ms | tok/sec: 11,305 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 7 | total time: 235.59m | eta: 1689.3m
step 09900/80000 (12.38%) | loss: 3.561733 | lrm: 1.00 | dt: 1442.59ms | tok/sec: 11,357 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 11 | total time: 237.99m | eta: 1686.9m
step 10000/80000 (12.50%) | loss: 3.590211 | lrm: 1.00 | dt: 1457.85ms | tok/sec: 11,238 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 16 | total time: 240.40m | eta: 1684.5m
step 10100/80000 (12.62%) | loss: 3.534300 | lrm: 1.00 | dt: 1449.99ms | tok/sec: 11,299 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 20 | total time: 242.82m | eta: 1682.1m
step 10200/80000 (12.75%) | loss: 3.489466 | lrm: 1.00 | dt: 1444.07ms | tok/sec: 11,345 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 24 | total time: 245.23m | eta: 1679.8m
step 10300/80000 (12.88%) | loss: 3.564761 | lrm: 1.00 | dt: 1441.58ms | tok/sec: 11,365 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 29 | total time: 247.64m | eta: 1677.4m
Step 10400 | Validation bpb: 1.089483
<|bos|>The capital of France is the capital of the capital of Bantland, with a population of 2,300 and an area of 8
<|bos|>The chemical symbol of gold is a chemical name for a series of chemical compounds that are the chemical abbreviations and abbreviations of chemical abbreviations. A chemical
<|bos|>If yesterday was Friday, then tomorrow will be the day with the world's leading scientists taking a look at the global effects of climate change. We are not talking about the
<|bos|>The opposite of hot is the hot one. It's more popular than hot, but the best thing that you can do is make it hot! This
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Saturn, and Saturn. The moon that I was in and the other stars that I visited.
<|bos|>My favorite color is blue. That's why I use blue the entire time. It's basically just a little blue in color, and that's
<|bos|>If 5*x + 3 = 13, then x is the same as the same value.
2.14.2018
Royalty, hardcore
This is one of
step 10400/80000 (13.00%) | loss: 3.581143 | lrm: 1.00 | dt: 1468.28ms | tok/sec: 11,158 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 33 | total time: 250.05m | eta: 1675.0m
step 10500/80000 (13.12%) | loss: 3.554607 | lrm: 1.00 | dt: 1450.08ms | tok/sec: 11,298 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 37 | total time: 252.46m | eta: 1672.7m
step 10600/80000 (13.25%) | loss: 3.600921 | lrm: 1.00 | dt: 1437.77ms | tok/sec: 11,395 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 41 | total time: 254.88m | eta: 1670.3m
step 10700/80000 (13.38%) | loss: 3.503989 | lrm: 1.00 | dt: 1448.03ms | tok/sec: 11,314 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 46 | total time: 257.28m | eta: 1667.9m
step 10800/80000 (13.50%) | loss: 3.510505 | lrm: 1.00 | dt: 1449.10ms | tok/sec: 11,306 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 50 | total time: 259.70m | eta: 1665.6m
step 10900/80000 (13.62%) | loss: 3.572264 | lrm: 1.00 | dt: 1449.37ms | tok/sec: 11,304 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 54 | total time: 262.12m | eta: 1663.2m
step 11000/80000 (13.75%) | loss: 3.552735 | lrm: 1.00 | dt: 1450.22ms | tok/sec: 11,297 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 59 | total time: 264.53m | eta: 1660.8m
step 11100/80000 (13.88%) | loss: 3.572337 | lrm: 1.00 | dt: 1453.10ms | tok/sec: 11,275 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 63 | total time: 266.94m | eta: 1658.5m
Step 11200 | Validation bpb: 1.084062
<|bos|>The capital of France is the capital of the country, in French countries, and is known for its great diversity in diversity and variety of cultures. The
<|bos|>The chemical symbol of gold is a symbol of gold. It's one of the most widely used symbols in chemistry and chemistry. This chemical symbol is made up
<|bos|>If yesterday was Friday, then tomorrow will be Friday, tomorrow will be Friday, this must be Friday, the night before tomorrow. So, today, you can just imagine
<|bos|>The opposite of hot is hot. Water is a substance which is absorbed by the body, absorbed by the body. The term 'solarity'
<|bos|>The opposite of hot is hot. Water is a substance which is absorbed by the body, absorbed by the body. The term 'solarity'
<|bos|>The planets of the solar system are: 1. The solar system orbits on a massive star, 2. There is no horizon for planets that are so close
<|bos|>My favorite color is orange. For example red, green, and blue. What I don't see is blue. For red, I can't
<|bos|>If 5*x + 3 = 13, then x is 5*x = 6*8*11*12 = 3*5*3. (This is the
step 11200/80000 (14.00%) | loss: 3.527073 | lrm: 1.00 | dt: 1464.69ms | tok/sec: 11,185 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 67 | total time: 269.35m | eta: 1656.1m
step 11300/80000 (14.12%) | loss: 3.504342 | lrm: 1.00 | dt: 1447.00ms | tok/sec: 11,322 | bf16<|bos|>The opposite of hot is hot. Water is a substance which is absorbed by the body, absorbed by the body. The term 'solarity'
<|bos|>The planets of the solar system are: 1. The solar system orbits on a massive star, 2. There is no horizon for planets that are so close
<|bos|>My favorite color is orange. For example red, green, and blue. What I don't see is blue. For red, I can't
<|bos|>If 5*x + 3 = 13, then x is 5*x = 6*8*11*12 = 3*5*3. (This is the
step 11200/80000 (14.00%) | loss: 3.527073 | lrm: 1.00 | dt: 1464.69ms | tok/sec: 11,185 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 67 | total time: 269.35m | eta: 1656.1m
<|bos|>The opposite of hot is hot. Water is a substance which is absorbed by the body, absorbed by the body. The term 'solarity'
<|bos|>The planets of the solar system are: 1. The solar system orbits on a massive star, 2. There is no horizon for planets that are so close
<|bos|>My favorite color is orange. For example red, green, and blue. What I don't see is blue. For red, I can't
<|bos|>If 5*x + 3 = 13, then x is 5*x = 6*8*11*12 = 3*5*3. (This is the
<|bos|>The opposite of hot is hot. Water is a substance which is absorbed by the body, absorbed by the body. The term 'solarity'
<|bos|>The planets of the solar system are: 1. The solar system orbits on a massive star, 2. There is no horizon for planets that are so close
<|bos|>My favorite color is orange. For example red, green, and blue. What I don't see is blue. For red, I can't
<|bos|>The opposite of hot is hot. Water is a substance which is absorbed by the body, absorbed by the body. The term 'solarity'
<|bos|>The planets of the solar system are: 1. The solar system orbits on a massive star, 2. There is no horizon for planets that are so close
<|bos|>My favorite color is orange. For example red, green, and blue. What I don't see is blue.<|bos|>The opposite of hot is hot. Water is a substance which is absorbed by the body, absorbed by the body. The term 'solarity'
<|bos|>The planets of the solar system are: 1. The solar system orbits on a massive star, 2. There is no horizon for planets that are so close
<|bos|>The planets of the solar system are: 1. The solar system orbits on a massive star, 2. There is no horizon for planets that are so close
ere is no horizon for planets that are so close
<|bos|>My favorite color is orange. For example red, green, and blue. What I don't see is blue. For red, I can't
<|bos|>If 5*x + 3 = 13, then x is 5*x = 6*8*11*12 = 3*5*3. (This is the
step 11200/80000 (14.00%) | loss: 3.527073 | lrm: 1.00 | dt: 1464.69ms | tok/sec: 11,185 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 67 | total time: 269.35m | eta: 1656.1m
step 11300/80000 (14.12%) | loss: 3.504342 | lrm: 1.00 | dt: 1447.00ms | tok/sec: 11,322 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 72 | total time: 271.77m | eta: 1653.7m
step 11400/80000 (14.25%) | loss: 3.491344 | lrm: 1.00 | dt: 1485.02ms | tok/sec: 11,032 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 76 | total time: 274.19m | eta: 1651.4m
step 11500/80000 (14.38%) | loss: 3.480033 | lrm: 1.00 | dt: 1458.64ms | tok/sec: 11,232 | bf16_mfu: 0.00 | epoch: 1 pq: 5 rg: 80 | total time: 276.66m | eta: 1649.4m
step 11600/80000 (14.50%) | loss: 3.477474 | lrm: 1.00 | dt: 1499.86ms | tok/sec: 10,923 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 2 | total time: 279.11m | eta: 1647.2m
step 11700/80000 (14.62%) | loss: 3.497940 | lrm: 1.00 | dt: 1470.49ms | tok/sec: 11,141 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 6 | total time: 281.57m | eta: 1645.1m
step 11800/80000 (14.75%) | loss: 3.536323 | lrm: 1.00 | dt: 1453.32ms | tok/sec: 11,273 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 10 | total time: 284.01m | eta: 1642.9m
step 11900/80000 (14.88%) | loss: 3.553719 | lrm: 1.00 | dt: 1459.06ms | tok/sec: 11,229 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 14 | total time: 286.45m | eta: 1640.6m
Step 12000 | Validation bpb: 1.080607
<|bos|>The capital of France is the capital of the French capital, and it is the capital of the world and the most important of which is the highest point
<|bos|>The chemical symbol of gold is a symbol of gold. It is sometimes a symbol of gold that is still widely used today. In ancient times, it was
<|bos|>If yesterday was Friday, then tomorrow will be the day in which the moon has finally lifted. In the next big one the moon will be in a much larger version,
<|bos|>The opposite of hot is the hot gas. The gas itself is used to generate the hot gas. In the past, gas has been used as a
<|bos|>The planets of the solar system are: 1. The Sun, with its close proximity to the moon, is on the east side of the solar system;
<|bos|>My favorite color is red. You can see my list of color that I've always loved. In the end, there is no reason not to
<|bos|>If 5*x + 3 = 13, then x is the number of moles. That will add 300 moles to my head!
I am sure it would be a better bet
2026-03-15 08:50:45,380 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_012000.pt
2026-03-15 08:50:45,386 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_012000.json
2026-03-15 08:50:47,989 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_012000_rank0.pt
step 12000/80000 (15.00%) | loss: 3.550828 | lrm: 1.00 | dt: 1887.06ms | tok/sec: 8,682 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 19 | total time: 288.93m | eta: 1638.7m
step 12100/80000 (15.12%) | loss: 3.493379 | lrm: 1.00 | dt: 1474.60ms | tok/sec: 11,110 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 23 | total time: 291.40m | eta: 1636.6m
step 12200/80000 (15.25%) | loss: 3.560994 | lrm: 1.00 | dt: 1493.69ms | tok/sec: 10,968 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 27 | total time: 293.86m | eta: 1634.5m
step 12300/80000 (15.38%) | loss: 3.459941 | lrm: 1.00 | dt: 1504.26ms | tok/sec: 10,891 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 32 | total time: 296.34m | eta: 1632.4m
step 12400/80000 (15.50%) | loss: 3.465815 | lrm: 1.00 | dt: 1504.98ms | tok/sec: 10,886 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 36 | total time: 298.83m | eta: 1630.4m
step 12500/80000 (15.62%) | loss: 3.527264 | lrm: 1.00 | dt: 1485.72ms | tok/sec: 11,027 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 40 | total time: 301.33m | eta: 1628.5m
step 12600/80000 (15.75%) | loss: 3.520087 | lrm: 1.00 | dt: 1477.56ms | tok/sec: 11,088 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 44 | total time: 303.83m | eta: 1626.5m
step 12700/80000 (15.88%) | loss: 3.532888 | lrm: 1.00 | dt: 1485.29ms | tok/sec: 11,030 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 49 | total time: 306.33m | eta: 1624.6m
Step 12800 | Validation bpb: 1.076644
<|bos|>The capital of France is a city known for the world's top international conglomerate. The famous French language, which has a rich culture and
<|bos|>The chemical symbol of gold is gold. Its chemical formula is gold. Gold is a substance that naturally occurs in nature, and this chemical compound is also used
<|bos|>If yesterday was Friday, then tomorrow will be Friday, Monday, Monday, Sunday, Monday, Monday, Sunday, Monday, Monday, Monday, Friday, Monday, Wednesday
<|bos|>The opposite of hot is the term used to refer to when hot before baking. The term hot is commonly used in the US for a variety of reasons
<|bos|>The planets of the solar system are: Mars, Jupiter, Jupiter, Saturn, and moon. The planet may be in the sun's largest and most powerful solar system
<|bos|>My favorite color is red. For example red is good for breakfast, but red is only good for breakfast. Red is the most popular color of
<|bos|>If 5*x + 3 = 13, then x is 13+1* = 0. To sum up, 5= 10* = 7* + 
step 12800/80000 (16.00%) | loss: 3.502078 | lrm: 1.00 | dt: 1474.95ms | tok/sec: 11,108 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 53 | total time: 308.78m | eta: 1622.4m
step 12900/80000 (16.12%) | loss: 3.521876 | lrm: 1.00 | dt: 1482.31ms | tok/sec: 11,053 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 57 | total time: 311.24m | eta: 1620.2m


step 13000/80000 (16.25%) | loss: 3.475437 | lrm: 1.00 | dt: 1468.99ms | tok/sec: 11,153 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 62 | total time: 313.70m | eta: 1618.0m
step 13100/80000 (16.38%) | loss: 3.567357 | lrm: 1.00 | dt: 1479.93ms | tok/sec: 11,070 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 66 | total time: 316.15m | eta: 1615.8m
step 13200/80000 (16.50%) | loss: 3.540492 | lrm: 1.00 | dt: 1489.22ms | tok/sec: 11,001 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 70 | total time: 318.62m | eta: 1613.6m
step 13300/80000 (16.62%) | loss: 3.449044 | lrm: 1.00 | dt: 1470.76ms | tok/sec: 11,139 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 75 | total time: 321.08m | eta: 1611.4m
step 13200/80000 (16.50%) | loss: 3.540492 | lrm: 1.00 | dt: 1489.22ms | tok/sec: 11,001 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 70 | total time: 318.62m | eta: 1613.6m
step 13300/80000 (16.62%) | loss: 3.449044 | lrm: 1.00 | dt: 1470.76ms | tok/sec: 11,139 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 75 | total time: 321.08m | eta: 1611.4m
_mfu: 0.00 | epoch: 1 pq: 6 rg: 70 | total time: 318.62m | eta: 1613.6m
step 13300/80000 (16.62%) | loss: 3.449044 | lrm: 1.00 | dt: 1470.76ms | tok/sec: 11,139 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 75 | total time: 321.08m | eta: 1611.4m
step 13300/80000 (16.62%) | loss: 3.449044 | lrm: 1.00 | dt: 1470.76ms | tok/sec: 11,139 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 75 | total time: 321.08m | eta: 1611.4m
_mfu: 0.00 | epoch: 1 pq: 6 rg: 75 | total time: 321.08m | eta: 1611.4m
step 13400/80000 (16.75%) | loss: 3.587270 | lrm: 1.00 | dt: 1479.38ms | tok/sec: 11,074 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 79 | total time: 323.54m | eta: 1609.2m
step 13400/80000 (16.75%) | loss: 3.587270 | lrm: 1.00 | dt: 1479.38ms | tok/sec: 11,074 | bf16_mfu: 0.00 | epoch: 1 pq: 6 rg: 79 | total time: 323.54m | eta: 1609.2m
_mfu: 0.00 | epoch: 1 pq: 6 rg: 79 | total time: 323.54m | eta: 1609.2m
step 13500/80000 (16.88%) | loss: 3.499869 | lrm: 1.00 | dt: 1475.75ms | tok/sec: 11,102 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 1 | total time: 326.00m | eta: 1607.0m
Step 13600 | Validation bpb: 1.073071
step 13500/80000 (16.88%) | loss: 3.499869 | lrm: 1.00 | dt: 1475.75ms | tok/sec: 11,102 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 1 | total time: 326.00m | eta: 1607.0m
Step 13600 | Validation bpb: 1.073071
<|bos|>The capital of France is the capital of the French Republic.
The capital of the French Republic is the capital of the French Republic
Step 13600 | Validation bpb: 1.073071
<|bos|>The capital of France is the capital of the French Republic.
The capital of the French Republic is the capital of the French Republic
The capital of the
<|bos|>The capital of France is the capital of the French Republic.
The capital of the French Republic is the capital of the French Republic
The capital of the
The capital of the French Republic is the capital of the French Republic
The capital of the
<|bos|>The chemical symbol of gold is a symbol of wealth, wealth, and independence. It represenThe capital of the
<|bos|>The chemical symbol of gold is a symbol of wealth, wealth, and independence. It represents the right to earn wealth through wealth and prosperity, wealth accumulation,
<|bos|>The chemical symbol of gold is a symbol of wealth, wealth, and independence. It represents the right to earn wealth through wealth and prosperity, wealth accumulation,
<|bos|>If yesterday was Friday, then tomorrow will be Friday, and today is Friday. However, Sunts the right to earn wealth through wealth and prosperity, wealth accumulation,
<|bos|>If yesterday was Friday, then tomorrow will be Friday, and today is Friday. However, Sunday is Friday, and for a few years, the last thing I thought I
<|bos|>If yesterday was Friday, then tomorrow will be Friday, and today is Friday. However, Sunday is Friday, and for a few years, the last thing I thought I
day is Friday, and for a few years, the last thing I thought I
<|bos|>The opposite of hot is the cold. Hot air is usually very hot, and it is much more pleasant than a hot one.
A hot engine is
<|bos|>The planets of the solar system are: 1. The solar system: 2. The solar system: 3. The solar system: 4. The
<|bos|>My favorite color is the red one. It's so much fun to see. It's beautiful in the background, makes the space look cool and
<|bos|>If 5*x + 3 = 13, then x is the number of 3 and d is the number of 5 and d is the number of 4 and d is the
step 13600/80000 (17.00%) | loss: 3.451103 | lrm: 1.00 | dt: 1466.12ms | tok/sec: 11,175 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 5 | total time: 328.46m | eta: 1604.8m
step 13700/80000 (17.12%) | loss: 3.572202 | lrm: 1.00 | dt: 1454.43ms | tok/sec: 11,264 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 10 | total time: 330.92m | eta: 1602.6m
step 13800/80000 (17.25%) | loss: 3.537184 | lrm: 1.00 | dt: 1467.99ms | tok/sec: 11,160 | bf16und, makes the space look cool and
<|bos|>If 5*x + 3 = 13, then x is the number of 3 and d is the number of 5 and d is the number of 4 and d is the
step 13600/80000 (17.00%) | loss: 3.451103 | lrm: 1.00 | dt: 1466.12ms | tok/sec: 11,175 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 5 | total time: 328.46m | eta: 1604.8m
step 13700/80000 (17.12%) | loss: 3.572202 | lrm: 1.00 | dt: 1454.43ms | tok/sec: 11,264 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 10 | total time: 330.92m | eta: 1602.6m
und, makes the space look cool and
<|bos|>If 5*x + 3 = 13, then x is the number of 3 and d is the number of 5 and d is the number of 4 and d is the
und, makes the space look cool and
<|bos|>If 5*x + 3 = 13, then x is the number of 3 and d is the number of 5 and d is the number of 4 and d is the
step 13600/80000 (17.00%) | loss: 3.451103 | lrm: 1.00 | dt: 1466.12ms | tok/sec: 11,175 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 5 | total time: 328.46m | eta: 1604.8m
step 13700/80000 (17.12%) | loss: 3.572202 | lrm: 1.00 | dt: 1454.43ms | tok/sec: 11,264 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 10 | total time: 330.92m | eta: 1602.6m
step 13800/80000 (17.25%) | loss: 3.537184 | lrm: 1.00 | dt: 1467.99ms | tok/sec: 11,160 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 14 | total time: 333.37m | eta: 1600.4m
step 13900/80000 (17.38%) | loss: 3.480143 | lrm: 1.00 | dt: 1457.06ms | tok/sec: 11,244 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 18 | total time: 335.83m | eta: 1598.2m
step 14000/80000 (17.50%) | loss: 3.499990 | lrm: 1.00 | dt: 1471.70ms | tok/sec: 11,132 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 23 | total time: 338.28m | eta: 1595.9m
step 14100/80000 (17.62%) | loss: 3.552388 | lrm: 1.00 | dt: 1469.76ms | tok/sec: 11,147 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 27 | total time: 340.74m | eta: 1593.6m
step 14200/80000 (17.75%) | loss: 3.469567 | lrm: 1.00 | dt: 1497.99ms | tok/sec: 10,937 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 31 | total time: 343.18m | eta: 1591.3m
step 14300/80000 (17.88%) | loss: 3.501817 | lrm: 1.00 | dt: 1451.83ms | tok/sec: 11,285 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 36 | total time: 345.62m | eta: 1589.0m
Step 14400 | Validation bpb: 1.070168
<|bos|>The capital of France is the capital of the world. France is the capital of the world anStep 14400 | Validation bpb: 1.070168
<|bos|>The chemical symbol of gold is a symbol of the world's oldest and most noble minerals. The origin of it is not known, however, and is still
<|bos|>If yesterday was Friday, then tomorrow will be the day. The morning that morning before Monday was the day of the meeting of the American Association of the American Association of the
Step 14400 | Validation bpb: 1.070168


<|bos|>If yesterday was Friday, then tomorrow will be the day. The morning that morning before Monday was the day of the meeting of the American Association of the American Association of the
<|bos|>`The opposite of hot is cold`. If you are sitting upright, you will feel that your brain is in a state of deep rest, and this will
<|bos|>The planets of the solar system are: Venus, Venus, Mars, Jupiter, Venus, Venus, Mars, Venus, Mars, Venus, Mercury, Venus, Venus
<|bos|>My favorite color is black. A common shade of gray to 20-25% browns.
I'm not sure what color you have.
<|bos|>If 5*x + 3 = 13, then x is the sum of the 0, 1, 2, 3, 4, 5, 6,
step 14400/80000 (18.00%) | loss: 3.535816 | lrm: 1.00 | dt: 1500.12ms | tok/sec: 10,921 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 40 | total time: 348.08m | eta: 1586.8m



step 15100/80000 (18.88%) | loss: 3.508820 | lrm: 1.00 | dt: 1486.46ms | tok/sec: 11,022 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 70 | total time: 365.24m | eta: 1570.9m
Step 15200 | Validation bpb: 1.066858
<|bos|>The capital of France is the largest cities in France, accounting for 35% of the territory. Its capital, France, was originally part of the
<|bos|>The chemical symbol of gold is a symbol of gold, meaning its alloying with gold and silver. It can be found in `gold alloy`, silver alloy,
<|bos|>·`If yesterday was Friday, then tomorrow will be Saturday·`. There will also be Friday for July 11th, from 11:00 am – 6:00 pm
<|bos|>·`The opposite of hot is the cold·`. The cold is an intense set of temperatures that are so intense that it causes the brain to send signals back to
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, and Saturn. Jupiter is the largest part of the solar system and can reach a diameter of 
<|bos|>My favorite color is red. My favorite is orange. I'll see you.
My favorite color is red. I'll see you.
What are
<|bos|>If 5*x + 3 = 13, then x is 5*x = 3! A plus is 0 to 4! The standard way is to use a plus
step 15200/80000 (19.00%) | loss: 3.450793 | lrm: 1.00 | dt: 1481.73ms | tok/sec: 11,057 | bf16_mfu: 0.00 | epoch: 1 pq: 7 rg: 74 | total time: 367.69m | eta: 1568.5m



<|bos|>The chemical symbol of gold is gold. When it is ground with acid, it releases a strong odour. However, in some cases the smell is quite unpleasant
<|bos|>If yesterday was Friday, then tomorrow will be the day the world will be closed. Maybe at least it's Saturday this time. Maybe it's Friday at the same time
<|bos|>The opposite of hot is the hot-tennis. Hot-tennis is a common practice in the United States, in many countries, as an
<|bos|>The planets of the solar system are: Jupiter, Saturn, Uranus, Neptune, and Pluto. Venus, for instance, is not named for it's
<|bos|>My favorite color is black. That's what I want to highlight and color on a website. So, it's really just black. You can
<|bos|>If 5*x + 3 = 13, then x is 13+5, then x is 13+5+x +4. 5 +x = 2+
2026-03-15 10:30:24,319 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_016000.pt
2026-03-15 10:30:24,322 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_016000.json
2026-03-15 10:30:26,461 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_016000_rank0.pt





Step 18400 | Validation bpb: 1.056929
<|bos|>The capital of France is the capital of the country and one of the most important and important political developments in the world. A great majority of countries,
<|bos|>The chemical symbol of gold is gold. However, gold is an example of a mineral that has a non-conducted state with an inner core.
How
<|bos|>If yesterday was Friday, then tomorrow will be Friday. `Saturday` is the last Friday of the month.
You will want to keep your computer running and working, but not for
<|bos|>The opposite of hot is the hot food. It is also called hot food. It is in the shape of hot food as the main ingredient and it
<|bos|>The planets of the solar system are: the planets of the `sun, Venus, Earth, Mars, Jupiter, Saturn, Uranus,` Uranus, Uranus,
<|bos|>My favorite color is red. If you have a large blue background, you can make your blue/red statement by choosing red with a light hue
<|bos|>If 5*x + 3 = 13, then x is 137.2*2.343*2.271*2.27*1.27*2


step 19100/80000 (23.88%) | loss: 3.514424 | lrm: 1.00 | dt: 1271.00ms | tok/sec: 12,890 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 78 | total time: 452.25m | eta: 1442.8m
Step 19200 | Validation bpb: 1.054546
<|bos|>The capital of France is the capital of the French capital of the world. The capital is divided into capital and capital. It is an important factor that
<|bos|>The chemical symbol of gold is a gold mineral that belongs to the family of nuclidean diatomic gas. It is named as the gold standard,
<|bos|>If yesterday was Friday, then tomorrow will be the day we will go out on the school day. The day you will do the work will be Friday. The school will
<|bos|>The opposite of hot is the fact that the temperature of the surface of the water is too low. For example, the temperature at which the water is
<|bos|>The planets of the solar system are: the sun, the stars, the moon, the moon, the moon, the sun, the moon. We can use the
<|bos|>My favorite color is red. My favorite is red because it's always bold and bold. Red's a lot more vibrant in color than black and
<|bos|>If 5*x + 3 = 13, then x is the number of times you put x in the line. For example, if x is the number of times you put it in
step 19200/80000 (24.00%) | loss: 3.401889 | lrm: 1.00 | dt: 1277.35ms | tok/sec: 12,826 | bf16_mfu: 0.00 | epoch: 1 pq: 9 rg: 82 | total time: 454.39m | eta: 1439.7m
step 19300/80000 (24.12%) | loss: 3.437625 | lrm: 1.00 | dt: 1288.71ms | tok/sec: 12,713 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 4 | total time: 456.56m | eta: 1436.6m


step 19900/80000 (24.88%) | loss: 3.448765 | lrm: 1.00 | dt: 1254.16ms | tok/sec: 13,063 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 29 | total time: 469.20m | eta: 1417.7m
Step 20000 | Validation bpb: 1.052438
<|bos|>The capital of France is the capital of France. The capital of France is the capital of France. A large area of about 4,600 
<|bos|>The chemical symbol of gold is gold. These metals have a symbol of silver and are highly sought after by jewelry makers because of their high strength and ductility
<|bos|>If yesterday was Friday, then tomorrow will be the day it was, no worries. From it, it was still Saturday night. So, tomorrow is it. No worries
<|bos|>The opposite of hot is the heat which is generated by air bubbles trapped in the air. An air bubble is made up of two gases that contain hydrogen
<|bos|>The planets of the solar system are: the Sun, the moon, and the Moon. The sun is the ultimate star. It is the most important planet of the
<|bos|>My favorite color is the red. I like the pink. I get a lot of white. When I go to the doctor I get it.
<|bos|>If 5*x + 3 = 13, then x is the number of times that the player takes less than 5, this means he has the same number of times that we need
2026-03-15 16:21:40,479 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_020000.pt
2026-03-15 16:21:40,506 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_020000.json
2026-03-15 16:21:42,450 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_020000_rank0.pt


step 20100/80000 (25.12%) | loss: 3.379871 | lrm: 1.00 | dt: 1369.69ms | tok/sec: 11,961 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 40 | total time: 473.82m | eta: 1412.7m
step 20400/80000 (25.50%) | loss: 3.362027 | lrm: 1.00 | dt: 1369.11ms | tok/sec: 11,966 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 53 | total time: 480.69m | eta: 1405.0m
step 20700/80000 (25.88%) | loss: 3.445846 | lrm: 1.00 | dt: 1387.92ms | tok/sec: 11,804 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 66 | total time: 487.55m | eta: 1397.4m
Step 20800 | Validation bpb: 1.050791
<|bos|>The capital of France is the capital of the world. The capital of France is the capital of the world. It is one of the most significant and
<|bos|>The chemical symbol of gold is a symbol of purity, purity, and indomitable spirit. If you find a gold ring near your home, consider it
<|bos|>If yesterday was Friday, then tomorrow will be Friday, Monday, Thursday, Wednesday, Friday, Friday, and Friday, but Friday, Friday, Thursday, Friday, and
<|bos|>The opposite of hot is the opposite of hot, in terms of physical usage. The word hot is commonly used to describe a wide variety of different physical
<|bos|>The planets of the solar system are: the Sun, the Earth, and the Moon. The planets are all at the same distance from each other, and are often
<|bos|>My favorite color is the red in the center of your room. Your favorite color is the orange in the middle of your bedroom, and it's
<|bos|>If 5*x + 3 = 13, then x is 5*x = 1. Then x is 1 = 10. This is an interesting observation. For example
step 21000/80000 (26.25%) | loss: 3.422749 | lrm: 1.00 | dt: 1377.63ms | tok/sec: 11,892 | bf16_mfu: 0.00 | epoch: 1 pq: 10 rg: 79 | total time: 494.54m | eta: 1390.1m
step 21300/80000 (26.62%) | loss: 3.394658 | lrm: 1.00 | dt: 1371.45ms | tok/sec: 11,946 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 10 | total time: 501.51m | eta: 1382.8m
step 21600/80000 (27.00%) | loss: 3.482685 | lrm: 1.00 | dt: 1364.47ms | tok/sec: 12,007 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 23 | total time: 508.36m | eta: 1375.1m
step 21900/80000 (27.38%) | loss: 3.486387 | lrm: 1.00 | dt: 1367.33ms | tok/sec: 11,982 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 36 | total time: 515.20m | eta: 1367.4m
step 22200/80000 (27.75%) | loss: 3.391373 | lrm: 1.00 | dt: 1355.70ms | tok/sec: 12,085 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 48 | total time: 522.05m | eta: 1359.8m
Step 22400 | Validation bpb: 1.047839
<|bos|>The capital of France is the capital of France, as we often focus on the vast and ever-changing territory of France. And as we move through the
<|bos|>The chemical symbol of gold is gold. Most of us can trace the first two or three of the above mentioned metals back to ancient Roman Rome. But in
<|bos|>If yesterday was Friday, then tomorrow will be the day after the school lunch was filled. Since the school is running through its school day, school is closed. We're
<|bos|>The opposite of hot is the heat it takes. Once your thermostat becomes a hot one, the electricity in your home will go back to the furnace to
<|bos|>The planets of the solar system are: Venus, Mars, Saturn, and the ones that are beyond the atmosphere. And the solar system itself has been a planet for
<|bos|>My favorite color is red. In fact red is one of the most commonly used color today. And in a word, this color is the result
<|bos|>If 5*x + 3 = 13, then x is the number of times that the x + 3 = 13 will be used to create 150 + 2.

step 22500/80000 (28.12%) | loss: 3.469285 | lrm: 1.00 | dt: 1469.06ms | tok/sec: 11,152 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 61 | total time: 528.92m | eta: 1352.3m
step 22800/80000 (28.50%) | loss: 3.493319 | lrm: 1.00 | dt: 1363.34ms | tok/sec: 12,017 | bf16_mfu: 0.00 | epoch: 1 pq: 11 rg: 74 | total time: 535.79m | eta: 1344.8m
step 23100/80000 (28.88%) | loss: 3.414238 | lrm: 1.00 | dt: 1361.73ms | tok/sec: 12,031 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 5 | total time: 542.63m | eta: 1337.2m
step 23400/80000 (29.25%) | loss: 3.295700 | lrm: 1.00 | dt: 1383.04ms | tok/sec: 11,846 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 18 | total time: 549.46m | eta: 1329.6m
step 23700/80000 (29.62%) | loss: 3.438520 | lrm: 1.00 | dt: 1377.72ms | tok/sec: 11,892 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 31 | total time: 556.35m | eta: 1322.2m
Step 24000 | Validation bpb: 1.044290
<|bos|>The capital of France is the capital of France. The country is generally composed of 2 nations, each with their own distinctive style of style, and
<|bos|>The chemical symbol of gold is gold. All gold has the same chemical formula, but the chemical formula is also different from the chemical element for gold.
What
<|bos|>If yesterday was Friday, then tomorrow will be Friday. That is the most common Monday's we will be looking at over the weekend.

"Watching a baby screams
<|bos|>The opposite of hot is the opposite of cold, i.e. heating water up, but instead of heat, heat is added, as in, the
<|bos|>The planets of the solar system are: Mercury, Venus, Earth, Venus, Mars, Saturn, and in the next few months, Earth. The planet Mercury will
<|bos|>My favorite color is black. There are a lot of shades of orange, red, and yellow and I've been getting them for years. The
<|bos|>If 5*x + 3 = 13, then x is 13 + 3 = 3. Now, we have a whole equation for 13*3 + 3 =
2026-03-15 19:51:30,707 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_024000.pt
2026-03-15 19:51:30,712 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_024000.json
2026-03-15 19:51:33,068 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_024000_rank0.pt
step 24000/80000 (30.00%) | loss: 3.364525 | lrm: 1.00 | dt: 1550.69ms | tok/sec: 10,565 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 44 | total time: 563.23m | eta: 1314.8m
step 24300/80000 (30.38%) | loss: 3.444072 | lrm: 1.00 | dt: 1397.50ms | tok/sec: 11,723 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 57 | total time: 570.17m | eta: 1307.5m
step 24600/80000 (30.75%) | loss: 3.375800 | lrm: 1.00 | dt: 1331.21ms | tok/sec: 12,307 | bf16_mfu: 0.00 | epoch: 1 pq: 12 rg: 70 | total time: 577.07m | eta: 1300.1m
step 24900/80000 (31.12%) | loss: 3.418135 | lrm: 1.00 | dt: 1318.05ms | tok/sec: 12,430 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 0 | total time: 583.75m | eta: 1292.3m
step 25200/80000 (31.50%) | loss: 3.399780 | lrm: 1.00 | dt: 1342.78ms | tok/sec: 12,201 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 13 | total time: 590.48m | eta: 1284.6m
step 25500/80000 (31.88%) | loss: 3.367206 | lrm: 1.00 | dt: 1321.84ms | tok/sec: 12,394 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 26 | total time: 597.22m | eta: 1276.9m
Step 25600 | Validation bpb: 1.041524
<|bos|>The capital of France is the capital of the French Polynesia, is the capital of the archipelago of Polynesia and the capital of the
<|bos|>The chemical symbol of gold is the symbol of the gold flower of the Sultan of Sultan.
The gold flower was used in the Greek to
<|bos|>If yesterday was Friday, then tomorrow will be Friday, the last week of this week when we will be moving toward the week of Thursday. For those who are looking for
<|bos|>The opposite of hot is the "dry" (fresh) means of being cooled by running water to a specific temperature (the temperature of air,
<|bos|>The planets of the solar system are: Mars, Jupiter, Uranus, Neptune, and Saturn. Mercury (the second most advanced planetary system) is the
<|bos|>My favorite color is red. In the summer, this is more abundant in the summer and here, it's one of our most favorite colors.
<|bos|>If 5*x + 3 = 13, then x is 5 x 3 = 9.0 x 13/13= 2.1/1.9 =
step 25800/80000 (32.25%) | loss: 3.394616 | lrm: 1.00 | dt: 1360.84ms | tok/sec: 12,039 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 39 | total time: 603.97m | eta: 1269.3m
step 26100/80000 (32.62%) | loss: 3.417984 | lrm: 1.00 | dt: 1327.03ms | tok/sec: 12,346 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 51 | total time: 610.69m | eta: 1261.6m
step 26400/80000 (33.00%) | loss: 3.351615 | lrm: 1.00 | dt: 1348.31ms | tok/sec: 12,151 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 64 | total time: 617.46m | eta: 1254.1m
step 26700/80000 (33.38%) | loss: 3.386454 | lrm: 1.00 | dt: 1345.35ms | tok/sec: 12,178 | bf16_mfu: 0.00 | epoch: 1 pq: 13 rg: 77 | total time: 624.23m | eta: 1246.6m
step 27000/80000 (33.75%) | loss: 3.460517 | lrm: 1.00 | dt: 1374.60ms | tok/sec: 11,919 | bf16_mfu: 0.00 | epoch: 1 pq: 14 rg: 7 | total time: 631.09m | eta: 1239.3m
Step 27200 | Validation bpb: 1.039111
<|bos|>The capital of France is the capital of the country, located in the heart of the city of deplos in the country's capital, France.
<|bos|>The chemical symbol of gold is gold. For example, the symbol of gold is gold. Gold is a common element and isy's capital, France.
<|bos|>The chemical symbol of gold is gold. For example, the symbol of gold is gold. Gold is a common element and is widely used in jewelry making.
<|bos|>If yesterday was Friday, then tomorrow will be Friday. And if tomorrow was Thursday, then tomorrow will be Friday. As you can see, it was very good yesterday.
<|bos|>The opposite of hot is the opposite of hot. A car in a warmer climate will be pushed inward to the left and then out to the right.
<|bos|>The planets of the solar system are: Earth, Sun, Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Saturn, Neptune,
<|bos|>My favorite color is blue. When I first picked up my painting class, I thought, "what color would you pick?" 
So, we started
<|bos|>If 5*x + 3 = 13, then x is 13+13=6+(362 = 1; 3 = 1 + 362 =
step 27300/80000 (34.12%) | loss: 3.445975 | lrm: 1.00 | dt: 1369.55ms | tok/sec: 11,963 | bf16_mfu: 0.00 | epoch: 1 widely used in jewelry making.
<|bos|>If yesterday was Friday, then tomorrow will be Friday. And if tomorrow was Thursday, then tomorrow will be Friday. As you can see, it was very good yesterday.
<|bos|>The opposite of hot is the opposite of hot. A car in a warmer climate will be pushed inward to the left and then out to the right.
<|bos|>The planets of the solar system are: Earth, Sun, Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Saturn, Neptune,
<|bos|>My favorite color is blue. When I first picked up my painting class, I thought, "what color would you pick?" 
So, we started
<|bos|>If 5*x + 3 = 13, then x is 13+13=6+(362 = 1; 3 = 1 + 362 =
step 27300/80000 (34.12%) | loss: 3.445975 | lrm: 1.00 | dt: 1369.55ms | tok/sec: 11,963 | bf16_mfu: 0.00 | epoch: 1iday. As you can see, it was very good yesterday.
<|bos|>The opposite of hot is the opposite of hot. A car in a warmer climate will be pushed inward to the left and then out to the right.
<|bos|>The planets of the solar system are: Earth, Sun, Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Saturn, Neptune,
<|bos|>My favorite color is blue. When I first picked up my painting class, I thought, "what color would you pick?" 
So, we started
<|bos|>If 5*x + 3 = 13, then x is 13+13=6+(362 = 1; 3 = 1 + 362 =
step 27300/80000 (34.12%) | loss: 3.445975 | lrm: 1.00 | dt: 1369.55ms | tok/sec: 11,963 | bf16_mfu: 0.00 | epoch: 1hen out to the right.
<|bos|>The planets of the solar system are: Earth, Sun, Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Saturn, Neptune,
<|bos|>My favorite color is blue. When I first picked up my painting class, I thought, "what color would you pick?" 
So, we started
<|bos|>If 5*x + 3 = 13, then x is 13+13=6+(362 = 1; 3 = 1 + 362 =
step 27300/80000 (34.12%) | loss: 3.445975 | lrm: 1.00 | dt: 1369.55ms | tok/sec: 11,963 | bf16_mfu: 0.00 | epoch: 1 pq: 14 rg: 20 | total time: 638.02m | eta: 1232.1m
step 27600/80000 (34.50%) | loss: 3.456059 | lrm: 1.00 | dt: 1379.87ms | tok/sec: 11,873 | bf16_mfu: 0.00 | epoch: 1<|bos|>My favorite color is blue. When I first picked up my painting class, I thought, "what color would you pick?" 
So, we started
<|bos|>If 5*x + 3 = 13, then x is 13+13=6+(362 = 1; 3 = 1 + 362 =
step 27300/80000 (34.12%) | loss: 3.445975 | lrm: 1.00 | dt: 1369.55ms | tok/sec: 11,963 | bf16_mfu: 0.00 | epoch: 1 pq: 14 rg: 20 | total time: 638.02m | eta: 1232.1m
step 27600/80000 (34.50%) | loss: 3.456059 | lrm: 1.00 | dt: 1379.87ms | tok/sec: 11,873 | bf16_mfu: 0.00 | epoch: 1<|bos|>If 5*x + 3 = 13, then x is 13+13=6+(362 = 1; 3 = 1 + 362 =
step 27300/80000 (34.12%) | loss: 3.445975 | lrm: 1.00 | dt: 1369.55ms | tok/sec: 11,963 | bf16_mfu: 0.00 | epoch: 1 pq: 14 rg: 20 | total time: 638.02m | eta: 1232.1m
step 27600/80000 (34.50%) | loss: 3.456059 | lrm: 1.00 | dt: 1379.87ms | tok/sec: 11,873 | bf16_mfu: 0.00 | epoch: 1 pq: 14 rg: 20 | total time: 638.02m | eta: 1232.1m
step 27600/80000 (34.50%) | loss: 3.456059 | lrm: 1.00 | dt: 1379.87ms | tok/sec: 11,873 | bf16_mfu: 0.00 | epoch: 1 pq: 14 rg: 33 | total time: 645.01m | eta: 1225.0m
step 27900/80000 (34.88%) | loss: 3.353916 | lrm: 1.00 | dt: 1403.81ms | tok/sec: 11,671 | bf16_mfu: 0.00 | epoch: 1 pq: 14 rg: 46 | total time: 652.00m | eta: 1218.0m
2026-03-15 21:23:07,861 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_028000.pt
2026-03-15 21:23:07,862 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_028000.json
2026-03-15 21:23:10,039 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_028000_rank0.pt
step 28200/80000 (35.25%) | loss: 3.383280 | lrm: 1.00 | dt: 1357.96ms | tok/sec: 12,065 | bf16_mfu: 0.00 | epoch: 1 pq: 14 rg: 59 | total time: 658.90m | eta: 1210.7m

#

step 28200/80000 (35.25%) | loss: 3.383280 | `lrm: 1.00` | dt: 1357.96ms | tok/sec: 12,065 | bf16_mfu: 0.00 | epoch: 1 pq: 14 rg: 59 | total time: 658.90m | eta: 1210.7m
step 28500/80000 (35.62%) | loss: 3.347931 | `lrm: 0.99` | dt: 1384.76ms | tok/sec: 11,831 | bf16_mfu: 0.00 | epoch: 1 pq: 14 rg: 72 | total time: 665.78m | eta: 1203.5m

#
step 28800/80000 (36.00%) | loss: 3.375728 | lrm: 0.99 | dt: 1389.12ms | tok/sec: 11,794 | bf16_mfu: 0.00 | epoch: 1 pq: 15 rg: 2 | total time: 672.65m | eta: 1196.2m



##
<|bos|>The capital of France is the capital of the world, where France is the capital of the world. Its capital, St. John, is 25
<|bos|>The chemical symbol of gold is gold. When you see gold in a pool, it is the type of silver that can be extracted. However, in this
<|bos|>If yesterday was Friday, then tomorrow will be Friday, while today is Friday. Then that will be Friday, 4pm. This will be the Friday of Saturday and
<|bos|>The opposite of hot is hot. In fact hot is what causes what we call "hot" in some cultures, most countries are "hot." It
<|bos|>The planets of the solar system are: Mercury, Venus, Mercury, Venus, Earth, Venus, and Earth. Each of these planets orbit at different speeds, depending
<|bos|>My favorite color is the red one. I would have the entire landscape with the color as the first color of the sky and then the entire landscape
<|bos|>If 5*x + 3 = 13, then x is 13 * 5 = 6, then x is 5 * 6 = 3 = 6 = 
step 28800/80000 (36.00%) | loss: 3.375728 | lrm: 0.99 | dt: 1389.12ms | tok/sec: 11,794 | bf16_mfu: 0.00 | epoch: 1 pq: 15 rg: 2 | total time: 672.65m | eta: 1196.2m
step 29100/80000 (36.38%) | loss: 3.316895 | lrm: 0.98 | dt: 1379.70ms | tok/sec: 11,875 | bf16_mfu: 0.00 | epoch: 1 pq: 15 rg: 15 | total time: 679.70m | eta: 1189.3m
step 29400/80000 (36.75%) | loss: 3.391577 | lrm: 0.97 | dt: 1379.00ms | tok/sec: 11,881 | bf16_mfu: 0.00 | epoch: 1 pq: 15 rg: 28 | total time: 686.59m | eta: 1182.1m
step 29700/80000 (37.12%) | loss: 3.381497 | lrm: 0.97 | dt: 1377.32ms | tok/sec: 11,895 | bf16_mfu: 0.00 | epoch: 1 pq: 15 rg: 41 | total time: 693.47m | eta: 1174.9m
step 30000/80000 (37.50%) | loss: 3.422521 | lrm: 0.96 | dt: 1387.19ms | tok/sec: 11,810 | bf16_mfu: 0.00 | epoch: 1 pq: 15 rg: 54 | total time: 700.34m | eta: 1167.6m
step 30300/80000 (37.88%) | loss: 3.388202 | lrm: 0.96 | dt: 1380.28ms | tok/sec: 11,870 | bf16_mfu: 0.00 | epoch: 1 pq: 15 rg: 66 | total time: 707.22m | eta: 1160.4m
Step 30400 | Validation bpb: 1.030997
<|bos|>The capital of France is the capital of the French Alps. It is the main hub for economic development for the French people of France and the Swiss people
<|bos|>The chemical symbol of gold is Au. With the chemical symbol Au, the substance is a soft, slightly shiny, yellow, silvery white, and white.
<|bos|>If yesterday was Friday, then tomorrow will be Friday. And if it is Saturday, then Wednesday will be Monday. Remember: the time should be on Sunday. You are
<|bos|>The opposite of hot is the hotness of a piece of jewellery. At the same time, the shape of a piece of jewellery can be measured in
<|bos|>The planets of the solar system are: Mars, Jupiter, and Saturn. If, however, they were designed to fit in a smaller, less compact, and more
<|bos|>My favorite color is the red/green/red, yellow/green, and green, not green. You can choose a color that complements your
<|bos|>If 5*x + 3 = 13, then x is 5 + 5 = 1.5*10 = 10^-7 = 0.5, 
step 30600/80000 (38.25%) | loss: 3.454386 | lrm: 0.95 | dt: 1379.51ms | tok/sec: 11,876 | bf16_mfu: 0.00 | epoch: 1 pq: 15 rg: 79 | total time: 714.09m | eta: 1153.2m
step 30900/80000 (38.62%) | loss: 3.297061 | lrm: 0.95 | dt: 1375.83ms | tok/sec: 11,908 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 9 | total time: 720.95m | eta: 1146.0m
step 31200/80000 (39.00%) | loss: 3.361789 | lrm: 0.94 | dt: 1374.19ms | tok/sec: 11,922 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 22 | total time: 727.82m | eta: 1138.8m
step 31500/80000 (39.38%) | loss: 3.326835 | lrm: 0.94 | dt: 1377.90ms | tok/sec: 11,890 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 35 | total time: 734.68m | eta: 1131.5m
step 31800/80000 (39.75%) | loss: 3.312265 | lrm: 0.93 | dt: 1374.70ms | tok/sec: 11,918 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 48 | total time: 741.54m | eta: 1124.3m
Step 32000 | Validation bpb: 1.027053
<|bos|>The capital of France is the capital of the world. There are dozens of different capital letters but these include the French capital "French", the Dutch capital
<|bos|>The chemical symbol of gold is gold. It is the 12th person in the world and, is used to refer to an object or substance which is
<|bos|>If yesterday was Friday, then tomorrow will be Friday. In the meantime, today will be Friday. It will be Friday in the evening. We'll have to take care
<|bos|>The opposite of hot is the opposite of cold, as you need to add heat to your pool to get it's desired temperature, then you must add
<|bos|>The planets of the solar system are: Mars, Jupiter, Uranus, Neptune, Pluto, Cygnus, Pluto, Jupiter, Uranus, Ne
<|bos|>My favorite color is blue. But I love to find that lightness in the sky (on a dark background) so it's a color that
<|bos|>If 5*x + 3 = 13, then x is 13 x 3 = 14. For example, if x = 5, then x + 3 = 
2026-03-15 22:55:41,023 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_032000.pt
2026-03-15 22:55:41,025 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_032000.json
2026-03-15 22:55:42,843 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_032000_rank0.pt
step 32100/80000 (40.12%) | loss: 3.256588 | lrm: 0.93 | dt: 1380.79ms | tok/sec: 11,865 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 61 | total time: 748.41m | eta: 1117.1m
step 32400/80000 (40.50%) | loss: 3.324320 | lrm: 0.92 | dt: 1377.16ms | tok/sec: 11,896 | bf16_mfu: 0.00 | epoch: 1 pq: 16 rg: 73 | total time: 755.28m | eta: 1109.9m
step 32700/80000 (40.88%) | loss: 3.302519 | lrm: 0.91 | dt: 1374.59ms | tok/sec: 11,919 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 3 | total time: 762.16m | eta: 1102.8m
step 33000/80000 (41.25%) | loss: 3.347833 | lrm: 0.91 | dt: 1378.62ms | tok/sec: 11,884 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 16 | total time: 769.02m | eta: 1095.6m
step 33300/80000 (41.62%) | loss: 3.316806 | lrm: 0.90 | dt: 1373.43ms | tok/sec: 11,929 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 29 | total time: 775.89m | eta: 1088.4m
Step 33600 | Validation bpb: 1.023150
<|bos|>The capital of France is the capital of France, and a great number of people have been identified as it. This is largely because it is located in
<|bos|>The chemical symbol of gold is P. The chemical name of the symbol of gold is P. The chemical symbol of the symbol of silver is P. The
<|bos|>If yesterday was Friday, then tomorrow will be Friday. That's the same with any business. So, what happens if today was Sunday, the business would be on the
<|bos|>The opposite of hot is the opposite of hot is the opposite of hot. It is the opposite of hotter. This is what is known as hot vs
<|bos|>The planets of the solar system are: 1. The sun, 2. The moon, 3. The planets of the solar system, 4.
<|bos|>My favorite color is black. If you've ever wanted to throw a color that is dark, then you know that this one is a perfect combination
<|bos|>If 5*x + 3 = 13, then x is 13.
If x + 3 = 13, then x is 13.
If 13 + 5 =
step 33600/80000 (42.00%) | loss: 3.354364 | lrm: 0.90 | dt: 1411.02ms | tok/sec: 11,611 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 42 | total time: 782.75m | eta: 1081.3m
step 33900/80000 (42.38%) | loss: 3.365518 | lrm: 0.89 | dt: 1378.57ms | tok/sec: 11,884 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 55 | total time: 789.62m | eta: 1074.1m
step 34200/80000 (42.75%) | loss: 3.282230 | lrm: 0.89 | dt: 1377.67ms | tok/sec: 11,892 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 68 | total time: 796.48m | eta: 1066.9m
step 34500/80000 (43.12%) | loss: 3.307513 | lrm: 0.88 | dt: 1374.87ms | tok/sec: 11,916 | bf16_mfu: 0.00 | epoch: 1 pq: 17 rg: 81 | total time: 803.35m | eta: 1059.8m
step 34800/80000 (43.50%) | loss: 3.318493 | lrm: 0.88 | dt: 1373.99ms | tok/sec: 11,924 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 11 | total time: 810.20m | eta: 1052.6m
step 35100/80000 (43.88%) | loss: 3.280490 | lrm: 0.87 | dt: 1380.77ms | tok/sec: 11,865 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 23 | total time: 817.07m | eta: 1045.5m
Step 35200 | Validation bpb: 1.019372
<|bos|>The capital of France is the capital of the United States. The capital of the United States is the capital of the United Kingdom and is the only capital
<|bos|>The chemical symbol of gold is a symbol of gold, and this is largely due to the fact that it's a rare metal with the atomic number 25
<|bos|>If yesterday was Friday, then tomorrow will be the same
The question is "I didn't know how long there was so much heat in a day when the earth was
<|bos|>The opposite of hot is the opposite of hot is the opposite of hot is the opposite of hot is the opposite of hot is the opposite of hot is
<|bos|>The planets of the solar system are: the sun, the planets of the solar system, the planets of the solar system, and the planets of the solar system.
<|bos|>My favorite color is the purple. I have the blue one when I was a kid, and that is why I am the only color in my
<|bos|>If 5*x + 3 = 13, then x is 5 + 13 = 16. When 5*x + 3 = 3 is 16, then
step 35400/80000 (44.25%) | loss: 3.264351 | lrm: 0.86 | dt: 1382.81ms | tok/sec: 11,848 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 36 | total time: 823.93m | eta: 1038.4m
step 35700/80000 (44.62%) | loss: 3.310223 | lrm: 0.86 | dt: 1375.59ms | tok/sec: 11,910 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 49 | total time: 830.80m | eta: 1031.2m
2026-03-16 00:27:44,425 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_036000.pt
2026-03-16 00:27:44,425 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_036000.json
2026-03-16 00:27:46,242 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_036000_rank0.pt
step 36000/80000 (45.00%) | loss: 3.264821 | lrm: 0.85 | dt: 1541.51ms | tok/sec: 10,628 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 62 | total time: 837.68m | eta: 1024.1m
step 36300/80000 (45.38%) | loss: 3.281560 | lrm: 0.85 | dt: 1355.17ms | tok/sec: 12,090 | bf16_mfu: 0.00 | epoch: 1 pq: 18 rg: 75 | total time: 844.54m | eta: 1017.0m
step 36600/80000 (45.75%) | loss: 3.361003 | lrm: 0.84 | dt: 1377.40ms | tok/sec: 11,894 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 6 | total time: 851.41m | eta: 1009.9m
Step 36800 | Validation bpb: 1.015350
<|bos|>The capital of France is the capital of France. The capital of France is the capital of France. You can also find capital on the internet.

The
<|bos|>The chemical symbol of gold is a gold chemical symbol, which indicates the position of the metal in the periodic table. In the modern context of the periodic table
<|bos|>If yesterday was Friday, then tomorrow will be Friday. Now, if Monday is Friday, then tomorrow will be Friday. This is what I will say on the 5
<|bos|>The opposite of hot is hot. A hot body is more resistant to heat exhaustion and heat stroke. This is due to the heat's ability to move
<|bos|>The planets of the solar system are: Earth, Uranus and Neptune. However, the solar system is also a solar system.
How is the planet called
<|bos|>My favorite color is black. If you have a red light, just add a bit of orange in the center. Then add a bit of green
<|bos|>If 5*x + 3 = 13, then x is 13 times 5 * 13 = 13*13 (millionth of a meter).
For example, if
step 36900/80000 (46.12%) | loss: 3.358488 | lrm: 0.84 | dt: 1378.90ms | tok/sec: 11,881 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 19 | total time: 858.28m | eta: 1002.8m
step 37200/80000 (46.50%) | loss: 3.338379 | lrm: 0.83 | dt: 1391.10ms | tok/sec: 11,777 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 32 | total time: 865.14m | eta: 995.6m
step 37500/80000 (46.88%) | loss: 3.321258 | lrm: 0.83 | dt: 1369.92ms | tok/sec: 11,959 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 45 | total time: 872.01m | eta: 988.5m
step 37800/80000 (47.25%) | loss: 3.308165 | lrm: 0.82 | dt: 1362.51ms | tok/sec: 12,024 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 58 | total time: 878.87m | eta: 981.4m
step 38100/80000 (47.62%) | loss: 3.379364 | lrm: 0.82 | dt: 1364.36ms | tok/sec: 12,008 | bf16_mfu: 0.00 | epoch: 1 pq: 19 rg: 71 | total time: 885.73m | eta: 974.3m
Step 38400 | Validation bpb: 1.012315
<|bos|>The capital of France is the capital of the world. This is the major city of France, built from 1513 to 1600 AD.
<|bos|>The chemical symbol of gold is a symbol of the element symbol and may be used to indicate a material's status in a chemistry test, chemical analysis or laboratory
<|bos|>If yesterday was Friday, then tomorrow will be Friday. So, what if today was Friday? That's what they did for the 2019 election.
The idea that
<|bos|>The opposite of hot is hot. There are no hot foods available if you are a hot-eater in the food chain. The term hot can also
<|bos|>The planets of the solar system are: Jupiter, Saturn, and their moons. You can see them all as an open planet, but in a different place than Earth
<|bos|>My favorite color is the blue in the sky. You can still see the blue sky without thinking that it is blue! The sky is the color
<|bos|>If 5*x + 3 = 13, then x is 5 x 13 = 127. (127 = 6.6 x 5, 3.
step 38400/80000 (48.00%) | loss: 3.343480 | lrm: 0.81 | dt: 1409.27ms | tok/sec: 11,625 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 0 | total time: 892.60m | eta: 967.2m
step 38700/80000 (48.38%) | loss: 3.274912 | lrm: 0.80 | dt: 1378.61ms | tok/sec: 11,884 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 12 | total time: 899.47m | eta: 960.1m
step 39000/80000 (48.75%) | loss: 3.240651 | lrm: 0.80 | dt: 1361.34ms | tok/sec: 12,035 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 25 | total time: 906.33m | eta: 953.1m
step 39300/80000 (49.12%) | loss: 3.283572 | lrm: 0.79 | dt: 1358.12ms | tok/sec: 12,063 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 38 | total time: 913.20m | eta: 946.0m
step 39600/80000 (49.50%) | loss: 3.269523 | lrm: 0.79 | dt: 1375.64ms | tok/sec: 11,910 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 51 | total time: 920.06m | eta: 938.9m
step 39900/80000 (49.88%) | loss: 3.305108 | lrm: 0.78 | dt: 1376.25ms | tok/sec: 11,904 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 64 | total time: 926.94m | eta: 931.8m
Step 40000 | Validation bpb: 1.008221
<|bos|>The capital of France is the capital of the United Kingdom, and the capital of France is the capital of the world. French history and culture. French
<|bos|>The chemical symbol of gold is Ag. The chemical formula of gold is Ag+. The chemical formula of gold is Ag+. The chemical formula of gold
<|bos|>If yesterday was Friday, then tomorrow will be Friday. No, I don't know at all. I don't know. I'm not ready. I'm not ready
<|bos|>The opposite of hot is the opposite of cold, it is just, this is a very useful analogy in the same way as you can use this analogy
<|bos|>The planets of the solar system are: Mars, Jupiter, and Saturn.

The planets of the solar system are: Mars, Jupiter, and Saturn. The planets of
<|bos|>My favorite color is the blue/green/red combination. All the colors are so color matched with each other. My only problem is this.
<|bos|>If 5*x + 3 = 13, then x is 13 times 5*. For example, if you want to set your 1*x = 13 = 
2026-03-16 02:00:01,424 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_040000.pt
2026-03-16 02:00:01,424 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_040000.json
2026-03-16 02:00:03,203 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_040000_rank0.pt
step 40200/80000 (50.25%) | loss: 3.323632 | lrm: 0.78 | dt: 1373.70ms | tok/sec: 11,926 | bf16_mfu: 0.00 | epoch: 1 pq: 20 rg: 77 | total time: 933.81m | eta: 924.7m
step 40500/80000 (50.62%) | loss: 3.200016 | lrm: 0.77 | dt: 1371.56ms | tok/sec: 11,945 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 7 | total time: 940.68m | eta: 917.7m
step 40800/80000 (51.00%) | loss: 3.262296 | lrm: 0.77 | dt: 1380.13ms | tok/sec: 11,871 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 20 | total time: 947.55m | eta: 910.6m
step 41100/80000 (51.38%) | loss: 3.339324 | lrm: 0.76 | dt: 1372.85ms | tok/sec: 11,934 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 33 | total time: 954.41m | eta: 903.5m
step 41400/80000 (51.75%) | loss: 3.307470 | lrm: 0.76 | dt: 1379.24ms | tok/sec: 11,879 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 46 | total time: 961.29m | eta: 896.5m
Step 41600 | Validation bpb: 1.004787
<|bos|>The capital of France is the capital of the French Empire. It is home to a large number of cities, but the main ones are the Paris,
<|bos|>The chemical symbol of gold is a "b" or "t" often used to refer to the Latin for "diamond" and "cream
<|bos|>If yesterday was Friday, then tomorrow will be Friday. That's right, we won't be going to school and that'll be our best days, but we are going
<|bos|>The opposite of hot is the hot. This means that it is colder than the surrounding environment and when this is true, we have a lot of heat
<|bos|>The planets of the solar system are: Mars, Jupiter, Venus, Earth, Mars, Jupiter, Saturn, Mercury, Venus, Mars, Earth, Mars, Saturn
<|bos|>My favorite color is the blue (though it's usually less brown than the other colors). White was the main color, so it's been a
<|bos|>If 5*x + 3 = 13, then x is 13/3 and x is 3/3. The second and third terms are 2, 3, 
step 41700/80000 (52.12%) | loss: 3.255281 | lrm: 0.75 | dt: 1376.19ms | tok/sec: 11,905 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 59 | total time: 968.16m | eta: 889.4m
step 42000/80000 (52.50%) | loss: 3.300124 | lrm: 0.74 | dt: 1383.24ms | tok/sec: 11,844 | bf16_mfu: 0.00 | epoch: 1 pq: 21 rg: 72 | total time: 975.04m | eta: 882.4m
step 42300/80000 (52.88%) | loss: 3.249133 | lrm: 0.74 | dt: 1377.38ms | tok/sec: 11,895 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 1 | total time: 981.91m | eta: 875.3m
step 42600/80000 (53.25%) | loss: 3.311953 | lrm: 0.73 | dt: 1356.83ms | tok/sec: 12,075 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 14 | total time: 988.78m | eta: 868.3m
step 42900/80000 (53.62%) | loss: 3.304760 | lrm: 0.73 | dt: 1370.89ms | tok/sec: 11,951 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 26 | total time: 995.63m | eta: 861.2m
Step 43200 | Validation bpb: 1.001203
<|bos|>The capital of France is the capital of the French Empire, and it is the largest city in the city of Paris. In Paris, the capital of
<|bos|>The chemical symbol of gold is 'm' and it is an alloy of metals and non-metals. Its chemical symbol is 'N'. It is a
<|bos|>If yesterday was Friday, then tomorrow will be Friday. As a consequence, today will be Friday. There are only 1,400,000 people, which
<|bos|>The opposite of hot is hot. If you are an open water driver, you need to wear a hot seat. If you have a hot seat,
<|bos|>The planets of the solar system are: Earth, Uranus and Neptune. Uranus is a planet of 148 billion years, so it's one
<|bos|>My favorite color is the one we use to decorate our kitchen. That's because the light shines brightly on our eat,
<|bos|>The planets of the solar system are: Earth, Uranus and Neptune. Uranus is a planet of 148 billion years, so it's one
eat,
<|bos|>The planets of the solar system are: Earth, Uranus and Neptune. Uranus is a planet of 148 billion years, so ieat,
eat,
<|bos|>The planets of the solar system are: Earth, Uranus and Neptune. Uranus is a planet of 148 billion years, so i<|bos|>The planets of the solar system are: Earth, Uranus and Neptune. Uranus is a planet of 148 billion years, so it's one
<|bos|>My favorite color is the one we use to decorate our kitchen. That's because the light shines brightly on our kitchen surfaces and creates a stunning accent
<|bos|>If 5*x + 3 = 13, then x is 130, so the value of x is 130 times the value of the original x, which is 13      
step 43200/80000 (54.00%) | loss: 3.281638 | lrm: 0.72 | dt: 1416.42ms | tok/sec: 11,567 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 39 | total time: 1002.50m | eta: 854.2m
step 43500/80000 (54.38%) | loss: 3.257848 | lrm: 0.72 | dt: 1377.05ms | tok/sec: 11,897 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 52 | total time: 1009.38m | eta: 847.1m
step 43800/80000 (54.75%) | loss: 3.267504 | lrm: 0.71 | dt: 1375.49ms | tok/sec: 11,911 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 65 | total time: 1016.24m | eta: 840.1m
2026-03-16 03:32:07,124 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_044000.pt
2026-03-16 03:32:07,126 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_044000.json
2026-03-16 03:32:08,985 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_044000_rank0.pt
step 44100/80000 (55.12%) | loss: 3.259719 | lrm: 0.71 | dt: 1382.04ms | tok/sec: 11,854 | bf16_mfu: 0.00 | epoch: 1 pq: 22 rg: 78 | total time: 1023.10m | eta: 833.1m


##
Step 44800 | Validation bpb: 0.998405
<|bos|>The capital of France is the capital of the world, a land of a rich history, cultural heritage and a unique art form that has been for more
<|bos|>The chemical symbol of gold is gold. 1. The gold of the periodic table is a white 14.8 g of gold. 2.
Step 44800 | Validation bpb: 0.998405
<|bos|>The capital of France is the capital of the world, a land of a rich history, cultural heritage and a unique art form that has been for more
<|bos|>The chemical symbol of gold is gold. 1. The gold of the periodic table is a white 14.8 g of gold. 2.
<|bos|>The capital of France is the capital of the world, a land of a rich history, cultural heritage and a unique art form that has been for more
<|bos|>The chemical symbol of gold is gold. 1. The gold of the periodic table is a white 14.8 g of gold. 2.
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. That's why I decided to bring in a few of you if tomrt form that has been for more
<|bos|>The chemical symbol of gold is gold. 1. The gold of the periodic table is a white 14.8 g of gold. 2.
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. That's why I decided to bring in a few of you if tomorrow is tomorrow. But if you're still stuck
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. That's why I decided to bring in a few of you if tomorrow is tomorrow. But if you're still stuck
orrow is tomorrow. But if you're still stuck
<|bos|>The opposite of hot is hot. Heat is a natural energy created by the sun, which in turn moves heat out of the atmosphere into the atmosphere.
<|bos|>The planets of the solar system are: Earth, Mars, Saturn, Neptune, Venus, Earth, Mars, Venus, Saturn, Uranus, Nept
<|bos|>My favorite color is the blue. I like it as a deep blue with a dark center. But I'm not really that into it. 
I
<|bos|>If 5*x + 3 = 13, then x is 13
If x is 8, then x is 1
And if 5*x = 7,
step 45000/80000 (56.25%) | loss: 3.242533 | lrm: 0.69 | dt: 1391.54ms | tok/sec: 11,774 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 34 | total time: 1043.73m | eta: 812.0m
step 45300/80000 (56.62%) | loss: 3.233842 | lrm: 0.68 | dt: 1378.11ms | tok/sec: 11,888 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 46 | total time: 1050.62m | eta: 805.0m
step 45600/80000 (57.00%) | loss: 3.236990 | lrm: 0.68 | dt: 1370.46ms | tok/sec: 11,955 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 59 | total time: 1057.49m | eta: 797.9m
step 45900/80000 (57.38%) | loss: 3.203924 | lrm: 0.67 | dt: 1367.68ms | tok/sec: 11,979 | bf16_mfu: 0.00 | epoch: 1 pq: 23 rg: 72 | total time: 1064.36m | eta: 790.9m
step 46200/80000 (57.75%) | loss: 3.297517 | lrm: 0.67 | dt: 1380.11ms | tok/sec: 11,871 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 2 | total time: 1071.24m | eta: 783.9m
Step 46400 | Validation bpb: 0.995141
<|bos|>The capital of France is the city of Paris. It is the third largest city in the world, followed by Milan and Paris.
The capital of
<|bos|>The chemical symbol of gold is gold. Most of its common properties are melting and boiling points, electrical conductivity, melting point, thermal properties and melting temperature.
<|bos|>If yesterday was Friday, then tomorrow will be Friday. You can see, yesterday was Friday. Now, tomorrow will be tomorrow. I'm talking the last 12 days
<|bos|>The opposite of hot is cold. Heat is the same everywhere else. There is no "normal" temperature for any given day of the year. However
<|bos|>The planets of the solar system are: Mars, Jupiter, Neptune, Saturn, Venus, Earth, Mars, Jupiter, Saturn, Mercury, Venus, Mars
<|bos|>My favorite color is black. I love the cool color of it. I love it bright and the color is more than enough. I think the
<|bos|>If 5*x + 3 = 13, then x is the number of moles. (5*x + 3 = 7.) The 7x and 7x are
step 46500/80000 (58.12%) | loss: 3.176358 | lrm: 0.66 | dt: 1369.94ms | tok/sec: 11,959 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 15 | total time: 1078.12m | eta: 776.9m
step 46800/80000 (58.50%) | loss: 3.173378 | lrm: 0.66 | dt: 1379.79ms | tok/sec: 11,874 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 28 | total time: 1084.99m | eta: 769.9m
step 47100/80000 (58.88%) | loss: 3.211622 | lrm: 0.65 | dt: 1378.68ms | tok/sec: 11,883 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 41 | total time: 1091.85m | eta: 762.8m
step 47400/80000 (59.25%) | loss: 3.167495 | lrm: 0.65 | dt: 1377.93ms | tok/sec: 11,890 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 54 | total time: 1098.72m | eta: 755.8m
step 47700/80000 (59.62%) | loss: 3.234136 | lrm: 0.64 | dt: 1380.05ms | tok/sec: 11,871 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 66 | total time: 1105.59m | eta: 748.8m
Step 48000 | Validation bpb: 0.990793
<|bos|>The capital of France is the capital of the United States, and the total number of cities in the US is the largest (the US Census Bureau Census
<|bos|>The chemical symbol of gold is gold. However, the common name for gold is gold-plated glass. A gold-plated glass can be made in several
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. You can get an exact version of the app in the top left side of the page (which has a few other
<|bos|>The opposite of hot is cold. In the case of this is due to the fact that they are all made the same size, but the shape of
<|bos|>The planets of the solar system are: Earth, sun, planets, sun, moon, stars, planets, moon, stars, planets, stars, planets, moon
<|bos|>My favorite color is the blue. It is a favorite color of any color person, with a color palette of green, red, and orange.
<|bos|>If 5*x + 3 = 13, then x is the number of pixels, and a <20 + 3 = 13 would be 13 pixels.
Now, we need
2026-03-16 05:04:29,722 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_048000.pt
2026-03-16 05:04:29,724 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_048000.json
2026-03-16 05:04:31,490 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_048000_rank0.pt
step 48000/80000 (60.00%) | loss: 3.218691 | lrm: 0.63 | dt: 1529.85ms | tok/sec: 10,709 | bf16_mfu: 0.00 | epoch: 1 pq: 24 rg: 79 | total time: 1112.45m | eta: 741.8m
step 48300/80000 (60.38%) | loss: 3.179760 | lrm: 0.63 | dt: 1369.67ms | tok/sec: 11,961 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 9 | total time: 1119.32m | eta: 734.8m
step 48600/80000 (60.75%) | loss: 3.151099 | lrm: 0.62 | dt: 1373.86ms | tok/sec: 11,925 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 22 | total time: 1126.19m | eta: 727.8m
step 48900/80000 (61.12%) | loss: 3.175086 | lrm: 0.62 | dt: 1377.08ms | tok/sec: 11,897 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 35 | total time: 1133.06m | eta: 720.8m
step 49200/80000 (61.50%) | loss: 3.191231 | lrm: 0.61 | dt: 1373.36ms | tok/sec: 11,929 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 48 | total time: 1139.94m | eta: 713.8m
step 49500/80000 (61.88%) | loss: 3.244856 | lrm: 0.61 | dt: 1371.72ms | tok/sec: 11,944 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 61 | total time: 1146.81m | eta: 706.8m
Step 49600 | Validation bpb: 0.987973
<|bos|>The capital of France is the capital of the United States, and the capital of the United Kingdom is Queen Victoria and Queen Victoria respectively. The capital of
<|bos|>The chemical symbol of gold is Pt. 14. A white crystal of gold is a very soft material. A gold crystal has a melting point of
<|bos|>If yesterday was Friday, then tomorrow will be Friday. And if today was Monday, then tomorrow will be Friday. Why not? Why not? It is a little easier
<|bos|>The opposite of hot is cold. So, the higher the temperature, the more hot you take it and the hotter it is, the more it will
<|bos|>The planets of the solar system are: Earth, space, the Sun, the moon, the stars, the planets, the moon, the planets, the asteroids,
<|bos|>My favorite color is black. Black is a color that is slightly richer than the color of the light. This is because black is a slightly richer
<|bos|>If 5*x + 3 = 13, then x is 13.
If x is 12, then x is 4.
For example, if 4 is 14,
step 49800/80000 (62.25%) | loss: 3.201834 | lrm: 0.60 | dt: 1372.02ms | tok/sec: 11,941 | bf16_mfu: 0.00 | epoch: 1 pq: 25 rg: 74 | total time: 1153.67m | eta: 699.8m
step 50100/80000 (62.62%) | loss: 3.181556 | lrm: 0.60 | dt: 1375.99ms | tok/sec: 11,907 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 4 | total time: 1160.54m | eta: 692.8m
step 50400/80000 (63.00%) | loss: 3.273690 | lrm: 0.59 | dt: 1367.69ms | tok/sec: 11,979 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 17 | total time: 1167.42m | eta: 685.8m
step 50700/80000 (63.38%) | loss: 3.205648 | lrm: 0.59 | dt: 1387.51ms | tok/sec: 11,808 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 30 | total time: 1174.29m | eta: 678.8m
step 51000/80000 (63.75%) | loss: 3.111880 | lrm: 0.58 | dt: 1373.74ms | tok/sec: 11,926 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 42 | total time: 1181.15m | eta: 671.8m
Step 51200 | Validation bpb: 0.984271
<|bos|>The capital of France is the capital of the French Empire, in France, the capital of the world (or capital of the French Empire) is the
<|bos|>The chemical symbol of gold is gold. A gold bar is an abbreviation of gold. Gold is a very common element in rocks, as it is used
<|bos|>If yesterday was Friday, then tomorrow will be Friday, because the weather is going to be cool and rainy, and people should be wearing suits that keep you dry, and
<|bos|>The opposite of hot is cold. So, the word hot is only used to refer to liquid water (or even to steam), whereas the word cold
<|bos|>The planets of the solar system are: Earth, our solar, our planet, our sun, our moon, our planet, our sun, the sun, and the
<|bos|>My favorite color is red. You can also use the black with all the other colors if you have a dark background or you can use black in
<|bos|>If 5*x + 3 = 13, then x is the number of digits of the
stackexchange.com answer. 5 *x + 3 = 13
step 51300/80000 (64.12%) | loss: 3.194928 | lrm: 0.57 | dt: 1370.95ms | tok/sec: 11,950 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 55 | total time: 1188.03m | eta: 664.8m
step 51600/80000 (64.50%) | loss: 3.131367 | lrm: 0.57 | dt: 1385.04ms | tok/sec: 11,829 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 68 | total time: 1194.90m | eta: 657.8m
step 51900/80000 (64.88%) | loss: 3.091756 | lrm: 0.56 | dt: 1360.26ms | tok/sec: 12,044 | bf16_mfu: 0.00 | epoch: 1 pq: 26 rg: 81 | total time: 1201.76m | eta: 650.8m
2026-03-16 06:36:36,073 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_052000.pt
2026-03-16 06:36:36,075 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_052000.json
2026-03-16 06:36:37,988 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_052000_rank0.pt
step 52200/80000 (65.25%) | loss: 3.229016 | lrm: 0.56 | dt: 1375.89ms | tok/sec: 11,907 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 12 | total time: 1208.63m | eta: 643.8m
step 52500/80000 (65.62%) | loss: 3.207940 | lrm: 0.55 | dt: 1364.48ms | tok/sec: 12,007 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 25 | total time: 1215.49m | eta: 636.8m
Step 52800 | Validation bpb: 0.981169
<|bos|>The capital of France is the capital of the country, but a bit more complicated than that. The main cities are the Gatineau (now
<|bos|>The chemical symbol of gold is the gold leaf. It is made up of carbon, hydrogen and nitrogen. When the metal is added to water, it reacts
<|bos|>If yesterday was Friday, then tomorrow will be Friday. There will also be Tuesday, Wednesday, and Friday. On that occasion, there will be Tuesday and Wednesday.
As
<|bos|>The opposite of hot is cold. You can get more sleep but no wake up call. Hot is better than cold. If the air is too hot
<|bos|>The planets of the solar system are: Jupiter, Saturn, Saturn, Uranus, and Neptune. Jupiter orbits the Sun at approximately 6.7 x
<|bos|>My favorite color is the blue I've seen the most in the past few years. However, when I went to go for the first ever "
<|bos|>If 5*x + 3 = 13, then x is 13 if x = 4*x + 3 = 1 or 3 = 2.5.
I
is too hot
<|bos|>The planets of the solar system are: Jupiter, Saturn, Saturn, Uranus, and Neptune. Jupiter orbits the Sun at approximately 6.7 x
<|bos|>My favorite color is the blue I've seen the most in the past few years. However, when I went to go for the first ever "
<|bos|>If 5*x + 3 = 13, then x is 13 if x = 4*x + 3 = 1 or 3 = 2.5.
is too hot
<|bos|>The planets of the solar system are: Jupiter, Saturn, Saturn, Uranus, and Neptune. Jupiter orbits the Sun at approximately 6.7 x
<|bos|>My favorite color is the blue I've seen the most in the past few years. However, when I went to go for the fiis too hot
<|bos|>The planets of the solar system are: Jupiter, Saturn, Saturn, Uranus, and Neptune. Jupiter orbits the Sun at approximately 6.7 x
is too hot
is too hot
<|bos|>The planets of the solar system are: Jupiter, Saturn, Saturn, Uranus, and Neptune. Jupiter orbits the Sun at approximately 6.7 x
<|bos|>My favorite color is the blue I've seen the most in the past few years. However, when I went to go for the first ever "
<|bos|>If 5*x + 3 = 13, then x is 13 if x = 4*x + 3 = 1 or 3 = 2.5.
I
step 52800/80000 (66.00%) | loss: 3.155075 | lrm: 0.55 | dt: 1413.18ms | tok/sec: 11,593 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 38 | total time: 1222.36m | eta: 629.8m
step 53100/80000 (66.38%) | loss: 3.147370 | lrm: 0.54 | dt: 1380.54ms | tok/sec: 11,867 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 51 | total time: 1229.23m | eta: 622.8m
step 53400/80000 (66.75%) | loss: `3.192002` | lrm: 0.54 | dt: 1364.69ms | tok/sec: 12,005 | bf16_mfu: 0.00 | epoch: 1 pq: 27 rg: 64 | total time: 1236.10m | eta: 615.8m


#
step 54300/80000 (67.88%) | loss: 3.174473 | lrm: 0.52 | dt: 1369.01ms | tok/sec: 11,967 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 20 | total time: 1256.73m | eta: 594.9m
Step 54400 | Validation bpb: 0.978392
<|bos|>The capital of France is Paris, so it is important to keep it pretty simple. It is important not to make a huge mess, but you can
<|bos|>The chemical symbol of gold is the gold/Gold, commonly found in 60s, 70s, and 80s jewelry, and the symbol
<|bos|>If yesterday was Friday, then tomorrow will be Friday. Friday is a holiday of holiday festivities. It's supposed to be a time to make some food, eat some
<|bos|>The opposite of hot is cold. In the United States, a particular body of water is generally much cold than that of a country that is a few
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Saturn, Uranus, and Neptune (the planets in our solar system). These planets
<|bos|>My favorite color is the blue. I like to see the little blue of the sky all over and the warm blue and white of the ocean.
<|bos|>If 5*x + 3 = 13, then x is 13
If x is 6, then x is 6
There are three ways to do this. 1
step 54600/80000 (68.25%) | loss: 3.228408 | lrm: 0.51 | dt: 1380.61ms | tok/sec: 11,867 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 32 | total time: 1263.60m | eta: 587.9m
step 54900/80000 (68.62%) | loss: 3.209473 | lrm: 0.51 | dt: 1381.03ms | tok/sec: 11,863 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 45 | total time: 1270.48m | eta: 581.0m
step 55200/80000 (69.00%) | loss: 3.182904 | lrm: 0.50 | dt: 1375.87ms | tok/sec: 11,908 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 58 | total time: 1277.36m | eta: 574.0m
step 55500/80000 (69.38%) | loss: 3.276376 | lrm: 0.50 | dt: 1373.79ms | tok/sec: 11,926 | bf16_mfu: 0.00 | epoch: 1 pq: 28 rg: 71 | total time: 1284.24m | eta: 567.0m
step 55800/80000 (69.75%) | loss: 3.065210 | lrm: 0.49 | dt: 1370.34ms | tok/sec: 11,956 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 2 | total time: 1291.10m | eta: 560.0m
Step 56000 | Validation bpb: 0.975136
<|bos|>The capital of France is the city of the 20th century, and it is home to more than 500 monuments and monuments.
<|bos|>The chemical symbol of gold is a yellow-green color, with a metallic luster. The metal is abundant in nature, and in nature it is one of
<|bos|>`If yesterday was Friday, then tomorrow will be Saturday`. And if you can't watch TESOL, then we don't have to watch any TV or watch the T
<|bos|>`The opposite of hot is cold`. If you are trying to understand how to tell if you are not drinking enough, you can look for the color of
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Saturn, Jupiter, Saturn, Saturn, Jupiter (the other planets) Jupiter, Saturn, Jupiter
<|bos|>`My favorite color is red`. That's why I've made a small version of this colorful paper bag. I started making the paper bag at
<|bos|>If 5*x + 3 = 13, then x is 13x +3 = 10. (1/13 = 12.8, 10 = 9.
2026-03-16 08:08:58,236 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_056000.pt
2026-03-16 08:08:58,236 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_056000.json
2026-03-16 08:09:00,100 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_056000_rank0.pt
step 56100/80000 (70.12%) | loss: 3.165471 | lrm: 0.49 | dt: 1364.96ms | tok/sec: 12,003 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 15 | total time: 1297.98m | eta: 553.1m
step 56400/80000 (70.50%) | loss: 3.103401 | lrm: 0.48 | dt: 1377.10ms | tok/sec: 11,897 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 28 | total time: 1304.85m | eta: 546.1m
step 56700/80000 (70.88%) | loss: 3.193868 | lrm: 0.48 | dt: 1376.90ms | tok/sec: 11,899 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 41 | total time: 1311.71m | eta: 539.1m
step 57000/80000 (71.25%) | loss: 3.159663 | `lrm: 0.47` | dt: 1375.50ms | tok/sec: 11,911 | bf16_mfu: 0.00 | epoch: 1 `pq: 29` rg: 54 | total time: 1318.59m | eta: 532.2m


##

step 57300/80000 (71.62%) | loss: 3.207848 | lrm: 0.46 | dt: 1358.46ms | tok/sec: 12,060 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 67 | total time: 1325.47m | eta: 525.2m
Step 57600 | Validation bpb: 0.971392
<|bos|>The capital of France is the capital of the Roman Empire. The capital of France is the capital of the Roman Empire, the Republic of France is the
<|bos|>The chemical symbol of gold is Au. For example gold is the same as
Gold, but because it is a rare metal, gold is not
available
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. But if yesterday was Tuesday, then Tuesday will be Wednesday. 5:30-11 a.m. Sunday.
<|bos|>The opposite of hot is cold. If you are having trouble heating your home, you might want to get a professional heating installation in New York City.
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Jupiter, Saturn, Uranus, and Neptune.

The moon Uranus is called Jupiter
<|bos|>My favorite color is black. We love black because it doesn't require a lot of ink, doesn't require a lot of maintenance, and doesn
<|bos|>If 5*x + 3 = 13, then x is 13 or 5 or 13:\x + 13\
If x is 0 or 0,
step 57600/80000 (72.00%) | loss: 3.204349 | lrm: 0.46 | dt: 1413.94ms | tok/sec: 11,587 | bf16_mfu: 0.00 | epoch: 1 pq: 29 rg: 80 | total time: 1332.35m | eta: 518.2m
step 57900/80000 (72.38%) | loss: 3.183834 | lrm: 0.45 | dt: 1380.59ms | tok/sec: 11,867 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 9 | total time: 1339.23m | eta: 511.3m
step 58200/80000 (72.75%) | loss: 3.044582 | lrm: 0.45 | dt: 1373.71ms | tok/sec: 11,926 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 22 | total time: 1346.10m | eta: 504.3m
step 58500/80000 (73.12%) | loss: 3.108521 | lrm: 0.44 | dt: 1371.57ms | tok/sec: 11,945 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 35 | total time: 1352.99m | eta: 497.3m
step 58800/80000 (73.50%) | loss: 3.223050 | lrm: 0.44 | dt: 1382.71ms | tok/sec: 11,849 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 48 | total time: 1359.85m | eta: 490.4m
step 59100/80000 (73.88%) | loss: 3.144274 | lrm: 0.43 | dt: 1376.68ms | tok/sec: 11,901 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 61 | total time: 1366.73m | eta: 483.4m
Step 59200 | Validation bpb: 0.969650
<|bos|>The capital of France is the capital of the Roman Empire, located in the region of Nunalie, a city that has a rich history and
<|bos|>The chemical symbol of gold is gold. There are two metals of gold, so gold is called gold. There are four metals in this world, namely iron
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. You can get out in the weather just like any other time of day. This is great if you are looking for
<|bos|>The opposite of hot is cold. As a matter of fact, 25% of the 245 million global economy's annual output is dependent on
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Uranus, Neptune, and Uranus. The planets of the solar system are:
<|bos|>My favorite color is pink. When I think of this color, I think of pink people. But pink is also my favorite color. When I
<|bos|>If 5*x + 3 = 13, then x is the number of 13th letters of the alphabet. So if 5 x 5 = 10, then the sum
step 59400/80000 (74.25%) | loss: 3.183990 | lrm: 0.43 | dt: 1377.23ms | tok/sec: 11,896 | bf16_mfu: 0.00 | epoch: 1 pq: 30 rg: 74 | total time: 1373.60m | eta: 476.4m
step 59700/80000 (74.62%) | loss: 3.150168 | lrm: 0.42 | dt: 1376.64ms | tok/sec: 11,901 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 4 | total time: 1380.47m | eta: 469.5m
2026-03-16 09:41:08,246 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_060000.pt
2026-03-16 09:41:08,247 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_060000.json
2026-03-16 09:41:10,238 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_060000_rank0.pt
step 60000/80000 (75.00%) | loss: 3.122398 | lrm: 0.42 | dt: 1573.91ms | tok/sec: 10,409 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 17 | total time: 1387.34m | eta: 462.5m
step 60300/80000 (75.38%) | loss: 3.146600 | lrm: 0.41 | dt: 1357.91ms | tok/sec: 12,065 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 30 | total time: 1394.20m | eta: 455.6m
step 60600/80000 (75.75%) | loss: 3.179588 | lrm: 0.40 | dt: 1360.05ms | tok/sec: 12,046 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 43 | total time: 1401.07m | eta: 448.6m
Step 60800 | Validation bpb: 0.966468
<|bos|>The capital of France is the capital of the world, but this is not the only city known to have a strong influence on French culture. There are
<|bos|>The chemical symbol of gold is gold. It is the hardest and most popular metal for making jewelry and other decorative items. It is used for making necklaces
<|bos|>If yesterday was Friday, then tomorrow will be Friday. Now, the news about the ongoing investigation of the source of the water that the Westerns have been studying,
<|bos|>The opposite of hot is cold. In the winter, a small amount of energy is used to increase the temperature. In the summer, the body uses
<|bos|>The planets of the solar system are: Jupiter, Saturn, Saturn, Venus, Jupiter, Saturn, Mars, Jupiter, Saturn, Jupiter, Jupiter, Saturn, and
<|bos|>My favorite color is red. If you're feeling really fancy, your favorite color is usually red. It's a favorite for those who have a
<|bos|>If 5*x + 3 = 13, then x is 5.
If x is 12, then x is 12.
In this way, we get x = 12
step 60900/80000 (76.12%) | loss: 3.112745 | lrm: 0.40 | dt: 1375.09ms | tok/sec: 11,914 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 56 | total time: 1407.93m | eta: 441.6m
step 61200/80000 (76.50%) | loss: 3.119179 | lrm: 0.39 | dt: 1373.83ms | tok/sec: 11,925 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 68 | total time: 1414.81m | eta: 434.7m
step 61500/80000 (76.88%) | loss: 3.186226 | lrm: 0.39 | dt: 1376.29ms | tok/sec: 11,904 | bf16_mfu: 0.00 | epoch: 1 pq: 31 rg: 81 | total time: 1421.68m | eta: 427.7m
step 61800/80000 (77.25%) | loss: 3.146390 | lrm: 0.38 | dt: 1379.49ms | tok/sec: 11,876 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 11 | total time: 1428.55m | eta: 420.8m
step 62100/80000 (77.62%) | loss: 3.056560 | lrm: 0.38 | dt: 1381.91ms | tok/sec: 11,856 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 24 | total time: 1435.42m | eta: 413.8m
Step 62400 | Validation bpb: 0.963439
<|bos|>The capital of France is Paris, just 10 miles west of the capital city of Saint Pierre (French: Saint Pierre-Louis-M
<|bos|>The chemical symbol of gold is Ag. The chemical formula of gold is Ag^2S^T^, where S is the number of atoms, T
<|bos|>If yesterday was Friday, then tomorrow will be Friday. You can get on today for one day, and tomorrow Friday for another day. Even with all of the time that
<|bos|>The opposite of hot is cold. That's because of the reason that both are the same kind of body temperature, and neither are hot. So when
<|bos|>The planets of the solar system are: Mercury, Venus, Mars, Saturn, Jupiter, Uranus, Neptune, Pluto, Haleakal Nada
<|bos|>My favorite color is the blue-gray, with just a hint of purple. The fact that blue is one of the colors used in this image
<|bos|>If 5*x + 3 = 13, then x is the number of times 5, 3, 13, 13 + 3 = 5. So we know
step 62400/80000 (78.00%) | loss: 3.066649 | lrm: 0.37 | dt: 1411.48ms | tok/sec: 11,607 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 37 | total time: 1442.29m | eta: 406.9m
step 62700/80000 (78.38%) | loss: 3.162652 | lrm: 0.37 | dt: 1383.44ms | tok/sec: 11,842 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 50 | total time: 1449.16m | eta: 399.9m
step 63000/80000 (78.75%) | loss: 3.072813 | lrm: 0.36 | dt: 1376.97ms | tok/sec: 11,898 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 63 | total time: 1456.02m | eta: 393.0m
step 63300/80000 (79.12%) | loss: 3.105628 | lrm: 0.36 | dt: 1368.20ms | tok/sec: 11,974 | bf16_mfu: 0.00 | epoch: 1 pq: 32 rg: 76 | total time: 1462.89m | eta: 386.0m
step 63600/80000 (79.50%) | loss: 3.198911 | lrm: 0.35 | dt: 1361.04ms | tok/sec: 12,037 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 6 | total time: 1469.75m | eta: 379.1m
step 63900/80000 (79.88%) | loss: 3.105501 | lrm: 0.34 | dt: 1377.39ms | tok/sec: 11,894 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 19 | total time: 1476.62m | eta: 372.1m
Step 64000 | Validation bpb: 0.960231
<|bos|>The capital of France is the capital of the Republic of France. The capital of France is St. Gilder, and its name is Bret
<|bos|>The chemical symbol of gold is gold. There are 2,000 known forms of gold. The simplest form of gold is silver, which is made
<|bos|>If yesterday was Friday, then tomorrow will be Friday. You can see this today when you buy a new car to spend money on.

Isn't that what happens with
<|bos|>The opposite of hot is cold. Heat is a term that's defined as the ability to work without energy. When you feel hot, you feel warm
<|bos|>The planets of the solar system are: Jupiter, Saturn, Saturn, Uranus, Neptune, Uranus and Neptune. They are the smallest planet
<|bos|>My favorite color is the blue. It is a warm, peaceful, and relaxing color. It makes you feel good when you are relaxing.
If
<|bos|>If 5*x + 3 = 13, then x is 13 times 5. That means that 13 x 13 = 13 is 13.
And if you assume
2026-03-16 11:13:25,994 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_064000.pt
2026-03-16 11:13:25,994 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_064000.json
2026-03-16 11:13:27,873 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_064000_rank0.pt
step 64200/80000 (80.25%) | loss: 3.078056 | lrm: 0.34 | dt: 1371.08ms | tok/sec: 11,949 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 31 | total time: 1483.48m | eta: 365.2m
step 64500/80000 (80.62%) | loss: 3.153523 | lrm: 0.33 | dt: 1375.93ms | tok/sec: 11,907 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 44 | total time: 1490.35m | eta: 358.2m
step 64800/80000 (81.00%) | loss: 3.156335 | lrm: 0.33 | dt: 1355.15ms | tok/sec: 12,090 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 57 | total time: 1497.23m | eta: 351.3m
step 65100/80000 (81.38%) | loss: 3.068030 | lrm: 0.32 | dt: 1382.49ms | tok/sec: 11,851 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 70 | total time: 1504.09m | eta: 344.3m
step 65400/80000 (81.75%) | loss: 3.166681 | lrm: 0.32 | dt: 1378.54ms | tok/sec: 11,885 | bf16_mfu: 0.00 | epoch: 1 pq: 33 rg: 83 | total time: 1510.96m | eta: 337.4m
Step 65600 | Validation bpb: 0.958053
<|bos|>The capital of France is the capital of the Republic of Serbia, which is located in the south of France. It was established in 15
<|bos|>The chemical symbol of gold is Ag. The chemical formula of gold is Ag+ (aq) which means 2-3 hydrogen. It is found in
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. And if today was Monday, then the day will be Saturday. If today was Friday, then Wednesday will be Sunday
<|bos|>The opposite of hot is cold. Heat is a source of energy, and in the case of waves the energy is kinetic energy (that is, the
<|bos|>The planets of the solar system are: Jupiter, Saturn, Neptune, Uranus, Neptune, Jupiter, Neptune. Jupiter is the largest
<|bos|>My favorite color is black. In fact I was very pleased with the color. It's soft and warm. It also gives a sense of warmth
<|bos|>If 5*x + 3 = 13, then x is the number of points that need to be recorded (and hence x is the number of points that need to be recorded, in
step 65700/80000 (82.12%) | loss: 2.998422 | lrm: 0.31 | dt: 1374.53ms | tok/sec: 11,919 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 12 | total time: 1517.83m | eta: 330.4m
step 66000/80000 (82.50%) | loss: 3.120629 | lrm: 0.31 | dt: 1379.38ms | tok/sec: 11,877 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 25 | total time: 1524.69m | eta: 323.5m
step 66300/80000 (82.88%) | loss: 3.118458 | lrm: 0.30 | dt: 1369.06ms | tok/sec: 11,967 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 38 | total time: 1531.56m | eta: 316.5m
step 66600/80000 (83.25%) | loss: 3.087834 | lrm: 0.29 | dt: 1367.14ms | tok/sec: 11,984 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 51 | total time: 1538.43m | eta: 309.6m
step 66900/80000 (83.62%) | loss: 3.013821 | lrm: 0.29 | dt: 1368.60ms | tok/sec: 11,971 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 64 | total time: 1545.30m | eta: 302.6m
Step 67200 | Validation bpb: 0.955113
<|bos|>The capital of France is the capital of the British Empire of France. The capital of the British Empire is the capital of the United States of America.
<|bos|>The chemical symbol of gold is gold. It is the third most abundant element in the earth's crust. In the past, gold was produced by gold mining
<|bos|>If yesterday was Friday, then tomorrow will be Friday. That's because of the days from Sunday to Friday. In the 1950s, it was a Sunday.
<|bos|>The opposite of hot is cold. In the winter, we use the term "cold" for those with a low body temperature and for those that do
<|bos|>The planets of the solar system are: Earth, the moon and the sun. I found them in a lot of interesting places, but not the most interesting places.
<|bos|>My favorite color is the one my mother always put in her fridge. She's a brown bonnet that's 4″ tall, so
<|bos|>If 5*x + 3 = 13, then x is the number of times that the first character of the sentence is changed from 2 to 3 or vice versa. So the
step 67200/80000 (84.00%) | loss: 3.069637 | lrm: 0.28 | dt: 1418.33ms | tok/sec: 11,551 | bf16_mfu: 0.00 | epoch: 1 pq: 34 rg: 76 | total time: 1552.17m | eta: 295.7m
step 67500/80000 (84.38%) | loss: 3.065363 | lrm: 0.28 | dt: 1370.24ms | tok/sec: 11,957 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 7 | total time: 1559.04m | eta: 288.8m
step 67800/80000 (84.75%) | loss: 3.063693 | lrm: 0.27 | dt: 1373.44ms | tok/sec: 11,929 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 20 | total time: 1565.90m | eta: 281.8m
2026-03-16 12:45:31,179 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_068000.pt
2026-03-16 12:45:31,181 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_068000.json
2026-03-16 12:45:33,032 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_068000_rank0.pt
step 68100/80000 (85.12%) | loss: 3.065484 | lrm: 0.27 | dt: 1364.25ms | tok/sec: 12,009 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 33 | total time: 1572.77m | eta: 274.9m
step 68400/80000 (85.50%) | loss: 3.010094 | lrm: 0.26 | dt: 1363.49ms | tok/sec: 12,016 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 46 | total time: 1579.64m | eta: 267.9m
step 68700/80000 (85.88%) | loss: 3.051830 | lrm: 0.26 | dt: 1380.65ms | tok/sec: 11,866 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 59 | total time: 1586.51m | eta: 261.0m
Step 68800 | Validation bpb: 0.952318
<|bos|>The capital of France is the capital of the world, it is bordered by the Atlantic Ocean to the north, the Netherlands to the south, and
<|bos|>The chemical symbol of gold is Ag. It is the mineral, common green, yellow, blue, pink, purple, red, green, yellow, red
<|bos|>If yesterday was Friday, then tomorrow will be Sunday. Today will be Sunday 12th Monday. Today will be Sunday 13th Monday. The date will be
<|bos|>The opposite of hot is cold. You can get rid of these behaviors by using a heating mat. Heat mats work by absorbing excess heat from your clothes
<|bos|>The planets of the solar system are: Jupiter, Saturn, Neptune, Uranus, Neptune, Saturn, Pluto, Pluto, Neptune,
<|bos|>My favorite color is blue. You can find it anywhere in the world, from the deep blue ocean to the rugged mountains and mountains. In this
<|bos|>If 5*x + 3 = 13, then x is 5*x+3+3 = 13. If 5*x+3 = 2, then x
step 69000/80000 (86.25%) | loss: 3.127566 | lrm: 0.25 | dt: 1372.45ms | tok/sec: 11,937 | bf16_mfu: 0.00 | epoch: 1 pq: 35 rg: 72 | total time: 1593.38m | eta: 254.1m
step 69300/80000 (86.62%) | loss: 3.153322 | lrm: 0.25 | dt: 1381.37ms | tok/sec: 11,860 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 3 | total time: 1600.26m | eta: 247.1m
step 69600/80000 (87.00%) | loss: 3.092173 | lrm: 0.24 | dt: 1377.23ms | tok/sec: 11,896 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 16 | total time: 1607.14m | eta: 240.2m
step 69900/80000 (87.38%) | loss: 3.059889 | lrm: 0.23 | dt: 1375.39ms | tok/sec: 11,912 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 29 | total time: 1614.01m | eta: 233.2m
step 70200/80000 (87.75%) | loss: 3.083220 | lrm: 0.23 | dt: 1367.59ms | tok/sec: 11,980 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 41 | total time: 1620.89m | eta: 226.3m
Step 70400 | Validation bpb: 0.950303
<|bos|>The capital of France is the capital of the world. France is the richest country in the world. This is a highly respected place for people and for
<|bos|>The chemical symbol of gold is gold. It is the main ingredient in gold which is used in the production of jewelry and other industrial objects. The gold is
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. But today will be Thursday. Or in other words, you will need to go to work today. You are likely
<|bos|>The opposite of hot is cold. A hot car is always hot. I mean, the hottest that the car will be, is cold. And I
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Uranus, Neptune, Uranus, Neptune, and Uranus. These
<|bos|>My favorite color is blue. That's because it's the classic dark blue. You want to have a nice, light, and airy look to
<|bos|>If 5*x + 3 = 13, then x is 13/5=10*6=24, so the distance between each point in the circle is 5*24
step 70500/80000 (88.12%) | loss: 2.993781 | lrm: 0.22 | dt: 1379.73ms | tok/sec: 11,874 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 54 | total time: 1627.76m | eta: 219.4m
step 70800/80000 (88.50%) | loss: 3.108373 | lrm: 0.22 | dt: 1378.05ms | tok/sec: 11,889 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 67 | total time: 1634.63m | eta: 212.4m
step 71100/80000 (88.88%) | loss: 3.022424 | lrm: 0.21 | dt: 1370.65ms | tok/sec: 11,953 | bf16_mfu: 0.00 | epoch: 1 pq: 36 rg: 80 | total time: 1641.50m | eta: 205.5m
step 71400/80000 (89.25%) | loss: 3.138603 | lrm: 0.21 | dt: 1379.69ms | tok/sec: 11,875 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 11 | total time: 1648.38m | eta: 198.6m
step 71700/80000 (89.62%) | loss: 3.030662 | lrm: 0.20 | dt: 1373.72ms | tok/sec: 11,926 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 24 | total time: 1655.24m | eta: 191.6m
Step 72000 | Validation bpb: 0.947615
<|bos|>The capital of France is the capital of the world, with approximately 8,000 acres of the country. The capital has a rich history and
<|bos|>The chemical symbol of gold is gold. It is an element found in nature in the form of silver, lead, etc. There are also other names for
<|bos|>If yesterday was Friday, then tomorrow will be Friday. And if today was Thursday, then tomorrow will be Friday. Which day is right? That depends on what time the
<|bos|>The opposite of hot is cold. Hot is a term used to describe warm objects, but you can have both hot and cold objects. Hot objects are
<|bos|>The planets of the solar system are: Mars, Jupiter, Venus, Earth, Saturn, Uranus, Neptune, Jupiter, Saturn, Uranus, Ne
<|bos|>My favorite color is the blue. I like the look of the blue on the walls. I don't want to see a lot of green.
<|bos|>If 5*x + 3 = 13, then x is 13+13=13=13=13=13=13=13=13=13=13=13=
2026-03-16 14:17:54,376 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_072000.pt
2026-03-16 14:17:54,378 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_072000.json
2026-03-16 14:17:56,339 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_072000_rank0.pt
step 72000/80000 (90.00%) | loss: 3.083898 | lrm: 0.20 | dt: 1533.72ms | tok/sec: 10,682 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 37 | total time: 1662.12m | eta: 184.7m
step 72300/80000 (90.38%) | loss: 3.109227 | lrm: 0.19 | dt: 1364.08ms | tok/sec: 12,010 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 50 | total time: 1668.99m | eta: 177.8m
step 72600/80000 (90.75%) | loss: 3.111117 | lrm: 0.19 | dt: 1372.05ms | tok/sec: 11,941 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 63 | total time: 1675.85m | eta: 170.8m
step 72900/80000 (91.12%) | loss: 3.115190 | lrm: 0.18 | dt: 1363.46ms | tok/sec: 12,016 | bf16_mfu: 0.00 | epoch: 1 pq: 37 rg: 76 | total time: 1682.73m | eta: 163.9m
step 73200/80000 (91.50%) | loss: 3.077969 | lrm: 0.17 | dt: 1368.62ms | tok/sec: 11,971 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 6 | total time: 1689.61m | eta: 157.0m
step 73500/80000 (91.88%) | loss: 3.082948 | lrm: 0.17 | dt: 1358.99ms | tok/sec: 12,056 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 19 | total time: 1696.50m | eta: 150.1m
Step 73600 | Validation bpb: 0.945592
<|bos|>The capital of France is the capital of the world, it is the capital of the world that is 10.50 million square miles. 15
<|bos|>The chemical symbol of gold is gold. The chemical formula of gold is . The chemical symbol of gold is the symbol for gold.
What is the formula for
<|bos|>If yesterday was Friday, then tomorrow will be Friday. But if yesterday was Wednesday, then that was Thursday. No, you're not going to get it. The way
<|bos|>The opposite of hot is cold. You can get it either way. Most people use a steam spin, but that's probably a waste of energy.
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Uranus, Neptune, Uranus, Neptune, Uranus, Nept
<|bos|>My favorite color is blue. When I first started to read this book, I thought I'd start with this color to see how it got me
<|bos|>If 5*x + 3 = 13, then x is 13+13=13 + 13 = 13+13 = 13, so there's a 4+
step 73800/80000 (92.25%) | loss: 3.056981 | lrm: 0.16 | dt: 1374.26ms | tok/sec: 11,922 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 31 | total time: 1703.36m | eta: 143.1m
step 74100/80000 (92.62%) | loss: 3.105106 | lrm: 0.16 | dt: 1379.87ms | tok/sec: 11,873 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 44 | total time: 1710.23m | eta: 136.2m
step 74400/80000 (93.00%) | loss: 3.008095 | lrm: 0.15 | dt: 1377.27ms | tok/sec: 11,895 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 57 | total time: 1717.09m | eta: 129.3m
step 74700/80000 (93.38%) | loss: 3.062831 | lrm: 0.15 | dt: 1375.01ms | tok/sec: 11,915 | bf16_mfu: 0.00 | epoch: 1 pq: 38 rg: 70 | total time: 1723.96m | eta: 122.3m
step 75000/80000 (93.75%) | loss: 2.986440 | lrm: 0.14 | dt: 1384.95ms | tok/sec: 11,830 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 0 | total time: 1730.83m | eta: 115.4m
Step 75200 | Validation bpb: 0.943431
<|bos|>The capital of France is the capital of the world, France is the world's largest economy and the third largest city. In fact, the United States
<|bos|>The chemical symbol of gold is X, and the chemical symbol of gold is X. The chemical symbol of gold is X, and the chemical symbol of gold
<|bos|>If yesterday was Friday, then tomorrow will be Friday. But if yesterday was Wednesday, then tomorrow will be Thursday. Today and tomorrow are the first days of the month and
<|bos|>The opposite of hot is cold. In the world of fashion, many people are excited about the world of fashion, but some of us are scared of
<|bos|>The planets of the solar system are: Jupiter, Saturn, Jupiter, Saturn, Uranus, Neptune, Uranus and Neptune. The Sun is
<|bos|>My favorite color is blue. When I first started learning I wasn't sure how to start. So I thought I might as well start from scratch
<|bos|>If 5*x + 3 = 13, then x is 5*x+3+3 = 13. If 5*x+3 = 6, then x
step 75300/80000 (94.12%) | loss: 3.065701 | lrm: 0.14 | dt: 1376.09ms | tok/sec: 11,906 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 13 | total time: 1737.70m | eta: 108.5m
step 75600/80000 (94.50%) | loss: 3.171860 | lrm: 0.13 | dt: 1376.93ms | tok/sec: 11,898 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 26 | total time: 1744.58m | eta: 101.5m
step 75900/80000 (94.88%) | loss: 3.092988 | lrm: 0.12 | dt: 1366.64ms | tok/sec: 11,988 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 39 | total time: 1751.45m | eta: 94.6m
2026-03-16 15:50:01,776 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\model_076000.pt
2026-03-16 15:50:01,776 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_076000.json
2026-03-16 15:50:03,607 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_076000_rank0.pt
step 76200/80000 (95.25%) | loss: 3.110854 | lrm: 0.12 | dt: 1370.49ms | tok/sec: 11,954 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 52 | total time: 1758.31m | eta: 87.7m
step 76500/80000 (95.62%) | loss: 3.075452 | lrm: 0.11 | dt: 1373.70ms | tok/sec: 11,926 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 65 | total time: 1765.17m | eta: 80.8m
Step 76800 | Validation bpb: 0.941428
<|bos|>The capital of France is the capital of the country of Spain. The capital of Spain is Vero Beach, in the heart of the city, which
<|bos|>The chemical symbol of gold is gold. It is the purest form of gold. It is chemically stable at room temperature and in liquid state. Gold is
<|bos|>If yesterday was Friday, then tomorrow will be Friday. We will have a little while to discuss the different kinds of technology. We will be doing this activity in small groups
<|bos|>The opposite of hot is cold. You can get that hot thing but it's not a drink. Hot is cold, you don't drink cold.
<|bos|>The planets of the solar system are: Earth, the moon and the sun. One of the planets, Neptune, was discovered in 1859 and has
<|bos|>My favorite color is the blue! I like the look of the blue! I love the shape of the blue! I like the color, I
<|bos|>If 5*x + 3 = 13, then x is 13.
If x is 9, then x is 17.
This is an answer in a specific case where
step 76800/80000 (96.00%) | loss: 3.078369 | lrm: 0.11 | dt: 1402.82ms | tok/sec: 11,679 | bf16_mfu: 0.00 | epoch: 1 pq: 39 rg: 77 | total time: 1772.04m | eta: 73.8m
step 77100/80000 (96.38%) | loss: 3.079396 | lrm: 0.10 | dt: 1381.41ms | tok/sec: 11,860 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 8 | total time: 1778.91m | eta: 66.9m
step 77400/80000 (96.75%) | loss: 3.050076 | lrm: 0.10 | dt: 1378.73ms | tok/sec: 11,883 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 21 | total time: 1785.79m | eta: 60.0m
step 77700/80000 (97.12%) | loss: 3.088910 | lrm: 0.09 | dt: 1358.16ms | tok/sec: 12,063 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 34 | total time: 1792.66m | eta: 53.1m
step 78000/80000 (97.50%) | loss: 3.039630 | lrm: 0.09 | dt: 1369.76ms | tok/sec: 11,961 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 47 | total time: 1799.52m | eta: 46.1m
step 78300/80000 (97.88%) | loss: 3.026992 | lrm: 0.08 | dt: 1376.75ms | tok/sec: 11,900 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 60 | total time: 1806.40m | eta: 39.2m
Step 78400 | Validation bpb: 0.939296
<|bos|>The capital of France is the capital of the world. In 1885, the French Prime Minister of France, Louvé, took his
<|bos|>`The chemical symbol of gold is gold`. It is the only natural metal that is pure. The physical properties of gold include its physical, chemical, and chemical
<|bos|>`If yesterday was Friday, then tomorrow will be Saturday`. And if today was Thursday, then tomorrow will be Sunday. Just don't go out, you have to make the
<|bos|>`The opposite of hot is cold`. A hot car is more fun than a cold one, but it will not get you through to the next person.
<|bos|>T`he planets of the solar system are: Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Neptune. Mercury is the largest`
<|bos|>`My favorite color is the blue. I like it in a new way. It's beautiful and bright`. I love how blue light is like a
<|bos|>If 5*x + 3 = 13, then x is 13 * 13 = 6.2. So x + 1 = 13 = 4.7.
step 78600/80000 (98.25%) | loss: 3.030495 | lrm: 0.08 | dt: 1388.71ms | tok/sec: 11,797 | bf16_mfu: 0.00 | epoch: 1 pq: 40 rg: 73 | total time: 1813.26m | eta: 32.3m
step 78900/80000 (98.62%) | loss: 3.115520 | lrm: 0.07 | dt: 1398.80ms | tok/sec: 11,712 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 3 | total time: 1820.11m | eta: 25.4m
step 79200/80000 (99.00%) | loss: 3.091536 | lrm: 0.06 | dt: 1390.13ms | tok/sec: 11,785 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 16 | total time: 1826.97m | eta: 18.5m
step 79500/80000 (99.38%) | loss: 3.014011 | lrm: 0.06 | dt: 1355.50ms | tok/sec: 12,087 | bf16_mfu: 0.00 | epoch: 1 pq: 41 rg: 29 | total time: 1833.77m | eta: 11.5m
step 79800/80000 (99.75%) | loss: `2.960526` | `·lrm: 0.05 `| dt: 1398.21ms | tok/sec: 11,717 | bf16_mfu: 0.00 | epoch: 1 `pq: 41 rg: 41` | total time: 1840.59m | eta: 4.6m
Step 80000 | Validation bpb: 0.937755
<|bos|>`The capital of France is Paris`, with a total area of 704,000 km2 and a total population of about 1,40        
<|bos|>`The chemical symbol of gold is Ag`. The chemical formula of gold is Ag (Ag = Ago) + 24. A common type of metal in
<|bos|>`If yesterday was Friday, then tomorrow will be Saturday`. And if today was Thursday, then tomorrow will be Saturday. Not that I would suggest a Sunday. I don't
<|bos|>`The opposite of hot is cold. That's because air can't always remain at a constant temperature`. So, when you take your dog out, the
<|bos|>The planets of the solar system are: `Jupiter, Saturn, Saturn, Uranus, Neptune`, Uranus and Neptune. They are the 10
<|bos|>`My favorite color is the blue I love to play with`. I usually make my own playa or playbeds with a lot of my own
<|bos|>If 5*x + 3 = 13, then x is 13.
If x is 14, then x is 14.
What is the unit of the product of two integers
2026-03-16 17:22:16,303 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\ `model_080000.pt`
2026-03-16 17:22:16,304 - nanochat.checkpoint_manager - INFO - Saved metadata to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\meta_080000.json
2026-03-16 17:22:18,464 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongf\.cache\nanochat\base_checkpoints\d16\optim_080000_rank0.pt
Peak memory usage: 7690.55MiB
Total training time: `1845.17m`
Minimum validation bpb: `0.937755`

(my_project_env) C:\Users\hongf\miniconda3.1\envs\nana_GPT\nanochat>
