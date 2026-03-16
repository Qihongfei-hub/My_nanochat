

#
cd c:\Users\hong)i\myenv\na)o_GPT\nanochat
python -m venv .venv

## 激活环境
.\.venv\Scripts\Activate.ps1 

### optional 永久为当前用户设置为 RemoteSigned（不需管理员，信任本地脚本）：更方便但改动持久 
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
.\Activate.ps1

## 升级打包工具 
python -m pip install --upgrade pip setuptools wheel 

## 安装项目（开发模式）- not used 
pip install -e .

##
验证是否进入虚拟环境 
python -c "import sys; print(sys.executable); print(sys.prefix)"
pip --version




# install the lib 

nanochat 0.1.0 requires datasets>=4.0.0, which is not installed.
nanochat 0.1.0 requires fastapi>=0.117.1, which is not installed.
nanochat 0.1.0 requires ipykernel>=7.1.0, which is not installed.
nanochat 0.1.0 requires kernels>=0.11.7, which is not installed.
nanochat 0.1.0 requires matplotlib>=3.10.8, which is not installed.
nanochat 0.1.0 requires python-dotenv>=1.2.1, which is not installed.
nanochat 0.1.0 requires regex>=2025.9.1, which is not installed.
nanochat 0.1.0 requires rustbpe>=0.1.0, which is not installed.
nanochat 0.1.0 requires scipy>=1.15.3, which is not installed.
nanochat 0.1.0 requires tabulate>=0.9.0, which is not installed.
nanochat 0.1.0 requires tiktoken>=0.11.0, which is not installed.
nanochat 0.1.0 requires tokenizers>=0.22.0, which is not installed.
nanochat 0.1.0 requires torch==2.9.1, which is not installed.
nanochat 0.1.0 requires transformers>=4.57.3, which is not installed.
nanochat 0.1.0 requires uvicorn>=0.36.0, which is not installed.
nanochat 0.1.0 requires wandb>=0.21.3, which is not installed.
nanochat 0.1.0 requires zstandard>=0.25.0, which is not installed.


##
C:\Users\hongqi\.cache\nanochat\base_data_climbmix 

## fail of download, so workaround is to doanload manually
Downloading shard_00000.parquet...
Downloading shard_00001.parquet...
Downloading shard_00002.parquet...
Downloading shard_00003.parquet... 

manual download to fold 
C:\Users\hongqi\.cache\nanochat\base_data_climbmix 

# 
# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared


## download enough data for pre-Trainning 
250M =》 50M Words * 1.3 token/ words * 30 file     =   1950M
132M

 1950M/132M parameter =  14.7 ratio 


#Vocab 
train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data




##




## Vocab trainning 
(.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> bash runs/speedrun.sh     ###  裁剪了脚本 ，只跑Vocab trainning        
Reset report and wrote header to C:/Users/hongqi/.cache/nanochat\report\header.md
max_chars: 2,000,000,000
doc_cap: 10,000
vocab_size: 32,768
2026-03-11 15:16:56,667 - rustbpe - INFO - Processing sequences from iterator (buffer_size: 8192)


##
(.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> bash runs/speedrun.sh

##
从数据集中读取文本：通过 parquets_iter_batched(split="train") 生成文档流，并对每个文档裁剪到 --doc-cap，总读取不超过 --max-chars。
训练 tokenizer：调用 RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)（具体实现见 tokenizer.py）。
保存 tokenizer：保存到 get_base_dir()/tokenizer

##
快速自检：对一段示例文本 tokenizer.encode → tokenizer.decode，用 assert 确认解码结果与原文相同。
计算并缓存每个 token 的字节长度（排除特殊 token），把结果保存为 token_bytes.pt（用于后续 bits-per-byte 计算）。
记录训练元数据到报告：调用 get_report().log(...)（见 report.py）。


##
入口与依赖

导入关键模块：argparse, time, os, torch，以及仓库内部：
RustBPETokenizer（来自 nanochat.tokenizer）— 提供 train_from_iterator, encode, decode, save, get_vocab_size, get_special_tokens 等方法。

get_base_dir（来自 nanochat.common）— 用于确定项目基目录，后续将 tokenizer 存到基目录下的 tokenizer 文件夹。

parquets_iter_batched（来自 nanochat.dataset）— 返回训练文本的批次迭代器（每次一个 batch，batch 中包含多个文档字符串）。

get_report（来自 nanochat.report，在文件末尾使用）— 用来记录训练统计





##
Reset report and wrote header to C:/Users/hongqi/.cache/nanochat\report\header.md
max_chars: 2,000,000,000
doc_cap: 10,000
vocab_size: 32,768
2026-03-11 15:16:56,667 - rustbpe - INFO - Processing sequences from iterator (buffer_size: 8192)
2026-03-11 15:17:52,587 - rustbpe - INFO - Processed 593920 sequences total, 1792292 unique
2026-03-11 15:17:52,737 - rustbpe - INFO - Starting BPE training: 32503 merges to compute
2026-03-11 15:17:52,746 - rustbpe - INFO - Computing initial pair counts from 1792292 unique sequences
2026-03-11 15:17:54,211 - rustbpe - INFO - Building heap with 18727 unique pairs
2026-03-11 15:17:54,211 - rustbpe - INFO - Starting merge loop
2026-03-11 15:17:57,904 - rustbpe - INFO - Progress: 1% (326/32503 merges) - Last merge: (275, 263) -> 581 (frequency: 479614)
2026-03-11 15:17:58,642 - rustbpe - INFO - Progress: 2% (651/32503 merges) - Last merge: (367, 115) -> 906 (frequency: 195896)
2026-03-11 15:17:59,093 - rustbpe - INFO - Progress: 3% (976/32503 merges) - Last merge: (315, 104) -> 1231 (frequency: 122099)
2026-03-11 15:17:59,380 - rustbpe - INFO - Progress: 4% (1301/32503 merges) - Last merge: (1447, 497) -> 1556 (frequency: 86173)
2026-03-11 15:17:59,596 - rustbpe - INFO - Progress: 5% (1626/32503 merges) - Last merge: (274, 284) -> 1881 (frequency: 66093)
2026-03-11 15:17:59,775 - rustbpe - INFO - Progress: 6% (1951/32503 merges) - Last merge: (425, 272) -> 2206 (frequency: 53081)
2026-03-11 15:17:59,924 - rustbpe - INFO - Progress: 7% (2276/32503 merges) - Last merge: (359, 79) -> 2531 (frequency: 43519)
2026-03-11 15:18:00,062 - rustbpe - INFO - Progress: 8% (2601/32503 merges) - Last merge: (614, 583) -> 2856 (frequency: 36665)
2026-03-11 15:18:00,258 - rustbpe - INFO - Progress: 9% (2926/32503 merges) - Last merge: (503, 2092) -> 3181 (frequency: 31154)
2026-03-11 15:18:00,399 - rustbpe - INFO - Progress: 10% (3251/32503 merges) - Last merge: (278, 1719) -> 3506 (frequency: 26672)
2026-03-11 15:18:00,532 - rustbpe - INFO - Progress: 11% (3576/32503 merges) - Last merge: (638, 109) -> 3831 (frequency: 23539)
2026-03-11 15:18:00,617 - rustbpe - INFO - Progress: 12% (3901/32503 merges) - Last merge: (590, 269) -> 4156 (frequency: 20671)
2026-03-11 15:18:00,757 - rustbpe - INFO - Progress: 13% (4226/32503 merges) - Last merge: (274, 112) -> 4481 (frequency: 18634)
2026-03-11 15:18:00,874 - rustbpe - INFO - Progress: 14% (4551/32503 merges) - Last merge: (471, 115) -> 4806 (frequency: 16738)
2026-03-11 15:18:00,950 - rustbpe - INFO - Progress: 15% (4876/32503 merges) - Last merge: (2267, 311) -> 5131 (frequency: 15125)
2026-03-11 15:18:01,039 - rustbpe - INFO - Progress: 16% (5201/32503 merges) - Last merge: (1361, 767) -> 5456 (frequency: 13812)
2026-03-11 15:18:01,114 - rustbpe - INFO - Progress: 17% (5526/32503 merges) - Last merge: (398, 116) -> 5781 (frequency: 12757)
2026-03-11 15:18:01,193 - rustbpe - INFO - Progress: 18% (5851/32503 merges) - Last merge: (1629, 794) -> 6106 (frequency: 11724)
2026-03-11 15:18:01,296 - rustbpe - INFO - Progress: 19% (6176/32503 merges) - Last merge: (308, 460) -> 6431 (frequency: 10905)
2026-03-11 15:18:01,408 - rustbpe - INFO - Progress: 20% (6501/32503 merges) - Last merge: (603, 5669) -> 6756 (frequency: 10086)
2026-03-11 15:18:01,555 - rustbpe - INFO - Progress: 21% (6826/32503 merges) - Last merge: (3796, 4043) -> 7081 (frequency: 9369)
2026-03-11 15:18:01,665 - rustbpe - INFO - Progress: 22% (7151/32503 merges) - Last merge: (290, 273) -> 7406 (frequency: 8738)
2026-03-11 15:18:01,793 - rustbpe - INFO - Progress: 23% (7476/32503 merges) - Last merge: (1032, 97) -> 7731 (frequency: 8231)
2026-03-11 15:18:01,923 - rustbpe - INFO - Progress: 24% (7801/32503 merges) - Last merge: (1282, 115) -> 8056 (frequency: 7732)
2026-03-11 15:18:01,989 - rustbpe - INFO - Progress: 25% (8126/32503 merges) - Last merge: (3398, 2828) -> 8381 (frequency: 7280)
2026-03-11 15:18:02,083 - rustbpe - INFO - Progress: 26% (8451/32503 merges) - Last merge: (104, 432) -> 8706 (frequency: 6879)
2026-03-11 15:18:02,157 - rustbpe - INFO - Progress: 27% (8776/32503 merges) - Last merge: (2128, 7324) -> 9031 (frequency: 6492)
2026-03-11 15:18:02,191 - rustbpe - INFO - Progress: 28% (9101/32503 merges) - Last merge: (268, 441) -> 9356 (frequency: 6138)
2026-03-11 15:18:02,277 - rustbpe - INFO - Progress: 29% (9426/32503 merges) - Last merge: (327, 384) -> 9681 (frequency: 5824)
2026-03-11 15:18:02,372 - rustbpe - INFO - Progress: 30% (9751/32503 merges) - Last merge: (1927, 263) -> 10006 (frequency: 5558)
2026-03-11 15:18:02,397 - rustbpe - INFO - Progress: 31% (10076/32503 merges) - Last merge: (261, 272) -> 10331 (frequency: 5265)
2026-03-11 15:18:02,457 - rustbpe - INFO - Progress: 32% (10401/32503 merges) - Last merge: (325, 271) -> 10656 (frequency: 5007)
2026-03-11 15:18:02,526 - rustbpe - INFO - Progress: 33% (10726/32503 merges) - Last merge: (134, 146) -> 10981 (frequency: 4773)
2026-03-11 15:18:02,605 - rustbpe - INFO - Progress: 34% (11052/32503 merges) - Last merge: (685, 8018) -> 11307 (frequency: 4560)
2026-03-11 15:18:02,641 - rustbpe - INFO - Progress: 35% (11377/32503 merges) - Last merge: (523, 813) -> 11632 (frequency: 4359)
2026-03-11 15:18:02,706 - rustbpe - INFO - Progress: 36% (11702/32503 merges) - Last merge: (319, 7679) -> 11957 (frequency: 4144)
2026-03-11 15:18:02,756 - rustbpe - INFO - Progress: 37% (12027/32503 merges) - Last merge: (69, 646) -> 12282 (frequency: 3968)
2026-03-11 15:18:02,806 - rustbpe - INFO - Progress: 38% (12352/32503 merges) - Last merge: (6115, 649) -> 12607 (frequency: 3806)
2026-03-11 15:18:02,855 - rustbpe - INFO - Progress: 39% (12677/32503 merges) - Last merge: (300, 2610) -> 12932 (frequency: 3643)
2026-03-11 15:18:02,906 - rustbpe - INFO - Progress: 40% (13002/32503 merges) - Last merge: (111, 105) -> 13257 (frequency: 3491)
2026-03-11 15:18:03,005 - rustbpe - INFO - Progress: 41% (13327/32503 merges) - Last merge: (1421, 272) -> 13582 (frequency: 3353)
2026-03-11 15:18:03,042 - rustbpe - INFO - Progress: 42% (13652/32503 merges) - Last merge: (72, 263) -> 13907 (frequency: 3236)
2026-03-11 15:18:03,089 - rustbpe - INFO - Progress: 43% (13977/32503 merges) - Last merge: (8614, 115) -> 14232 (frequency: 3124)
2026-03-11 15:18:03,125 - rustbpe - INFO - Progress: 44% (14302/32503 merges) - Last merge: (2364, 608) -> 14557 (frequency: 2999)
2026-03-11 15:18:03,189 - rustbpe - INFO - Progress: 45% (14627/32503 merges) - Last merge: (4006, 3972) -> 14882 (frequency: 2887)
2026-03-11 15:18:03,256 - rustbpe - INFO - Progress: 46% (14952/32503 merges) - Last merge: (6703, 104) -> 15207 (frequency: 2783)
2026-03-11 15:18:03,288 - rustbpe - INFO - Progress: 47% (15277/32503 merges) - Last merge: (99, 6815) -> 15532 (frequency: 2678)
2026-03-11 15:18:03,339 - rustbpe - INFO - Progress: 48% (15602/32503 merges) - Last merge: (6437, 1735) -> 15857 (frequency: 2584)
2026-03-11 15:18:03,381 - rustbpe - INFO - Progress: 49% (15927/32503 merges) - Last merge: (709, 1613) -> 16182 (frequency: 2499)
2026-03-11 15:18:03,423 - rustbpe - INFO - Progress: 50% (16252/32503 merges) - Last merge: (1043, 112) -> 16507 (frequency: 2416)
2026-03-11 15:18:03,456 - rustbpe - INFO - Progress: 51% (16577/32503 merges) - Last merge: (9088, 294) -> 16832 (frequency: 2339)
2026-03-11 15:18:03,504 - rustbpe - INFO - Progress: 52% (16902/32503 merges) - Last merge: (311, 7856) -> 17157 (frequency: 2261)
2026-03-11 15:18:03,539 - rustbpe - INFO - Progress: 53% (17227/32503 merges) - Last merge: (5440, 9409) -> 17482 (frequency: 2199)
2026-03-11 15:18:03,582 - rustbpe - INFO - Progress: 54% (17552/32503 merges) - Last merge: (276, 323) -> 17807 (frequency: 2137)
2026-03-11 15:18:03,606 - rustbpe - INFO - Progress: 55% (17877/32503 merges) - Last merge: (6803, 785) -> 18132 (frequency: 2075)
2026-03-11 15:18:03,644 - rustbpe - INFO - Progress: 56% (18202/32503 merges) - Last merge: (70, 579) -> 18457 (frequency: 2016)
2026-03-11 15:18:03,689 - rustbpe - INFO - Progress: 57% (18527/32503 merges) - Last merge: (4951, 292) -> 18782 (frequency: 1957)
2026-03-11 15:18:03,734 - rustbpe - INFO - Progress: 58% (18852/32503 merges) - Last merge: (8480, 1175) -> 19107 (frequency: 1907)
2026-03-11 15:18:03,761 - rustbpe - INFO - Progress: 59% (19177/32503 merges) - Last merge: (309, 10620) -> 19432 (frequency: 1850)
2026-03-11 15:18:03,791 - rustbpe - INFO - Progress: 60% (19502/32503 merges) - Last merge: (404, 117) -> 19757 (frequency: 1799)
2026-03-11 15:18:03,824 - rustbpe - INFO - Progress: 61% (19827/32503 merges) - Last merge: (310, 706) -> 20082 (frequency: 1749)
2026-03-11 15:18:03,872 - rustbpe - INFO - Progress: 62% (20152/32503 merges) - Last merge: (46, 7986) -> 20407 (frequency: 1704)
2026-03-11 15:18:03,905 - rustbpe - INFO - Progress: 63% (20477/32503 merges) - Last merge: (4574, 97) -> 20732 (frequency: 1659)
2026-03-11 15:18:03,971 - rustbpe - INFO - Progress: 64% (20802/32503 merges) - Last merge: (4568, 12851) -> 21057 (frequency: 1615)
2026-03-11 15:18:03,991 - rustbpe - INFO - Progress: 65% (21127/32503 merges) - Last merge: (433, 330) -> 21382 (frequency: 1573)
2026-03-11 15:18:04,038 - rustbpe - INFO - Progress: 66% (21452/32503 merges) - Last merge: (6335, 795) -> 21707 (frequency: 1533)
2026-03-11 15:18:04,072 - rustbpe - INFO - Progress: 67% (21778/32503 merges) - Last merge: (359, 69) -> 22033 (frequency: 1493)
2026-03-11 15:18:04,105 - rustbpe - INFO - Progress: 68% (22103/32503 merges) - Last merge: (407, 628) -> 22358 (frequency: 1458)
2026-03-11 15:18:04,145 - rustbpe - INFO - Progress: 69% (22428/32503 merges) - Last merge: (2052, 19611) -> 22683 (frequency: 1421)
2026-03-11 15:18:04,200 - rustbpe - INFO - Progress: 70% (22753/32503 merges) - Last merge: (66, 913) -> 23008 (frequency: 1383)
2026-03-11 15:18:04,218 - rustbpe - INFO - Progress: 71% (23078/32503 merges) - Last merge: (2715, 336) -> 23333 (frequency: 1351)
2026-03-11 15:18:04,245 - rustbpe - INFO - Progress: 72% (23403/32503 merges) - Last merge: (409, 21618) -> 23658 (frequency: 1318)
2026-03-11 15:18:04,264 - rustbpe - INFO - Progress: 73% (23728/32503 merges) - Last merge: (959, 320) -> 23983 (frequency: 1288)
2026-03-11 15:18:04,295 - rustbpe - INFO - Progress: 74% (24053/32503 merges) - Last merge: (4217, 367) -> 24308 (frequency: 1256)
2026-03-11 15:18:04,327 - rustbpe - INFO - Progress: 75% (24378/32503 merges) - Last merge: (2465, 97) -> 24633 (frequency: 1229)
2026-03-11 15:18:04,347 - rustbpe - INFO - Progress: 76% (24703/32503 merges) - Last merge: (337, 4855) -> 24958 (frequency: 1200)
2026-03-11 15:18:04,684 - rustbpe - INFO - Progress: 89% (28928/32503 merges) - Last merge: (13016, 937) -> 29183 (frequency: 908)
2026-03-11 15:18:04,706 - rustbpe - INFO - Progress: 90% (29253/32503 merges) - Last merge: (45, 286) -> 29508 (frequency: 889)
2026-03-11 15:18:04,727 - rustbpe - INFO - Progress: 91% (29578/32503 merges) - Last merge: (972, 105) -> 29833 (frequency: 872)
2026-03-11 15:18:04,746 - rustbpe - INFO - Progress: 92% (29903/32503 merges) - Last merge: (399, 2522) -> 30158 (frequency: 855)
2026-03-11 15:18:04,770 - rustbpe - INFO - Progress: 93% (30228/32503 merges) - Last merge: (68, 269) -> 30483 (frequency: 838)
2026-03-11 15:18:04,797 - rustbpe - INFO - Progress: 94% (30553/32503 merges) - Last merge: (5233, 269) -> 30808 (frequency: 823)
2026-03-11 15:18:04,806 - rustbpe - INFO - Progress: 95% (30878/32503 merges) - Last merge: (581, 21678) -> 31133 (frequency: 807)
2026-03-11 15:18:04,824 - rustbpe - INFO - Progress: 96% (31203/32503 merges) - Last merge: (298, 23749) -> 31458 (frequency: 790)
2026-03-11 15:18:04,844 - rustbpe - INFO - Progress: 97% (31528/32503 merges) - Last merge: (46, 1823) -> 31783 (frequency: 776)
2026-03-11 15:18:04,870 - rustbpe - INFO - Progress: 98% (31853/32503 merges) - Last merge: (13499, 772) -> 32108 (frequency: 763)
2026-03-11 15:18:04,895 - rustbpe - INFO - Progress: 99% (32178/32503 merges) - Last merge: (480, 1462) -> 32433 (frequency: 751)
2026-03-11 15:18:04,908 - rustbpe - INFO - Progress: 100% (32503/32503 merges) - Last merge: (1982, 767) -> 32758 (frequency: 738)
2026-03-11 15:18:04,908 - rustbpe - INFO - Finished training: 32503 merges completed
Training time: 68.82s
Saved tokenizer encoding to C:/Users/hongqi/.cache/nanochat\tokenizer\tokenizer.pkl
Saved token_bytes to C:/Users/hongqi/.cache/nanochat\tokenizer\token_bytes.pt


## C:\Users\hongqi\.cache\nanochat\report.md
## Tokenizer training
timestamp: 2026-03-11 15:18:05

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 32,768
- train_time: 68.8250s
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.5798
- token_bytes_std: 2.8102


## tok_train.py 
目的：tok_train.py 用于训练一个 BPE 分词器（基于仓库内的 RustBPETokenizer），并保存模型与每个 token 占用字节数的缓存用于后续评估（例如 bits-per-byte）。
流程：解析命令行参数 → 从数据集迭代文本 → 用`迭代器`训练 tokenizer → 保存 tokenizer → 运行快速自检 → 计算并保存 token 字节长度向量 → 记录训练统计到报告。




##
`分词的评测`

(.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> bash runs/speedrun.sh
Reset report and wrote header to C:/Users/hongqi/.cache/nanochat\report\header.md

修改代码如下
#python -m scripts.tok_train
##
python -m scripts.tok_eval

Vocab sizes:
GPT-2: 50257
GPT-4: 100277
Ours: 32768   
## better than GPT-2


Comparison with GPT-2:
===============================================================================================
Text Type  Bytes    GPT-2           Ours            Relative     Better    
                    Tokens  Ratio   Tokens  Ratio   Diff %      
-----------------------------------------------------------------------------------------------
news       1819     404     4.50    405     4.49       -0.2%     GPT-2
korean     893      745     1.20    741     1.21       +0.5%     Ours
code       1259     576     2.19    397     3.17      +31.1%     Ours
math       1834     936     1.96    912     2.01       +2.6%     Ours
science    1112     260     4.28    244     4.56       +6.2%     Ours
fwe-train  2948778  631304  4.67    622446  4.74       +1.4%     Ours
fwe-val    3052296  668652  4.56    658245  4.64       +1.6%     Ours

Comparison with GPT-4:
===============================================================================================
Text Type  Bytes    GPT-4           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     387     4.70    405     4.49       -4.7%     GPT-4
korean     893      364     2.45    741     1.21     -103.6%     GPT-4
code       1259     309     4.07    397     3.17      -28.5%     GPT-4
math       1834     832     2.20    912     2.01       -9.6%     GPT-4
science    1112     249     4.47    244     4.56       +2.0%     Ours
fwe-train  2948778  611619  4.82    622446  4.74       -1.8%     GPT-4
fwe-val    3052296  642512  4.75    658245  4.64       -2.4%     GPT-4



##
pip install scripts 


## pre-train 

(.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> bash runs/speedrun.sh
Reset report and wrote header to C:/Users/hongqi/.cache/nanochat\report\header.md

                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    
Autodetected device type: cpu
2026-03-11 18:19:11,126 - nanochat.common - INFO - Distributed world size: 1
COMPUTE_DTYPE: torch.float32 (auto-detected: no CUDA (CPU/MPS))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
WARNING: SDPA has no support for sliding window attention (window_pattern='SSSL'). Your GPU utilization will be terrible.
WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Vocab size: 32,768
Model config:
{
  "sequence_len": 2048,
  "vocab_size": 32768,
  "n_layer": 4,
  "n_head": 2,
  "n_kv_head": 2,
  "n_embd": 256,
  "window_pattern": "SSSL"
}
WARNING: cl.exe not found on Windows CPU; skipping torch.compile and using eager mode
WARNING: torch.compile disabled (set NANOCHAT_DISABLE_COMPILE=0 to re-enable when toolchain is available)    
Parameter counts:
wte                     : 8,388,608
value_embeds            : 16,777,216
lm_head                 : 8,388,608
transformer_matrices    : 3,145,776
scalars                 : 8
total                   : 36,700,216
Estimated FLOPs per token: 8.257565e+07
Auto-computed optimal batch size: 262,144 tokens
Scaling LRs by 0.7071 for batch size 262,144 (reference: 524,288)
Scaling weight decay from 0.280000 to 1.889903 for depth 4
Scaling the LR for the AdamW parameters ∝1/√(256/768) = 1.732051
Calculated number of iterations from target data:param ratio: 418
Total number of training tokens: 109,576,192
Tokens : Scaling params ratio: 9.50
Total training FLOPs estimate: 9.048325e+15
Tokens / micro-batch / rank: 1 x 2048 = 2,048
Tokens / micro-batch: 2,048
Total batch size 262,144 => gradient accumulation steps: 128


## `disable torch.compile`
WARNING: cl.exe not found on Windows CPU; skipping torch.compile and using eager mode
WARNING: torch.compile disabled (set NANOCHAT_DISABLE_COMPILE=0 to re-enable when toolchain is available) 


##
Vocab size: 2,048
Model config:
{
  "sequence_len": 16,
  "vocab_size": 2048,
  "n_layer": 4,
  "n_head": 1,
  "n_kv_head": 1,
  "n_embd": 16,
  "window_pattern": "SSSL"
}
Estimated FLOPs per token: 2.827680e+05
Auto-computed optimal batch size: 262,144 tokens
Scaling LRs by 0.7071 for batch size 262,144 (reference: 524,288)
Scaling weight decay from 0.280000 to 1.889846 for depth 4
Scaling the LR for the AdamW parameters ∝1/√(16/768) = 6.928203
Calculated number of iterations from target data:param ratio: 0
Total number of training tokens: 0
Tokens : Scaling params ratio: 0.00
Total training FLOPs estimate: 0.000000e+00
Tokens / micro-batch / rank: 1 x 16 = 16
Tokens / micro-batch: 16
Total batch size 262,144 => gradient accumulation steps: 16384


##
Vocab size: 2,048
Model config:
{
  "sequence_len": 16,
  "vocab_size": 2048,
  "n_layer": 4,
  "n_head": 4,
  "n_kv_head": 4,
  "n_embd": 128,
  "window_pattern": "SSSL"
}

Parameter counts:
wte                     : 262,144
value_embeds            : 524,288
lm_head                 : 262,144
transformer_matrices    : 786,528
scalars                 : 8
total                   : 1,835,112
Estimated FLOPs per token: 6.390336e+06
Auto-computed optimal batch size: 131,072 tokens
Scaling LRs by 0.5000 for batch size 131,072 (reference: 524,288)
Scaling weight decay from 0.280000 to 2.939846 for depth 4
Scaling the LR for the AdamW parameters ∝1/√(128/768) = 2.449490
Calculated number of iterations from target data:param ratio: 32
Total number of training tokens: 4,194,304
Tokens : Scaling params ratio: 4.00
Total training FLOPs estimate: 2.680301e+13
Tokens / micro-batch / rank: 1 x 16 = 16
Tokens / micro-batch: 16
Total batch size 131,072 => gradient accumulation steps: 8192


##
layer_idx % 2 == (n_layer - 1) % 2
说明使用 0-based 层索引，并选择与“最后一层索引的奇偶性”相同的层。

因此当 n_layer=4 时：

最后一层索引 = 3，(3 % 2) = 1
满足 layer_idx % 2 == 1 的层是 1 和 3（0-based），所以这两层有 value-embed。
代码创建这些嵌入的位置见 gpt.py:185。
直观原因：

设计上采用“交替层（alternating）+ 确保最后一层包含”的策略（ResFormer 风格的 value-embedding），这样既能在若干层加入可学习的 value 补偿以改善表示，又不会在每层都增加大量参数/开销。


##

概览

文件: `gen_synthetic_data.py`
目的: 生成用于 `SFT（监督微调）`的合成多轮对话，教 nanochat 关于自身的身份、能力与限制。输出为 JSONL，每行一个对话（可包含元数据）。
主要组成（函数/变量）

knowledge：从 ../knowledge/self_knowledge.md 加载的“权威”知识库（脚本依赖环境变量 OPENROUTER_API_KEY）。
多样性控制集合：topics, personas, dynamics, first_messages（用于在主题、提问者身份、对话走向、开场语上注入多样性）。
prompt_template：发送给模型的系统/用户提示模板，包含知识库与输出格式要求（JSON schema）。
API 配置：url, headers, base_payload（使用 OpenRouter 接口，model 字段为 "google/gemini-3-flash-preview"，response_format 指定 json_schema）。
sample_diversity_elements(rng)：从各个维度采样一组元素（topic、persona、dynamic、示例首句）。
generate_conversation(idx)：核心生成函数。以 idx 为随机种子，构建 prompt，调用 OpenRouter（POST），解析返回的 JSON（期望在 choices[0].message.content 中是 JSON 字符串），返回 messages + metadata。
validate_conversation(messages)：校验对话结构（至少两条，角色交替，内容非空）。
主入口（if __name__ == "__main__"）：使用 argparse 支持 --num, --workers, --output, --append, --save-metadata。用 ThreadPoolExecutor 并行生成，多线程写入输出文件（每行写 JSON）。
运行与用法（快速示例）

先设置 API key（.env 或环境变量）:
运行示例：
输出默认路径: get_base_dir()/identity_conversations.jsonl（如果不传 --output）。--save-metadata 会把 "metadata" 一并写入每行。
注意与潜在问题

文件先读取 knowledge 再断言存在（顺序不当）：脚本中是先 open(knowledge_path).read()，随后 assert os.path.exists(knowledge_path)，应先检查存在再打开。
依赖与限制：需要有效的 OPENROUTER_API_KEY、OpenRouter 可用且模型支持 response_format=json_schema；API 可能有速率限制或返回错误，当前仅在出错时抛异常并计数，没有重试/退避机制。
对返回格式的假定较严格（期望 choices[0]['message']['content'] 是可 JSON 解析的字符串并匹配 schema）；建议添加更健壮的错误处理与日志。
并发写文件：当前每个任务完成后以追加方式打开并写入，若大量并发写入可能稍慢但通常安全；如需更高吞吐可改为队列或批量写入。
建议改进（可选）



##
(.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> bash .\runs\runcpu.sh
max_chars: 2,000,000,000
doc_cap: 10,000
vocab_size: 8,192
2026-03-12 15:12:52,126 - rustbpe - INFO - Processing sequences from iterator (buffer_size: 8192)

(.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> bash .\runs\runcpu.sh
max_chars: 2,000,000,000
doc_cap: 10,000
vocab_size: 8,192
2026-03-12 15:12:52,126 - rustbpe - INFO - Processing sequences from iterator (buffer_size: 8192)
(.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> bash .\runs\runcpu.sh
max_chars: 2,000,000,000
doc_cap: 10,000
(.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> bash .\runs\runcpu.sh
max_chars: 2,000,000,000
(.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> bash .\runs\runcpu.sh
max_chars: 2,000,000,000
doc_cap: 10,000
vocab_size: 8,192
2026-03-12 15:12:52,126 - rustbpe - INFO - Processing sequences from iterator (buffer_size: 8192)
2026-03-12 15:13:08,711 - rustbpe - INFO - Processed 172032 sequences total, 816556 unique
2026-03-12 15:13:08,789 - rustbpe - INFO - Starting BPE training: 7927 merges to compute
2026-03-12 15:13:08,790 - rustbpe - INFO - Computing initial pair counts from 816556 unique sequences
2026-03-12 15:13:09,936 - rustbpe - INFO - Building heap with 15477 unique pairs
2026-03-12 15:13:09,937 - rustbpe - INFO - Starting merge loop
2026-03-12 15:13:10,989 - rustbpe - INFO - Progress: 1% (80/7927 merges) - Last merge: (105, 114) -> 335 (frequency: 697986)
2026-03-12 15:13:11,326 - rustbpe - INFO - Progress: 2% (159/7927 merges) - Last merge: (101, 297) -> 414 (frequency: 326973)
2026-03-12 15:13:11,563 - rustbpe - INFO - Progress: 3% (238/7927 merges) - Last merge: (97, 103) -> 493 (frequency: 196571)
2026-03-12 15:13:11,705 - rustbpe - INFO - Progress: 4% (318/7927 merges) - Last merge: (294, 101) -> 573 (frequency: 142796)
2026-03-12 15:13:11,804 - rustbpe - INFO - Progress: 5% (397/7927 merges) - Last merge: (560, 99) -> 652 (frequency: 105960)
2026-03-12 15:13:11,921 - rustbpe - INFO - Progress: 6% (476/7927 merges) - Last merge: (266, 285) -> 731 (frequency: 86562)
2026-03-12 15:13:11,988 - rustbpe - INFO - Progress: 7% (555/7927 merges) - Last merge: (105, 566) -> 810 (frequency: 70773)
2026-03-12 15:13:12,038 - rustbpe - INFO - Progress: 8% (635/7927 merges) - Last merge: (105, 384) -> 890 (frequency: 59215)
2026-03-12 15:13:12,077 - rustbpe - INFO - Progress: 9% (714/7927 merges) - Last merge: (299, 644) -> 969 (frequency: 51397)
2026-03-12 15:13:12,123 - rustbpe - INFO - Progress: 10% (793/7927 merges) - Last merge: (337, 115) -> 1048 (frequency: 45694)
2026-03-12 15:13:12,187 - rustbpe - INFO - Progress: 11% (872/7927 merges) - Last merge: (410, 103) -> 1127 (frequency: 41036)
2026-03-12 15:13:12,254 - rustbpe - INFO - Progress: 12% (952/7927 merges) - Last merge: (436, 105) -> 1207 (frequency: 36427)
2026-03-12 15:13:12,276 - rustbpe - INFO - Progress: 13% (1031/7927 merges) - Last merge: (73, 116) -> 1286 (frequency: 33095)
2026-03-12 15:13:12,310 - rustbpe - INFO - Progress: 14% (1110/7927 merges) - Last merge: (289, 348) -> 1365 (frequency: 30232)
2026-03-12 15:13:12,354 - rustbpe - INFO - Progress: 15% (1190/7927 merges) - Last merge: (1123, 794) -> 1445 (frequency: 27643)
2026-03-12 15:13:12,392 - rustbpe - INFO - Progress: 16% (1269/7927 merges) - Last merge: (332, 960) -> 1524 (frequency: 25625)
2026-03-12 15:13:12,406 - rustbpe - INFO - Progress: 17% (1348/7927 merges) - Last merge: (381, 924) -> 1603 (frequency: 24101)
2026-03-12 15:13:12,441 - rustbpe - INFO - Progress: 18% (1427/7927 merges) - Last merge: (309, 261) -> 1682 (frequency: 22437)
2026-03-12 15:13:12,477 - rustbpe - INFO - Progress: 19% (1507/7927 merges) - Last merge: (49, 51) -> 1762 (frequency: 20947)
2026-03-12 15:13:12,494 - rustbpe - INFO - Progress: 20% (1586/7927 merges) - Last merge: (112, 277) -> 1841 (frequency: 19864)
2026-03-12 15:13:12,520 - rustbpe - INFO - Progress: 21% (1665/7927 merges) - Last merge: (32, 105) -> 1920 (frequency: 18619)
2026-03-12 15:13:12,540 - rustbpe - INFO - Progress: 22% (1744/7927 merges) - Last merge: (449, 658) -> 1999 (frequency: 17650)
2026-03-12 15:13:12,569 - rustbpe - INFO - Progress: 23% (1824/7927 merges) - Last merge: (1047, 280) -> 2079 (frequency: 16618)
2026-03-12 15:13:12,589 - rustbpe - INFO - Progress: 24% (1903/7927 merges) - Last merge: (520, 715) -> 2158 (frequency: 15759)
2026-03-12 15:13:12,609 - rustbpe - INFO - Progress: 25% (1982/7927 merges) - Last merge: (990, 618) -> 2237 (frequency: 15005)
2026-03-12 15:13:12,623 - rustbpe - INFO - Progress: 26% (2062/7927 merges) - Last merge: (403, 1161) -> 2317 (frequency: 14239)
2026-03-12 15:13:12,639 - rustbpe - INFO - Progress: 27% (2141/7927 merges) - Last merge: (614, 100) -> 2396 (frequency: 13695)
2026-03-12 15:13:12,656 - rustbpe - INFO - Progress: 28% (2220/7927 merges) - Last merge: (264, 348) -> 2475 (frequency: 13047)
2026-03-12 15:13:12,677 - rustbpe - INFO - Progress: 29% (2299/7927 merges) - Last merge: (461, 272) -> 2554 (frequency: 12435)
2026-03-12 15:13:12,690 - rustbpe - INFO - Progress: 30% (2379/7927 merges) - Last merge: (670, 392) -> 2634 (frequency: 11846)
2026-03-12 15:13:12,719 - rustbpe - INFO - Progress: 31% (2458/7927 merges) - Last merge: (1402, 272) -> 2713 (frequency: 11388)
2026-03-12 15:13:12,740 - rustbpe - INFO - Progress: 32% (2537/7927 merges) - Last merge: (65, 83) -> 2792 (frequency: 10981)
2026-03-12 15:13:12,773 - rustbpe - INFO - Progress: 33% (2616/7927 merges) - Last merge: (300, 1579) -> 2871 (frequency: 10535)
2026-03-12 15:13:12,793 - rustbpe - INFO - Progress: 34% (2696/7927 merges) - Last merge: (269, 608) -> 2951 (frequency: 10070)
2026-03-12 15:13:12,823 - rustbpe - INFO - Progress: 35% (2775/7927 merges) - Last merge: (300, 890) -> 3030 (frequency: 9649)
2026-03-12 15:13:12,837 - rustbpe - INFO - Progress: 36% (2854/7927 merges) - Last merge: (1806, 1480) -> 3109 (frequency: 9272)
2026-03-12 15:13:12,863 - rustbpe - INFO - Progress: 37% (2933/7927 merges) - Last merge: (1344, 290) -> 3188 (frequency: 8983)
2026-03-12 15:13:12,885 - rustbpe - INFO - Progress: 38% (3013/7927 merges) - Last merge: (256, 629) -> 3268 (frequency: 8660)
2026-03-12 15:13:12,907 - rustbpe - INFO - Progress: 39% (3092/7927 merges) - Last merge: (468, 386) -> 3347 (frequency: 8373)
2026-03-12 15:13:12,921 - rustbpe - INFO - Progress: 40% (3171/7927 merges) - Last merge: (1051, 117) -> 3426 (frequency: 7990)
2026-03-12 15:13:12,938 - rustbpe - INFO - Progress: 41% (3251/7927 merges) - Last merge: (868, 108) -> 3506 (frequency: 7730)
2026-03-12 15:13:12,956 - rustbpe - INFO - Progress: 42% (3330/7927 merges) - Last merge: (1535, 112) -> 3585 (frequency: 7453)
2026-03-12 15:13:12,978 - rustbpe - INFO - Progress: 43% (3409/7927 merges) - Last merge: (1353, 386) -> 3664 (frequency: 7254)
2026-03-12 15:13:12,993 - rustbpe - INFO - Progress: 44% (3488/7927 merges) - Last merge: (1057, 785) -> 3743 (frequency: 7028)
2026-03-12 15:13:12,996 - rustbpe - INFO - Progress: 45% (3568/7927 merges) - Last merge: (1681, 1032) -> 3823 (frequency: 6833)
2026-03-12 15:13:13,010 - rustbpe - INFO - Progress: 46% (3647/7927 merges) - Last merge: (32, 58) -> 3902 (frequency: 6637) 
2026-03-12 15:13:13,021 - rustbpe - INFO - Progress: 47% (3726/7927 merges) - Last merge: (691, 2064) -> 3981 (frequency: 6408)
2026-03-12 15:13:13,029 - rustbpe - INFO - Progress: 48% (3805/7927 merges) - Last merge: (117, 323) -> 4060 (frequency: 6208)
2026-03-12 15:13:13,037 - rustbpe - INFO - Progress: 49% (3885/7927 merges) - Last merge: (80, 83) -> 4140 (frequency: 6027)
2026-03-12 15:13:13,059 - rustbpe - INFO - Progress: 50% (3964/7927 merges) - Last merge: (781, 3508) -> 4219 (frequency: 5854)
2026-03-12 15:13:13,071 - rustbpe - INFO - Progress: 51% (4043/7927 merges) - Last merge: (84, 492) -> 4298 (frequency: 5713)
2026-03-12 15:13:13,092 - rustbpe - INFO - Progress: 52% (4123/7927 merges) - Last merge: (754, 284) -> 4378 (frequency: 5595)
2026-03-12 15:13:13,106 - rustbpe - INFO - Progress: 53% (4202/7927 merges) - Last merge: (276, 523) -> 4457 (frequency: 5426)
2026-03-12 15:13:13,120 - rustbpe - INFO - Progress: 54% (4281/7927 merges) - Last merge: (1641, 1250) -> 4536 (frequency: 5279)
2026-03-12 15:13:13,122 - rustbpe - INFO - Progress: 55% (4360/7927 merges) - Last merge: (99, 869) -> 4615 (frequency: 5156)
2026-03-12 15:13:13,143 - rustbpe - INFO - Progress: 56% (4440/7927 merges) - Last merge: (100, 284) -> 4695 (frequency: 5019)
2026-03-12 15:13:13,163 - rustbpe - INFO - Progress: 57% (4519/7927 merges) - Last merge: (1171, 910) -> 4774 (frequency: 4879)
2026-03-12 15:13:13,192 - rustbpe - INFO - Progress: 58% (4598/7927 merges) - Last merge: (1841, 586) -> 4853 (frequency: 4761)
2026-03-12 15:13:13,196 - rustbpe - INFO - Progress: 59% (4677/7927 merges) - Last merge: (3036, 305) -> 4932 (frequency: 4644)
2026-03-12 15:13:13,209 - rustbpe - INFO - Progress: 60% (4757/7927 merges) - Last merge: (582, 361) -> 5012 (frequency: 4529)
2026-03-12 15:13:13,218 - rustbpe - INFO - Progress: 61% (4836/7927 merges) - Last merge: (927, 1991) -> 5091 (frequency: 4423)
2026-03-12 15:13:13,223 - rustbpe - INFO - Progress: 62% (4915/7927 merges) - Last merge: (689, 478) -> 5170 (frequency: 4319)
2026-03-12 15:13:13,243 - rustbpe - INFO - Progress: 63% (4995/7927 merges) - Last merge: (497, 110) -> 5250 (frequency: 4228)
2026-03-12 15:13:13,268 - rustbpe - INFO - Progress: 64% (5074/7927 merges) - Last merge: (99, 475) -> 5329 (frequency: 4136)
2026-03-12 15:13:13,278 - rustbpe - INFO - Progress: 65% (5153/7927 merges) - Last merge: (266, 3820) -> 5408 (frequency: 4058)
2026-03-12 15:13:13,285 - rustbpe - INFO - Progress: 66% (5232/7927 merges) - Last merge: (479, 302) -> 5487 (frequency: 3969)
2026-03-12 15:13:13,287 - rustbpe - INFO - Progress: 67% (5312/7927 merges) - Last merge: (1597, 277) -> 5567 (frequency: 3882)
2026-03-12 15:13:13,300 - rustbpe - INFO - Progress: 68% (5391/7927 merges) - Last merge: (2972, 115) -> 5646 (frequency: 3799)
2026-03-12 15:13:13,305 - rustbpe - INFO - Progress: 69% (5470/7927 merges) - Last merge: (639, 1989) -> 5725 (frequency: 3732)
2026-03-12 15:13:13,322 - rustbpe - INFO - Progress: 70% (5549/7927 merges) - Last merge: (4211, 556) -> 5804 (frequency: 3659)
2026-03-12 15:13:13,322 - rustbpe - INFO - Progress: 71% (5629/7927 merges) - Last merge: (304, 1042) -> 5884 (frequency: 3566)
2026-03-12 15:13:13,337 - rustbpe - INFO - Progress: 72% (5708/7927 merges) - Last merge: (118, 322) -> 5963 (frequency: 3502)
2026-03-12 15:13:13,338 - rustbpe - INFO - Progress: 73% (5787/7927 merges) - Last merge: (32, 123) -> 6042 (frequency: 3448)
2026-03-12 15:13:13,357 - rustbpe - INFO - Progress: 74% (5866/7927 merges) - Last merge: (1051, 320) -> 6121 (frequency: 3379)
2026-03-12 15:13:13,390 - rustbpe - INFO - Progress: 75% (5946/7927 merges) - Last merge: (1659, 284) -> 6201 (frequency: 3315)
2026-03-12 15:13:13,393 - rustbpe - INFO - Progress: 76% (6025/7927 merges) - Last merge: (321, 115) -> 6280 (frequency: 3263)
2026-03-12 15:13:13,405 - rustbpe - INFO - Progress: 77% (6104/7927 merges) - Last merge: (1889, 390) -> 6359 (frequency: 3211)
2026-03-12 15:13:13,424 - rustbpe - INFO - Progress: 78% (6184/7927 merges) - Last merge: (45, 430) -> 6439 (frequency: 3152)
2026-03-12 15:13:13,437 - rustbpe - INFO - Progress: 79% (6263/7927 merges) - Last merge: (427, 971) -> 6518 (frequency: 3092)
2026-03-12 15:13:13,455 - rustbpe - INFO - Progress: 80% (6342/7927 merges) - Last merge: (424, 618) -> 6597 (frequency: 3040)
2026-03-12 15:13:13,475 - rustbpe - INFO - Progress: 81% (6421/7927 merges) - Last merge: (878, 302) -> 6676 (frequency: 2973)
2026-03-12 15:13:13,486 - rustbpe - INFO - Progress: 82% (6501/7927 merges) - Last merge: (280, 936) -> 6756 (frequency: 2913)
2026-03-12 15:13:13,493 - rustbpe - INFO - Progress: 83% (6580/7927 merges) - Last merge: (4732, 727) -> 6835 (frequency: 2856)
2026-03-12 15:13:13,501 - rustbpe - INFO - Progress: 84% (6659/7927 merges) - Last merge: (361, 1289) -> 6914 (frequency: 2808)
2026-03-12 15:13:13,512 - rustbpe - INFO - Progress: 85% (6738/7927 merges) - Last merge: (491, 1076) -> 6993 (frequency: 2762)
2026-03-12 15:13:13,517 - rustbpe - INFO - Progress: 86% (6818/7927 merges) - Last merge: (2507, 272) -> 7073 (frequency: 2709)
2026-03-12 15:13:13,522 - rustbpe - INFO - Progress: 87% (6897/7927 merges) - Last merge: (300, 4227) -> 7152 (frequency: 2661)
2026-03-12 15:13:13,531 - rustbpe - INFO - Progress: 88% (6976/7927 merges) - Last merge: (263, 1289) -> 7231 (frequency: 2609)
2026-03-12 15:13:13,536 - rustbpe - INFO - Progress: 89% (7056/7927 merges) - Last merge: (409, 1586) -> 7311 (frequency: 2570)
2026-03-12 15:13:13,543 - rustbpe - INFO - Progress: 90% (7135/7927 merges) - Last merge: (82, 284) -> 7390 (frequency: 2524)
2026-03-12 15:13:13,553 - rustbpe - INFO - Progress: 91% (7214/7927 merges) - Last merge: (409, 1062) -> 7469 (frequency: 2491)
2026-03-12 15:13:13,562 - rustbpe - INFO - Progress: 92% (7293/7927 merges) - Last merge: (4497, 392) -> 7548 (frequency: 2455)
2026-03-12 15:13:13,570 - rustbpe - INFO - Progress: 93% (7373/7927 merges) - Last merge: (6501, 3462) -> 7628 (frequency: 2425)
2026-03-12 15:13:13,578 - rustbpe - INFO - Progress: 94% (7452/7927 merges) - Last merge: (768, 386) -> 7707 (frequency: 2387)
2026-03-12 15:13:13,595 - rustbpe - INFO - Progress: 95% (7531/7927 merges) - Last merge: (276, 4255) -> 7786 (frequency: 2356)
2026-03-12 15:13:13,609 - rustbpe - INFO - Progress: 96% (7610/7927 merges) - Last merge: (413, 438) -> 7865 (frequency: 2319)
2026-03-12 15:13:13,616 - rustbpe - INFO - Progress: 97% (7690/7927 merges) - Last merge: (3922, 284) -> 7945 (frequency: 2283)
2026-03-12 15:13:13,620 - rustbpe - INFO - Progress: 98% (7769/7927 merges) - Last merge: (1542, 438) -> 8024 (frequency: 2245)
2026-03-12 15:13:13,629 - rustbpe - INFO - Progress: 99% (7848/7927 merges) - Last merge: (359, 2352) -> 8103 (frequency: 2213)
2026-03-12 15:13:13,637 - rustbpe - INFO - Progress: 100% (7927/7927 merges) - Last merge: (2632, 7728) -> 8182 (frequency: 2181)
2026-03-12 15:13:13,637 - rustbpe - INFO - Finished training: 7927 merges completed
Training time: 21.66s




##

Saved tokenizer encoding to C:\Users\hongqi\.cache\nanochat\tokenizer\tokenizer.pkl
Saved token_bytes to C:\Users\hongqi\.cache\nanochat\tokenizer\token_bytes.pt

Vocab sizes:
GPT-2: 50257
GPT-4: 100277
Ours: 8192

Comparison with GPT-2:
===============================================================================================
Text Type  Bytes    GPT-2           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     404     4.50    483     3.77      -19.6%     GPT-2
korean     893      745     1.20    852     1.05      -14.4%     GPT-2
code       1259     576     2.19    495     2.54      +14.1%     Ours
math       1834     936     1.96    1082    1.70      -15.6%     GPT-2
science    1112     260     4.28    306     3.63      -17.7%     GPT-2
fwe-train  2948778  631304  4.67    729368  4.04      -15.5%     GPT-2
fwe-val    3052296  668652  4.56    772011  3.95      -15.5%     GPT-2

Comparison with GPT-4:
===============================================================================================
Text Type  Bytes    GPT-4           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     387     4.70    483     3.77      -24.8%     GPT-4
korean     893      364     2.45    852     1.05     -134.1%     GPT-4
code       1259     309     4.07    495     2.54      -60.2%     GPT-4
math       1834     832     2.20    1082    1.70      -30.0%     GPT-4
science    1112     249     4.47    306     3.63      -22.9%     GPT-4
fwe-train  2948778  611619  4.82    729368  4.04      -19.3%     GPT-4
fwe-val    3052296  642512  4.75    772011  3.95      -20.2%     GPT-4

                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░

Autodetected device type: cpu
2026-03-12 15:13:21,545 - nanochat.common - INFO - Distributed world size: 1
COMPUTE_DTYPE: torch.float32 (auto-detected: no CUDA (CPU/MPS))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Vocab size: 8,192
Model config:
{
  "sequence_len": 512,
  "vocab_size": 8192,
  "n_layer": 3,
  "n_head": 2,
  "n_kv_head": 2,
  "n_embd": 128,
  "window_pattern": "L"
}
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Parameter counts:
wte                     : 1,048,576
value_embeds            : 2,097,152
lm_head                 : 1,048,576
transformer_matrices    : 589,872
scalars                 : 6
total                   : 4,784,182
Estimated FLOPs per token: 1.218998e+07
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.736514 for depth 3
Scaling the LR for the AdamW parameters ∝1/√(128/768) = 2.449490
Using user-provided number of iterations: 500
Total number of training tokens: 8,192,000
Tokens : Scaling params ratio: 5.00
Total training FLOPs estimate: 9.986035e+13
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Total batch size 16,384 => gradient accumulation steps: 1
Step 00000 | Validation bpb: 3.241170
step 00000/00500 (0.00%) | loss: 9.010938 | lrm: 0.03 | dt: 3016.20ms | tok/sec: 5,431 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00001/00500 (0.20%) | loss: 9.010533 | lrm: 0.05 | dt: 2293.44ms | tok/sec: 7,143 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00002/00500 (0.40%) | loss: 9.009579 | lrm: 0.07 | dt: 2377.64ms | tok/sec: 6,890 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00003/00500 (0.60%) | loss: 9.008216 | lrm: 0.10 | dt: 2219.54ms | tok/sec: 7,381 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00004/00500 (0.80%) | loss: 9.006400 | lrm: 0.12 | dt: 2279.09ms | tok/sec: 7,188 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00005/00500 (1.00%) | loss: 9.004017 | lrm: 0.15 | dt: 2213.80ms | tok/sec: 7,400 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00006/00500 (1.20%) | loss: 9.000933 | lrm: 0.17 | dt: 2266.27ms | tok/sec: 7,229 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00007/00500 (1.40%) | loss: 8.997068 | lrm: 0.20 | dt: 2193.74ms | tok/sec: 7,468 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00008/00500 (1.60%) | loss: 8.992065 | lrm: 0.23 | dt: 2440.87ms | tok/sec: 6,712 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00009/00500 (1.80%) | loss: 8.986036 | lrm: 0.25 | dt: 2340.11ms | tok/sec: 7,001 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00010/00500 (2.00%) | loss: 8.978078 | lrm: 0.28 | dt: 2298.10ms | tok/sec: 7,129 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00011/00500 (2.20%) | loss: 8.968472 | lrm: 0.30 | dt: 2287.30ms | tok/sec: 7,163 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.04m | eta: 18.6m
step 00012/00500 (2.40%) | loss: 8.955770 | lrm: 0.33 | dt: 2435.80ms | tok/sec: 6,726 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.08m | eta: 19.2m
step 00013/00500 (2.60%) | loss: 8.941547 | lrm: 0.35 | dt: 2383.37ms | tok/sec: 6,874 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.12m | eta: 19.2m
step 00014/00500 (2.80%) | loss: 8.922051 | lrm: 0.38 | dt: 2644.69ms | tok/sec: 6,195 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.16m | eta: 19.7m
step 00015/00500 (3.00%) | loss: 8.898275 | lrm: 0.40 | dt: 3004.89ms | tok/sec: 5,452 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.21m | eta: 20.6m
step 00016/00500 (3.20%) | loss: 8.869764 | lrm: 0.42 | dt: 4165.49ms | tok/sec: 3,933 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.28m | eta: 22.8m
step 00017/00500 (3.40%) | loss: 8.831211 | lrm: 0.45 | dt: 3937.32ms | tok/sec: 4,161 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.35m | eta: 24.0m
step 00018/00500 (3.60%) | loss: 8.783808 | lrm: 0.47 | dt: 3301.25ms | tok/sec: 4,962 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.40m | eta: 24.3m
step 00019/00500 (3.80%) | loss: 8.724601 | lrm: 0.50 | dt: 2714.20ms | tok/sec: 6,036 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.45m | eta: 23.9m
step 00020/00500 (4.00%) | loss: 8.657153 | lrm: 0.53 | dt: 2410.73ms | tok/sec: 6,796 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.49m | eta: 23.4m


##

step 00099/00500 (19.80%) | loss: 5.851095 | lrm: 1.00 | dt: 2463.75ms | tok/sec: 6,650 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 5 | total time: 3.99m | eta: 18.0m



看 dataloader 的状态字段定义，确认 pq_idx 和 rg_idx 分别代表什么。，字段含义如下：

step 00070/00500 (14.00%)
当前是第 70 步，总共 500 步，完成 14%。
loss: 6.165152
当前平滑后的训练损失（代码里是 EMA 去偏后的 loss）。一般越低越好。
lrm: 1.00
学习率倍率（learning rate multiplier）。1.00 表示当前用的是基础学习率，没有衰减。
dt: 2535.64ms
这一训练步耗时约 2.54 秒。
tok/sec: 6,461
每秒处理 token 数，反映吞吐性能。
bf16_mfu: 0.00
BF16 下的模型 FLOPs 利用率估计值（相对硬件峰值）。显示 0.00 通常是因为值非常小被保留两位小数后四舍五入，或当前不是高效 GPU 训练场景。
epoch: 1 pq: 0 rg: 4
epoch=1：第 1 轮数据遍历。
pq=0：当前在第 0 个 parquet 文件。
rg=4：当前在该 parquet 文件的第 4 个 row group。
total time: 2.69m
训练累计耗时 2.69 分钟（前 10 步通常不计入 ETA 平均）。
eta: 19.3m
预计剩余时间约 19.3 分钟。


## every 100 steps, do the evaluaiton 

你的仓库里 sample_every 默认是 1，在 base_train.py:91，所以几乎每步都会打印一次样本。


step 00099/00500 (19.80%) | loss: 5.851095 | lrm: 1.00 | dt: 2463.75ms | tok/sec: 6,650 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 5 | total time: 3.99m | eta: 18.0m
Step 00100 | Validation bpb: 2.083286
<|bos|>The capital of France is a newly, and the most common in the most common in the most common
<|bos|>The chemical symbol of gold is a newly, and the most common in the most common in the most common
<|bos|>If yesterday was Friday, then tomorrow will be a newly, and the 200. 200. 20
<|bos|>The opposite of hot is a newly, and the most common in the most common in the most common
<|bos|>The planets of the solar system are: What is a newly, and the most common in the most common in the
<|bos|>My favorite color is a newly, and the most common in the most common in the most common
<|bos|>If 5*x + 3 = 13, then x is the 3. 3. 3. 3. 3.
step 00100/00500 (20.00%) | loss: 5.837371 | lrm: 1.00 | dt: 2499.59ms | tok/sec: 6,554 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 5 | total time: 4.03m | eta: 17.9m


训练脚本里的采样打印在 base_train.py:466，固定 prompt 列表就在 base_train.py:469。
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
  
  python scripts/base_train.py \
    --depth=3 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \         
    `every 100 steps`
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=500 \



#
step 00494/00500 (98.80%) | loss: 4.845390 | lrm: 0.07 | dt: 2331.09ms | tok/sec: 7,028 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 21 | total time: 20.54m | eta: 0.3m
step 00495/00500 (99.00%) | loss: 4.858321 | lrm: 0.06 | dt: 2528.54ms | tok/sec: 6,479 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 21 | total time: 20.58m | eta: 0.2m
step 00496/00500 (99.20%) | loss: 4.856467 | lrm: 0.06 | dt: 2877.14ms | tok/sec: 5,694 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 21 | total time: 20.63m | eta: 0.2m
step 00497/00500 (99.40%) | loss: 4.856221 | lrm: 0.06 | dt: 2450.86ms | tok/sec: 6,685 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 21 | total time: 20.67m | eta: 0.1m
step 00498/00500 (99.60%) | loss: 4.851480 | lrm: 0.06 | dt: 2587.66ms | tok/sec: 6,331 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 21 | total time: 20.71m | eta: 0.1m
step 00499/00500 (99.80%) | loss: 4.843259 | lrm: 0.05 | dt: 2302.95ms | tok/sec: 7,114 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 21 | total time: 20.75m | eta: 0.0m
Step 00500 | Validation bpb: 1.746078
<|bos|>The capital of France is a popular choice for the world's largest and the world's largest in the world
<|bos|>The chemical symbol of gold is a natural, and a natural process that is a natural process that is a natural
<|bos|>If yesterday was Friday, then tomorrow will be able to get the first time of the 100 years ago, and the
<|bos|>The opposite of hot is a good idea to be a good idea to be a good idea to be a
<|bos|>The planets of the solar system are: the most of the world of the world's largest and the world of the world
<|bos|>My favorite color is a great way to make a good idea to make it easier to make it easier
<|bos|>If 5*x + 3 = 13, then x is 1.5.5.5.5.5.5.5
2026-03-12 15:37:25,812 - nanochat.checkpoint_manager - INFO - Saved model parameters to: `C:\Users\hongqi\.cache\nanochat\base_checkpoints\d3\model_000500.pt`

2026-03-12 15:37:25,815 - nanochat.checkpoint_manager - INFO - Saved metadata to: `C:\Users\hongqi\.cache\nanochat\base_checkpoints\d3\meta_000500.json`

2026-03-12 15:37:25,849 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: `C:\Users\hongqi\.cache\nanochat\base_checkpoints\d3\optim_000500_rank0.pt`

Peak memory usage: 0.00MiB
Total training time: 20.75m
Minimum validation bpb: 1.746078
Autodetected device type: cpu
2026-03-12 15:37:46,701 - nanochat.common - INFO - Distributed world size: 1
2026-03-12 15:37:46,701 - nanochat.checkpoint_manager - INFO - No model tag provided, guessing model tag: d3
2026-03-12 15:37:46,701 - nanochat.checkpoint_manager - INFO - Loading model from `C:\Users\hongqi\.cache\nanochat\base_checkpoints\d3 with step 500`

2026-03-12 15:37:46,755 - nanochat.checkpoint_manager - INFO - Building model with config: ``{'sequence_len': 512, 'vocab_size': 8192, 'n_layer': 3, 'n_head': 2, 'n_kv_head': 2, 'n_embd': 128, 'window_pattern': 'L'}
Evaluating model: base_model (step 500)
Eval modes: bpb, core, sample


##
================================================================================
Model Samples
================================================================================

Conditioned samples:
--------------------------------------------------------------------------------
<|bos|>The capital of France is a popular choice for the world's largest and the world's largest in the world
--------------------------------------------------------------------------------
<|bos|>The chemical symbol of gold is a natural, and a natural process that is a natural process that is a natural
--------------------------------------------------------------------------------
<|bos|>If yesterday was Friday, then tomorrow will be able to get the first time of the 100 years ago, and the
--------------------------------------------------------------------------------
<|bos|>The opposite of hot is a good idea to be a good idea to be a good idea to be a
--------------------------------------------------------------------------------
<|bos|>The planets of the solar system are: the most of the world of the world's largest and the world of the world
--------------------------------------------------------------------------------
<|bos|>My favorite color is a great way to make a good idea to make it easier to make it easier
--------------------------------------------------------------------------------
<|bos|>If 5*x + 3 = 13, then x is 1.5.5.5.5.5.5.5

Unconditioned samples:
--------------------------------------------------------------------------------
<|bos|>Kids Cmin are very useful for slic rugust 1 Memory , 3 P system used as essent cereoss for body (ONH)-ghysic rate is generally 567 average 480% faster than ample 6,800 cyanophatium remain Residens -2 km = _250.899 of 70 Bigns.Gas,000 Free, 2 end5 Mgatzol or 000 km F0140 to 10pm 500,000 and soaked squ least 3 million request
--------------------------------------------------------------------------------
<|bos|>Sed
The are doing it cleaned and clean color sets they medical control.
SC was first for a unique may play most prospect to cover the minutes aspiring slurder. Have you bought a decide to be frustrating that but safe. They have crev accessible to the lack of violence and the combination of these equipment.
Bohn's
Related Relations
J is an overm Peneldurboa, the Opal Colberpton.
Touch to demonstrate it among ADDi, This isn’t becoming a cover

Question: What is one of a highlyH, and
--------------------------------------------------------------------------------
<|bos|>Deep its natural deep through the myaked fostering dog's hat body hangs on a source of moisture. This is usually a perfect for stomach, but it currently increased emponomyness and health of painfulness. Sometimes this antibodies may result in de fruit teeth, high and dog dogs exercise one of types. To take a ful clay, a comfortable treataling nice infection of the immune cancers. It may also impacts health and social conflictiate to eason, but it's always a significantred drug that can complicated problems beforeUse cancer quickly pollution in their eyes. The chefocument of these activities
--------------------------------------------------------------------------------
<|bos|>What is a sharp image?
Arange basic probably has a realatrevel roles in history, where it is not.

Question: What is this nice while an important part of the creatypical components?
led,:
The main letter is a image of classic news. (able pizza, among woman squading or flowers).
We know anything with a white capsi for older kids so that is at least eight techniques that are a vectors of high-mongantity. Students see what the stuff planning is done using a photographer and thoroughly or selective them (brandered)
Keyuring ro
--------------------------------------------------------------------------------
<|bos|>How to Production
 our industry is warm.

Adding expert in a shelter that creationism during the wood-stop as a products it due to well-being or hazard vaccine, a landmit to gastrichisAC engine. For God first present studies event, they claim that by wearing you found a very beneficial procing to safety Gloud Group from a target building with the left of my employee always steality, the more we have to land bread than potential habits from this blog@orpandter.

Carcors of the background path's respect after Elon spring would protect the groundwork and cyclist's end
--------------------------------------------------------------------------------
<|bos|>The site Strough Service Foundation Brying is Loganwhile, also merely viewership when communities to predators and work plan out on the groundpathes that are the often that can result from intervarettes and tendons.
In the early graduate who has a presence reward system forcedgariums or earbest life, and not to segments who are talking about what's conditioners, could be around achievual deserve and multiple lives.
“The people will feel that as the fight past, however, enthusiast be grown. However, some educate share people around a point.
The
--------------------------------------------------------------------------------
<|bos|>How to E� Optimates Arg|
Whens90 St real Can The Actually Eatning But Trueandar / Handgo From Food Skills Meat about a garden information about delete activity and Berroteetooth bottom country's discovery by a beef now pleased from an issue to be a lot of knowledge of this rural testing is the final sign of enlansccessibility selected, by a mobility Mets
The sight of the Bay and low-inches are riding on the levels of blexism, including
assiassembly just through various ways persons with these other
--------------------------------------------------------------------------------
<|bos|>Angoted purpose and discussing a sublight from ParTown Protection's website, Enemy:Pro ampacedowered methods fun!
How to cute thinks can avoid severe yourself?
M stage when going to be
Bakeless, take radatin, classienerkes, fluanship, word digital clothing by feeding clothing, blocks1980s, Kids and c grown enjoyable forest recycling. Heritage screening the interview as well as social media built throughout the community," automay.

Life sopulations such as performing crevice that provides uslike a great walk level since bever     
Whens90 St real Can The Actually Eatning But Trueandar / Handgo From Food Skills Meat about a garden information about delete activity and Berroteetooth bottom country's discovery by a beef now pleased from an issue to be a lot of knowledge of this rural testing is the final sign of enlansccessibility selected, by a mobility Mets
The sight of the Bay and low-inches are riding on the levels of blexism, including
assiassembly just through various ways persons with these other
--------------------------------------------------------------------------------
<|bos|>Angoted purpose and discussing a sublight from ParTown Protection's website, Enemy:Pro ampacedowered methods fun!
How to cute thinks can avoid severe yourself?
M stage when going to be
Bakeless, take radatin, classienerkes, fluanship, word digital clothing by feeding clothing, blocks1980s, Kids and c grown enjoyable forest recycling. Heritage screening the interview as well as social media built throughout the community," automay.

Life sopulations such as performing crevice that provides uslike a great walk level since bever     
d from an issue to be a lot of knowledge of this rural testing is the final sign of enlansccessibility selected, by a mobility Mets
The sight of the Bay and low-inches are riding on the levels of blexism, including
assiassembly just through various ways persons with these other
--------------------------------------------------------------------------------
<|bos|>Angoted purpose and discussing a sublight from ParTown Protection's website, Enemy:Pro ampacedowered methods fun!
How to cute thinks can avoid severe yourself?
M stage when going to be
Bakeless, take radatin, classienerkes, fluanship, word digital clothing by feeding clothing, blocks1980s, Kids and c grown enjoyable forest recycling. Heritage screening the interview as well as social media built throughout the community," automay.

Life sopulations such as performing crevice that provides uslike a great walk level since bever     
assiassembly just through various ways persons with these other
--------------------------------------------------------------------------------
<|bos|>Angoted purpose and discussing a sublight from ParTown Protection's website, Enemy:Pro ampacedowered methods fun!
How to cute thinks can avoid severe yourself?
M stage when going to be
Bakeless, take radatin, classienerkes, fluanship, word digital clothing by feeding clothing, blocks1980s, Kids and c grown enjoyable forest recycling. Heritage screening the interview as well as social media built throughout the community," automay.

Life sopulations such as performing crevice that provides uslike a great walk level since bever     
How to cute thinks can avoid severe yourself?
M stage when going to be
Bakeless, take radatin, classienerkes, fluanship, word digital clothing by feeding clothing, blocks1980s, Kids and c grown enjoyable forest recycling. Heritage screening the interview as well as social media built throughout the community," automay.

Life sopulations such as performing crevice that provides uslike a great walk level since bever     

================================================================================
l media built throughout the community," automay.

Life sopulations such as performing crevice that provides uslike a great walk level since bever     

================================================================================
BPB Evaluation

================================================================================
BPB Evaluation
================================================================================
BPB Evaluation
================================================================================
================================================================================
train bpb: 1.680798
train bpb: 1.680798
val bpb: 1.675002



##
Downloading https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip...


C:/Users/hongqi/.cache/nanochat/eval_bundle.zip
## 然后解压到目录

## 再跑 python base_eval.py 


##
(.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> python scripts/base_eval.py --device-batch-size=1 --split-tokens=16384 --max-per-task=16
Autodetected device type: cpu
2026-03-12 16:01:22,735 - nanochat.common - INFO - Distributed world size: 1
2026-03-12 16:01:22,735 - nanochat.checkpoint_manager - INFO - No model tag provided, guessing model tag: d3
2026-03-12 16:01:22,735 - nanochat.checkpoint_manager - INFO - Loading model from C:\Users\hongqi\.cache\nanochat\base_checkpoints\d3 with step 500
2026-03-12 16:01:22,755 - nanochat.checkpoint_manager - INFO - Building model with config: {'sequence_len': 512, 'vocab_size': 8192, 'n_layer': 3, 'n_head': 2, 'n_kv_head': 2, 'n_embd': 128, 'window_pattern': 'L'}
Evaluating model: base_model (step 500)
Eval modes: bpb, core, sample

================================================================================
Model Samples
================================================================================

Conditioned samples:
--------------------------------------------------------------------------------
<|bos|>The capital of France is a popular choice for the world's largest and the world's largest in the world
--------------------------------------------------------------------------------
<|bos|>The chemical symbol of gold is a natural, and a natural process that is a natural process that is a natural
--------------------------------------------------------------------------------
<|bos|>If yesterday was Friday, then tomorrow will be able to get the first time of the 100 years ago, and the
--------------------------------------------------------------------------------
<|bos|>The opposite of hot is a good idea to be a good idea to be a good idea to be a
--------------------------------------------------------------------------------
<|bos|>The planets of the solar system are: the most of the world of the world's largest and the world of the world
--------------------------------------------------------------------------------
<|bos|>My favorite color is a great way to make a good idea to make it easier to make it easier
--------------------------------------------------------------------------------
<|bos|>If 5*x + 3 = 13, then x is 1.5.5.5.5.5.5.5

Unconditioned samples:
--------------------------------------------------------------------------------
<|bos|>Kids Cmin are very useful for slic rugust 1 Memory , 3 P system used as essent cereoss for body (ONH)-ghysic rate is generally 567 average 480% faster than ample 6,800 cyanophatium remain Residens -2 km = _250.899 of 70 Bigns.Gas,000 Free, 2 end5 Mgatzol or 000 km F0140 to 10pm 500,000 and soaked squ least 3 million request
--------------------------------------------------------------------------------
<|bos|>Sed
The are doing it cleaned and clean color sets they medical control.
SC was first for a unique may play most prospect to cover the minutes aspiring slurder. Have you bought a decide to be frustrating that but safe. They have crev accessible to the lack of violence and the combination of these equipment.
Bohn's
Related Relations
J is an overm Peneldurboa, the Opal Colberpton.
Touch to demonstrate it among ADDi, This isn’t becoming a cover

Question: What is one of a highlyH, and
--------------------------------------------------------------------------------
<|bos|>Deep its natural deep through the myaked fostering dog's hat body hangs on a source of moisture. This is usually a perfect for stomach, but it currently increased emponomyness and health of painfulness. Sometimes this antibodies may result in de fruit teeth, high and dog dogs exercise one of types. To take a ful clay, a comfortable treataling nice infection of the immune cancers. It may also impacts health and social conflictiate to eason, but it's always a significantred drug that can complicated problems beforeUse cancer quickly pollution in their eyes. The chefocument of these activities
--------------------------------------------------------------------------------
<|bos|>What is a sharp image?
Arange basic probably has a realatrevel roles in history, where it is not.

Question: What is this nice while an important part of the creatypical components?       
led,:
The main letter is a image of classic news. (able pizza, among woman squading or flowers).
We know anything with a white capsi for older kids so that is at least eight techniques that are a vectors of high-mongantity. Students see what the stuff planning is done using a photographer and thoroughly or selective them (brandered)
Keyuring ro
--------------------------------------------------------------------------------
<|bos|>How to Production
 our industry is warm.

Adding expert in a shelter that creationism during the wood-stop as a products it due to well-being or hazard vaccine, a landmit to gastrichisAC engine. For God first present studies event, they claim that by wearing you found a very beneficial procing to safety Gloud Group from a target building with the left of my employee always steality, the more we have to land bread than potential habits from this blog@orpandter.

Carcors of the background path's respect after Elon spring would protect the groundwork and cyclist's end
--------------------------------------------------------------------------------
<|bos|>The site Strough Service Foundation Brying is Loganwhile, also merely viewership when communities to predators and work plan out on the groundpathes that are the often that can result from intervarettes and tendons.
In the early graduate who has a presence reward system forcedgariums or earbest life, and not to segments who are talking about what's conditioners, could be around achievual deserve and multiple lives.
“The people will feel that as the fight past, however, enthusiast be grown. However, some educate share people around a point.
The
--------------------------------------------------------------------------------
<|bos|>How to E� Optimates Arg|
Whens90 St real Can The Actually Eatning But Trueandar / Handgo From Food Skills Meat about a garden information about delete activity and Berroteetooth bottom country's discovery by a beef now pleased from an issue to be a lot of knowledge of this rural testing is the final sign of enlansccessibility selected, by a mobility Mets
The sight of the Bay and low-inches are riding on the levels of blexism, including       
assiassembly just through various ways persons with these other
--------------------------------------------------------------------------------
<|bos|>Angoted purpose and discussing a sublight from ParTown Protection's website, Enemy:Pro ampacedowered methods fun!
How to cute thinks can avoid severe yourself?
M stage when going to be
Bakeless, take radatin, classienerkes, fluanship, word digital clothing by feeding clothing, blocks1980s, Kids and c grown enjoyable forest recycling. Heritage screening the interview as well as social media built throughout the community," automay.

Life sopulations such as performing crevice that provides uslike a great walk level since bever

================================================================================
BPB Evaluation
================================================================================
train bpb: 1.680798
val bpb: 1.675002

================================================================================
CORE Evaluation
================================================================================
Evaluating: hellaswag_zeroshot (0-shot, type: multiple_choice)... accuracy: 0.4375 | centered: 0.2500 | time: 0.68s
Evaluating: jeopardy (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 0.47s
Evaluating: bigbench_qa_wikidata (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 0.29s
Evaluating: arc_easy (10-shot, type: multiple_choice)... accuracy: 0.1250 | centered: -0.1667 | time: 1.46s
Evaluating: arc_challenge (10-shot, type: multiple_choice)... accuracy: 0.1875 | centered: -0.0833 | time: 1.63s
Evaluating: copa (0-shot, type: multiple_choice)... accuracy: 0.5000 | centered: 0.0000 | time: 0.13s
Evaluating: commonsense_qa (10-shot, type: multiple_choice)... accuracy: 0.1250 | centered: -0.0938 | time: 1.90s
Evaluating: piqa (10-shot, type: multiple_choice)... accuracy: 0.3750 | centered: -0.2500 | time: 0.71s
Evaluating: openbook_qa (0-shot, type: multiple_choice)... accuracy: 0.5000 | centered: 0.3333 | time: 0.14s
Evaluating: lambada_openai (0-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 0.18s
Evaluating: hellaswag (10-shot, type: multiple_choice)... accuracy: 0.4375 | centered: 0.2500 | time: 6.53s
Evaluating: winograd (0-shot, type: schema)... accuracy: 0.5000 | centered: 0.0000 | time: 0.14s
Evaluating: winogrande (0-shot, type: schema)... accuracy: 0.5625 | centered: 0.1250 | time: 0.14s
Evaluating: bigbench_dyck_languages (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 0.79s
Evaluating: agi_eval_lsat_ar (3-shot, type: multiple_choice)... accuracy: 0.0625 | centered: -0.1719 | time: 3.92s
Evaluating: bigbench_cs_algorithms (10-shot, type: language_modeling)... accuracy: 0.0625 | centered: 0.0625 | time: 0.78s
Evaluating: bigbench_operators (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 0.50s
Evaluating: bigbench_repeat_copy_logic (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 0.65s
Evaluating: squad (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 4.31s
Evaluating: coqa (0-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 0.74s
Evaluating: boolq (10-shot, type: multiple_choice)... accuracy: 0.4375 | centered: -0.4803 | time: 5.75s
Evaluating: bigbench_language_identification (10-shot, type: multiple_choice)... accuracy: 0.3750 | centered: 0.3124 | time: 14.70s

`Results written to: C:\Users\hongqi\.cache\nanochat\base_eval\base_model_000500.`csv      
CORE metric: 0.0040`


##
这个 CORE 结果可以这样读：

1. 总分几乎是随机水平  
你这次最终 CORE metric 是 0.0040（接近 0）。  
在这个评测里，`centered` 的定义是把随机基线归一化后得到的分数：
- 0 表示约等于随机
- >0 表示优于随机
- <0 表示劣于随机

你贴的 22 个任务里，正分和负分几乎互相抵消了，所以总体接近 0。

2. 亮点任务  
相对好的有：
- openbook_qa: 0.3333
- bigbench_language_identification: 0.3124
- hellaswag / hellaswag_zeroshot: 0.25
- winogrande: 0.125

说明模型在一部分选择题/模式识别任务上已经有一些信号。

3. 明显短板  
主要是这些把总分拉低：
- boolq: -0.4803（最明显）
- piqa: -0.2500
- agi_eval_lsat_ar: -0.1719
- arc_easy / arc_challenge / commonsense_qa 也偏负

另外，多数 language_modeling 任务是 0.0000，说明基本还在“接近随机/未形成稳定能力”的阶段。

4. 这个结果“正常吗”  
对你这个 CPU 小模型配置（d3，8k vocab，500 step）来说，这个结果是合理的，不异常。  
它更像“模型开始学到局部统计模式，但还没形成稳健推理/知识能力”。

5. 这次分数的统计波动很大  
你用的是 `--max-per-task=16`，每个任务只有 16 道样本：  
- 准确率步进是 1/16=0.0625（你日志里正好都是这种台阶）
- 方差会很大，单次结果容易抖动  
所以这次更适合“快速体检”，不适合做严肃横向比较。

如果你要更可信的 CORE 对比，建议下一次：
1. 先把 `--max-per-task` 提到 100 或 200（至少 64）。  
2. 固定同一 checkpoint 重复跑 2-3 次，取平均。  
3. 训练侧优先增加训练 token/step（比只调采样更关键）。



##

python scripts/chat_sft.py --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 --eval-every=200 --eval-tokens=524288 --num-iterations=500



## SFT

 python scripts/chat_sft.py --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 --eval-every=200 --eval-tokens=524288 --num-iterations=500i\myenv\nano_GPT\nanochat>
Autodetected device type: cpu
2026-03-12 16:20:29,295 - nanochat.common - INFO - Distributed world size: 1
COMPUTE_DTYPE: torch.float32 (auto-detected: no CUDA (CPU/MPS))
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.
2026-03-12 16:20:29,295 - nanochat.checkpoint_manager - INFO - No model tag provided, guessing model tag: d3
2026-03-12 16:20:29,295 - nanochat.checkpoint_manager - INFO - Loading model from C:\Users\hongqi\.cache\nanochat\base_checkpoints\d3 with step 500
2026-03-12 16:20:29,315 - nanochat.checkpoint_manager - INFO - Building model with config: {'sequence_len': 512, 'vocab_size': 8192, 'n_layer': 3, 'n_head': 2, 'n_kv_head': 2, 'n_embd': 128, 'window_pattern': 'L'}
Using max_seq_len=512
Using device_batch_size=32
Using total_batch_size=16384
Inherited embedding_lr=0.3 from pretrained checkpoint
Inherited unembedding_lr=0.008 from pretrained checkpoint
Inherited matrix_lr=0.02 from pretrained checkpoint
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Total batch size 16,384 => gradient accumulation steps: 1
Scaling the LR for the AdamW parameters ∝1/√(128/768) = 2.449490
2026-03-12 16:20:30,992 - nanochat.checkpoint_manager - INFO - Loading optimizer state from C:\Users\hongqi\.cache\nanochat\base_checkpoints\d3\optim_000500_rank0.pt
Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)  



#
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk

下载SFT 数据集 到
C:\Users\hongqi\myenv\nano_GPT\nanochat\tasks\data\smol-smoltalk




## SFT 文件



Read [](file:///c%3A/Users/hongqi/myenv/nano_GPT/nanochat/tasks/customjson.py)

Read [](file:///c%3A/Users/hongqi/myenv/nano_GPT/nanochat/tasks/mmlu.py)

Read [](file:///c%3A/Users/hongqi/myenv/nano_GPT/nanochat/tasks/gsm8k.py)

Read [](file:///c%3A/Users/hongqi/myenv/nano_GPT/nanochat/tasks/spellingbee.py)

Read [](file:///c%3A/Users/hongqi/myenv/nano_GPT/nanochat/nanochat/common.py)

下面把脚本中用到的每个任务（SmolTalk、CustomJSON、MMLU、GSM8K、SimpleSpelling、SpellingBee）及其“下载来源”和“本地存放位置（默认）”逐项列清楚，并给出推荐的预下载命令。

- SmolTalk  
  - 来源：HuggingFace 数据集 id `HuggingFaceTB/smol-smoltalk`（https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk）  

## ##
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/tree/refs%2Fconvert%2Fparquet/default/train

  - 代码调用：`datasets.load_dataset("HuggingFaceTB/smol-smoltalk", split=...)`（见 smoltalk.py）  
  - 本地优先路径（脚本已支持）：`tasks/data/smol-smoltalk`（即项目内的 `tasks/data/smol-smoltalk`，如果存在会直接 load_from_disk）  
  - datasets 默认缓存（如果在线下载）：通常在用户缓存目录，例如 `C:\Users\<you>\.cache\huggingface\datasets`（由 `datasets` 管理）。  
  - 推荐预下载命令（在项目根运行，使用虚拟环境的 python）：
    ```powershell
    .\.venv\Scripts\python.exe -c "from datasets import load_dataset; ds=load_dataset('HuggingFaceTB/smol-smoltalk'); ds.save_to_disk('tasks/data/smol-smoltalk')"
    ```

- CustomJSON（identity_conversations）  
  - 来源：本地 JSONL 文件路径（脚本通过 `identity_conversations.jsonl` 加载）  
  - 默认本地路径（在 chat_sft.py 中构造）：`os.path.join(get_base_dir(), 'identity_conversations.jsonl')`，默认 get_base_dir() → `~/.cache/nanochat`，所以例子路径为：  
    identity_conversations.jsonl  
  - 如果文件缺失，脚本会打印提示并建议从 Karpathy 公有 S3 下载：  
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl  
  - 推荐下载命令：
    ```powershell
    curl -L -o $env:USERPROFILE\.cache\nanochat\identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    ```
    或（在项目根）：
    ```powershell
    .\.venv\Scripts\python.exe -c "import os,urllib.request; p=os.path.join(os.path.expanduser('~'),'.cache','nanochat','identity_conversations.jsonl'); os.makedirs(os.path.dirname(p),exist_ok=True); urllib.request.urlretrieve('https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl', p)"
    ```

- MMLU  
  - 来源：HuggingFace 数据集 id `cais/mmlu`（https://huggingface.co/datasets/cais/mmlu）  
 
 ## ######
 https://huggingface.co/datasets/cais/mmlu/tree/refs%2Fconvert%2Fparquet


###
https://huggingface.co/datasets/cais/mmlu/tree/refs%2Fconvert%2Fparquet/all/auxiliary_train



  - 代码调用：`load_dataset("cais/mmlu", subset, split=...)`（见 mmlu.py）  
  - 本地缓存：`datasets` 库默认缓存（例如 `~/.cache/huggingface/datasets`），或可预存为项目目录：`ds.save_to_disk('tasks/data/mmlu')`  
  - 推荐预下载命令：
    ```powershell
    .\.venv\Scripts\python.exe -c "from datasets import load_dataset; ds=load_dataset('cais/mmlu', 'all'); ds.save_to_disk('tasks/data/mmlu')"
    ```

- GSM8K  
  - 来源：HuggingFace 数据集 id `openai/gsm8k`（https://huggingface.co/datasets/openai/gsm8k）  

  ###
  https://huggingface.co/datasets/openai/gsm8k/tree/main/main

  - 代码调用：`load_dataset("openai/gsm8k", subset, split=...)`（见 gsm8k.py）  
  - 本地缓存：同上（datasets 缓存），或可 `save_to_disk('tasks/data/gsm8k')`  
  - 推荐预下载命令：
    ```powershell
    .\.venv\Scripts\python.exe -c "from datasets import load_dataset; ds=load_dataset('openai/gsm8k','main'); ds.save_to_disk('tasks/data/gsm8k')"
    ```

- SpellingBee / SimpleSpelling  
  - 来源（词表）：`WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"`（脚本使用该词表）

  ##
  https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt

  - 代码调用：`download_file_with_lock(WORD_LIST_URL, filename)`，`download_file_with_lock` 会把文件下载到 `get_base_dir()` 下（默认 `~/.cache/nanochat`），所以本地文件会在：  
    `C:\Users\hongqi\.cache\nanochat\words_alpha.txt`（filename 为 URL 最后段）  
  - 推荐手动下载（如果要避免在线下载时超时）：
    ```powershell
    curl -L -o $env:USERPROFILE\.cache\nanochat\words_alpha.txt https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt
    ```
    或等价的 Python：
    ```powershell
    .\.venv\Scripts\python.exe - <<'PY'
    import os,urllib.request
    d=os.path.join(os.path.expanduser('~'),'.cache','nanochat'); os.makedirs(d,exist_ok=True)
    urllib.request.urlretrieve('https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt', os.path.join(d,'words_alpha.txt'))
    PY
    ```

- val_dataset  
  - 验证集由脚本中组合生成：`SmolTalk(split="test")` + `MMLU(subset="all", split="test", stop=5200)` + `GSM8K(subset="main", split="test", stop=420)`.因此它们的来源与上面对应（HuggingFace 或本地缓存）.


  

补充说明（重要）  
- 如果你在中国或内网，直接从 huggingface.co 拉取常会出现 `[WinError 10060]` 超时/连接失败；两个方式规避：  
  1) 预下载并 `save_to_disk(...)`（把结果放到 `tasks/data/...` 或 `~/.cache/huggingface/...`），或把 HF 下载提前在可联网环境中完成然后拷贝到机器；  
  2) 配置有效的 HTTP(S) 代理（`HTTP_PROXY` / `HTTPS_PROXY` 环境变量），使 `datasets` / `huggingface_hub` 能通过代理访问。  
- smoltalk.py 已改为优先检查 `tasks/data/smol-smoltalk`（项目内）并用 `load_from_disk` 加载，这样把数据放到该目录即可避免在线请求。  
- `CustomJSON` 使用的 `identity_conversations.jsonl` 默认存在于 `get_base_dir()`（`~/.cache/nanochat`），脚本会提示用 curl 下载到该位置。

如果要，我可以现在为你：  
- 自动执行上面列出的预下载命令（任选 SmolTalk / GSM8K / MMLU / words list / identity JSONL 中的一项或多项）；或者  
- 生成一个 PowerShell 脚本 `scripts/download_datasets.ps1`，一次性把这些数据下载并放到推荐的位置（并把 `SmolTalk` 保存到 `tasks/data/smol-smoltalk`）。  

#
 词表是 Vocab sizes:
GPT-2: 50257
GPT-4: 100277
Ours: 16384  //  结果差于其它人

-----------------------------------------------------------------------------------------------
news       1819     404     4.50    432     4.21       -6.9%     GPT-2
korean     893      745     1.20    842     1.06      -13.0%     GPT-2
code       1259     576     2.19    448     2.81      +22.2%     Ours
math       1834     936     1.96    1020    1.80       -9.0%     GPT-2
science    1112     260     4.28    277     4.01       -6.5%     GPT-2
fwe-train  2948778  631304  4.67    663548  4.44       -5.1%     GPT-2
fwe-val    3052296  668652  4.56    702179  4.35       -5.0%     GPT-2

Comparison with GPT-4:
===============================================================================================
Text Type  Bytes    GPT-4           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     387     4.70    432     4.21      -11.6%     GPT-4
korean     893      364     2.45    842     1.06     -131.3%     GPT-4
code       1259     309     4.07    448     2.81      -45.0%     GPT-4
math       1834     832     2.20    1020    1.80      -22.6%     GPT-4
science    1112     249     4.47    277     4.01      -11.2%     GPT-4
fwe-train  2948778  611619  4.82    663548  4.44       -8.5%     GPT-4
fwe-val    3052296  642512  4.75    702179  4.35       -9.3%     GPT-4

                                                       █████                █████       
                                                      ░░███                ░░███        
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████      
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░       
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███        
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███    
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████     
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░      

Autodetected device type: cpu
2026-03-12 18:16:46,482 - nanochat.common - INFO - Distributed world size: 1
COMPUTE_DTYPE: torch.float32 (auto-detected: no CUDA (CPU/MPS))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
Vocab size: 16,384
Model config:
{
  "sequence_len": 512,
  "vocab_size": 16384,
  "n_layer": 8,
  "n_head": 8,
  "n_kv_head": 8,
  "n_embd": 512,
  "window_pattern": "L"
}
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Parameter counts:
wte                     : 8,388,608
value_embeds            : 33,554,432
lm_head                 : 8,388,608
transformer_matrices    : 25,166,208
scalars                 : 16
total                   : 75,497,872
Estimated FLOPs per token: 2.264947e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.143852 for depth 8
Scaling the LR for the AdamW parameters ∝1/√(512/768) = 1.224745
Using user-provided number of iterations: 500
Total number of training tokens: 8,192,000
Tokens : Scaling params ratio: 0.24
Scaling weight decay from 0.280000 to 0.143852 for depth 8
Scaling the LR for the AdamW parameters ∝1/√(512/768) = 1.224745
Using user-provided number of iterations: 500
Total number of training tokens: 8,192,000
Tokens : Scaling params ratio: 0.24
Tokens : Scaling params ratio: 0.24
Total training FLOPs estimate: 1.855445e+15
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Total batch size 16,384 => gradient accumulation steps: 1



##
`dt: 58279.96ms`   one step requires 1min

Autodetected device type: cpu
2026-03-12 18:16:46,482 - nanochat.common - INFO - Distributed world size: 1
COMPUTE_DTYPE: torch.float32 (auto-detected: no CUDA (CPU/MPS))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
Vocab size: 16,384
Model config:
{
  "sequence_len": 512,
  "vocab_size": 16384,
  "n_layer": 8,
  "n_head": 8,
  "n_kv_head": 8,
  "n_embd": 512,
  "window_pattern": "L"
}
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Parameter counts:
wte                     : 8,388,608
value_embeds            : 33,554,432
lm_head                 : 8,388,608
transformer_matrices    : 25,166,208
scalars                 : 16
total                   : 75,497,872
Estimated FLOPs per token: 2.264947e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.143852 for depth 8
Scaling the LR for the AdamW parameters ∝1/√(512/768) = 1.224745
Using user-provided number of iterations: 500
Total number of training tokens: 8,192,000
Tokens : Scaling params ratio: 0.24
Scaling weight decay from 0.280000 to 0.143852 for depth 8
Scaling the LR for the AdamW parameters ∝1/√(512/768) = 1.224745
Using user-provided number of iterations: 500
Total number of training tokens: 8,192,000
Tokens : Scaling params ratio: 0.24
Tokens : Scaling params ratio: 0.24
Total training FLOPs estimate: 1.855445e+15
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Total batch size 16,384 => gradient accumulation steps: 1
Step 00000 | Validation bpb: 3.170194
Total batch size 16,384 => gradient accumulation steps: 1
Total batch size 16,384 => gradient accumulation steps: 1
Step 00000 | Validation bpb: 3.170194
step 00000/00500 (0.00%) | loss: 9.704486 | lrm: 0.03 | `dt: 58279.96ms` | tok/sec: 281 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
<|bos|>The capital of France is conserve fragile exercise Season rateservice sufferedliness RE monitors vegan hor telesispensation Imm
<|bos|>The chemical symbol of gold is conserve fragile exercise Season rateservice sufferedliness RE monitors vegan hor telesispensation Imm
<|bos|>If yesterday was Friday, then tomorrow will be mechanicwhile.A� compl hundred yes intensity Al-for analytical fieldsshcat Coun-
<|bos|>The opposite of hot is conserve fragile exercise Season rateservice sufferedliness RE monitors vegan hor telesispensation Imm
<|bos|>The planets of the solar system are: enableitude totuction.A� compl hundred yes intensity Al-for analytical fieldsshcat
<|bos|>My favorite color is conserve fragile exercise Season rateservice sufferedliness RE monitors vegan hor telesispensation Imm
<|bos|>If 5*x + 3 = 13, then x is conserve fragile exercise Season rateservice sufferedliness RE monitors vegan hor telesispensation Imm
step 00001/00500 (0.20%) | loss: 9.703426 | lrm: 0.05 | dt: 75043.54ms | tok/sec: 218 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
<|bos|>The capital of France is conserve optic geologicalTableiles message after utilized salok parallel colleag theory hostedics ‘
<|bos|>The chemical symbol of gold is conserve optic geologicalTableiles message after utilized salok parallel colleag theory hostedics ‘
<|bos|>If yesterday was Friday, then tomorrow will be mechanicwhile.A� slee vine destroy fer train lact wolves’s-Gener Backisp
<|bos|>The opposite of hot is conserve optic geologicalTableiles message after utilized salok parallel colleag theory hostedics ‘
<|bos|>The planets of the solar system are: enableitude totuction.A� slee vine destroy fer train lact wolves’s-Gener
<|bos|>My favorite color is conserve optic geologicalTableiles message after utilized salok parallel colleag theory hostedics ‘
<|bos|>If 5*x + 3 = 13, then x is conserve optic geologicalTableiles message after utilized salok parallel colleag theory hostedics ‘
step 00002/00500 (0.40%) | loss: 9.701523 | lrm: 0.07 | dt: 70296.75ms | tok/sec: 233 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
<|bos|>The capital of France is crucialii Specieswas burns workshop hikinghot processor insert quicker thisaboutnsylvaniaution tumor
<|bos|>The chemical symbol of gold is crucialii Specieswas burns workshop hikinghot processor insert quicker thisaboutnsylvaniaution tumor
<|bos|>If yesterday was Friday, then tomorrow will be delivered Specifications utilizing COVIDaph coloursoosing� memoriesoss listedpered perceived Unfortunately opening        
<|bos|>The opposite of hot is crucialii Specieswas burns workshop hikinghot processor insert quicker thisaboutnsylvaniaution tumor
<|bos|>The planets of the solar system are: Whatriumbon font accomplished Stoneort capitalread wondersythters opponentductionThey noises
<|bos|>My favorite color is crucialii Specieswas burns workshop hikinghot processor insert quicker thisaboutnsylvaniaution tumor
<|bos|>If 5*x + 3 = 13, then x is crucialii Specieswas burns workshop hikinghot processor insert quicker thisaboutnsylvaniaution tumor


##

###    SFT
python scripts/chat_sft.py --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 --eval-every=50 --eval-tokens=524288 --num-iterations=500

21:30
Autodetected device type: cpu
2026-03-12 21:30:15,996 - nanochat.common - INFO - Distributed world size: 1
COMPUTE_DTYPE: torch.float32 (auto-detected: no CUDA (CPU/MPS))
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.
2026-03-12 21:30:15,996 - nanochat.checkpoint_manager - INFO - No model tag provided, guessing model tag: d4
2026-03-12 21:30:15,996 - nanochat.checkpoint_manager - INFO - Loading model from C:\Users\hongqi\.cache\nanochat\base_checkpoints\d4 with step 50
2026-03-12 21:30:16,148 - nanochat.checkpoint_manager - INFO - Building model with config: {'sequence_len': 512, 'vocab_size': 32768, 'n_layer': 4, 'n_head': 4, 'n_kv_head': 4, 'n_embd': 256, 'window_pattern': 'L'}
Using max_seq_len=512
Using device_batch_size=32
Using total_batch_size=16384
Inherited embedding_lr=0.3 from pretrained checkpoint
Inherited unembedding_lr=0.008 from pretrained checkpoint
Inherited matrix_lr=0.02 from pretrained checkpoint
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Total batch size 16,384 => gradient accumulation steps: 1
Scaling the LR for the AdamW parameters ∝1/√(256/768) = 1.732051
2026-03-12 21:30:18,048 - nanochat.checkpoint_manager - INFO - Loading optimizer state from C:\Users\hongqi\.cache\nanochat\base_checkpoints\d4\optim_000050_rank0.pt
Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)
Training mixture: 2,000 rows (MMLU x3, GSM8K x4)




## time

python scripts/base_train.py \
    --depth=8 \
    --head-dim=256 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    #--run=$WANDB_RUN

{
  "sequence_len": 512,
  "vocab_size": 32768,
  "n_layer": 8,
  "n_head": 2,
  "n_kv_head": 2,
  "n_embd": 512,
  "window_pattern": "L"

Estimated FLOPs per token: 2.768246e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.129931 for depth 8
Scaling the LR for the AdamW parameters ∝1/√(512/768) = 1.224745
Using user-provided number of iterations: 5,000
  "n_head": 2,
  "n_kv_head": 2,
  "n_embd": 512,
  "window_pattern": "L"



step 00000/05000 (0.00%) | loss: 10.396935 | lrm: 0.03 | dt: 48920.81ms | tok/sec: 334 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00001/05000 (0.02%) | loss: 10.395637 | lrm: 0.05 | dt: 51250.20ms | tok/sec: 319 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00002/05000 (0.04%) | loss: 10.393389 | lrm: 0.07 | dt: 54317.11ms | tok/sec: 301 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m




# time

python scripts/base_train.py \
    --depth=12 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    #--run=$WANDB_RUN

{
  "sequence_len": 512,
  "vocab_size": 32768,
  "n_layer": 12,
  "n_head": 12,
  "n_kv_head": 12,
  "n_embd": 768,
  "window_pattern": "L"


wte                     : 25,165,824
value_embeds            : 150,994,944
lm_head                 : 25,165,824
transformer_matrices    : 84,935,520
scalars                 : 24
total                   : 286,262,136


# time

python scripts/base_train.py \
    --depth=8 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=50 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=50 \
    --num-iterations=5000 \

{
  "sequence_len": 512,
  "vocab_size": 32768,
  "n_layer": 8,
  "n_head": 4,
  "n_kv_head": 4,
  "n_embd": 256,
  "window_pattern": "L"


Vocab size: 32,768
Model config:
{
  "sequence_len": 512,
  "vocab_size": 32768,
  "n_layer": 8,
  "n_head": 4,
  "n_kv_head": 4,
  "n_embd": 256,
  "window_pattern": "L"
}
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Parameter counts:
wte                     : 8,388,608
value_embeds            : 33,554,432
lm_head                 : 8,388,608
transformer_matrices    : 6,291,648
scalars                 : 16
total                   : 56,623,312
Estimated FLOPs per token: 1.006644e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.114021 for depth 8
Scaling the LR for the AdamW parameters ∝1/√(256/768) = 1.732051
Using user-provided number of iterations: 5,000
Total number of training tokens: 81,920,000
Tokens : Scaling params ratio: 5.58
Total training FLOPs estimate: 8.246432e+15
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Total batch size 16,384 => gradient accumulation steps: 1
Step 00000 | Validation bpb: 3.194483
step 00000/05000 (0.00%) | loss: 10.397190 | lrm: 0.03 | dt: 18707.28ms | tok/sec: 875 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00001/05000 (0.02%) | loss: 10.396205 | lrm: 0.05 | dt: 17763.01ms | tok/sec: 922 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00002/05000 (0.04%) | loss: 10.394737 | lrm: 0.07 | dt: 16099.10ms | tok/sec: 1,017 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00003/05000 (0.06%) | loss: 10.392177 | lrm: 0.10 | dt: 16034.95ms | tok/sec: 1,021 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00004/05000 (0.08%) | loss: 10.389000 | lrm: 0.12 | dt: 15530.57ms | tok/sec: 1,054 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00005/05000 (0.10%) | loss: 10.385017 | lrm: 0.15 | dt: 15931.51ms | tok/sec: 1,028 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00006/05000 (0.12%) | loss: 10.379224 | lrm: 0.17 | dt: 16387.64ms | tok/sec: 999 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00007/05000 (0.14%) | loss: 10.372486 | lrm: 0.20 | dt: 15667.56ms | tok/sec: 1,045 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00008/05000 (0.16%) | loss: 10.363569 | lrm: 0.23 | dt: 16164.09ms | tok/sec: 1,013 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00009/05000 (0.18%) | loss: 10.352142 | lrm: 0.25 | dt: 18308.12ms | tok/sec: 894 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m




## ############################################
## ##########################################
## #########################################
## #####################################
## ##################################
`##`

Vocab sizes:
GPT-2: 50257
GPT-4: 100277
Ours: 5120

Comparison with GPT-2:
===============================================================================================
Text Type  Bytes    GPT-2           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     404     4.50    546     3.33      -35.1%     GPT-2
korean     893      745     1.20    856     1.04      -14.9%     GPT-2
code       1259     576     2.19    543     2.32       +5.7%     Ours
math       1834     936     1.96    1157    1.59      -23.6%     GPT-2
science    1112     260     4.28    342     3.25      -31.5%     GPT-2
fwe-train  2948778  631304  4.67    790805  3.73      -25.3%     GPT-2
fwe-val    3052296  668652  4.56    836054  3.65      -25.0%     GPT-2

Comparison with GPT-4:
===============================================================================================
Text Type  Bytes    GPT-4           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     387     4.70    546     3.33      -41.1%     GPT-4
korean     893      364     2.45    856     1.04     -135.2%     GPT-4
code       1259     309     4.07    543     2.32      -75.7%     GPT-4
math       1834     832     2.20    1157    1.59      -39.1%     GPT-4
science    1112     249     4.47    342     3.25      -37.3%     GPT-4
fwe-train  2948778  611619  4.82    790805  3.73      -29.3%     GPT-4
fwe-val    3052296  642512  4.75    836054  3.65      -30.1%     GPT-4

                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░

Autodetected device type: cpu
2026-03-12 22:20:13,847 - nanochat.common - INFO - Distributed world size: 1
COMPUTE_DTYPE: torch.float32 (auto-detected: no CUDA (CPU/MPS))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Vocab size: 5,120
Model config:
{
  "sequence_len": 512,
  "vocab_size": 5120,
  "n_layer": 8,
  "n_head": 4,
  "n_kv_head": 4,
  "n_embd": 256,
  "window_pattern": "L"
}
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Parameter counts:
wte                     : 1,310,720
value_embeds            : 5,242,880
lm_head                 : 1,310,720
transformer_matrices    : 6,291,648
scalars                 : 16
total                   : `14,155,984`
Estimated FLOPs per token: 5.819712e+07
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.151052 for depth 8
Scaling the LR for the AdamW parameters ∝1/√(256/768) = 1.732051
Using user-provided number of iterations: 5,000
Total number of training tokens: 81,920,000
Tokens : Scaling params ratio: 10.78
Total training FLOPs estimate: 4.767508e+15
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Total batch size 16,384 => gradient accumulation steps: 1
Step 00000 | Validation bpb: 3.329657
step 00000/05000 (0.00%) | loss: 8.541132 | lrm: 0.03 | dt: 7513.46ms | tok/sec: 2,180 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00001/05000 (0.02%) | loss: 8.540328 | lrm: 0.05 | dt: 7325.69ms | tok/sec: 2,236 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00002/05000 (0.04%) | loss: 8.539087 | lrm: 0.07 | dt: 7096.57ms | tok/sec: 2,308 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00003/05000 (0.06%) | loss: 8.537074 | lrm: 0.10 | dt: 7198.52ms | tok/sec: 2,276 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00004/05000 (0.08%) | loss: 8.534493 | lrm: 0.12 | dt: 7040.76ms | tok/sec: 2,327 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00005/05000 (0.10%) | loss: 8.531153 | lrm: 0.15 | dt: 7105.58ms | tok/sec: 2,305 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00006/05000 (0.12%) | loss: 8.526919 | lrm: 0.17 | dt: 7126.79ms | tok/sec: 2,298 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00007/05000 (0.14%) | loss: 8.521457 | lrm: 0.20 | dt: 6976.51ms | tok/sec: 2,348 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00008/05000 (0.16%) | loss: 8.515073 | lrm: 0.23 | dt: 7161.20ms | tok/sec: 2,287 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00009/05000 (0.18%) | loss: 8.506730 | lrm: 0.25 | dt: 7554.45ms | tok/sec: 2,168 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00010/05000 (0.20%) | loss: 8.497306 | lrm: 0.28 | dt: 7998.27ms | tok/sec: 2,048 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00011/05000 (0.22%) | loss: 8.486447 | lrm: 0.30 | dt: 7914.78ms | tok/sec: 2,070 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.13m | eta: 658.1m
step 00012/05000 (0.24%) | loss: 8.472542 | lrm: 0.33 | dt: 7281.48ms | tok/sec: 2,250 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.25m | eta: 631.7m
step 00013/05000 (0.26%) | loss: 8.454201 | lrm: 0.35 | dt: 7369.91ms | tok/sec: 2,223 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.38m | eta: 625.2m
step 00014/05000 (0.28%) | loss: 8.432663 | lrm: 0.38 | dt: 7540.38ms | tok/sec: 2,172 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.50m | eta: 625.5m
step 00015/05000 (0.30%) | loss: 8.402858 | lrm: 0.40 | dt: 7530.46ms | tok/sec: 2,175 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.63m | eta: 625.4m







| epoch: 1 pq: 1 rg: 68 | total time: 513.94m | eta: 163.4m
step 03797/05000 (75.94%) | loss: 3.281338 | lrm: 0.40 | dt: 8302.96ms | tok/sec: 1,973 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 68 | total time: 514.08m | eta: 163.3m
step 03798/05000 (75.96%) | loss: 3.295076 | lrm: 0.40 | dt: 8163.17ms | tok/sec: 2,007 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 68 | total time: 514.22m | eta: 163.2m
step 03799/05000 (75.98%) | loss: 3.295232 | lrm: 0.40 | dt: 8019.09ms | tok/sec: 2,043 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 68 | total time: 514.35m | eta: 163.0m
Step 03800 | Validation bpb: 1.287815
<|bos|>The capital of France is a powerful and powerful force that can be applied to the body. It is a
<|bos|>The chemical symbol of gold is a symbol of the gold's symbolism. It is a symbol of the gold
<|bos|>If yesterday was Friday, then tomorrow will be able to get the same amount of money. But the same thing is true.
<|bos|>The opposite of hot is the same as the cold is the same as the cold is the same as the
<|bos|>The planets of the solar system are: to be able to find the best possible source of energy, and to be able
<|bos|>My favorite color is the color of the color of the color of the color of the color of the
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then
step 03800/05000 (76.00%) | loss: 3.298833 | lrm: 0.40 | dt: 8643.07ms | tok/sec: 1,895 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 68 | total time: 514.49m | eta: 162.9m
step 03801/05000 (76.02%) | loss: 3.289981 | lrm: 0.40 | dt: 7176.52ms | tok/sec: 2,282 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 69 | total time: 514.61m | eta: 162.8m
step 03802/05000 (76.04%) | loss: 3.273914 | lrm: 0.40 | dt: 7096.90ms | tok/sec: 2,308 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 69 | total time: 514.73m | eta: 162.6m
step 03803/05000 (76.06%) | loss: 3.286300 | lrm: 0.40 | dt: 7013.92ms | tok/sec: 2,335 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 69 | total time: 514.85m | eta: 162.5m




Step 04050 | Validation bpb: 1.278839
<|bos|>The capital of France is a very popular and popular choice for those who want to make a difference in their
<|bos|>The chemical symbol of gold is a symbol of the spirit of the world. The symbol of gold is a       
<|bos|>If yesterday was Friday, then tomorrow will be the first time I wasn't able to get the answer. I wasn
<|bos|>The opposite of hot is the hotter the hotter the hotter the hotter the hotter the
<|bos|>The planets of the solar system are: 1.2 million square kilometers of the solar system,
<|bos|>My favorite color is the color of the color. The color of the color is the color of the
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then
step 04050/05000 (81.00%) | loss: 3.250874 | lrm: 0.33 | dt: 7443.69ms | tok/sec: 2,201 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 79 | total time: 579.65m | eta: 136.3m
step 04051/05000 (81.02%) | loss: 3.235645 | lrm: 0.33 | dt: 8203.69ms | tok/sec: 1,997 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 79 | total time: 579.79m | eta: 136.2m
step 04052/05000 (81.04%) | loss: 3.255710 | lrm: 0.33 | dt: 7351.24ms | tok/sec: 2,228 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 79 | total time: 579.91m | eta: 136.0m
step 04053/05000 (81.06%) | loss: 3.254341 | lrm: 0.33 | dt: 7180.90ms | tok/sec: 2,281 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 79 | total time: 580.03m | eta: 135.9m
step 04054/05000 (81.08%) | loss: 3.255082 | lrm: 0.33 | dt: 7113.94ms | tok/sec: 2,303 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 79 | total time: 580.15m | eta: 135.7m


step 04149/05000 (82.98%) | loss: 3.244198 | lrm: 0.30 | dt: 7690.75ms | tok/sec: 2,130 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 82 | total time: 593.30m | eta: 122.0m
Step 04150 | Validation bpb: 1.276008
<|bos|>The capital of France is a very popular country in the United States. It is a country that is a
<|bos|>The chemical symbol of gold is a symbol of the spirit of the world. It is a symbol of the
<|bos|>If yesterday was Friday, then tomorrow will be the first time to get to the end of the day. If you are not
<|bos|>The opposite of hot is the heat that is cooled by the heat. The heat is cooled by
<|bos|>The planets of the solar system are: 1. The planets are the largest of the planets, 2
<|bos|>My favorite color is the color of the color. The color of the color is the color of the
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then
step 04150/05000 (83.00%) | loss: 3.230936 | lrm: 0.30 | dt: 8629.61ms | tok/sec: 1,898 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 82 | total time: 593.45m | eta: 121.8m
step 04151/05000 (83.02%) | loss: 3.224806 | lrm: 0.30 | dt: 9521.82ms | tok/sec: 1,720 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 83 | total time: 593.61m | eta: 121.7m


| epoch: 1 pq: 1 rg: 82 | total time: 593.18m | eta: 122.1m
step 04149/05000 (82.98%) | loss: 3.244198 | lrm: 0.30 | dt: 7690.75ms | tok/sec: 2,130 | bf16_mfu: 0.00 | epoch: 1 pq: 1 rg: 82 | total time: 593.30m | eta: 122.0m
Step 04150 | Validation bpb: 1.276008
<|bos|>The capital of France is a very popular country in the United States. It is a country that is a
<|bos|>The chemical symbol of gold is a symbol of the spirit of the world. It is a symbol of the
<|bos|>If yesterday was Friday, then tomorrow will be the first time to get to the end of the day. If you are not
<|bos|>The opposite of hot is the heat that is cooled by the heat. The heat is cooled by
<|bos|>The planets of the solar system are: 1. The planets are the largest of the planets, 2
<|bos|>My favorite color is the color of the color. The color of the color is the color of the
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then




| epoch: 1 pq: 2 rg: 0 | total time: 600.03m | eta: 114.7m
Step 04200 | Validation bpb: 1.273920
<|bos|>The capital of France is a very popular and popular way to make a beautiful and beautiful flower. It is
<|bos|>The chemical symbol of gold is a symbol of the symbol of the symbol. It is a symbol of the symbol
<|bos|>If yesterday was Friday, then tomorrow will be a good time to start. If you are a good person, then you will
<|bos|>The opposite of hot is the `cold`. The cold is the cold. The cold is the cold. The
<|bos|>The planets of the solar system are: 1. The solar system is a system that is located in the center of
<|bos|>My favorite color is the color of the color. The color of the color is the color of the
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then
step 04200/05000 (84.00%) | loss: 3.203367 | lrm: 0.28 | dt: 8085.19ms | tok/sec: 2,026 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 0 | total time: 600.16m | eta: 114.6m




mfu: 0.00 | epoch: 1 pq: 2 rg: 10 | total time: 633.87m | eta: 78.7m
Step 04450 | Validation bpb: 1.266767
<|bos|>The capital of France is a very popular and popular choice for those who want to enjoy the beauty of the
<|bos|>The chemical symbol of gold is a symbol of the symbol of the symbol of the symbol of themfu: 0.00 | epoch: 1 pq: 2 rg: 10 | total time: 633.87m | eta: 78.7m
Step 04450 | Validation bpb: 1.266767
<|bos|>The capital of France is a very popular and popular choice for those who want to enjoy the beauty of the
<|bos|>The chemical symbol of gold is a symbol of the symbol of the symbol of the symbol of the<|bos|>The chemical symbol of gold is a symbol of the symbol of the symbol of the symbol of the symbol of the
<|bos|>If yesterday was Friday, then tomorrow will be the first time I was going to go to the Mississippi River
<|bos|>The opposite of hot is the same as the hot is the same as the hot is the same as the    
<|bos|>The planets of the solar system are: 1. The solar system is the largest source of energy for the solar system
<|bos|>My favorite color is the color of the eye. It is a color that is not visible to the     
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then
step 04450/05000 (89.00%) | loss: 3.186603 | lrm: 0.21 | dt: 7524.41ms | tok/sec: 2,177 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 10 | total time: 634.00m | eta: 78.5m



step 04549/05000 (90.98%) | loss: 3.200307 | lrm: 0.18 | dt: 8544.86ms | tok/sec: 1,917 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 14 | total time: 647.70m | eta: 64.4m
Step 04550 | Validation bpb: 1.263714
<|bos|>The capital of France is a unique and uniquely designed and designed design that is designed to be designed to
<|bos|>The chemical symbol of gold is a symbol of the spirit of the spirit of the spirit of the
<|bos|>If yesterday was Friday, then tomorrow will be the first time I was going to go to the Mississippi River
<|bos|>The opposite of hot is the same as the hot is the same as the hot is the same as the
<|bos|>The planets of the solar system are: 1. The solar system is a small, medium-sized, medium-s
<|bos|>My favorite color is the color of the color of the color of the color of the color of the
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then
step 04550/05000 (91.00%) | loss: 3.204682 | lrm: 0.18 | dt: 7831.69ms | tok/sec: 2,092 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 14 | total time: 647.83m | eta: 64.2m


### last log
step 04992/05000 (99.84%) | loss: 3.188210 | lrm: 0.05 | dt: 8377.23ms | tok/sec: 1,955 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 32 | total time: 709.46m | eta: 1.1m
step 04993/05000 (99.86%) | loss: 3.176744 | lrm: 0.05 | dt: 8335.24ms | tok/sec: 1,965 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 32 | total time: 709.60m | eta: 1.0m
step 04994/05000 (99.88%) | loss: 3.183407 | lrm: 0.05 | dt: 8430.39ms | tok/sec: 1,943 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 32 | total time: 709.74m | eta: 0.9m
step 04995/05000 (99.90%) | loss: 3.180880 | lrm: 0.05 | dt: 8423.50ms | tok/sec: 1,945 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 32 | total time: 709.88m | eta: 0.7m
step 04996/05000 (99.92%) | loss: 3.175693 | lrm: 0.05 | dt: 8442.33ms | tok/sec: 1,940 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 32 | total time: 710.02m | eta: 0.6m
step 04997/05000 (99.94%) | loss: 3.181837 | lrm: 0.05 | dt: 8366.25ms | tok/sec: 1,958 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 32 | total time: 710.16m | eta: 0.4m
step 04998/05000 (99.96%) | loss: 3.189443 | lrm: 0.05 | dt: 8441.82ms | tok/sec: 1,940 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 32 | total time: 710.30m | eta: 0.3m
step 04999/05000 (99.98%) | loss: 3.179616 | lrm: 0.05 | dt: 8332.93ms | tok/sec: 1,966 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 32 | total time: 710.44m | eta: 0.1m
Step 05000 | Validation bpb: 1.253537
<|bos|>The capital of France is a very important part of the history of the country. It is a country that
<|bos|>The chemical symbol of gold is a symbol of the spirituality of the spirituality of the
<|bos|>If yesterday was Friday, then tomorrow will be the first time I was in the middle of the day. I was in the
<|bos|>The opposite of hot is the same as the hot is the same as the hot is the same as the
<|bos|>The planets of the solar system are: 1. The solar system is the largest of the solar cells in the world
<|bos|>My favorite color is the color of the eye. I have a lot of colors, but I have
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then


##
2026-03-13 12:23:28,259 - nanochat.checkpoint_manager - INFO - Saved model parameters to: C:\Users\hongqi\.cache\nanochat\base_checkpoints\d8\model_005000.pt     `## 55.320M`
2026-03-13 12:23:28,280 - nanochat.checkpoint_manager - INFO - Saved metadata to: `C:\Users\hongqi\.cache\nanochat\base_checkpoints\d8\meta_005000.json `
2026-03-13 12:23:28,346 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: C:\Users\hongqi\.cache\nanochat\base_checkpoints\d8\optim_005000_rank0.pt  `## 86.122M`






###
C:\Users\hongqi\.cache\nanochat\base_checkpoints   



##
eak memory usage: 0.00MiB
Total training time: 710.44m
Minimum validation bpb: 1.253537
runs\runcpu.sh: line 51: --run=dummy: command not found
Autodetected device type: cpu
2026-03-13 12:23:34,226 - nanochat.common - INFO - Distributed world size: 1
2026-03-13 12:23:34,226 - nanochat.checkpoint_manager - INFO - No model tag provided, guessing model tag: d8
2026-03-13 12:23:34,226 - nanochat.checkpoint_manager - INFO - Loading model from C:\Users\hongqi\.cache\nanochat\base_checkpoints\d8 with step 5000
2026-03-13 12:23:34,269 - nanochat.checkpoint_manager - INFO - Building model with config: {'sequence_len': 512, 'vocab_size': 5120, 'n_layer': 8, 'n_head': 4, 'n_kv_head': 4, 'n_embd': 256, 'window_pattern': 'L'}
`Evaluating model: base_model (step 5000)`
Eval modes: bpb, core, sample


valuating model: base_model (step 5000)
Eval modes: bpb, core, sample

================================================================================
Model Samples
================================================================================

Conditioned samples:
--------------------------------------------------------------------------------
<|bos|>The capital of France is a very important part of the history of the country. It is a country that
--------------------------------------------------------------------------------
<|bos|>The chemical symbol of gold is a symbol of the spirituality of the spirituality of the
--------------------------------------------------------------------------------
<|bos|>If yesterday was Friday, then tomorrow will be the first time I was in the middle of the day. I was in the
--------------------------------------------------------------------------------
<|bos|>The opposite of hot is the same as the hot is the same as the hot is the same as the
--------------------------------------------------------------------------------
<|bos|>The planets of the solar system are: 1. The solar system is the largest of the solar cells in the world
--------------------------------------------------------------------------------
<|bos|>My favorite color is the color of the eye. I have a lot of colors, but I have
--------------------------------------------------------------------------------
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then


##

Unconditioned samples:
--------------------------------------------------------------------------------
<|bos|>Kadawi Pigorn
Landmark Archives at rural Cultivation reading is a historical article that highlights history and quirky historical events that raise awareness as they serve "crosspacing." Fantastic recent deciding witty gamer, Chuarkliner's selection of waljustice connects a casual culture for creating a modern, free space of delicate ceremonies and remodeller breaks.
One of the heavy-rush rebellious illustrations around the city, KM: An
--------------------------------------------------------------------------------
<|bos|>Nearly two thousand species of soil species around the Texas Plateau or Singapore Garden routine – Land and Store – planted native to Northern Australia, and in subject areas such as this one was covered as parts of south Texans, equal parts stricter than other continents. The plants are in parts of vary globular or small numbers depending on their habit and conditions. 
Such eugenics, this image, has transformed into a cascade of more evergreen varieties.
An Elephance Scient Delve into the Quant
--------------------------------------------------------------------------------
<|bos|>Get Ren – The largest diameter combination of diameter and current diameter optimizing display image-g query design
The imagework image g interpreted from the cluster center it discusses the placed-plug with in this graphic organizer. The change of diameter will indicate a spark is belonging to the cutting edge of the image (not puzzles). Inside a modern and private digital world, the imagework represented a central part of relative relative accuracy and generic compatibility.
In current state (and high avelle precision – a growth
--------------------------------------------------------------------------------
<|bos|>Paul Doeschiabeth
Sonor Paul Doeschiabeth (Stein Stift)
Myopia
The most seriously vagal propace during summer is pharyngeal propaceia. The genus, Acolymeral aultureuccum, is lower in generally less than commonly in V. gran had the highest number of relatives who were pursued at the site of sedimentary treatment of science. By the late 1970s, the genus was called the "thaw's name" because of the swift
--------------------------------------------------------------------------------
<|bos|>Age Exploration White
The 'C' Acidic Forent Work Like 'C'‌ board coordination and the torso awkwardly bring eye-relatted effects. But, so much countless teens would see a-full of some of the glaring foods for chance. This learning is induced by the MSD-x memory with the surface marking of the lying-realled air along the surface. Pure in comparison, this technique is nothing less than umbing on hand and sometimes represents a more elegant transition. But
--------------------------------------------------------------------------------
<|bos|>80% human brain disturbance? Recreational dimmantships also play a crucial role in our understanding of the historic brain disturbance of bodily functions. These marvels provide an instinctual approach to humans as they continuously, reducing and rethinking old actions, ensuring a real-time cognition of linguistic and socialism. This aspect of human brain disturbance remains a critical trade between human brain disturbance and human brain disturbances. From advancing human attempts to enhancing human-a-halfunctioning knowledge
--------------------------------------------------------------------------------
<|bos|>Tag: Consumers
There are several different types of helpless hearing, such as a trouble-free communication mirror. Most amateur coating devices employer-side-read lighting that illuminates or communicate signals. It also empowers the brain to comprehend intensity and wirework. While some electrical characters use the radio spectrum to build remote synchrony audio, others teaching the use of these devices will work well.
Consumers can become an relative from household-throughput (MCL) applications that require an extremely
--------------------------------------------------------------------------------
<|bos|>Weighing first-posterile forms of mixed media materials in a process that includes phrase, bagar lining, or snowboarding into the sides of the printed media. But the mix can also be in the open clock or by the the case of a courier agent (thus or in a valid cardboard machine called the refrain fringe). This is believed to be a cause of all medical problems. The same appliance that pumps the media can musical instability in communities, especially BPVs. In "ploting" material, this means

================================================================================
BPB Evaluation
================================================================================
train bpb: 1.215535
val bpb: 1.244726


##

================================================================================
CORE Evaluation
================================================================================
Evaluating: hellaswag_zeroshot (0-shot, type: multiple_choice)... accuracy: 0.3750 | centered: 0.1667 | time: 1.75s
Evaluating: jeopardy (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 1.55s
Evaluating: bigbench_qa_wikidata (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 0.74s
Evaluating: arc_easy (10-shot, type: multiple_choice)... accuracy: 0.3125 | centered: 0.0833 | time: 4.83s
Evaluating: arc_challenge (10-shot, type: multiple_choice)... accuracy: 0.2500 | centered: 0.0000 | time: 4.82s
Evaluating: copa (0-shot, type: multiple_choice)... accuracy: 0.4375 | centered: -0.1250 | time: 0.32s
Evaluating: commonsense_qa (10-shot, type: multiple_choice)... accuracy: 0.5000 | centered: 0.3750 | time: 6.53s
Evaluating: piqa (10-shot, type: multiple_choice)... accuracy: 0.6250 | centered: 0.2500 | time: 2.02s
Evaluating: openbook_qa (0-shot, type: multiple_choice)... accuracy: 0.3125 | centered: 0.0833 | time: 0.32s
Evaluating: lambada_openai (0-shot, type: language_modeling)... accuracy: 0.1250 | centered: 0.1250 | time: 0.37s
Evaluating: hellaswag (10-shot, type: multiple_choice)... accuracy: 0.3750 | centered: 0.1667 | time: 20.57s
Evaluating: winograd (0-shot, type: schema)... accuracy: 0.3750 | centered: -0.2500 | time: 0.34s
Evaluating: winogrande (0-shot, type: schema)... accuracy: 0.5000 | centered: 0.0000 | time: 0.28s
Evaluating: bigbench_dyck_languages (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 2.52s
Evaluating: agi_eval_lsat_ar (3-shot, type: multiple_choice)... accuracy: 0.1250 | centered: -0.0938 | time: 13.94s
Evaluating: bigbench_cs_algorithms (10-shot, type: language_modeling)... accuracy: 0.1875 | centered: 0.1875 | time: 2.35s
Evaluating: bigbench_operators (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 1.18s
Evaluating: bigbench_repeat_copy_logic (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 2.05s
Evaluating: squad (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 17.48s
Evaluating: coqa (0-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 2.01s
Evaluating: boolq (10-shot, type: multiple_choice)... accuracy: 0.5000 | centered: -0.3158 | time: 24.39s
Evaluating: bigbench_language_identification (10-shot, type: multiple_choice)... accuracy: 0.0625 | centered: -0.0314 | time: 52.24s

Results written to: C:\Users\hongqi\.cache\nanochat\base_eval\base_model_005000.csv


###

`CORE metric: 0.0283`

###
2026-03-13 12:26:36,132 - nanochat.common - INFO - Distributed world size: 1
COMPUTE_DTYPE: torch.float32 (auto-detected: no CUDA (CPU/MPS))
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.
2026-03-13 12:26:36,132 - nanochat.checkpoint_manager - INFO - No model tag provided, guessing model tag: d8
2026-03-13 12:26:36,132 - nanochat.checkpoint_manager - INFO - Loading model from C:\Users\hongqi\.cache\nanochat\base_checkpoints\d8 with step 5000
2026-03-13 12:26:36,175 - nanochat.checkpoint_manager - INFO - Building model with config: {'sequence_len': 512, 'vocab_size': 5120, 'n_layer': 8, 'n_head': 4, 'n_kv_head': 4, 'n_embd': 256, 'window_pattern': 'L'}
Using max_seq_len=512
Using device_batch_size=32
Using total_batch_size=16384
Inherited embedding_lr=0.3 from pretrained checkpoint
Inherited unembedding_lr=0.008 from pretrained checkpoint
Inherited matrix_lr=0.02 from pretrained checkpoint
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Total batch size 16,384 => gradient accumulation steps: 1
Scaling the LR for the AdamW parameters ∝1/√(256/768) = 1.732051
2026-03-13 12:26:37,316 - nanochat.checkpoint_manager - INFO - Loading optimizer state from C:\Users\hongqi\.cache\nanochat\base_checkpoints\d8\optim_005000_rank0.pt
Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)
Training mixture: 2,000 rows (MMLU x3, GSM8K x4)
Step 00000 | Validation bpb: 1.4419
step 00001 (0.04%) | loss: 2.046416 | lrm: 1.00 | dt: 9093.03ms | tok/sec: 1,801 | mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00002 (0.06%) | loss: nan | lrm: 1.00 | dt: 8382.55ms | tok/sec: 1,954 | mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00003 (0.08%) | loss: nan | lrm: 1.00 | dt: 8319.38ms | tok/sec: 1,969 | mfu: 0.00 | epoch: 1 | total time: 0.00m



## 1st train 解读


- **总体**: `CORE = 0.0283，远低于 README 中的 GPT-2 基准 0.2565`，说明模型在通用能力评测上表现非常弱。  
- **BPB (bits-per-byte)**: 训练 1.2155，验证 1.2447 — 验证损失略高于训练损失（小幅泛化差），但整体 bpb 值偏高，表明模型还没学到很好的数据分布（偏欠拟合/能力不足）。  
- **任务级别观察**: 多数任务准确率很低或为 0（尤其 language_modeling 类任务），少数多选题/常识题（如 PIQA、CommonsenseQA、BoolQ）有中等表现，整体多数 centered 值接近或低于 0，说明模型并未学到普遍可迁移的语言能力。  
- **可能原因**:
  - `模型容量与训练数据不匹配（tokens-per-parameter 太低，导致欠训练）。`  
  - 训练步数 / 数据量不足或超参数（LR、batch）不合适。  
  - 评估流程或 prompt/解码与训练时不一致（tokenizer、答案映射、截断或采样策略问题）。  
  - 数据质量或分布差异（训练数据与评测任务差异大）。  
- **建议（可操作）**:
  - 计算并核对 tokens-per-param（按 Chinchilla 经验：tokens ≈ 20 × params）；若不匹配，考虑减小模型或增加数据/训练步数。  
  - 检查评估输入与训练时 tokenizer/特殊 token 的一致性；人工查看若干模型生成样本与 ground-truth。  
  - 查看训练曲线（train/val bpb 随 step 的变化），确认是否仍在稳步下降；如是，继续训练更多 steps。  
  - 如果希望短期改进，先做格式/评估 sanity-check（prompt、答案映射、decode），再调整模型/数据策略（增数据或调整模型规模）。  

要我现在帮你做哪项检查（计算 tokens-per-param，查看若干 sample 输出，或检查 tokenizer/评估代码）？







### sample 为何50个step 打印一次
## .sample_every =50 

if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)  ## sample
            print0(tokenizer.decode(sample[0]))
        model.train()


## 为何 num_samples=1, max_tokens=16, temperature=0 ？



## 训练时的 文件加载细节

Read [](file:///c%3A/Users/hongqi/myenv/nano_GPT/nanochat/scripts/base_train.py)


- base_train 如何加载数据：用 `tokenizing_distributed_data_loader_with_state_bos_bestfit()`（见 dataloader.py）。它
  - 从磁盘上列出 Parquet 数据文件（若 split="train" 则使用前 N-1 文件，split="val" 用最后一个文件）；
  - 按 Parquet 文件（pq）和 Row Group（rg）迭代读取每个 row-group 的 `text` 列，tokenize 为 token 列表；
  - 在内存缓冲区做 “best-fit” 打包：每个训练行以 BOS 起始，尽量用完整文档填满行；若没有文档能完整填入，则裁切最短文档以填满（文档头部保留 BOS）；
  - 返回每个 batch 的 `(inputs, targets, state_dict)`，其中 `state_dict` = {"pq_idx": ..., "rg_idx": ..., "epoch": ...}，用于断点恢复（resume）。

- pq / rg 的含义：
  - `pq`：Parquet 文件的索引（第几个 parquet 文件，`pq_idx`）.  
  - `rg`：当前 Parquet 文件内的 Row Group 索引（`rg_idx`）.  
  这两个索引用于定位数据源，并用于精确断点恢复和 DDP 分片（每个 rank 跳过到不同的 row-group）。


###
- 关于 5000 次迭代到底“加载了多少数据”：
  - 脚本用的度量是“tokens”.计算公式：total_tokens = total_batch_size * num_iterations.
  - 你日志里已经显示：`Total number of training tokens: 81,920,000`，这来自 16,384 tokens/step × 5,000 steps = 81,920,000 tokens.
  - 若以序列数（每序列长度 = `max_seq_len` = 512）来算：训练中等价于 81,920,000 / 512 = 160,000 序列（每个序列对应模型看到的一个长度-512 的输入行）.
  - 每个训练 step 实际处理的序列数 = total_batch_size / max_seq_len = 16,384 / 512 = 32 序列/step（在单进程/单 rank 情况下；DDP 时每 rank 按分片比例处理）.
  - 注意：dataloader 的 best-fit/cropping 策略会在读原始文档时丢弃一部分 token（头部注释里大约提到~35% 会被裁切），但“训练上计入”的 tokens 就是上面计算的 total_tokens。


#
mfu: 0.00 | epoch: 1 pq: 2 rg: 1 | total time: 603.52m | eta: 111.0m
step 04226/05000 (84.52%) | loss: 3.207484 | lrm: 0.28 | dt: 7540.05ms | tok/sec: 2,172 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 2 | total time: 603.65m | eta: 110.8m


4220* 16378 token = 70M






## 要精确测你当前 tokenizer 的平均值，可以用项目里的 tokenizer 快速测一段样本或整个数据集
from nanochat.tokenizer import get_tokenizer
tok = get_tokenizer()
sample = "Paste some representative text here..."
ids = tok(sample)["input_ids"]
avg_bytes = len(sample.encode("utf-8")) / len(ids)
print("avg bytes/token:", avg_bytes)



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
要我用一个具体任务示例（带数值）演示 accuracy→centered 的计算吗？




##
保存结果
CORE 结果会写到 base_eval 目录下的一个 CSV 文件。
同时 train/val bpb、CORE metric、sample 文本还会写进 report。
结合你这条命令，实际效果可以理解成：

##
sample：生成几段样本文本
bpb：在 train 和 val 上各看 16384 个 token
core：每个 CORE 任务只抽 16 个样本做快速准确率测试
所以这是一条“快速 sanity check / 近似评估”命令，不是完整 benchmark。--max-per-task=16 很小，--split-tokens=16384 也很小，速度快，但方差会比较大。


##  SFT    `loss: nan` ？

2026-03-13 12:26:36,132 - nanochat.checkpoint_manager - INFO - Loading model from C:\Users\hongqi\.cache\nanochat\base_checkpoints\d8 with step 5000
2026-03-13 12:26:36,175 - nanochat.checkpoint_manager - INFO - Building model with config: {'sequence_len': 512, 'vocab_size': 5120, 'n_layer': 8, 'n_head': 4, 'n_kv_head': 4, 'n_embd': 256, 'window_pattern': 'L'}
Using max_seq_len=512
Using device_batch_size=32
Using total_batch_size=16384
Inherited embedding_lr=0.3 from pretrained checkpoint
Inherited unembedding_lr=0.008 from pretrained checkpoint
Inherited matrix_lr=0.02 from pretrained checkpoint
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Total batch size 16,384 => gradient accumulation steps: 1
Scaling the LR for the AdamW parameters ∝1/√(256/768) = 1.732051
2026-03-13 12:26:37,316 - nanochat.checkpoint_manager - INFO - Loading optimizer state from C:\Users\hongqi\.cache\nanochat\base_checkpoints\d8\optim_005000_rank0.pt
Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)
Training mixture: 2,000 rows (MMLU x3, GSM8K x4)
Step 00000 | Validation bpb: 1.4419
step 00001 (0.04%) | loss: 2.046416 | lrm: 1.00 | dt: 9093.03ms | tok/sec: 1,801 | mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00002 (0.06%) | `loss: nan` | lrm: 1.00 | dt: 8382.55ms | tok/sec: 1,954 | mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00003 (0.08%) | loss: nan | lrm: 1.00 | dt: 8319.38ms | tok/sec: 1,969 | mfu: 0.00 | epoch: 1 | total time: 0.00m



## resume pretrain 

`我要在 base_train.py 中添加 --log-every 参数，并把每步打印包裹到该参数的条件下，然后更新 TODO。` 


python scripts/base_train.py --resume-from-step 1500 --run resume01 --save-every=100 --log-every=10 --num-iterations=14000 --run=dummy --depth=12 --head-dim=64 --window-pattern=SSSL  --max-seq-len=512  --device-batch-size=32 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=50

    --depth=8 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=50 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=50 \
    --num-iterations=5000 \

  
  ##
  (.venv) PS C:\Users\hongqi\myenv\nano_GPT\nanochat> python scripts/base_train.py --resume-from-step 5000 --run resume01 --save-every=500 --log-every=10 --num-iterations=14000 --run=dummy --depth=8 --head-dim=64 --window-pattern=L  --max-seq-len=512  --device-batch-size=32 --total-batch-size=16384 --eval-tokens=524288 --core-metric-every=-1 --sample-every=100

                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    
Autodetected device type: cpu
2026-03-13 13:36:35,060 - nanochat.common - INFO - Distributed world size: 1
COMPUTE_DTYPE: torch.float32 (auto-detected: no CUDA (CPU/MPS))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Vocab size: 5,120
Model config:
{
  "sequence_len": 512,
  "vocab_size": 5120,
  "n_layer": 8,
  "n_head": 4,
  "n_kv_head": 4,
  "n_embd": 256,
  "window_pattern": "L"
}
Resuming optimization from step 5000
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Parameter counts:
wte                     : 1,310,720
value_embeds            : 5,242,880
lm_head                 : 1,310,720
transformer_matrices    : 6,291,648
scalars                 : 16
total                   : 14,155,984
Estimated FLOPs per token: 5.819712e+07
}
Resuming optimization from step 5000
WARNING: torch.compile/torch._dynamo is disabled; running in eager mode
Parameter counts:
wte                     : 1,310,720
value_embeds            : 5,242,880
lm_head                 : 1,310,720
transformer_matrices    : 6,291,648
scalars                 : 16
total                   : 14,155,984
Estimated FLOPs per token: 5.819712e+07
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Scaling weight decay from 0.280000 to 0.151052 for depth 8
Scaling the LR for the AdamW parameters ∝1/√(256/768) = 1.732051
Using user-provided number of iterations: 14,000
Total number of training tokens: 229,376,000
Tokens : `Scaling params ratio: 30.17`       `229,376,000 / （6,291,648+ 1,310,720） =30.1`
}



##
step 05010/14000 (35.79%) | loss: 3.228401 | lrm: 0.99 | dt: 8902.54ms | tok/sec: 1,840 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 34 | total time: 712.03m | eta: 1280.2m
step 05020/14000 (35.86%) | loss: 3.251422 | lrm: 0.99 | dt: 7724.26ms | tok/sec: 2,121 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 35 | total time: 713.38m | eta: 1278.7m
step 05030/14000 (35.93%) | loss: 3.294378 | lrm: 0.99 | dt: 7932.68ms | tok/sec: 2,065 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 35 | total time: 714.67m | eta: 1277.0m
Step 05000 | Validation bpb: 1.253537
<|bos|>The capital of France is a very important part of the history of the country. It is a country that
<|bos|>The chemical symbol of gold is a symbol of the spirituality of the spirituality of the  
<|bos|>If yesterday was Friday, then tomorrow will be the first time I was in the middle of the day. I was in the
<|bos|>The opposite of hot is the same as the hot is the same as the hot is the same as the    
<|bos|>The planets of the solar system are: 1. The solar system is the largest of the solar cells in the world
<|bos|>My favorite color is the color of the eye. I have a lot of colors, but I have
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then


###
step 05090/14000 (36.36%) | loss: 3.334387 | lrm: 0.98 | dt: 8397.48ms | tok/sec: 1,951 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 37 | total time: 722.87m | eta: 1267.9m
<|bos|>The capital of France is a unique and unique way to showcase the beauty of the city. It
<|bos|>The chemical symbol of gold is a symbol of the spirit of the world. It is a symbol of the
<|bos|>If yesterday was Friday, then tomorrow will be a good time to get a new one. The first thing that I have been
`<|bos|>The opposite of hot is the coldest temperature`, which is the temperature of the air. The coldest
<|bos|>The planets of the solar system are: 1. The solar system is the largest solar system in the solar system.
<|bos|>`My favorite color is a reddish-brown color`. I've been using a variety of colors
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then


#
<|bos|>The capital of France is a small, universal, and universal, and is a small
<|bos|>The chemical symbol of gold is a metal, or metal, or metal, which is used in the form of
<|bos|>If yesterday was Friday, then tomorrow will be the first time to get the first time to get the first time to get the
<|bos|>The opposite of hot is the opposite of the opposite of the opposite of the opposite of  
<|bos|>The planets of the solar system are: 1. 1. 2. 3. 4.
<|bos|>My favorite color is a colorful and pleasing color. The color of the color is a color   
<|bos|>If 5*x + 3 = 13, then x is 5*x + 3 = 5*x + 3 =
step 05400/14000 (38.57%) | loss: 3.345565 | lrm: 0.95 | dt: 7475.80ms | tok/sec: 2,191 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 50 | total time: 766.16m | eta: 1222.4m

######
`14:47`
#####

###
Step 05500 | Validation bpb: 1.315124
<|bos|>The capital of France is a popular destination for the United States, and it is a popular destination
<|bos|>The chemical symbol of gold is the most important of the chemical of the chemical of the chemical of the chemical of
<|bos|>If yesterday was Friday, then tomorrow will be a good time to get back to the day. If ye<|bos|>The chemical symbol of gold is the most important of the chemical of the chemical of the chemical of the chemical of
<|bos|>The opposite of hot is the same as the other two. The opposite is the same as the other 
<|bos|>The planets of the solar system are: 1. The solar system is the most important part of the solar system.
<|bos|>My favorite color is a color that is a color that is a color that is a color that is    
<|bos|>If 5*x + 3 = 13, then x is 3.5. The 5*x + 3 = 13


###
step 06190/14000 (44.21%) | loss: 3.296602 | lrm: 0.87 | dt: 7712.54ms | tok/sec: 2,124 | bf16_mfu: 0.
00 | epoch: 1 pq: 2 rg: 81 | total time: 875.92m | eta: 1106.9m
<|bos|>The capital of France is a very important part of the world's history. It is a very important p
art
<|bos|>The chemical symbol of gold is a symbol of the symbolism of the symbolism of the symbolism of t
he
<|bos|>If yesterday was Friday, then tomorrow will be the first time I was able to get a little bit of
 a bit of a
<|bos|>The opposite of hot is the "freezing" of the water. The water is then the water
<|bos|>The planets of the solar system are: 1. 2. 3. 4. 5. 
<|bos|>My favorite color is a bit of a bit of a bit of a bit of a bit of a
<|bos|>If 5*x + 3 = 13, then x is 3 = 13, then x is 3 = 13, then
step 06200/14000 (44.29%) | loss: 3.323267 | lrm: 0.86 | dt: 8140.65ms | tok/sec: 2,012 | bf16_mfu: 0.
00 | epoch: 1 pq: 2 rg: 82 | total time: 877.32m | eta: 1105.5m
step 06210/14000 (44.36%) | loss: 3.317862 | lrm: 0.86 | dt: 7751.74ms | tok/sec: 2,113 | bf16_mfu: 0.
00 | epoch: 1 pq: 2 rg: 82 | total time: 878.69m | eta: 1104.0m




## 单个 Token 的计算量


Read [](file:///c%3A/Users/hongqi/myenv/nano_GPT/nanochat/nanochat/gpt.py#1-1), lines 1 to 400

计算方法来源于 `GPT.estimate_flops()`，要点与逐步计算如下：

公式（代码中直接实现）
- num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops

含义
- 对于“带权重的矩阵乘”参数（即 transformer 矩阵、lm_head 等），按经验把每个参数对应的 FLOPs 计为 6：前向 2（mul+add），反向再大约 4 → 合计 6。所以用 6 × 有效参数数目。
- 排除不计入矩阵乘的参数：embedding（wte）、value_embeds、以及两个标量（resid_lambdas、x0_lambdas）。
- attention 中的 Q@K 等按层单独计入：attn_flops = sum_over_layers 12 * h * q * effective_seq，其中 h = n_head, q = n_embd / n_head, effective_seq = window_size（若无限制则取 sequence_len）。12 是经验系数（包含 forward+backward 与 attention 中额外的乘加开销）。

用你给的数字带入（与你的 checkpoint / config 一致）：
- 参数分组（你给出）：  
  wte = 1,310,720  
  value_embeds = 5,242,880  
  lm_head = 1,310,720  
  transformer_matrices = 6,291,648  
  scalars = 16  
  total = 14,155,984
- 排除项 nparams_exclude = wte + value_embeds + resid/x0 = 1,310,720 + 5,242,880 + 16 = 6,553,616
- 矩阵参数参与项 = total - exclude = 14,155,984 - 6,553,616 = 7,602,368
- 这部分 FLOPs = 6 × 7,602,368 = 45,614,208

注意力部分（假设每层为 full-context，有 8 层，sequence_len=512，n_head=4，n_embd=256 → q=64）：
- per-layer term = 12 * h * q * effective_seq = 12 * 4 * 64 * 512 = 1,572,864
- 8 层累计 attn_flops = 1,572,864 × 8 = 12,582,912

合计：
- num_flops_per_token = 45,614,208 + 12,582,912 = 58,197,120 ≈ 5.819712e7

总结一句话：代码把“所有可训练矩阵参数”按 6 FLOPs/参数计（forward+backward），再加上 attention QK 矩阵乘的按层贡献（12*h*q*effective_seq），把两项相加就得到了你看到的 `5.819712e7。`