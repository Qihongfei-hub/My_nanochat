#!/usr/bin/env pwsh

Write-Host "=== NanoChat 依赖安装脚本 ==="
Write-Host ""

# 检查Python版本
Write-Host "检查Python版本..."
python --version

# 升级pip
Write-Host "升级pip..."
python -m pip install --upgrade pip

# 安装基本依赖
Write-Host ""
Write-Host "安装基本依赖..."
python -m pip install "datasets>=4.0.0" "fastapi>=0.117.1" "ipykernel>=7.1.0" "kernels>=0.11.7" "matplotlib>=3.10.8" "psutil>=7.1.0" "python-dotenv>=1.2.1" "regex>=2025.9.1" "rustbpe>=0.1.0" "scipy>=1.15.3" "setuptools>=80.9.0" "tabulate>=0.9.0" "tiktoken>=0.11.0" "tokenizers>=0.22.0" "transformers>=4.57.3" "uvicorn>=0.36.0" "wandb>=0.21.3" "zstandard>=0.25.0"

# 选择PyTorch版本
Write-Host ""
Write-Host "请选择PyTorch版本："
Write-Host "1. CPU版本"
Write-Host '2. GPU版本 (CUDA 12.8)'
$choice = Read-Host '输入选项编号 (1-2)'



Write-Host "安装PyTorch GPU版本..."
python -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
python -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128

##  `work `
python -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-cache-dir
# 询问
### 是
conda run -n my_project_env python -m pip uninstall -y torch
##否安装开发依赖
conda run -n my_project_env python -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
conda run -n my_project_env python -m pip install "torch==2.9.1" -i https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/

#验证安装是否成功
conda run -n my_project_env python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

Write-Host "安装开发依赖..."
python -m pip install "pytest>=8.0.0"


# 安装项目本身
Write-Host ""
Write-Host "安装项目本身..."
python -m pip install -e .

Write-Host ""
Write-Host "=== 依赖安装完成 ==="
Write-Host "可以通过运行 'python scripts/chat_web.py' 启动Web界面"
Write-Host "或运行 'python scripts/chat_cli.py' 启动命令行界面"
