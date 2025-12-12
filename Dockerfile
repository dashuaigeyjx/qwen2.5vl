# 基于 Ubuntu 22.04 镜像
FROM ubuntu:22.04

# 替换 Debian 镜像源为阿里云（加速系统依赖下载）
RUN sed -i 's@archive.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirrors.aliyun.com@g' /etc/apt/sources.list

# 安装系统基础工具和 Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    gnupg2 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 安装 NVIDIA CUDA 12.6 和 cuDNN
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-6 libcudnn8 libcudnn8-dev && \
    rm -f cuda-keyring_1.0-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# 设置CUDA和cuDNN环境变量
ENV PATH=/usr/local/cuda-12.6/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV CUDNN_HOME=/usr/local/cuda-12.6
ENV CUDNN_INCLUDE_PATH=/usr/local/cuda-12.6/include
ENV CUDNN_LIBRARY_PATH=/usr/local/cuda-12.6/lib64

# 安装 Miniconda（管理 Conda 环境）
RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash

# 配置 Conda 国内镜像（清华源，加速下载）
RUN /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    /opt/conda/bin/conda config --set show_channel_urls yes

# 设置工作目录为 /app（与挂载路径对应）
WORKDIR /app

# 复制项目文件到镜像（包括 requirements.txt、roboviki.yml、verl 目录等）
COPY . .

# 跳过conda环境创建，直接使用pip安装依赖
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 配置 pip 镜像（清华源）并安装依赖
RUN /opt/conda/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    /opt/conda/bin/pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    /opt/conda/bin/pip install --upgrade pip && \
    /opt/conda/bin/pip install aiohttp==3.9.0


# 安装 requirements.txt 中的依赖
RUN /opt/conda/bin/pip install --no-cache-dir -r requirements.txt

# 安装 verl 框架（可编辑模式）
RUN cd verl && /opt/conda/bin/pip install -e .

# 验证CUDA和cuDNN安装
RUN nvcc --version && \
    /opt/conda/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

# 安装指定版本的 FlashAttention（适配 CUDA 12.6）
RUN /opt/conda/bin/pip install flash-attn --no-build-isolation

# 设置环境变量
ENV PYTHONPATH=/app
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTENTION
ENV HF_HOME=/home/dataset/huggingface
ENV HF_DATASETS_CACHE=/home/dataset/huggingface/datasets
ENV TRANSFORMERS_CACHE=/home/dataset/huggingface/transformers
ENV HF_ENDPOINT=https://hf-mirror.com

# 暴露端口（与启动命令映射一致）
EXPOSE 9900

# 启动命令：直接执行训练脚本
CMD ["bash", "train/7BGRPO/VIKI-L1/VIKI-R.sh"]
