# VIKI-L1 GRPO 强化学习微调启动指南

## 📋 前置条件检查

### 1. 环境准备

```bash
# 激活conda环境
conda activate roboviki
# 或使用完整路径
source /opt/conda/envs/roboviki/bin/activate

# 验证Python环境
python3 --version  # 应该是Python 3.10+
```

### 2. 依赖安装

```bash
# 安装VSPO相关依赖（如果使用VSPO功能）
pip install sentence-transformers

# 验证verl框架已安装
python3 -c "import verl; print(verl.__version__)"
```

### 3. 数据准备

确保数据集文件存在：
- 训练集：`/app/VIKI-R/VIKI-L1/train.parquet`
- 测试集：`/app/VIKI-R/VIKI-L1/test.parquet`

**重要**：如果使用VSPO功能，确保数据集中每个样本包含：
- `data_source` 字段设置为 `'viki_1_vspo'`
- `reward_model.ground_truth` 字段包含标准答案

### 4. 模型检查点准备

确保SFT模型检查点存在：
```
/path/to/models/qwen2.5_vl-3b/full/viki_1_sft_500/checkpoint-62
```

## 🚀 启动训练

### 方式1：标准GRPO训练（不含VSPO）

```bash
cd qwen2.5vl/train/3BGRPO/VIKI-L1

# 修改脚本中的路径配置
# 1. 修改数据集路径（第11-12行）
# 2. 修改模型路径（第19行）
# 3. 修改输出目录（第7行）

# 启动训练
bash VIKI-R.sh
```

### 方式2：VSPO集成的GRPO训练

```bash
cd qwen2.5vl/train/3BGRPO/VIKI-L1

# 修改脚本中的路径配置
# 1. 修改数据集路径（第24-25行）
# 2. 修改模型路径（第33行）
# 3. 修改输出目录（第12行）

# 可选：配置VSPO参数（环境变量）
export VSPO_ENABLED=true
export VSPO_WEIGHT=0.1
export VSPO_MODEL_NAME=all-MiniLM-L6-v2
export VSPO_THRESHOLD=0.7

# 启动训练
bash VIKI-R-VSPO.sh
```

### 方式3：使用自定义参数启动

```bash
cd qwen2.5vl/train/3BGRPO/VIKI-L1

# 直接传递参数覆盖默认值
bash VIKI-R-VSPO.sh vllm \
    data.train_files=/your/path/to/train.parquet \
    data.val_files=/your/path/to/test.parquet \
    actor_rollout_ref.model.path=/your/model/path \
    trainer.default_local_dir=/your/output/path \
    trainer.n_gpus_per_node=2
```

## ⚙️ 关键配置说明

### GPU配置

根据你的硬件（2张NVIDIA 4090 D，每张21GB显存）：

```bash
trainer.n_gpus_per_node=2  # 使用2张GPU
actor_rollout_ref.rollout.gpu_memory_utilization=0.6  # 每张GPU使用60%显存
```

### 训练参数调整

```bash
# 批次大小（根据显存调整）
data.train_batch_size=256
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10

# 学习率
actor_rollout_ref.actor.optim.lr=1e-6

# 训练轮数
trainer.total_epochs=5
```

### VSPO参数（仅VSPO脚本）

```bash
# VSPO权重（默认0.1，即10%）
VSPO_WEIGHT=0.1

# Sentence transformer模型（默认all-MiniLM-L6-v2）
VSPO_MODEL_NAME=all-MiniLM-L6-v2

# 语义相似度阈值（默认0.7）
VSPO_THRESHOLD=0.7
```

## 📊 监控训练

### WandB日志

训练会自动记录到WandB（如果配置了`trainer.logger=['console','wandb']`）：

```bash
# 确保设置了WANDB_API_KEY
export WANDB_API_KEY=your_api_key
```

### 控制台输出

训练过程中会输出：
- 每个epoch的进度
- 奖励分数统计
- 模型保存信息

### 检查点保存

检查点会保存在：
```
/path/to/checkpoints/${EXP_NAME}/checkpoint-{step}/
```

保存频率由 `trainer.save_freq=100` 控制。

## 🔧 常见问题

### 1. CUDA内存不足

**解决方案**：
- 减小 `data.train_batch_size`
- 减小 `actor_rollout_ref.rollout.gpu_memory_utilization`
- 启用 `actor_rollout_ref.actor.fsdp_config.param_offload=True`

### 2. Ray初始化失败

**解决方案**：
```bash
# 清理Ray临时文件
rm -rf /tmp/ray_tmp
mkdir -p /tmp/ray_tmp
export RAY_TMPDIR=/tmp/ray_tmp
```

### 3. 数据集路径错误

**解决方案**：
- 检查parquet文件是否存在
- 确认文件路径权限
- 验证数据格式正确

### 4. VSPO模型加载失败

**解决方案**：
```bash
# 安装sentence-transformers
pip install sentence-transformers

# 或禁用VSPO
export VSPO_ENABLED=false
```

## 📝 训练脚本参数说明

### 必需参数

- `data.train_files`: 训练数据集路径
- `data.val_files`: 验证数据集路径
- `actor_rollout_ref.model.path`: SFT模型检查点路径
- `trainer.default_local_dir`: 输出目录

### 可选参数

- `algorithm.adv_estimator`: 优势估计器（默认grpo）
- `trainer.n_gpus_per_node`: 每节点GPU数量
- `trainer.total_epochs`: 训练轮数
- `reward_model.reward_kwargs.*`: VSPO相关参数

## 🎯 快速启动示例

```bash
# 1. 激活环境
conda activate roboviki

# 2. 进入训练目录
cd qwen2.5vl/train/3BGRPO/VIKI-L1

# 3. 修改脚本中的路径（使用编辑器）
# vi VIKI-R.sh 或 nano VIKI-R.sh

# 4. 启动训练
bash VIKI-R.sh

# 或使用VSPO版本
bash VIKI-R-VSPO.sh
```

## 📚 相关文档

- GRPO算法说明：`qwen2.5vl/verl/verl/trainer/ppo/core_algos.py`
- VSPO集成说明：`qwen2.5vl/train/3BGRPO/VIKI-L1/VSPO_INTEGRATION_README.md`（如果存在）
- 项目README：`qwen2.5vl/README.md`
