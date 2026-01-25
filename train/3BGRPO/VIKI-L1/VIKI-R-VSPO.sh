#!/bin/bash
# VIKI-R training script with VSPO semantic validation integration
# This script integrates VSPO (Validating Semantic Pitfalls in Ontology)
# into the GRPO training pipeline for VIKI-L1

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_TMPDIR=/tmp/ray_tmp
# Ray 日志和临时目录配置 - 增强稳定性
export RAY_LOG_TO_STDERR=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_DEDUP_LOGS_AGG_WINDOW_S=5
export RAY_DASHBOARD_HOST=0.0.0.0
export RAY_DASHBOARD_PORT=8265
export RAY_IGNORE_UNHANDLED_ERRORS=1
# 如果Dashboard仍有问题，可以禁用它
# export RAY_DISABLE_DASHBOARD=1
export PYTHONPATH=/root/miniconda3/lib/python3.12/site-packages:/root/lz::/app/verl:$PYTHONPATH
# Safetensors相关环境变量 - 彻底解决HeaderTooLarge问题
export SAFETENSORS_FAST_GPU=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# 临时禁用GPU加载以避免HeaderTooLarge错误
export CUDA_VISIBLE_DEVICES=""
mkdir -p /tmp/ray_tmp
EXP_NAME='qwen2_5_vl_3b_VIKI_L1_rft_vspo'
OUTPUT_DIR="/path/to/checkpoints/${EXP_NAME}"

# Ray进程和端口清理函数 - 解决端口冲突问题
cleanup_ray() {
    echo "=== 清理Ray进程和端口 ==="
    
    # 停止所有Ray进程
    if command -v ray &> /dev/null; then
        echo "正在停止现有Ray进程..."
        ray stop --force 2>/dev/null || true
        sleep 2
    fi
    
    # 清理Ray临时文件
    if [ -d "/tmp/ray_tmp" ]; then
        echo "清理Ray临时文件..."
        rm -rf /tmp/ray_tmp/* 2>/dev/null || true
    fi
    
    # 查找并终止占用Ray相关端口的进程
    echo "检查并释放Ray相关端口..."
    for port in 44227 38555 8265 10001; do
        local pid=$(lsof -ti:$port 2>/dev/null || fuser $port/tcp 2>/dev/null | awk '{print $1}' || echo "")
        if [ ! -z "$pid" ]; then
            echo "发现端口 $port 被进程 $pid 占用，正在终止..."
            kill -9 $pid 2>/dev/null || true
            sleep 1
        fi
    done
    
    # 清理Python Ray相关进程
    pkill -f "ray::" 2>/dev/null || true
    pkill -f "dashboard" 2>/dev/null || true
    
    echo "✅ Ray清理完成"
    sleep 2
}

# 信号处理 - 确保脚本退出时清理Ray
trap 'echo "收到退出信号，清理Ray进程..."; cleanup_ray; exit' INT TERM EXIT

# 执行Ray清理
cleanup_ray

# VSPO Configuration
# Note: Ensure your dataset has 'data_source' field set to 'viki_1_vspo'
# VSPO parameters can be passed via dataset extra_info or config
VSPO_ENABLED=${VSPO_ENABLED:-true}
VSPO_WEIGHT=${VSPO_WEIGHT:-0.1}
VSPO_MODEL_NAME=${VSPO_MODEL_NAME:-all-MiniLM-L6-v2}
VSPO_THRESHOLD=${VSPO_THRESHOLD:-0.7}

# 模型路径配置 - 基于提示词模版中的模型结构
MODEL_PATH="/app/models/Qwen2.5VL-3B-Instruct-VIKI-R-1"

# 模型完整性检查函数 - 解决SafetensorError: HeaderTooLarge
check_model_integrity() {
    local model_path=$1

    echo "=== 检查模型完整性: ${model_path} ==="

    # 检查目录是否存在
    if [ ! -d "$model_path" ]; then
        echo "❌ 模型目录不存在: ${model_path}"
        echo "请确认模型已正确下载到指定路径"
        return 1
    fi

    # 检查必需的配置文件 (基于提示词模版)
    local required_files=("config.json" "tokenizer_config.json" "model.safetensors.index.json")
    for file in "${required_files[@]}"; do
        if [ ! -f "${model_path}/${file}" ]; then
            echo "❌ 缺少必需文件: ${file}"
            return 1
        fi
    done

    # 检查safetensors文件 (基于提示词模版中的model-00001-of-00002.safetensors等)
    local safetensor_count=$(find "$model_path" -name "*.safetensors" | wc -l)
    if [ "$safetensor_count" -eq 0 ]; then
        echo "❌ 未找到任何safetensors文件"
        echo "模型文件可能损坏或不完整"
        return 1
    fi

    # 检查文件大小是否合理 (至少1MB)
    local total_size=0
    for file in $(find "$model_path" -name "*.safetensors"); do
        local file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        total_size=$((total_size + file_size))
    done

    if [ "$total_size" -lt 1048576 ]; then  # 1MB
        echo "❌ 模型文件总大小异常: ${total_size} bytes"
        echo "模型文件可能损坏，请重新下载"
        return 1
    fi

    echo "✅ 模型完整性检查通过 (${safetensor_count}个safetensors文件, ${total_size} bytes)"
    return 0
}

# 检查模型完整性 - 预防HeaderTooLarge错误
if ! check_model_integrity "$MODEL_PATH"; then
    echo ""
    echo "🔧 修复建议:"
    echo "1. 检查模型文件是否完整下载"
    echo "2. 验证磁盘空间是否充足"
    echo "3. 确认文件权限设置正确"
    echo "4. 如有必要，重新下载模型文件"
    echo ""
    echo "模型应包含以下文件 (基于提示词模版):"
    echo "  - config.json, tokenizer_config.json, model.safetensors.index.json"
    echo "  - model-00001-of-00002.safetensors, model-00002-of-00002.safetensors"
    echo "  - generation_config.json, preprocessor_config.json"
    echo "  - tokenizer.json, vocab.json, merges.txt, chat_template.json"
    exit 1
fi

# 模型验证完成后重新启用GPU用于训练
export CUDA_VISIBLE_DEVICES="0,1"
echo "🚀 启动训练，使用GPU: $CUDA_VISIBLE_DEVICES"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/app/viki/VIKI-L1/train.parquet \
    data.val_files=/app/viki/VIKI-L1/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='VIKI-L1_3b_VSPO' \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=5 \
    +reward_model.reward_kwargs.vspo_enabled=${VSPO_ENABLED} \
    +reward_model.reward_kwargs.vspo_weight=${VSPO_WEIGHT} \
    +reward_model.reward_kwargs.vspo_model_name=${VSPO_MODEL_NAME} \
    +reward_model.reward_kwargs.vspo_threshold=${VSPO_THRESHOLD} \
    $@
