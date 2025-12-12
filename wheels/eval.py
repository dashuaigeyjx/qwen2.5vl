from evalscope import TaskConfig, run_task
task_cfg = TaskConfig(
    model='/models/Qwen3-235-Instruct',
    api_url='http://117.145.66.250:8/v1/chat/completions',
    eval_type='openai_api',
    datasets=[
        'data_collection',
    ],
    dataset_args={
        'data_collection': {
            'dataset_id': 'evalscope/Qwen3-Test-Collection',
            'shuffle': True,
        }
    },
    eval_batch_size=32,
    generation_config={
        'max_tokens': 32768,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.7,  # 采样温度 (qwen 报告推荐值)
        'top_p': 0.8,  # top-p采样 (qwen 报告推荐值)
        'top_k': 20,  # top-k采样 (qwen 报告推荐值)
    },
    timeout=60000,  # 超时时间
    stream=True,  # 是否使用流式输出
    #limit=1,  # 设置为100条数据进行测试
)

run_task(task_cfg=task_cfg)