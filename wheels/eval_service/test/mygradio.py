import json
import gradio as gr
from openai import OpenAI
from transformers import AutoTokenizer

# ========== 模型提示 ==========
MY_PREFIX = """Answer the given question by following the structured reasoning process below.
1. Always begin with <think> ... </think> for a brief internal reasoning sketch.
2. If your internal knowledge is sufficient, skip searching and directly provide: Summarize your internal reasoning with <reflect>...</reflect> and final concise answer with <answer>...</answer>
3. If your internal knowledge is not sufficient, Generate a short, keyword-based retrieval query inside tag <search> ... </search>.
The environment will return results in <information> ... </information>.
If the retrieved information is reliable and sufficient, integrate it with internal reasoning inside <reflect>...</reflect> and final concise answer with <answer>...</answer>.
If the retrieved information is insufficient or conflicts with internal knowledge, then explain the issue with <reflect></reflect> and search again.
You can search as many times as you want.
The final answer inside <answer> must always be concise and decisive.
For example: <answer> Beijing </answer>.
Question: """


# ========== 工具函数 ==========
def count_search_tags(response_content):
    """统计检索标签次数并返回内容"""
    start_tag, end_tag = "<search>", "</search>"
    starts = response_content.count(start_tag)
    ends = response_content.count(end_tag)
    contents = []

    if starts != ends:
        print("[WARN] <search> 和 </search> 标签数量不匹配")
        return 0, []

    for _ in range(starts):
        s_idx = response_content.find(start_tag)
        e_idx = response_content.find(end_tag, s_idx)
        if s_idx == -1 or e_idx == -1:
            break
        content = response_content[s_idx + len(start_tag):e_idx].strip()
        contents.append(content)
        response_content = response_content[e_idx + len(end_tag):]
    return starts, contents


# ========== 模型推理函数 ==========
def ask_model(question, temperature=0.7, max_tokens=32768, top_p=1.0):
    """调用本地模型进行推理"""
    model_name = "/root/aliyunpan-v0.3.7-linux-amd64/Downloads/9e25f2886bc841b7a99b478648e2393e/rag-qwen3-8b/huggingface"
    base_url = "http://localhost:5001"
    api_key = "sk-proj-1234567890"

    client = OpenAI(api_key=api_key, base_url=base_url)

    chat_messages = [
        {"role": "system", "content": MY_PREFIX},
        {"role": "user", "content": question}
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        response_content = completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] 模型推理失败: {e}", "", "", 0

    # 解析 <answer>
    if "<answer>" in response_content and "</answer>" in response_content:
        prediction = response_content.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        prediction = "(未检测到 <answer> 标签)"

    # 统计 <search> 标签
    search_count, search_contents = count_search_tags(response_content)
    search_text = "\n".join(search_contents) if search_contents else "无"

    return prediction, response_content, search_text, search_count


# ========== Gradio 前端 ==========
def run_interface():
    with gr.Blocks(title="RAG-Qwen3-8B Reasoning Demo") as demo:
        gr.Markdown("RAG-Qwen3-8B 智能问答演示\n输入你的问题，模型将展示推理结构、检索次数和最终答案。")

        with gr.Row():
            with gr.Column(scale=2):
                question_box = gr.Textbox(label="输入你的问题", placeholder="请输入问题...", lines=3)
                temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="temperature")
                submit_btn = gr.Button("生成回答", variant="primary")

            with gr.Column(scale=3):
                answer_box = gr.Textbox(label="最终答案 <answer>", lines=2)
                search_count_box = gr.Number(label="检索次数 <search> 标签计数")
                search_text_box = gr.Textbox(label="检索内容 <search> 内文本", lines=4)
                raw_output_box = gr.Textbox(label="完整模型输出", lines=10)

        submit_btn.click(
            fn=ask_model,
            inputs=[question_box, temperature],
            outputs=[answer_box, raw_output_box, search_text_box, search_count_box],
        )

        gr.Markdown("---\n© 2025 Qwen3 RAG Interactive Demo")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    run_interface()
