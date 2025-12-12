import json
import os
import random
from datetime import datetime
from openai import OpenAI
from transformers import AutoTokenizer

# ========== 模型提示 ==========
MY_PREFIX = """Answer the given question by following the structured reasoning process below. 
1. Always begin with <think> ... </think> for a brief internal reasoning sketch. 
2. If your internal knowledge is sufficient, skip searching and directly provide: Summarize your internal reasoning with <reflect>...</reflect> and final concise answer with <answer>...</answer> 
3. If your internal knowledge is not sufficient, Generate a short, keyword-based retrieval query inside tag <search> ... </search>. The environment will return results in <information> ... </information>. 
If the retrieved information is reliable and sufficient, integrate it with internal reasoning inside with <reflect>...</reflect> and final concise answer with <answer>...</answer> 
If the retrieved information is insufficient or conflicts with internal knowledge, then explain the issue with <reflect> </reflect> and search again. 
You can search as many times as you want.
The <search> block must only contain concise, factual keywords that are directly relevant to the query. These should include:
- Specific time expressions 
- Named entities
- Key factual event phrases
Do NOT include full sentences, question words, or reasoning terms in <search>. Specifically, avoid using words such as:
- "why", "what", "how", "when", "who", "whose", "which", "where"
The <search> block should never resemble a question. It should always be a concise keyword query designed to retrieve specific, factual information. Avoid redundant queries that only change a single word without adding new information.
The final answer inside <answer> must always be concise and decisive. For example: <answer> Beijing </answer>. Question:"""


# ========== 数据路径配置 ==========
INPUT_FILES = [
    "/root/hzr/FlashRAG-main-0.1.4/dataset/nq/test_decoded.jsonl",
    "/root/hzr/FlashRAG-main-0.1.4/dataset/popqa/test_decoded.jsonl",
    "/root/hzr/FlashRAG-main-0.1.4/dataset/triviaqa/test_decoded.jsonl",
    "/root/hzr/FlashRAG-main-0.1.4/dataset/musique/dev_decoded.jsonl",
    "/root/hzr/FlashRAG-main-0.1.4/dataset/2wikimultihopqa/dev_decoded.jsonl",
    "/root/hzr/FlashRAG-main-0.1.4/dataset/bamboogle/test_decoded.jsonl",
    "/root/hzr/FlashRAG-main-0.1.4/dataset/hotpotqa/dev_decoded.jsonl",
]


# ========== 工具函数 ==========
def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def count_search_tags(response_content):
    """统计检索标签次数并返回内容"""
    start_tag, end_tag = "<search>", "</search>"
    starts = response_content.count(start_tag)
    ends = response_content.count(end_tag)
    contents = []

    if starts != ends:
        print("[ERROR] <search> 和 </search> 标签数量不匹配")
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


def chat_with_token_logging(normalized_golden, normalized_prediction, client, model_name):
    """语义评测 + token 统计"""
    prompt = (
        "You are an evaluation assistant.\n"
        "Determine whether the prediction semantically matches any of the reference answers.\n"
        "Be flexible: if the meaning is the same, consider it a match.\n\n"
        f"[Prediction]: {normalized_prediction.strip()}\n"
        f"[Reference Answers]: {normalized_golden}\n\n"
        "Answer YES if semantically equivalent, NO otherwise."
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
    )

    pred_text = response.choices[0].message.content.strip()
    usage = getattr(response, "usage", None)
    tokens = usage.total_tokens if usage else 0

    return pred_text, tokens


# ========== 核心评测逻辑 ==========
def evaluate_dataset(file_path, client, eval_client, model_name, evaluation_model, temperature, max_tokens, top_p, n):
    dataset_name = os.path.basename(os.path.dirname(file_path))  # ✅ 提取数据集名称
    print(f"\n=== 开始评测数据集: {dataset_name} ===")

    data = read_jsonl(file_path)

    # ✅ 随机抽取 1000 条样本（seed 固定）
    random.seed(42)
    sample_size = min(1000, len(data))
    data = random.sample(data, sample_size)

    results = []
    total_searches = 0
    total_tokens_used = 0
    correct_predictions = 0

    for idx, item in enumerate(data):
        question = item.get('question', '')
        golden_answers = item.get('golden_answers', [])
        print(f"[{idx + 1}/{len(data)}] {question[:80]}...")

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
                n=n,
            )
            response_content = completion.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] 模型推理失败: {e}")
            continue

        # 提取 <answer>
        if "<answer>" in response_content and "</answer>" in response_content:
            prediction = response_content.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            prediction = ""

        # 检索统计
        search_count, search_contents = count_search_tags(response_content)
        total_searches += search_count

        # 语义评测
        try:
            is_correct, token_used = chat_with_token_logging(
                normalized_golden=" ".join(golden_answers),
                normalized_prediction=prediction,
                client=eval_client,
                model_name=evaluation_model,
            )
        except Exception as e:
            print(f"[ERROR] 评测失败: {e}")
            is_correct, token_used = "NO", 0

        total_tokens_used += token_used
        if is_correct.strip().upper() == "YES":
            correct_predictions += 1

        results.append({
            "index": idx + 1,
            "question": question,
            "golden_answers": golden_answers,
            "prediction": prediction,
            "is_correct": is_correct,
            "search_count": search_count,
            "search_contents": search_contents,
            "token_used": token_used,
        })

    # 汇总结果
    accuracy = correct_predictions / len(data) * 100 if data else 0
    avg_searches = total_searches / len(data) if data else 0
    summary = {
        "dataset": dataset_name,
        "sample_size": sample_size,
        "correct_predictions": correct_predictions,
        "accuracy": f"{accuracy:.2f}%",
        "total_searches": total_searches,
        "avg_searches_per_sample": round(avg_searches, 2),
        "total_tokens_used": total_tokens_used,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    results.append({"summary": summary})

    # ✅ 文件名只用数据集名，如 evaluation_result_nq.json
    output_path = f"evaluation_result_qwen3_8b_{dataset_name}_118.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ {dataset_name} 完成，结果已保存至 {output_path}")
    return summary


# ========== 主函数 ==========
def main():
    model_name = "/root/aliyunpan-v0.3.7-linux-amd64/Downloads/9e25f2886bc841b7a99b478648e2393e/117-2/huggingface"
    base_url = "http://localhost:5001"
    evaluation_base_url = "http://117.145.66.250:8/v1"
    evaluation_model = "/models/Qwen3-235-Instruct"
    #evaluation_base_url = "http://172.20.20.16:9997/v1"
    #evaluation_model = "qwen3"
    api_key = "sk-proj-1234567890"
    temperature = 0.7
    max_tokens = 32768
    top_p = 1.0
    n = 1

    client = OpenAI(api_key=api_key, base_url=base_url)
    eval_client = OpenAI(api_key=api_key, base_url=evaluation_base_url)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_summaries = []
    for file_path in INPUT_FILES:
        try:
            summary = evaluate_dataset(
                file_path=file_path,
                client=client,
                eval_client=eval_client,
                model_name=model_name,
                evaluation_model=evaluation_model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"[ERROR] 处理文件 {file_path} 时出错: {e}")
            continue

    # ✅ 保存总体汇总
    summary_path = "evaluation_summary_qwen3_8b_all.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\n=== 全部数据集评测完成 ===")
    for s in all_summaries:
        print(f"{s['dataset']}: {s['accuracy']} (avg searches {s['avg_searches_per_sample']})")
    print(f"所有汇总已保存至 {summary_path}")


if __name__ == "__main__":
    main()
