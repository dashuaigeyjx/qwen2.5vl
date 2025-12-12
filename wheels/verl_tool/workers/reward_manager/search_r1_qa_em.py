"""
Search-R1 style QA Exact Match Reward Manager
"""
import torch
import random
import regex as re
import json
import time
import os
import time
import os
import numpy as np
from typing import Dict, Any
from verl import DataProto
from verl.workers.reward_manager.registry import register
from .reward_score import _default_compute_score
from collections import defaultdict
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained("/home/unnet/project/model/Qwen3-1.7B")
import re
import string
import openai
import random
import math

def normalize_answer(s: str) -> str:
    """更强的归一化：
    - 转小写
    - 去标点
    - 去冠词
    - 去空格
    - 标准化数字（1,000 -> 1000）
    - 货币符号统一为 'usd' / 'eur' / 'yuan'
    - 百分号 -> 'percent'
    """

    def lower(text):
        return text.lower()

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation) - set(["%", "$", "€", "¥"])  # 保留常见货币/百分号
        return ''.join(ch for ch in text if ch not in exclude)

    def normalize_numbers(text):
        # 去掉千分位逗号：1,000 -> 1000
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
        return text

    def normalize_currency(text):
        text = text.replace("$", " usd ")
        text = text.replace("€", " eur ")
        text = text.replace("¥", " yuan ")
        return text

    def normalize_percent(text):
        text = text.replace("%", " percent ")
        return text

    if s is None:
        return ""
    text = lower(s)
    text = normalize_numbers(text)
    text = normalize_currency(text)
    text = normalize_percent(text)
    text = remove_articles(text)
    text = remove_punc(text)
    text = white_space_fix(text)

    return text.strip()


import re
from openai import OpenAI

def evaluate_search_block_global(text: str):
    """
    评估模型输出中所有 <search>...</search> 的合理性及多样性，并给出扣分逻辑。

    评分规则（S1版本，仅-1或0）：
      - 0 次搜索：score = 0
      - 1 次搜索：若包含疑问词（why/what/how/...）→ score = -1，否则 0分
      - ≥2 次搜索：大模型判断语义是否重复 → 重复 score = -1，否则 0分

    返回字段：
      search_blocks: List[str]
      is_effective_search: "YES" / "NO"
      is_comparable_search: "null" / "YES" / "NO"
      score: -1 or 0
      reason: str 扣分解释
      usage_summary: token消耗
    """

    client = OpenAI(api_key="null", base_url="http://117.145.66.250:8/v1")

    # === Step 1: 提取所有 <search> 块 ===
    search_matches = re.findall(r"<search>(.*?)</search>", text, re.DOTALL | re.IGNORECASE)
    search_blocks = [s.strip() for s in search_matches if s.strip()]

    # === Step 2: 无搜索 → score = 0 ===
    if not search_blocks:
        return {
            "search_blocks": [],
            "is_effective_search": "NO",
            "is_comparable_search": "null",
            "score": 0,
            "reason": "No search performed, no penalty applied.",
            "usage_summary": {}
        }

    # === Step 3: 单次搜索判定 ===
    question_words = ["why", "what", "how", "when", "where", "who", "whose", "which"]
    invalid_found = any(qw in search_blocks[0].lower() for qw in question_words) if len(search_blocks) == 1 else False

    if len(search_blocks) == 1:
        if invalid_found:
            return {
                "search_blocks": search_blocks,
                "is_effective_search": "NO",
                "is_comparable_search": "null",
                "score": -1,
                "reason": "Single search contains invalid question words, penalized.",
                "usage_summary": {}
            }
        else:
            return {
                "search_blocks": search_blocks,
                "is_effective_search": "YES",
                "is_comparable_search": "null",
                "score": 0,
                "reason": "Single valid search, no penalty.",
                "usage_summary": {}
            }


    # === Step 4: 多次搜索 → 调用大模型判断内容是否重合 ===
    joined_blocks = "\n".join([f"{i+1}. {blk}" for i, blk in enumerate(search_blocks)])
    prompt = (
    "You are a content comparison assistant.\n"
    "You are given multiple search queries produced by an LLM during multi-turn reasoning.\n"
    "Your task is to judge whether the search queries contain overlapping content or redundancy.\n\n"
    "Guidelines:\n"
    "- If the queries are nearly identical, such as having the same wording or very similar phrases → answer NO (redundant).\n"
    "- If the queries mention the same topic with different wordings but convey the same information → answer NO (redundant).\n"
    "- If the queries ask about distinct aspects, topics, or details → answer YES (diverse).\n\n"
    "Queries:\n"
    f"{joined_blocks}\n\n"
    "Respond in the following format:\n"
    "Reasoning: <brief>\n"
    "Judgement: YES or NO"
    )

    response, result_text = None, "No diversity check performed"
    usage = {}

    try:
        response = client.chat.completions.create(
            model="/models/Qwen3-235-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        result_text = response.choices[0].message.content.strip()
        usage = response.usage or {}
        if "YES" in result_text.upper():
            comparable_judge = "YES"
            score = 1
            reason = "Multiple searches are diverse, no penalty."
        else:
            comparable_judge = "NO"
            score = -2
            reason = "Multiple searches redundant, penalized."
    except Exception as e:
        print(f"[ERROR] LLM diversity check failed: {e}")
        comparable_judge = "NO"
        score = -1
        reason = "Model call failed; treat as redundant for safety."

    usage_summary = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }

    return {
        "search_blocks": search_blocks,
        "is_effective_search": "YES" if not invalid_found else "NO",
        "is_comparable_search": comparable_judge,
        "score": score,
        "reason": reason,
        "usage_summary": usage_summary
    }



def chat_with_token_logging(normalized_golden, normalized_prediction):
    client = openai.OpenAI(api_key="null", base_url="http://117.145.66.250:8/v1")
    prompt = (
        "You are an evaluation assistant.\n"
        "Your task is to determine whether the given prediction semantically matches any of the reference answers. "
        "This is a flexible match: even if the wording is different, if the meaning is the same, consider it a match.\n\n"
        f"[Prediction]: {normalized_prediction.strip()}\n"
        f"[Reference Answers]: {normalized_golden}\n\n"
        "Please answer with a single word:\n"
        "- YES → if the prediction expresses the same meaning as any of the reference answers.\n"
        "- NO → otherwise.\n"
        "Only respond with YES or NO."
    )
    response = client.chat.completions.create(
        model="/models/Qwen3-235-Instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=128,
    )
    # 提取回答文本
    pred_text = response.choices[0].message.content
    # pred_rea_text = getattr(response.choices[0].message, "reasoning_content", "")
    # 提取 token 用量
    usage = response.usage or {}
    prompt_tokens = getattr(usage, "prompt_tokens", 0)
    completion_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", 0)

    # 打印本次调用的 token 信息
    # print(f"[Token Usage] prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")

    # 返回回答和详细用量
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
    return pred_text, usage_dict


def em_check(prediction, golden_answers):
    """宽松版本：只要标准答案出现在模型预测里就算对"""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    print("========== [DEBUG] em_check ==========")
    print(f"[RAW PREDICTION] {prediction}")
    print(f"[RAW GOLDEN] {golden_answers}")

    normalized_prediction = normalize_answer(prediction)
    print(f"[NORMALIZED PREDICTION] {normalized_prediction}")

    for golden_answer in golden_answers:
        normalized_golden = normalize_answer(golden_answer)
        print(f"[NORMALIZED GOLDEN] {normalized_golden}")
        if normalized_golden in normalized_prediction:  # 允许包含匹配
            print("[MATCH] Prediction contains golden answer ✓")
            return 1
        pred, usage = chat_with_token_logging(normalized_golden, normalized_prediction)
        if pred == "YES" or pred == "yes":
            print("[MATCH] LLM Prediction YES ✓")
            return 1

    print("[NO MATCH] Prediction does not contain golden answers ✗")
    return 0


def extract_solution(solution_str: str) -> str:
    """Extract the final answer from <answer> tags in the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")
    return opening_tags, closing_tags


import re
from typing import List


# ------- 重复惩罚模块（仅作用于 <think>...</think>） -------- #


def compute_think_repetition_penalty(text: str, n=3, max_penalty=-1.0) -> float:
    """
    计算 <think> 与 <reflect> 中的 n-gram 重复惩罚。

    惩罚越大（更负）说明重复越多。
    当没有标签或长度太短时返回 0。
    """
    # === Step 1: 抽取 <think> 和 <reflect> 内容 ===
    segments = re.findall(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    segments += re.findall(r"<reflect>(.*?)</reflect>", text, re.DOTALL | re.IGNORECASE)

    if not segments:
        return 0.0

    total_penalty = 0.0
    total_segments = 0

    # === Step 2: 分别计算每个段落的重复率 ===
    for content in segments:
        tokens = content.lower().split()
        if len(tokens) < n:
            continue

        ngram_total = 0
        ngram_set = set()

        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i + n])
            ngram_set.add(ng)
            ngram_total += 1

        if ngram_total == 0:
            continue

        repetition_ratio = 1 - len(ngram_set) / ngram_total  # 重复比例 ∈ [0, 1]
        penalty = repetition_ratio * max_penalty
        total_penalty += penalty
        total_segments += 1

    # === Step 3: 平均惩罚（避免长段落权重过大） ===
    if total_segments > 0:
        total_penalty /= total_segments

    return round(total_penalty, 4)


def extract_model_output(solution_str: str) -> str:
    """
    Extract the model-generated output from a solution_str that includes prompt + response.
    We assume the model output starts after the special token '<｜Assistant｜>'.
    """
    assistant_marker = "<|im_start|>assistant"
    #assistant_marker = "<｜Assistant｜>"
    if assistant_marker in solution_str:
        # 截取 Assistant 部分
        return solution_str.split(assistant_marker, 1)[1].strip()
    else:
        # 如果没有明确分隔符，就默认整段都是模型输出
        return solution_str.strip()


import re

def check_tag_constraint_and_count(text: str):
    """
    检查模型输出格式是否合法，并统计检索次数。

    限制：
      - think 段内不允许包含其他标签（search / information / reflect / answer）
      - 仅支持以下两种合法结构：
          ① <think>...</think> + [ (<search>...</search><information>...</information><reflect>...</reflect>)* ] + <answer>...</answer>
          ② <think>...</think><reflect>...</reflect><answer>...</answer>

    返回:
        is_valid (bool): 是否结构合法
        search_count (int): 搜索块数量
    """
    if not text or not isinstance(text, str):
        return False, 0

    # === Step 1: 检查 think 和 answer 唯一性 ===
    required_pairs = ["think", "answer"]
    for tag in required_pairs:
        open_tag, close_tag = f"<{tag}>", f"</{tag}>"
        if text.count(open_tag) != 1 or text.count(close_tag) != 1:
            print(f"[ERROR] 标签 <{tag}> 次数错误: open={text.count(open_tag)}, close={text.count(close_tag)}")
            return False, 0

    # === Step 2: 检查 think 内部是否嵌套非法标签 ===
    think_content = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_content:
        inside = think_content.group(1)
        if re.search(r"<(information|reflect|answer)>", inside):
            print("[ERROR] think 段中包含非法标签（information/reflect/answer）！")
            return False, 0

    # === Step 3: 匹配两种合法模式 ===
    # ✅ 有检索结构（允许多轮 search → info → reflect）
    pattern_with_search = (
        r"^.*?<think>.*?</think>"  # think 段
        r"(?:\s*<search>.*?</search>\s*"  # 一轮 search
        r"<information>.*?</information>\s*"  # 一轮 information
        r"<reflect>.*?</reflect>\s*)+"  # 一轮 reflect（至少一次）
        r"<answer>.*?</answer>.*?$"  # 结尾 answer
    )

    # ✅ 无检索结构（仅 think → reflect → answer）
    #pattern_no_search = (r"^.*?<think>.*?</think>\s*<reflect>.*?</reflect>\s*<answer>.*?</answer>.*?$")
    # pattern_no_search = (r"^.*?<think>.*?</think>\s*<reflect>.*?</reflect>\s*(?!<search>).*?<answer>.*?</answer>.*?$")
    pattern_no_search = (r"^(?!.*<search>).*?<think>.*?</think>\s*<reflect>.*?</reflect>\s*<answer>.*?</answer>.*?$")

    # === Step 4: 判断匹配结构类型 ===
    if re.search(pattern_with_search, text, re.DOTALL):
        search_blocks = re.findall(r"<search>.*?</search>", text, re.DOTALL)
        search_count = len(search_blocks)

        first_reflect = text.find("<reflect>")
        first_search = text.find("<search>")
        if 0 <= first_reflect < first_search:
            print("[ERROR] reflect 出现在 search 之前，非法顺序！")
            return False, 0

        print(f"[OK] 标签顺序正确，共检测到 {search_count} 次搜索。")
        return True, search_count

    elif re.search(pattern_no_search, text, re.DOTALL):
        print("[OK] 标签顺序正确（无检索结构，包含 think→reflect→answer）。")
        return True, 0

    else:
        print("[ERROR] 标签顺序或结构错误！")
        return False, 0


TRANSITION_WORDS = [
    # 转折对比
    "however",
    "but",
    "yet",
    "although",
    "nevertheless",
    "on the other hand",
    "in contrast",
    "conversely",
    "let me think"

    # 反思修正
    "wait",
    "i should",
    "i'll",
    "i think",
    "i recall",
    "let me reconsider",

    # 替代方案
    "alternatively",
    "another",

    # 结构衔接
    "first",
    "in summary",
]


def analyze_content(text: str):
    """
    分析模型输出中的思考内容 (<think>) 与反思内容 (<reflect>)。

    返回:
    {
        "think_tokens": <int>,      # think 段的 token 数
        "reflect_tokens": <int>,    # reflect 段的 token 数
        "total_tokens": <int>,      # think + reflect 总 token 数
        "word_counts": {<phrase>: <count>, ...}  # think 段中转折词统计
    }
    """
    # === 提取 <think> 与 <reflect> 段 ===
    think_matches = re.findall(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    reflect_matches = re.findall(r"<reflect>(.*?)</reflect>", text, re.DOTALL | re.IGNORECASE)

    think_tokens = 0
    reflect_tokens = 0
    word_counter = Counter()

    # === 统计 think 标签内容 ===
    for m in think_matches:
        tokens = GLOBAL_TOKENIZER.encode(m, add_special_tokens=False)
        think_tokens += len(tokens)

        lowered = m.lower()
        for phrase in TRANSITION_WORDS:
            phrase_lower = phrase.lower()
            count = lowered.count(phrase_lower)
            if count > 0:
                word_counter[phrase] += count

    # === 统计 reflect 标签内容 ===
    for m in reflect_matches:
        tokens = GLOBAL_TOKENIZER.encode(m, add_special_tokens=False)
        reflect_tokens += len(tokens)

    total_tokens = think_tokens + reflect_tokens

    return {
        "think_tokens": think_tokens,
        "reflect_tokens": reflect_tokens,
        "total_tokens": total_tokens,
        "word_counts": dict(word_counter),
    }


def compute_score(
        solution_str: str,
        ground_truth=None,
        method="strict",
        format_score: float = 0.5,
        score: float = 0.5,
        repetition_ngram: int = 3,
        repetition_max_penalty: float = -1.0,
        debug: bool = True,
) -> float:
    model_output = extract_model_output(solution_str)
    print("-------------------************以下是模型输出***************------------------------")
    print(model_output)

    is_valid, search_count = check_tag_constraint_and_count(model_output)
    if is_valid:
        format_reward = 0.5
    else:
        return -2
    answer = extract_solution(model_output)
    open_count, close_count = count_answer_tags(model_output)

    if answer is None:
        target_answers = "null"
    else:
        target_answers = ground_truth.get('target', ground_truth) if isinstance(ground_truth, dict) else ground_truth
    # === 使用动态搜索次数计算答案奖励 ===
    if em_check(answer, target_answers):  # ✅ 答对 # 从 3.0 起，每检索一次减少 0.5，最低为 0
        answer_reward = max(0.0, round(2 - 0.5 * search_count, 3))
    else:  # ❌ 答错 # 从 -2.0 起，每检索一次惩罚减少 0.3，最低 -1.0
        answer_reward = min(-1.0, round(-2 + 0.5 * search_count, 3))
        #answer_reward = 0
    res = analyze_content(model_output)
    res2 = evaluate_search_block_global(model_output)
    search_quality_bonus = res2["score"]
    total_tokens = res["total_tokens"]
    total_counts = sum(res.get("word_counts", {}).values())
    if total_tokens == 0:
        coherence_penalty = -1.0
        repetition_penalty = -1.0
    else:
        coherence_penalty = -0.02 * total_counts
        coherence_penalty = max(coherence_penalty, -1.0)

        repetition_penalty = compute_think_repetition_penalty(
            model_output,
            n=repetition_ngram,
            max_penalty=repetition_max_penalty
        )
    #final_score = format_reward + repetition_penalty + coherence_penalty + answer_reward + search_quality_bonus
    final_score = format_reward + answer_reward + search_quality_bonus

    if debug:
        print("========== [DEBUG] Reward Breakdown ==========")
        #print("总 token 数:", total_tokens)
        #print("转折词统计:", total_counts)
        print("格式奖励:", format_reward)
        print("检索次数", search_count)
        print("检索质量",search_quality_bonus)
        #print("转折词乘法:", coherence_penalty)
        #print("重复乘法:", repetition_penalty)
        #print("答案奖励:", answer_reward)
        print("最终分数:", final_score)
        print("==============================================")

    return round(final_score, 4)


@register("search_r1_qa_em")
class SearchR1QAEMRewardManager:
    """
    Reward Manager for Search-R1 style QA tasks with Exact Match scoring.
    """
    name = "search_r1_qa_em"

    # fix the error: in reward.py force passing "reward_fn_key" param
    def __init__(self, tokenizer=None, num_examine=1, compute_score=compute_score, format_score=0.5, score=0.5,
                 run_id=None, **kwargs) -> None:
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("/home/unnet/project/model/Qwen3-1.7B")

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.format_score = format_score
        self.score = score
        self.step = None

    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards for Search-R1 style responses."""
        save_record = data.meta_info.get('save_record', True)

        if not hasattr(self, 'record_dir'):
            if hasattr(self, 'run_id'):
                self.record_dir = Path(__file__).parent.parent.parent.parent / "verl_step_records" / self.run_id
                self.record_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.record_dir = Path(
                    __file__).parent.parent.parent.parent / "verl_step_records" / f"torl-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
                self.record_dir.mkdir(parents=True, exist_ok=True)

        # check the last step index
        if self.step is None:
            last_step_idx = 0
            for file in os.listdir(self.record_dir):
                if self.num_examine == 1:
                    if re.search(r"step-val-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
                else:
                    if re.search(r"step-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
            self.step = last_step_idx + 1
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']

        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        scores = [{} for _ in range(len(data))]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        reward_extra_info = defaultdict(list)
        to_save_records = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch[
                'prompts']  # prompt_ids → 包含了 system + user + <|im_start|>assistant 的部分（模型输入） https://chatgpt.com/c/68dbba3a-38f4-8326-af4b-7ad77ad6c429
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][
                                  :prompt_length].sum()  # 计算的是 整个 prompt 部分的 token 数（包括 system + user + <|im_start|>assistant 这个开头）
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']  # response_ids → 从 <|im_start|>assistant 后面开始，模型自己生成的 token
            valid_response_length = data_item.batch['attention_mask'][
                                    prompt_length:].sum()  # 计算的是 assistant 生成的 token 数（模型预测输出的）
            valid_response_ids = response_ids[:valid_response_length]

            prompt_tokens = valid_prompt_length.item() if hasattr(valid_prompt_length, "item") else valid_prompt_length
            response_tokens = valid_response_length.item() if hasattr(valid_response_length,
                                                                      "item") else valid_response_length

            reward_extra_info['prompt_token_count'].append(prompt_tokens)
            reward_extra_info['response_token_count'].append(response_tokens)

            # 打印调试
            print(f"[DEBUG] sample {i}: prompt_token_count={prompt_tokens}, response_token_count={response_tokens}")

            # Decode the full sequence
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            # Get ground truth
            if 'reward_model' in data_item.non_tensor_batch:
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            else:
                # Fallback to direct ground truth or golden_answers
                ground_truth = data_item.non_tensor_batch.get('ground_truth',
                                                              data_item.non_tensor_batch.get('golden_answers', []))

            # Compute score
            score = compute_score(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score,
                score=self.score
            )
            # print(f"[DEBUG-compute_score] sample {i} return type={type(score)}, value={score}")
            if score > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

            # TODO: check if logic is correct
            # update this score to the scores
            scores[i] = {"score": score}

            reward_tensor[i, valid_response_length - 1] = score

            # Print examples for debugging
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"=== Search-R1 QA EM Reward Debug ===")
                print(f"Data source: {data_source}")
                print(f"Score: {score}")
                print(f"Sequence: {sequences_str}")
                print("=" * 50)

            # Save the records
            to_save_records.append({
                'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch[
                    'extra_info'] else None,
                'data_source': data_source,
                "prompt": self.tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=False),
                "response": self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False),
                'ground_truth': ground_truth,
                'score': score,
                'tool_interact_info': data[i].non_tensor_batch.get('tool_interact_info', None),
                'extra_info': data_item.non_tensor_batch.get('extra_info', None),
            })
            if "turns_stats" in data_item.non_tensor_batch:
                to_save_records[i]['num_turn'] = data[i].non_tensor_batch["turns_stats"]
                to_save_records[i]['num_valid_action'] = data[i].non_tensor_batch["valid_action_stats"]
                to_save_records[i]['is_done'] = not data[i].non_tensor_batch["active_mask"]
        if save_record:
            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"{self.name}-step-val-{self.step}.json"
            else:
                temp_file = self.record_dir / f"{self.name}-step-{self.step}.json"
            self.step += 1
            if temp_file.exists():
                with open(temp_file, "r") as f:
                    existing_records = json.load(f)
                existing_records.extend(to_save_records)
                with open(temp_file, "w") as f:
                    json.dump(existing_records, f, indent=4)
            else:
                with open(temp_file, "w") as f:
                    json.dump(to_save_records, f, indent=4)
            print(f"Saved records to {temp_file}")

        for i, score in enumerate(scores):
            if isinstance(score, dict):
                # convert the length to a Python int
                length_i = data[i].batch['attention_mask'][data[i].batch['prompts'].shape[-1]:].sum().item()
                reward_tensor[i, length_i - 1] = score['score']

                # 调试打印
                # print(f"[DEBUG] sample {i} scores[i] = {score}")
                # print(f"[DEBUG] keys in scores[i]: {list(score.keys())}")

                for k, v in score.items():
                    reward_extra_info[k].append(v)
            else:
                length_i = data[i].batch['attention_mask'][data[i].batch['prompts'].shape[-1]:].sum().item()
                reward_tensor[i, length_i - 1] = score

        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info[
            'correct_response_length'] else 0.0
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info[
            'wrong_response_length'] else 0.0
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
