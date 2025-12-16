import os
import re
from typing import TYPE_CHECKING, Dict, List, Union
from transformers import AutoTokenizer
import json
from collections import Counter
import string
import openai
import random
if TYPE_CHECKING:
    from swift.llm import InferRequest


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards

class Format2(ORM):
    def __init__(self,
                 full_score: float = 1.0,
                 partial_score: float = 0.5,
                 missing_score: float = 0.0,
                 multi_penalty: float = -1.0):
        """
        full_score: 严格格式匹配得分
        partial_score: 标签齐全但格式不严谨得分
        missing_score: 标签不全得分
        multi_penalty: 多组标签时的负分
        """
        self.full_score = full_score
        self.partial_score = partial_score
        self.missing_score = missing_score
        self.multi_penalty = multi_penalty

        # 预编译正则
        # 严格格式：全串 = <think>…</think> + (仅空白) + <answer>…</answer>
        self.full_pattern = re.compile(
            r'^<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>$',
            flags=re.DOTALL
        )
        # 单段捕获，用于计数
        self.think_block = re.compile(r'<think>[\s\S]*?</think>', flags=re.DOTALL)
        self.answer_block = re.compile(r'<answer>[\s\S]*?</answer>', flags=re.DOTALL)

        # 标签存在的快速检测（不计数）
        self.has_think = re.compile(r'<think>[\s\S]*?</think>', flags=re.DOTALL)
        self.has_answer = re.compile(r'<answer>[\s\S]*?</answer>', flags=re.DOTALL)

    def __call__(self, completions, **kwargs) -> List[float]:
        scores = []
        for content in completions:
            # 统计块出现次数
            n_think = len(self.think_block.findall(content))
            n_answer = len(self.answer_block.findall(content))

            # 多组：直接负分
            if n_think > 1 or n_answer > 1:
                scores.append(self.multi_penalty)
                continue

            # 严格格式：恰好一组，且整体结构完整、无多余内容
            if self.full_pattern.match(content):
                scores.append(self.full_score)
                continue

            # 单组但格式不严谨：两标签都恰好出现一次，但不满足严格结构
            if n_think == 1 and n_answer == 1:
                scores.append(self.partial_score)
                continue

            # 标签不全
            has_think = bool(self.has_think.search(content))
            has_answer = bool(self.has_answer.search(content))
            if has_think and has_answer:
                # 理论上不会到这里（已被 n_think/n_answer 覆盖），但留作兜底
                scores.append(self.partial_score)
            else:
                scores.append(self.missing_score)

        return scores
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # edge case
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards 
# 提取模型生成的 <answer> 标签内容
def extract_xml_answer(text: str) -> str:
    """
    从 <answer>…</answer> 中抽取内容；若无标签则返回空字符串。
    """
    m = re.search(r'<answer>([\s\S]*?)</answer>', text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

def is_mcq_answer(s: str) -> bool:
    """是否为单字母选项 A/B/C/D（大小写皆可）"""
    return bool(re.fullmatch(r'[A-Da-d]', s.strip()))

def is_numeric(s: str) -> bool:
    """是否为整数或小数格式"""
    return bool(re.fullmatch(r'-?\d+(\.\d+)?', s.strip()))

def is_latex(s: str) -> bool:
    """是否像 LaTeX 表达式（包含反斜杠指令）"""
    return '\\' in s

def normalize_latex(s: str) -> str:
    """移除所有空白后做精确比较"""
    return re.sub(r'\s+', '', s)

def numeric_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    """尝试将字符串转为浮点数并比较，或回退为子串匹配"""
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return b in a  # 退而求其次，只要包含目标数字即可

# 提取模型生成的 <reasoning> 标签内容
def extract_xml_reasoning(text: str) -> str:
    reasoning = text.split("<think>")[-1]
    reasoning = reasoning.split("</think>")[0]
    return reasoning.strip()

def chat_with_token_logging(normalized_golden, normalized_prediction):
    """调用 LLM 判断 prediction 是否等价于 golden answer"""
    client = OpenAI(api_key="null", base_url="http://117.145.66.250:7/v1")
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
        model="/models/Qwen3-Coder-480/",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )

    pred_text = response.choices[0].message.content.strip().upper()
    return pred_text

class EnhancedAccuracyReward(ORM):
    """
    针对三种答案类型给 1 或 0 分：
      - 选项题：A/B/C/D 精确匹配（不区分大小写）
      - 数字题：数值相等或包含
      - LaTeX 题：去空白后字符串相等
      - 其他文本：小写无关精确匹配
    特殊规则：
      - 如果出现多个 <answer>…</answer> 标签，则直接给 -1 分
    """
    def __call__(self,
                 completions: List[str],
                 solution: List[str],
                 **kwargs) -> List[float]:
        rewards: List[float] = []
        for comp, gold in zip(completions, solution):
            # 检查 <answer> 标签数量
            answer_matches = re.findall(r'<answer>([\s\S]*?)</answer>', comp, flags=re.IGNORECASE)
            if len(answer_matches) > 1:
                rewards.append(-1.0)
                continue  # 直接跳过后续判定

            # 检查 <think> 标签数量
            think_matches = re.findall(r'<think>([\s\S]*?)</think>', comp, flags=re.IGNORECASE)
            if len(think_matches) > 1:
                rewards.append(-1.0)
                continue  # 直接跳过后续判定

            pred = extract_xml_answer(comp)
            gold_str = gold.strip()

            if is_mcq_answer(gold_str):
                ok = pred.strip().upper() == gold_str.upper()
            elif is_numeric(gold_str):
                ok = numeric_equal(pred.strip(), gold_str)
            elif is_latex(gold_str) or is_latex(pred):
                ok = normalize_latex(pred) == normalize_latex(gold_str)
            else:
                ok = pred.strip().lower() == gold_str.lower()
            if chat_with_token_logging(gold_str,pred):
                ok = 1

            rewards.append(1.0 if ok else 0.0)

        return rewards
    
class Token(ORM):


    def __call__(self, completions, solution, **kwargs) -> List[float]:

        rewards = []
        for comp, sol in zip(completions, solution):
            answer_matches = re.findall(r'<answer>([\s\S]*?)</answer>', comp, flags=re.IGNORECASE)
            if len(answer_matches) > 1:
                rewards.append(-1.0)
                continue  # 直接跳过后续判定

            # 检查 <think> 标签数量
            think_matches = re.findall(r'<think>([\s\S]*?)</think>', comp, flags=re.IGNORECASE)
            if len(think_matches) > 1:
                rewards.append(-1.0)
                continue  # 直接跳过后续判定

            res = analyze_think_content(comp)
            total_tokens = res["total_tokens"]
            if total_tokens ==0 :
                reward = -1.0
            else:
                coherence_penalty = -((sum(res["word_counts"].values()) / max(total_tokens, 1)) * 2.0+ max(0, (total_tokens - 250) / 1000.0))
                reward = max(coherence_penalty, -1.0)
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


from typing import List
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
GLOBAL_TOKENIZER =AutoTokenizer.from_pretrained("/home/dataset-assist-0/verl-main/model/DeepSeek-R1-Distill-Qwen-7B")
def analyze_think_content(text: str):
    """
    返回统计结果:
    - total_tokens: think 标签中的总 token 数
    - word_counts: 转折词出现次数（大小写不敏感）
    """
    matches = re.findall(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    total_tokens = 0
    word_counter = Counter()
    knowledge_flag = "missing"


    for m in matches:
        # === 统计 token 数 ===
        tokens = GLOBAL_TOKENIZER.encode(m, add_special_tokens=False)
        total_tokens += len(tokens)

        # === 统计转折词 ===
        lowered = m.lower()
        for phrase in TRANSITION_WORDS:
            phrase_lower = phrase.lower()
            count = lowered.count(phrase_lower)
            if count > 0:
                word_counter[phrase] += count

    return {
        "total_tokens": total_tokens,
        "word_counts": dict(word_counter),
    }

class LengthControlReward(ORM):
    """
    Length Control Reward (SoftOverlong style)

    Provides a linearly decaying score when the response length t = |y|
    falls outside the ideal range [L_min, L_max] but within the buffer b,
    and zero beyond that.

    r_len(y) =
        1, if L_min <= t <= L_max
        1 - (L_min - t)/b, if L_min - b <= t < L_min
        1 - (t - L_max)/b, if L_max < t <= L_max + b
        0, otherwise
    """

    def __init__(self):
        self.soft_min_length = 1000
        self.soft_max_length = 2000
        self.soft_buffer = 500

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids', [])

        for ids in response_token_ids:
            t = len(ids)  # 输出长度

            if self.soft_min_length <= t <= self.soft_max_length:
                r = 1.0
            elif self.soft_min_length - self.soft_buffer <= t < self.soft_min_length:
                r = 1 - (self.soft_min_length - t) / self.soft_buffer
            elif self.soft_max_length < t <= self.soft_max_length + self.soft_buffer:
                r = 1 - (t - self.soft_max_length) / self.soft_buffer
            else:
                r = -1.0

            rewards.append(round(r, 4))

        return rewards

class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards

from openai import OpenAI

def extract_reward_simple(text: str) -> float:
    m = re.search(r'[-+]?(?:\d+\.?\d*|\.\d+)', text)
    if not m:
        return 0.0
    try:
        val = float(m.group(0))
    except Exception:
        return 0.0
    return max(-1.0, min(1.0, val))


class RMReward(ORM):

    def __init__(self):
        super().__init__()
        try:
            # 连接 OpenAI API
            self.client = OpenAI(
                api_key='sk-72tkvudyGLPMi',
                base_url='http://117.145.66.250:7/v1',  # 本地部署
            )
            self.verify_model_name = '/models/Qwen3-Coder-480/'
        except Exception as e:
            raise RuntimeError('Failed to connect to the model service. Please deploy the model '
                               "using 'swift deploy' or 'vllm serve'.") from e

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for completion, sol in zip(completions, solution):
            pred = extract_xml_reasoning(completion)
            if not pred:
                rewards.append(0.0)
                continue
            
            rm_prompt = (
    "You are a scoring model. Return one float between -1.0 and 1.0 assessing the following:\n"
    "Evaluate whether the given text contains excessive use of transitional or contrastive words "
    "(e.g., 'however', 'but', 'although', 'yet', 'on the other hand', 'alternatively', etc.).\n\n"
    "Guidelines:\n"
    "- If the text includes many such words that indicate hesitation, revision, or repeated self-correction, output a LOW score (near -1.0).\n"
    "- If the text uses them moderately or not at all, output a HIGH score (near +1.0).\n"
    "- Focus only on transition/contrast words, not general reasoning quality.\n\n"
    "Output only one float number between -1.0 and 1.0."
    f"{pred}")

            try:
                # 尝试调用接口，增加超时处理
                chat_response = self.client.chat.completions.create(
                    model=self.verify_model_name,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': rm_prompt}
                    ],
                    timeout=10000  # 超过10秒没有响应则超时
                )
                response = chat_response.choices[0].message.content.strip()
                reward = extract_reward_simple(response)
                rewards.append(reward)
            except Exception as e:
                # 捕获所有异常，超时或连接错误等，返回0
                print(f"Error occurred: {e}")
                rewards.append(0.0)

        return rewards
    
orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': EnhancedAccuracyReward,
    'format': Format2,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'model':RMReward,
    'length':LengthControlReward,
    'token':Token
}
