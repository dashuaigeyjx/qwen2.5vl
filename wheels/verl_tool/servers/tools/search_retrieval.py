"""
Search Retrieval Tool for verl-tool - Ray-safe + rollout-level isolation
Compatible with Search-R1 functionality
"""
from .base import BaseTool, register_tool
import regex as re
import requests
from typing import Tuple, Dict, Any, List, Optional
import logging, os, json

logger = logging.getLogger(__name__)

@register_tool
class SearchRetrievalTool(BaseTool):
    tool_type = "search_retrieval"

    def __init__(self, num_workers=1, retriever_url="http://127.0.0.1:8001/retrieve", topk=5, **kwargs):
        super().__init__(num_workers)
        self.retriever_url = kwargs.get('retriever_url', os.getenv('RETRIEVER_URL', retriever_url))
        self.topk = kwargs.get('topk', int(os.getenv('RETRIEVER_TOPK', str(topk))))
        # rollout-level in-memory cache; Ray-safe via PID+trajectory key
        self.env_cache: Dict[str, dict] = {}
        logger.info(f"✅ SearchRetrievalTool initialized with URL: {self.retriever_url}, topk: {self.topk}")

    # ========== 说明 ==========
    def get_usage_inst(self):
        return "You can search for information by putting your query between <search> and </search> tags."

    # ========== 解析 ==========
    def _parse_search_query(self, action: str) -> Tuple[str, bool]:
        """
        Returns: (query, is_valid)
        """
        if "</search>" in action:
            search_matches = re.findall(r"<search>(.*?)</search>", action, re.DOTALL)
            if search_matches:
                return search_matches[-1].strip(), True
        return "", False

    def _parse_answer_tags(self, action: str) -> Tuple[str, bool]:
        """
        Returns: (answer_text, is_valid)
        """
        if "</answer>" in action:
            matches = re.findall(r"<answer>(.*?)</answer>", action, re.DOTALL)
            if matches:
                return matches[-1].strip(), True
        return "", False

    def parse_action(self, action: str) -> Tuple[str, bool]:
        query, valid = self._parse_search_query(action)
        if valid:
            return query, True
        ans, valid = self._parse_answer_tags(action)
        if valid:
            return ans, True
        return "", False

    # ========== 优先级 ==========
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        if "</search>" in action:
            _, valid = self.parse_action(action)
            if valid:
                return 100
        _, valid = self.parse_action(action)
        return 0 if valid else -1

    # ========== 环境状态管理（Ray-safe） ==========
    def _get_env_key(self, trajectory_id: str) -> str:
        """Ensure per-rollout isolation using PID + trajectory_id"""
        pid = os.getpid()
        return f"{pid}_{trajectory_id}"

    def load_env(self, trajectory_id: str) -> dict:
        """
        首次创建或在标记完成(_finished=True)后重置，避免历史污染导致首轮检索全被去重。
        """
        key = self._get_env_key(trajectory_id)
        env = self.env_cache.get(key)
        if env is None:
            env = {"retrieved_docs": [], "_finished": False}
            self.env_cache[key] = env
            logger.debug(f"[Rollout {trajectory_id} | PID {os.getpid()}] env created")
        elif env.get("_finished", False):
            env = {"retrieved_docs": [], "_finished": False}
            self.env_cache[key] = env
            logger.debug(f"[Rollout {trajectory_id} | PID {os.getpid()}] env reset after finished")
        return env

    def save_env(self, trajectory_id: str, env: dict):
        key = self._get_env_key(trajectory_id)
        self.env_cache[key] = env  # in-memory, per rollout

    # ========== 检索主逻辑 ==========
    def conduct_action(self, trajectory_id, action, extra_field):
        parsed_query, is_valid = self._parse_search_query(action)
        env = self.load_env(trajectory_id)
        seen_ids = set(env.get("retrieved_docs", []))

        logger.debug(f"[Rollout {trajectory_id} | PID {os.getpid()}] Action parsed: {parsed_query!r}, Valid={is_valid}, seen_ids={len(seen_ids)}")

        if not is_valid:
            # 检测是否是回答结束
            parsed_query, is_valid = self._parse_answer_tags(action)
            if is_valid:
                # 标记本次 rollout 环境结束，下次加载会重置
                env["_finished"] = True
                observation = ""
                execution_result = ""
                done, valid = True, False
                logger.debug(f"[Rollout {trajectory_id}] Answer detected → mark finished")
            else:
                observation = ""
                execution_result = ""
                done, valid = False, False
        else:
            try:
                raw_results = self._batch_search(
                    [parsed_query],
                    seen_ids=seen_ids,
                    rollout_id=trajectory_id,
                    topk_override=extra_field.get("topk") if isinstance(extra_field, dict) else None
                )
                current_results = self._first_list(raw_results)

                formatted_results = self._passages2string(current_results)
                observation = f'\n\n<information>{formatted_results.strip()}</information>\n\n'
                execution_result = formatted_results
                done, valid = False, True

                # === 更新当前 rollout 的检索记录（仅加入有 id 的文档）
                if current_results:
                    new_cnt = 0
                    for doc in current_results:
                        doc_id = self._get_doc_id(doc)
                        if doc_id and doc_id not in seen_ids:
                            env["retrieved_docs"].append(doc_id)
                            new_cnt += 1
                    logger.info(f"[Rollout {trajectory_id}] ✅ Retrieved {len(current_results)} docs, recorded {new_cnt} new ids for query: {parsed_query!r}")
                else:
                    logger.info(f"[Rollout {trajectory_id}] ⚠️ No results for query: {parsed_query!r}")

            except Exception as e:
                logger.error(f"[Rollout {trajectory_id}] ❌ Search error: {e}")
                execution_result = f"Search error: {str(e)}"
                observation = f'\n\n<information>Search temporarily unavailable</information>\n\n'
                done, valid = False, False

        # === 保存状态 ===
        self.save_env(trajectory_id, env)
        return observation, done, valid

    # ========== 批量检索 + 防御性去重 ==========
    def _batch_search(
        self,
        queries: List[str],
        seen_ids: Optional[set] = None,
        rollout_id: Optional[str] = None,
        topk_override: Optional[int] = None
    ) -> List[List[Dict]]:
        payload = {
            "queries": queries,
            "topk": int(topk_override) if topk_override else self.topk,
            "return_scores": True
        }

        logger.debug(f"[Rollout {rollout_id}] POST {self.retriever_url} payload={payload}")

        try:
            response = requests.post(self.retriever_url, json=payload, timeout=30)
            response.raise_for_status()
            j = response.json() if response.content else {}
            result = j.get("result", [])

            # 统一结构为 List[List[dict]]
            if isinstance(result, dict):
                # 容错：服务端错误返回 dict 时，转为空
                logger.warning(f"[Rollout {rollout_id}] Unexpected result type dict; coercing to empty list")
                result = [[]]
            elif isinstance(result, list) and result and isinstance(result[0], dict):
                # 容错：部分实现直接返回一维 list
                result = [result]

            before_count = sum(len(r) for r in result)

            # === 去重：仅对当前 rollout 的 seen_ids 生效；若误删全部则回退
            if seen_ids:
                clean_seen = {sid for sid in seen_ids if sid}
                for i, results in enumerate(result):
                    if not results:
                        continue
                    filtered = []
                    for doc in results:
                        doc_id = self._get_doc_id(doc)
                        # 仅当确知 doc_id 在 seen 才过滤；缺少 id 的文档不去重
                        if (doc_id and doc_id in clean_seen):
                            continue
                        filtered.append(doc)
                    # 若过滤后为空且原始存在，回退原结果，避免“全部被去重”导致首轮空
                    if not filtered and results:
                        logger.warning(f"[Rollout {rollout_id}] Dedup removed all docs; reverting to original results for batch index {i}")
                        result[i] = results
                    else:
                        result[i] = filtered

            after_count = sum(len(r) for r in result)
            logger.debug(f"[Rollout {rollout_id}] Got {before_count} docs before dedup, {after_count} after dedup")

            return result

        except Exception as e:
            logger.error(f"[Rollout {rollout_id}] Retrieval service error: {e}")
            # 和 batch 大小对齐的空结果
            return [[] for _ in queries]

    # ========== 工具函数 ==========
    @staticmethod
    def _get_doc_id(doc_item: Dict) -> Optional[str]:
        if not isinstance(doc_item, dict):
            return None
        doc = doc_item.get("document", {})
        return doc.get("id") or doc_item.get("id")

    @staticmethod
    def _first_list(nested: Any) -> List[Dict]:
        """
        兼容多种返回形态：
          - [[{...}, {...}], ...] → 取第 0 组
          - [] → []
          - None → []
        """
        if not nested:
            return []
        if isinstance(nested, list):
            if nested and isinstance(nested[0], list):
                return nested[0]
            if nested and isinstance(nested[0], dict):
                # 容错：服务端返回一维 list
                return nested
        return []

    # ========== 格式化输出 ==========
    def _passages2string(self, retrieval_result: List[Dict]) -> str:
        """
        Format retrieval results into readable string.
        Adapted for Search-R1 schema: document has 'title' and 'text'
        """
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            doc = doc_item.get("document", {}) if isinstance(doc_item, dict) else {}
            title = doc.get("title", "No title")
            text = doc.get("text", "No content")
            format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"
        return format_reference
