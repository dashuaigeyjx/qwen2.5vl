import os
import re
import requests
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

from .base import BaseTool, register_tool


# --- Config ---
class OnlineSearchConfig:
    def __init__(
        self,
        search_url: str = "https://api.bochaai.com/v1/web-search",
        topk: int = 3,
        serp_api_key: Optional[str] = "sk-90e918a656524283ae0143058fd89806",  # 这里保留名字，但实际存放 bocha api key
        serp_engine: str = "bocha",          # 保留参数，但不再使用
    ):
        self.search_url = search_url
        self.topk = topk
        self.serp_api_key = serp_api_key    # 实际上就是 bocha_api_key
        self.serp_engine = serp_engine


# --- Online Search Wrapper ---
class OnlineSearchEngine:
    def __init__(self, config: OnlineSearchConfig):
        self.config = config

    def _search_query(self, query: str) -> Dict:
        """Execute a single search query using Bocha API."""
        if not self.config.serp_api_key:
            raise ValueError("Bocha API key is required")

        headers = {
            "Authorization": f"Bearer {self.config.serp_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "summary": True,
            "freshness": "oneYear",
            "count": self.config.topk
        }

        try:
            response = requests.post(self.config.search_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Search request failed: {str(e)}"}

    def batch_search(self, queries: List[str]) -> List[List[Dict]]:
        """Execute multiple search queries in parallel."""
        results = []
        with ThreadPoolExecutor() as executor:
            for result in executor.map(self._search_query, queries):
                results.append(self._process_result(result))
        return results

    def _process_result(self, search_result: Dict) -> List[Dict]:
        """Process Bocha API search results into standardized format."""
        results = []
        if "error" in search_result:
            return [{'document': {'contents': search_result["error"]}, 'url': '', 'type': 'error'}]

        web_pages = (
            search_result.get("data", {})
            .get("webPages", {})
            .get("value", [])
        )

        for item in web_pages[:self.config.topk]:
            title = item.get("name", "No title.")
            snippet = item.get("snippet") or item.get("summary", "No snippet available.")
            link = item.get("url", "")
            results.append({
                "document": {"contents": f'"{title}"\n{snippet}'},
                "url": link,
                "type": "organic"
            })

        return results


@register_tool
class SerpSearchTool(BaseTool):
    tool_type = "serp_search"

    def __init__(self, num_workers=1, search_url="https://api.bochaai.com/v1/web-search",
                 serp_api_key=None, serp_engine="bocha", topk=3):
        super().__init__(num_workers)

        self.serp_api_key = "sk-90e918a656524283ae0143058fd89806"  # 用 BOCHA_API_KEY 环境变量
        self.search_url = search_url
        self.serp_engine = serp_engine
        self.topk = topk

        if not self.serp_api_key:
            raise ValueError("Bocha API key must be provided either as parameter or environment variable (BOCHA_API_KEY)")

        self.config = OnlineSearchConfig(
            search_url=self.search_url,
            topk=self.topk,
            serp_api_key=self.serp_api_key,
            serp_engine=self.serp_engine
        )
        self.engine = OnlineSearchEngine(self.config)

    def get_usage_inst(self):
        return f"You can search the web using Bocha API. Provide search queries in <search>query</search> tags or ```search query``` code blocks."

    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string to extract search queries.
        """
        search_queries = re.findall(r"<search>(.*?)</search>", action, re.DOTALL)

        if not search_queries:
            search_queries = re.findall(r"```\n?search\n(.*?)```", action, re.DOTALL)

        if not search_queries:
            search_queries = re.findall(r"```search\n(.*?)\n```", action, re.DOTALL)

        if len(search_queries) == 0:
            return "", False

        parsed_query = search_queries[0].strip()
        return parsed_query, True

    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the search action using Bocha API.
        """
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            observation = "No valid search query found. Please provide search queries in <search>...</search> tags or ```search...``` code blocks."
            done = False
            valid = False
        else:
            try:
                search_results = self.engine.batch_search([parsed_action])[0]
                print("---------------")
                print(search_results)
                if search_results:
                    formatted_results = []
                    for i, result in enumerate(search_results, 1):
                        content = result['document']['contents']
                        url = result.get('url', 'N/A')
                        result_type = result.get('type', 'unknown')

                        type_indicator = {
                            'organic': 'Web Result',
                            'error': 'Error'
                        }.get(result_type, 'Result')

                        formatted_results.append(f"{type_indicator} {i}:\nURL: {url}\nContent: {content}")

                    observation = "\n\n".join(formatted_results)
                else:
                    observation = "No search results found."

                if action.endswith("</search>") or "</search>" in action:
                    observation = f"\n<search_results>\n{observation}\n</search_results>\n"
                elif action.strip().endswith("```") or "```search" in action:
                    observation = f"\n```search_results\n{observation}\n```\n"
                else:
                    observation = f"\nSearch Results:\n{observation}\n"

                done = False
                valid = True

            except Exception as e:
                observation = f"Search error: {str(e)}"
                done = False
                valid = False

        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)

        return observation, done, valid
