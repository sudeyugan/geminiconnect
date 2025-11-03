import requests
from typing import Dict, Any, List
from config import config


class APIClient:
    def __init__(self):
        self.base_url = config.BASE_URL
        self.token = config.TOKEN
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def search(self, db_name: str, query: str, top_k: int = None, expr: str = None) -> Dict[str, Any]:
        """
        调用 /search 接口。
        现在可以接受一个可选的 top_k 参数。
        """
        url = f"{self.base_url}/databases/{db_name}/search"

        # 修改点 1: 决定最终使用的 top_k 值
        # 如果调用时传入了 top_k，就用传入的值；否则，使用配置文件中的默认值。
        final_top_k = top_k if top_k is not None else config.TOP_K

        payload = {
            "token": self.token,
            "query": query,
            "top_k": final_top_k,  # 修改点 2: 使用我们最终确定的 top_k 值
            "metric_type": config.DEFAULT_METRIC_TYPE,
        }
        if expr:
            payload["expr"] = expr

        resp = self.session.post(url, json=payload)
        if resp.status_code != 200:
            raise Exception(f"Search API error: {resp.text}")

        # 返回 JSON 数据，但要确保 files 键存在且是一个列表
        data = resp.json()
        if "files" not in data or not isinstance(data["files"], list):
            # 如果API返回的数据格式不符合预期，返回一个空列表，避免后续代码出错
            return {"files": []}

        return data

    def dialogue(self, user_input: str) -> str:
        """调用 /dialogue 接口"""
        url = f"{self.base_url}/dialogue"
        payload = {"user_input": user_input,
                   "token": self.token,
                   "max_tokens": 1024
                   }
        resp = self.session.post(url, json=payload)
        if resp.status_code != 200:
            raise Exception(f"Dialogue API error: {resp.text}")
        return resp.json().get("response", "")
