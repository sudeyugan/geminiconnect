import os
from typing import NamedTuple

class Config(NamedTuple):
    BASE_URL: str = os.getenv("VECTOR_DB_BASE_URL", "http://10.1.0.220:9002/api")
    USER_NAME: str = os.getenv("USER_NAME", "Group4")
    TOKEN: str = os.getenv("TOKEN", "_QZ9BtHUWrgT8BrO4ihZFAPJpzju8PBnFG_VbGJUGDYSBkOEztl8FqxafKhh-Prb")
    DEFAULT_METRIC_TYPE: str = "cosine"
    MAX_CONTEXT_LENGTH: int = 2000  # 检索结果最大上下文长度
    TOP_K: int = 3                # 默认返回 top_k 个结果
    WAIT_TIME: int = 2            # 等待向量库flush的时间

config = Config()