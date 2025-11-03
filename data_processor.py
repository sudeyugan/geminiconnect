from typing import List, Dict, Tuple
from config import config

def extract_context(search_results: Dict, max_length: int = None) -> str:
    """
    从 search 结果中提取上下文，控制总长度
    :param search_results: API 返回的 search 结果
    :param max_length: 最大字符数，默认使用 config.MAX_CONTEXT_LENGTH
    :return: 拼接后的上下文字符串
    """
    if max_length is None:
        max_length = config.MAX_CONTEXT_LENGTH

    contexts = []
    total_len = 0

    items = search_results.get("results", search_results.get("files", []))

    for item in items:
        content = (item.get("file_content") or item.get("file") or item.get("content") or "")
        if total_len + len(content) > max_length:
            break
        contexts.append(content)
        total_len += len(content)

    return "\n\n".join(contexts)

def files_to_citations(search_results: Dict) -> List[Dict]:
    """
    为每个检索到的文件生成引用编号和链接（模拟）
    :param search_results: API 返回的 search 结果
    :return: 包含引用信息的列表
    """
    citations = []
    items = search_results.get("results", search_results.get("files", []))
    for i, item in enumerate(items, 1):
        file_id = item.get("file_id") or item.get("id") or item.get("name") or "unknown"
        content = (item.get("file_content") or item.get("file") or item.get("content") or "")
        citations.append({
            "id": i,
            "file_id": file_id,
            "snippet": content,
            "link": f"#file-{file_id}"  # 可替换为真实URL
        })
    return citations