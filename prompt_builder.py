from typing import List, Dict

def detect_personality(user_input: str) -> str:
    """基于用户输入识别期望的人格类型"""
    from config import PersonalityConfig
    
    # 检查用户是否明确指定了人格
    if "教学模式" in user_input or "老师模式" in user_input:
        return "TEACHER"
    if "查询模式" in user_input or "研究模式" in user_input:
        return "RESEARCHER"
    if "通用模式" in user_input or "正常模式" in user_input:
        return "GENERAL"
    
    # 基于关键词匹配
    for personality_type, config in [
        ("TEACHER", PersonalityConfig.TEACHER),
        ("RESEARCHER", PersonalityConfig.RESEARCHER),
        ("GENERAL", PersonalityConfig.GENERAL)
    ]:
        if any(keyword in user_input for keyword in config["keywords"]):
            return personality_type
    
    # 默认返回通用模式
    return "GENERAL"

def build_chat_prompt(
    history: List[Dict[str, str]], 
    user_input: str, 
    context: str, 
    citations: List[Dict],
    personality_type: str = "GENERAL"  # 新增参数
) -> str:
    """
    组合系统 Prompt + 历史对话 + 当前用户输入 + 上下文 + 引用
    :param history: 历史对话 [{"role": "user"/"assistant", "content": "..."}]
    :param user_input: 当前用户问题
    :param context: 检索到的相关上下文
    :param citations: 引用列表
    :return: 最终发送给 LLM 的 Prompt（包含对话历史）
    """
    # 只取最近的若干条历史，避免过长
    truncated = history[-10:] if len(history) > 10 else history
    history_text = "\n".join([
        f"{'【用户】' if m['role']=='user' else '【助手】'}{m['content']}"
        for m in truncated
    ])

    citation_text = "\n".join([
        f"[{c['id']}] {c['snippet']} (来源: {c['link']})"
        for c in citations
    ])

    from config import PersonalityConfig
    personalities = {
        "TEACHER": PersonalityConfig.TEACHER,
        "RESEARCHER": PersonalityConfig.RESEARCHER,
        "GENERAL": PersonalityConfig.GENERAL
    }
    
    selected_personality = personalities.get(personality_type, personalities["GENERAL"])
    system_prompt = selected_personality["system_prompt"]

    final_prompt = f"""{system_prompt}

【对话历史】
{history_text or '（无）'}

【用户问题】
{user_input}

【参考上下文】
{context}

请回答：
"""
    return final_prompt
