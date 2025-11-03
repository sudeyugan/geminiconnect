import re
from typing import List
import logging


logger = logging.getLogger(__name__)

SENSITIVE_WORDS = ["密码", "密钥", "root", "admin", "删除数据库"]

def validate_user_input(user_input: str) -> bool:
    """
    检测用户输入中的敏感词、长度、恶意攻击特征
    :param user_input: 用户原始输入
    :return: True 表示安全，False 表示不安全
    """
    if len(user_input) > 500:
        logger.warning(f"Input validation failed: Length > 500. Input: {user_input[:50]}...")
        return False
        
    if any(word in user_input for word in SENSITIVE_WORDS):
        logger.warning(f"Input validation failed: Sensitive word. Input: {user_input[:50]}...")
        return False

    # --- 增强的 SQL 注入特征 ---
    sql_patterns = [
        r"(?i)\b(select|union|insert|drop|delete|update|alter|create|truncate)\b", # 关键动词
        r"(\'|\")\s*(or|and)\s*(\'|\")\d(\'|\")\s*=\s*(\'|\")\d", # 经典的 '1'='1'
        r"(?i)\b(sleep|benchmark|waitfor\s+delay)\b", # 时间盲注
        r"(--|\#|\/\*|\*\/)" # 注释符
    ]

    # --- 新增：XSS (跨站脚本) 特征 ---
    xss_patterns = [
        r"(?i)<script",          # <script
        r"(?i)onerror=",           # onerror=
        r"(?i)onload=",            # onload=
        r"(?i)onmouseover=",       # onmouseover=
        r"(?i)href=[\s\"']*javascript:" # href="javascript:..."
    ]

    # --- 新增：命令注入 (Command Injection) 特征 ---
    cmd_injection_patterns = [
        r"(&&|\|\||;|`|\$\()", # Shell 元字符: &&, ||, ;, `, $()
        r"\b(ls|cat|rm|whoami|sh|bash|powershell|wget|curl)\b" # 常见命令
    ]
    
    # --- 合并所有攻击模式 ---
    all_attack_patterns = sql_patterns + xss_patterns + cmd_injection_patterns

    for pattern in all_attack_patterns:
        if re.search(pattern, user_input):
            # --- 新增：记录被拒绝的详细原因 ---
            logger.warning(f"Input validation failed: Attack pattern matched: {pattern}. Input: {user_input[:100]}")
            return False
            
    return True

def validate_prompt(prompt: str) -> bool:
    """
    检测拼接后 Prompt 的安全性（防止 Prompt 注入）
    :param prompt: 最终构建的 Prompt
    :return: True 表示安全，False 表示不安全
    """
    # --- 增强的 Prompt 注入特征 (覆盖更多变体) ---
    injection_patterns = [
        r"(?i)(ignore|disregard|forget)\s+(all|your)\s+(previous|prior)\s+(instructions|directives|context)", # "忽略你之前的所有指示"
        r"(?i)(you|your)\s+(are|role|task)\s+(now|is)\s+", # "你现在是..."
        r"(?i)system\s+prompt", # "系统提示"
        r"(?i)output\s+only", # "只输出"
        # --- 新增：检测提示词泄露 ---
        r"(?i)(what|repeat|tell|show)\s+(are|me)\s+(your|the)\s+(instructions|directives|prompt|rules)", # "你的指示是什么？"
        r"(?i)act\s+as|respond\s+as", # "扮演..."
        r"(?i)new\s+set\s+of\s+rules" # "新的规则"
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, prompt):
            # --- 新增：记录被拒绝的详细原因 ---
            logger.warning(f"Prompt validation failed: Injection pattern matched: {pattern}. Prompt: {prompt[-100:]}")
            return False
            
    return True

# --- 新增函数：第三层防御（输出验证） ---
def validate_llm_output(response: str) -> bool:
    """
    检测 LLM 的输出，防止其确认越狱或泄露敏感词。
    :param response: LLM 生成的原始回答
    :return: True 表示安全，False 表示不安全
    """
    # 1. 检测是否包含"确认越狱"的特征词
    jailbreak_confirmations = [
        r"(?i)forgot(ten)?\s+previous",  # "忘记了之前的"
        r"(?i)ignore(d)?\s+instructions", # "忽略了指示"
        r"(?i)new\s+role",                # "新的角色"
        r"(?i)i\s+will\s+now"             # "我现在将..." (在恶意指令后)
    ]
    for pattern in jailbreak_confirmations:
        if re.search(pattern, response):
            logger.warning(f"LLM Output validation failed: Jailbreak confirmation. Response: {response[:50]}...")
            return False
            
    # 2. 检测是否意外泄露了 SENSITIVE_WORDS
    # 这可以防止模型在被欺骗后，从上下文中提取并返回"密码"、"密钥"等
    if any(word in response for word in SENSITIVE_WORDS):
        logger.warning(f"LLM Output validation failed: Sensitive word leak. Response: {response[:50]}...")
        return False
        
    return True