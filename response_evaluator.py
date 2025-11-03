# response_evaluator.py
from typing import Dict, List, Optional, Tuple
from api_client import APIClient
import json
import logging
import re

logger = logging.getLogger(__name__)

def evaluate_response(question: str, context: str, response: str, max_retries: int = 2) -> Dict[str, any]:
    """
    评估模型回答的质量，并提供优化建议
    
    Args:
        question: 用户原始问题
        context: 检索到的上下文
        response: 模型生成的回答
        max_retries: 最大重试次数
        
    Returns:
        包含评分和建议的字典
    """
    evaluator_prompt = _build_evaluation_prompt(question, context, response)
    
    client = APIClient()
    
    # 尝试获取有效的JSON响应
    for attempt in range(max_retries + 1):
        try:
            evaluation_result = client.dialogue(evaluator_prompt)
            logger.info(f"Evaluation attempt {attempt + 1}: Raw response: {evaluation_result}")
            
            # 尝试从响应中提取JSON
            json_match = re.search(r'\{[\s\S]*\}', evaluation_result)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # 如果找不到JSON，尝试直接解析整个响应
            return json.loads(evaluation_result)
            
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"JSON解析失败 (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries:
                # 修改提示词，更明确地要求JSON格式
                evaluator_prompt = _build_evaluation_prompt(
                    question, context, response, 
                    additional_instruction="请务必严格按照JSON格式输出，不要包含任何额外说明或文本。"
                )
            else:
                # 最终失败，返回默认结构
                return _create_default_evaluation()
    
    return _create_default_evaluation()

def _build_evaluation_prompt(question: str, context: str, response: str, 
                            additional_instruction: str = "") -> str:
    """构建评估提示词"""
    return f"""
你是一个专业的AI回答质量评估专家。请根据以下标准评估一个AI助手对用户问题的回答质量：

评估标准：
1. 准确性（30分）：回答是否准确无误，是否基于提供的上下文，是否有事实错误
2. 相关性（25分）：回答是否紧扣用户问题，是否包含无关信息
3. 完整性（20分）：回答是否全面覆盖问题要点，是否遗漏重要信息
4. 清晰度（15分）：回答是否逻辑清晰，表达是否简洁明了
5. 格式与引用（10分）：是否正确标注引用，格式是否恰当

{additional_instruction}

请严格按照以下JSON Schema格式输出评估结果：
{{
    "accuracy_score": 0-30,
    "relevance_score": 0-25,
    "completeness_score": 0-20,
    "clarity_score": 0-15,
    "format_score": 0-10,
    "total_score": 0-100,
    "strengths": ["优点1", "优点2", ...],
    "weaknesses": ["缺点1", "缺点2", ...],
    "suggestions": ["改进建议1", "改进建议2", ...],
    "optimized_prompt": "优化后的prompt建议（如果需要）"
}}

【用户问题】
{question}

【参考上下文】
{context}

【AI回答】
{response}

请只输出JSON内容，不要包含任何其他说明：
"""

def _create_default_evaluation() -> Dict[str, any]:
    """创建默认的评估结果"""
    return {
        "accuracy_score": 0,
        "relevance_score": 0,
        "completeness_score": 0,
        "clarity_score": 0,
        "format_score": 0,
        "total_score": 0,
        "strengths": ["评估失败"],
        "weaknesses": ["无法获取有效评估结果"],
        "suggestions": ["检查评估提示词或模型配置"],
        "optimized_prompt": ""
    }

def format_evaluation_report(evaluation: Dict[str, any]) -> str:
    """
    格式化评估报告，生成易于阅读的文本
    
    Args:
        evaluation: 评估结果字典
        
    Returns:
        格式化的评估报告文本
    """
    total = evaluation.get("total_score", 0)
    grade = "优秀" if total >= 90 else "良好" if total >= 80 else "中等" if total >= 70 else "及格" if total >= 60 else "不及格"
    
    report = f"## 回答质量评估报告\n\n"
    report += f"**总体评分**: {total}/100 ({grade})\n\n"
    
    report += "### 详细评分\n"
    report += f"- 准确性: {evaluation.get('accuracy_score', 0)}/30\n"
    report += f"- 相关性: {evaluation.get('relevance_score', 0)}/25\n"
    report += f"- 完整性: {evaluation.get('completeness_score', 0)}/20\n"
    report += f"- 清晰度: {evaluation.get('clarity_score', 0)}/15\n"
    report += f"- 格式与引用: {evaluation.get('format_score', 0)}/10\n\n"
    
    report += "### 优点\n"
    for i, strength in enumerate(evaluation.get('strengths', []), 1):
        report += f"{i}. {strength}\n"
    
    report += "\n### 需要改进的地方\n"
    for i, weakness in enumerate(evaluation.get('weaknesses', []), 1):
        report += f"{i}. {weakness}\n"
    
    report += "\n### 优化建议\n"
    for i, suggestion in enumerate(evaluation.get('suggestions', []), 1):
        report += f"{i}. {suggestion}\n"
    
    optimized_prompt = evaluation.get('optimized_prompt' or '').strip()
    if optimized_prompt:
        report += "\n### 优化后的Prompt建议\n"
        report += f"```\n{optimized_prompt}\n```"
    
    return report

def integrate_with_rag_flow(original_response: str, user_input: str, context: str) -> Tuple[str, str]:
    """
    与RAG流程集成，评估回答质量并生成报告
    
    Args:
        original_response: RAG流程生成的原始回答
        user_input: 用户原始输入
        context: 检索到的上下文
        
    Returns:
        (原始回答, 评估报告)
    """
    # 评估回答质量
    evaluation = evaluate_response(user_input, context, original_response)
    
    # 生成评估报告
    report = format_evaluation_report(evaluation)
    
    return original_response, report