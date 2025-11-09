import json
import time
import os
import sys
import google.generativeai as genai
# 【新添加】导入特定的错误类型用于诊断
# import google.generativeai.errors as google_errors  <- 【已删除】这行导致了错误
from google.api_core import exceptions as google_api_exceptions

# 从环境变量中获取 API 密钥
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("错误：GEMINI_API_KEY 环境变量未设置。请先设置密钥。")

# 配置 Google Gemini 客户端
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"API 密钥配置失败: {e}", file=sys.stderr)
    sys.exit(1)

# 设置模型参数
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 4096,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# 初始化 Gemini Pro 模型
try:
    model = genai.GenerativeModel(model_name="gemini-pro",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
except Exception as e:
    print(f"模型初始化失败: {e}", file=sys.stderr)
    sys.exit(1)


class SyntheticDataGenerator:
    def __init__(self):
        self.corpus = []
        print("Google Gemini API 客户端初始化成功。")

    def generate_qa_pairs_for_topic(self, topic: str, count: int = 5):
        """
        为一个主题调用LLM生成Q&A对（带详细的错误返回）
        """
        print(f"正在为主题 '{topic}' 生成 {count} 个Q&A对...")
        
        prompt = f"""
        你是一名顶级的网络安全教授和密码学专家。
        请围绕以下主题，生成 {count} 个高质量的、有深度的问答(Q&A)对。
        
        主题: "{topic}"
        
        请严格按照以下JSON列表格式输出，不要包含任何JSON格式之外的解释性文本：
        [
          {{"q": "问题1...", "a": "答案1..."}},
          {{"q": "问题2...", "a": "答案2..."}}
        ]
        """
        
        response_text = "" # 初始化变量
        
        try:
            # 【新添加】增加调用日志
            print(f"  > 正在连接 Google API 并发送请求 (模型: gemini-pro)... (如果卡在这里，说明是网络问题)")
            
            # 这是会卡住的行
            response = model.generate_content(prompt)
            
            # 【新添加】增加返回日志
            print(f"  > API 已成功返回。正在解析JSON...")
            response_text = response.text
            
            # 清理和解析模型的JSON输出
            json_str = response_text.strip().lstrip("```json").rstrip("```")
            qa_list = json.loads(json_str)
            
            # 将生成的Q&A转换为RAG格式
            for qa in qa_list:
                if "q" in qa and "a" in qa:
                    entry = {
                        "file": qa["a"],  # 答案是文档内容
                        "metadata": {
                            "source": "synthetic",
                            "topic": topic,
                            "question": qa["q"] # 问题是元数据
                        }
                    }
                    self.corpus.append(entry)
            
            print(f"成功为 '{topic}' 生成 {len(qa_list)} 条数据。")
            return True

        # --- 【新添加】详细的错误捕获 ---
        
        # 【已修改】移除了 google_errors.PermissionDeniedError
        except (google_api_exceptions.PermissionDenied, google_api_exceptions.Unauthenticated) as e:
            print("\n" + "="*50, file=sys.stderr)
            print(f"  [严重错误]：API 密钥无效或未启用！", file=sys.stderr)
            print(f"  [详情]：{e}", file=sys.stderr)
            print(f"  [解决方案]：请确认您已撤销旧密钥，并正在使用一个 *全新* 的、*有效* 的 API 密钥。", file=sys.stderr)
            print("  请检查您的 'GEMINI_API_KEY' 环境变量设置是否正确。", file=sys.stderr)
            print("="*50 + "\n", file=sys.stderr)
            return False # 停止这个主题的生成

        except (google_api_exceptions.DeadlineExceeded, TimeoutError) as e:
            print("\n" + "="*50, file=sys.stderr)
            print(f"  [严重错误]：网络连接超时！", file=sys.stderr)
            print(f"  [详情]：{e}", file=sys.stderr)
            print(f"  [解决方案]：脚本无法在规定时间内连接到 Google 服务器。")
            print("  请检查您的防火墙、代理或网络连接是否阻止了对 'generativelanguage.googleapis.com' 的访问。", file=sys.stderr)
            print("="*50 + "\n", file=sys.stderr)
            return False
        
        except json.JSONDecodeError:
            print(f"  [错误]: 模型返回的JSON格式不正确。跳过主题: {topic}", file=sys.stderr)
            print(f"  原始回复: {response_text}", file=sys.stderr)
            return False
        
        except Exception as e:
            print("\n" + "="*50, file=sys.stderr)
            print(f"  [未知错误]：在调用 API 时发生意外错误。", file=sys.stderr)
            print(f"  [详情]：{e}", file=sys.stderr)
            print(f"  [解决方案]：请检查您的网络和API密钥。")
            print("="*50 + "\n", file=sys.stderr)
            return False

    def save_corpus(self, filename: str):
        """
        将所有生成的语料保存到文件
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.corpus, f, indent=2, ensure_ascii=False)
        print(f"--- 完成 ---")
        print(f"总共生成 {len(self.corpus)} 条数据，已保存到 {filename}")

# --- 主题列表保持不变 ---
TOPIC_LIST = [
    "SQL注入的原理与防范",
    "跨站脚本攻击(XSS)的类型和缓解措施",
    "OWASP Top 10 详解",
    "RSA非对称加密算法的数学原理",
    "AES对称加密算法的工作模式",
    "哈希函数(MD5, SHA-256)的特性与碰撞",
    "零信任网络架构(Zero Trust)的核心原则"
]

if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    
    for topic in TOPIC_LIST:
        success = generator.generate_qa_pairs_for_topic(topic, count=5) 
        if not success:
            print(f"因发生错误，已停止在主题: {topic}", file=sys.stderr)
            # 如果您希望在遇到错误时立即停止整个脚本，请取消下面一行的注释
            # break 
        
        if success:
            time.sleep(5) # 避免触发API的速率限制
            
    generator.save_corpus("synthetic_security_corpus.json")