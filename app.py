from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
from api_client import APIClient
from data_processor import extract_context, files_to_citations
from prompt_builder import build_chat_prompt
from guard import validate_user_input, validate_prompt
from response_evaluator import integrate_with_rag_flow
from config import config
import time
import requests
from typing import List, Dict, Tuple
import logging
import uuid
from sentence_transformers import CrossEncoder

app = Flask(__name__)
CORS(app)

# 全局变量存储对话历史和数据库名
conversations: Dict[str, Tuple[str, List[Dict[str, str]]]] = {} # (title, history_list)
db_name = None

print("⏳ 正在加载二次检索模型 (Re-ranker)...")
try:
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✅ 二次检索模型加载成功!")
except Exception as e:
    print(f"❌ 加载二次检索模型失败: {e}")
    reranker_model = None

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO。DEBUG日志将不显示，INFO, WARNING, ERROR 都会记录。
    filename='app_security.log',  # 指定日志输出到的文件名
    filemode='a',  # 'a' = append (追加模式), 'w' = write (覆盖模式)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # 定义日志行的格式
    encoding='utf-8' # 确保中文日志（如敏感词）不会乱码
)


client = APIClient()

# --- 新增 ---: 封装二次检索逻辑的辅助函数
def rerank_documents(query: str, documents: List[Dict], model: CrossEncoder, top_n: int = 5) -> List[Dict]:
    """
    使用 Cross-Encoder 模型对检索到的文档进行重新排序。
    """
    if not documents or not isinstance(documents, list) or not model:
        return documents[:top_n] if isinstance(documents, list) else []

    pairs = []
    for doc in documents:
        if isinstance(doc, dict):
            text = doc.get('file_content') or doc.get('file') or doc.get('content') or ''
        else:
            text = str(doc or '')
        pairs.append([query, text])
    
    # 模型预测，获取相关性分数
    scores = model.predict(pairs, show_progress_bar=False)
    
    # 将分数与原始文档绑定并排序
    combined_results = []
    for i in range(len(documents)):
        combined_results.append({
            'score': scores[i],
            'document': documents[i] 
        })
    combined_results.sort(key=lambda x: x['score'], reverse=True)
    
    # 提取排序后的前 N 个文档
    reranked_docs = [res['document'] for res in combined_results]
    
    return reranked_docs[:top_n]

def initialize_database():
    """初始化数据库"""
    global db_name
    db_name = f"student_{config.USER_NAME}_final"
    
    try:
        create_resp = requests.post(
            f"{config.BASE_URL}/databases",
            json={
                "database_name": db_name,
                "token": config.TOKEN,
                "metric_type": config.DEFAULT_METRIC_TYPE
            }
        )
        
        if create_resp.status_code != 200:
            print(f"创建数据库失败: {create_resp.text}")
            return False
            
        print(f"数据库创建成功: {db_name}")
        
        # 上传测试数据
        files = [
            {"file": "hello world, 网络安全测试", "metadata": {"description": "测试文件1"}},
            {"file": "第二条测试文本", "metadata": {"description": "测试文件2"}},
            {"file": "网络安全是指保护网络系统及其数据免受攻击、损坏或未经授权访问的过程。",
                "metadata": {"description": "网络安全定义"}},
            {"file": "防火墙是一种网络安全系统,用于监控和控制传入和传出的网络流量。",
                "metadata": {"description": "防火墙定义"}}
        ]
        
        payload = {
            "files": files,
            "token": config.TOKEN
        }
        
        resp = requests.post(
            f"{config.BASE_URL}/databases/{db_name}/files", json=payload)
            
        if resp.status_code == 200:
            print(f"测试数据上传成功")
            time.sleep(config.WAIT_TIME)
            return True
        else:
            print(f"数据上传失败: {resp.text}")
            return False
            
    except Exception as e:
        print(f"初始化数据库时出错: {e}")
        return False

#首页路由
@app.route('/')
def index():
    """返回根目录的 index.html"""
    return render_template('index.html')

@app.route('/history', methods=['GET'])
def get_history_list():
    """返回所有对话的ID和标题列表"""
    history_summary = [
        {"id": conv_id, "title": data[0]} 
        for conv_id, data in conversations.items()
    ]
    return jsonify(sorted(history_summary, key=lambda x: x['id'], reverse=True))

# --- 新增API：获取特定对话的完整内容 ---
@app.route('/history/<conversation_id>', methods=['GET'])
def get_conversation_history(conversation_id):
    """根据ID返回特定对话的完整消息历史"""
    if conversation_id in conversations:
        return jsonify({"messages": conversations[conversation_id][1]})
    return jsonify({"error": "Conversation not found"}), 404

# 聊天核心路由
@app.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求 - 集成了二次检索功能"""
    
    # ========== 1. 接收和验证输入 (不变) ==========
    data = request.get_json(silent=True) or {}
    msg = data.get('message', None)
    if isinstance(msg, dict):
        msg = msg.get('text') or msg.get('content') or msg.get('value')
    user_input = str(msg or '').strip()

    conversation_id = data.get('conversation_id')
    enable_evaluation = bool(data.get('enable_evaluation', False))

    if not user_input:
        return jsonify({'error': '消息不能为空，或 message 不是字符串'}), 400
    
    if not validate_user_input(user_input):
        return jsonify({'error': '您的输入包含敏感内容或过长，请修改后重试'}), 400
    
    if not conversation_id or conversation_id not in conversations:
        conversation_id = str(uuid.uuid4())
        title = user_input[:30] + "..." if len(user_input) > 30 else user_input
        conversations[conversation_id] = (title, [])
    
    current_history = conversations[conversation_id][1]

    try:
        # 一次检索：返回 { "files": [...] } 或 { "results": [...] }
        initial_results = client.search(db_name, user_input, top_k=20)

        # 提取出文档列表（兼容 'files' 或 'results'）
        initial_docs = initial_results.get('files', initial_results.get('results', []))
        
        # ========== 新增步骤: 2.5 二次检索 (Re-ranking) ==========
        # --- 新增 ---: 使用 rerank_documents 函数对初步结果进行精排。
        reranked_results = rerank_documents(
            query=user_input, 
            documents=initial_docs, 
            model=reranker_model, 
            top_n=5  # 最终选择最相关的 5 个文档
        )
        
        # ========== 3. 提取上下文和引用 ==========
         # 用二次检索后的结果构建上下文与引用（包一层保持原接口期望的字典结构）
        context = extract_context({"results": reranked_results})
        citations = files_to_citations({"results": reranked_results})
        
        # ========== 4. 构建 Prompt (不变) ==========
        prompt = build_chat_prompt(current_history, user_input, context, citations)
        
        # ========== 5. Prompt 安全检测 (不变) ==========
        if not validate_prompt(prompt):
            return jsonify({'error': '生成的提示词存在安全风险'}), 400
        
        # ========== 6. 生成回答 (不变) ==========
        response = client.dialogue(prompt)
        
        # ========== 7. 更新对话历史 (不变) ==========
        current_history.append({"role": "user", "content": user_input})
        current_history.append({"role": "assistant", "content": response})
        
        # ========== 8. 准备响应数据 (不变) ==========
        response_data = {
            'response': response,
            'citations': citations,
            'conversation_id': conversation_id
        }
        
        # ========== 9. 可选：回答质量评估 (不变) ==========
        if enable_evaluation:
            _, evaluation_report = integrate_with_rag_flow(
                response, user_input, context
            )
            response_data['evaluation'] = evaluation_report
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"处理请求时出错: {e}")
        return jsonify({'error': f'处理请求失败: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    """清空所有对话历史"""
    global conversations
    conversations = {}
    return jsonify({'status': 'success', 'message': 'All conversations cleared'})

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'database': db_name})


# ✅ 启动时的输出信息
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("⏳ 正在初始化数据库...")
    print("=" * 50 + "\n")
    
    if initialize_database():
        print("\n" + "=" * 50)
        print("🚀 服务启动成功！")
        print("📱 请在浏览器访问: http://localhost:5000/")
        print("💡 提示: 按 Ctrl+C 停止服务")
        print("=" * 50 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader = False)
    else:
        print("\n" + "=" * 50)
        print("❌ 数据库初始化失败，请检查配置")
        print("💡 检查项:")
        print("   - VECTOR_DB_BASE_URL 是否正确")
        print("   - TOKEN 是否有效")
        print("   - 向量库服务是否在运行")
        print("=" * 50 + "\n")