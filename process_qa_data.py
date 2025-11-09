import json
import re

# 输入文件现在是 .txt 文件
INPUT_FILE = "QA_DATA.txt"
# 输出文件保持不变
OUTPUT_FILE = "processed_qa_data.json"

def convert_qa_data_robustly():
    """
    使用正则表达式 robustly（健壮地）处理 QA_DATA.txt 文件，
    将其转换为RAG所需的JSON列表格式。
    这个版本可以容忍文件末尾的语法错误或不完整条目。
    """
    print(f"开始健壮地处理: {INPUT_FILE}")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except FileNotFoundError:
        print(f"错误: 文件 {INPUT_FILE} 未找到。请确保文件名正确。")
        return

    rag_corpus = []
    
    # 1. 定义用于查找 Q&A 对的正则表达式
    # re.DOTALL 标志允许 '.' 匹配换行符，这对于多行答案至关重要
    qa_pair_regex = re.compile(
        r'\(\s*"(.+?)"\s*,\s*"(.+?)"\s*\)', 
        re.DOTALL
    )
    
    # 2. 定义用于按主题分割文件的正则表达式
    # 这将匹配 "主题名": [ 并捕获 "主题名"
    topic_splitter = re.compile(r'"([^"]+)":\s*\[')
    
    # 3. 按主题分割文件
    # 这将产生一个列表，格式为: [文件头部杂项, '主题1', 'Q&A块1', '主题2', 'Q&A块2', ...]
    parts = topic_splitter.split(file_content)
    
    if len(parts) <= 1:
        print("错误: 未能在文件中找到任何主题（例如 '\"网络安全基础\": ['）。请检查文件格式。")
        return

    # 4. 遍历分割后的部分
    # 我们跳过列表中的第一项（文件头部杂项）
    # 每次跳 2 步：parts[i] 是主题名, parts[i+1] 是包含Q&A的文本块
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            current_topic = parts[i].strip()
            qa_block = parts[i+1]
            
            print(f"  > 正在查找主题: '{current_topic}'")
            
            # 5. 在当前主题的文本块中查找所有匹配的 Q&A 对
            matches = qa_pair_regex.findall(qa_block)
            
            if not matches:
                print(f"    - 未在 '{current_topic}' 下找到任何 *完整* 的Q&A对。")
                continue
                
            # 6. 将找到的 Q&A 对转换为 RAG 格式
            for question, answer in matches:
                entry = {
                    "file": answer.strip(),  # 答案
                    "metadata": {
                        "source": "manual_qa_txt",
                        "topic": current_topic,
                        "question": question.strip() # 问题
                    }
                }
                rag_corpus.append(entry)
            
            print(f"    - 成功找到 {len(matches)} 个Q&A对。")

    if not rag_corpus:
        print("错误: 未能从文件中解析出任何完整的Q&A对。")
        return

    # 7. 保存为新的JSON文件
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(rag_corpus, f, indent=2, ensure_ascii=False)
        
        print(f"\n--- 转换成功! ---")
        print(f"总共处理了 {len(rag_corpus)} 条 Q&A 对。")
        print(f"已保存到: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"保存到 {OUTPUT_FILE} 时出错: {e}")

if __name__ == "__main__":
    convert_qa_data_robustly()