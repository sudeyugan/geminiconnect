# 文件名: process_CQA_data.py (v4 - 移除 metadata)

import json
import re
import textwrap

# 输入文件 (包含 CQA 三元组)
INPUT_FILE = "QA_DATA.txt"
# 输出文件 (处理后的JSON语料库)
OUTPUT_FILE = "processed_cqa_corpus.json"

def convert_cqa_data_robustly():
    """
    使用正则表达式 robustly（健壮地）处理 QA_DATA.txt 文件，
    将其从 (Context, Question, Answer) 三元组格式
    转换为 RAG 语料库所需的JSON列表格式。
    
    [!] 更新 v4: 此版本只输出 {context, question, answer}，移除了 metadata。
    """
    print(f"--- 开始处理CQA三元组 (无 metadata, 清理全角括号): {INPUT_FILE} ---")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except FileNotFoundError:
        print(f"错误: 文件 {INPUT_FILE} 未找到。请确保文件名正确。")
        return

    rag_corpus = []
    
    # 1. 定义用于查找 C-Q-A 三元组的正则表达式
    cqa_triplet_regex = re.compile(
        r'\(\s*"(.+?)"\s*,\s*"(.+?)"\s*,\s*"(.+?)"\s*\)', 
        re.DOTALL
    )
    
    # 2. 定义用于按主题分割文件的正则表达式
    topic_splitter = re.compile(r'"([^"]+)":\s*\[')
    
    # 3. 按主题分割文件
    parts = topic_splitter.split(file_content)
    
    if len(parts) <= 1:
        print("错误: 未能在文件中找到任何主题（例如 '\"网络安全基础\": ['）。请检查文件格式。")
        return

    # 4. 遍历分割后的部分
    total_found = 0
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            current_topic = parts[i].strip()
            cqa_block = parts[i+1]
            
            print(f"  > 正在查找主题: '{current_topic}'")
            
            # 5. 在当前主题的文本块中查找所有匹配的 C-Q-A 三元组
            matches = cqa_triplet_regex.findall(cqa_block)
            
            if not matches:
                print(f"    - 未在 '{current_topic}' 下找到任何 *完整* 的 CQA 三元组。")
                continue
            
            print(f"    - 成功找到 {len(matches)} 个 CQA 三元组。")
            total_found += len(matches)
            
            # 6. 将找到的 C-Q-A 对转换为 RAG 格式
            for context, question, answer in matches:

                def clean_text(text):
                    cleaned = text.strip()
                    if cleaned.startswith(('"""', '"')):
                        cleaned = cleaned.strip('"""').strip('"')
                    cleaned = textwrap.dedent(cleaned)
                    
                    # --- [!] 核心修正：移除所有全角括号及其内容 ---
                    cleaned = re.sub(r'（.*?）', '', cleaned)
                    
                    cleaned = cleaned.strip()
                    return cleaned

                cleaned_context = clean_text(context)
                cleaned_question = clean_text(question)
                cleaned_answer = clean_text(answer)

                # 7. [!] 构建新的 JSON 条目 (无 metadata)
                entry = {
                    "context": cleaned_context,
                    "question": cleaned_question,
                    "answer": cleaned_answer
                }
                rag_corpus.append(entry)

    if not rag_corpus:
        print("错误: 未能从文件中解析出任何完整的 CQA 三元组。")
        return

    # 8. 保存为新的JSON文件
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(rag_corpus, f, indent=2, ensure_ascii=False)
        
        print(f"\n--- 转换成功! ---")
        print(f"总共处理了 {total_found} 条 CQA 三元组。")
        print(f"已保存到: {OUTPUT_FILE}")
        print(f"输出格式: {{context, question, answer}} (无 metadata)")
        
    except Exception as e:
        print(f"保存到 {OUTPUT_FILE} 时出错: {e}")

if __name__ == "__main__":
    convert_cqa_data_robustly()