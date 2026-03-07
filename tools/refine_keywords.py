import json

def add_phrase_to_keywords(input_json_path, output_json_path):
    print(f"正在读取文件: {input_json_path} ...")
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified_count = 0
    
    # 遍历数据集中的每一个字典
    for item in data:
        phrase = item.get("phrase", "")
        keywords = item.get("keywords", [])
        
        # 如果 phrase 不在 keywords 列表中，则将其加入
        if phrase and phrase not in keywords:
            # 插入到第 0 个位置，保持最具体在前
            keywords.insert(0, phrase)
            modified_count += 1
            
    # 将修改后的数据保存到新文件（或覆盖原文件）
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # indent=2 让输出的 JSON 保持良好的可读性
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"处理完成！共为 {modified_count} 个条目补充了 phrase。")
    print(f"新文件已保存至: {output_json_path}")

if __name__ == "__main__":
    # 替换为你实际的 JSON 文件路径
    INPUT_FILE = "yolo_dataset/home_ovd_filtered_with_keywords.json"
    OUTPUT_FILE = "yolo_dataset/home_ovd_filtered_with_keywords_fixed.json"
    
    add_phrase_to_keywords(INPUT_FILE, OUTPUT_FILE)