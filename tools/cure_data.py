import dspy
import json
import os
from tqdm import tqdm

class Filter(dspy.Signature):
    """Determine whether the picture is related to the theme of family environment."""
    image_1: dspy.Image = dspy.InputField()
    answer: str = dspy.OutputField(desc="Determine whether the picture is related to the theme of family environment. Only output Yes or No.")

def run_llm(json_path, images_dir, output_json_path):
    # 1. 初始化 DSPy 和 本地 VLM
    lm = dspy.LM(
        model="openai/qwen3.5", # 请确保你的本地模型支持多模态视觉输入
        api_base="http://127.0.0.1:8080/v1",
        api_key="unused",
        cache=False, 
    )
    dspy.configure(lm=lm)
    Judge = dspy.Predict(Filter)

    # 2. 读取原始 JSON 数据
    print(f"正在读取原始 JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. 提取独一无二的 image_id 集合，避免重复推理
    unique_image_ids = set(str(item['image_id']) for item in data)
    valid_image_ids = set() # 用来存放判定为 "家庭场景" 的图片 ID
    
    print(f"总共有 {len(unique_image_ids)} 张不重复的图片需要大模型判定。")

    # 4. 遍历所有图片进行推理
    for img_id in tqdm(unique_image_ids, desc="VLM 正在鉴别图片"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        
        if not os.path.exists(img_path):
            continue
            
        try:
            # 加载图片给 DSPy
            img_input = dspy.Image(img_path)
            
            # 让模型判定
            result = Judge(image_1=img_input)
            res = result.answer.strip()
            
            # 为了防止模型偶尔输出 "Yes." 或者包含大小写问题，做一点容错
            if "yes" in res.lower():
                valid_image_ids.add(img_id)
                
        except Exception as e:
            print(f"\n处理图片 {img_id}.jpg 时出错: {e}")
            
    # 5. 根据白名单 (valid_image_ids) 过滤原始 JSON 数据
    filtered_data = [item for item in data if str(item['image_id']) in valid_image_ids]
    
    # 6. 保存清洗后的新 JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*40)
    print("✅ 数据清洗完成！")
    print(f"图片清洗情况: 保留 {len(valid_image_ids)} 张 / 总计 {len(unique_image_ids)} 张")
    print(f"标注清洗情况: 保留 {len(filtered_data)} 条 / 总计 {len(data)} 条")
    print(f"清洗后的数据已保存至: {output_json_path}")
    print("="*40)

if __name__ == "__main__":
    # 请根据实际情况修改下面三个路径
    JSON_PATH = "yolo_dataset/home_ovd_samples.json"
    IMAGES_DIR = "yolo_dataset/images/val"
    OUTPUT_JSON = "yolo_dataset/home_ovd_filtered.json"
    
    run_llm(JSON_PATH, IMAGES_DIR, OUTPUT_JSON)
