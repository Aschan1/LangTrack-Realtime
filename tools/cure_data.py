import dspy
import json
import os
import threading
from tqdm import tqdm
import concurrent.futures

class Filter(dspy.Signature):
    """Determine whether the picture is related to the theme of family environment."""
    image_1: dspy.Image = dspy.InputField()
    answer: str = dspy.OutputField(desc="Determine whether the picture is related to the theme of family environment. Only output Yes or No.")

# 新增了 target_count 参数，控制需要的图片数量
def run_llm(json_path, images_dir, output_json_path, target_count=100, max_workers=4):
    # 1. Initialize DSPy and local VLM
    lm = dspy.LM(
        model="openai/qwen3.5", 
        api_base="http://127.0.0.1:8082/v1",
        api_key="unused",
        cache=False, 
    )
    dspy.configure(lm=lm)
    Judge = dspy.Predict(Filter)

    # 2. Read original JSON data
    print(f"Reading original JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. Extract unique image_id set
    unique_image_ids = list(data.keys())
    valid_image_ids = set() 
    
    print(f"There are {len(unique_image_ids)} unique images. We need to find {target_count} valid images.")

    # 引入停止标志，用于通知其他线程不要再发请求了
    stop_event = threading.Event()

    # 定义单张图片的处理逻辑，方便交给线程池
    def process_image(img_id):
        # 如果已经发出了停止信号，直接返回 False，不再请求 VLM
        if stop_event.is_set():
            return img_id, False

        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            return img_id, False
            
        try:
            img_input = dspy.Image(img_path)
            result = Judge(image_1=img_input)
            res = result.answer.strip()
            if "yes" in res.lower():
                return img_id, True
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            pass
            
        return img_id, False

    # 4. 使用线程池并发推理
    print(f"Starting inference with {max_workers} threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到线程池
        futures = {executor.submit(process_image, img_id): img_id for img_id in unique_image_ids}
        
        # 使用 tqdm 监控并发进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(unique_image_ids), desc="VLM is identifying images"):
            img_id, is_valid = future.result()
            
            if is_valid:
                valid_image_ids.add(img_id)
                # 检查是否达到了指定的数量
                if len(valid_image_ids) >= target_count:
                    print(f"\n🎉 已成功找到 {target_count} 张符合条件的图片，正在停止任务...")
                    
                    # 触发停止事件，让正在运行的线程跳过推理
                    stop_event.set()
                    
                    # 取消线程池队列中还没开始的 future 任务
                    for f in futures:
                        f.cancel()
                        
                    # 跳出收集循环
                    break
            
    # 5. Filter original JSON data
    filtered_data = {k: v for k, v in data.items() if k in valid_image_ids}
    
    # 6. Save the cleaned new JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*40)
    print("✅ Data cleaning completed!")
    print(f"Image cleaning status: Retained {len(valid_image_ids)} / Total Search Space {len(unique_image_ids)}")
    print(f"Cleaned data saved to: {output_json_path}")
    print("="*40)

if __name__ == "__main__":
    JSON_PATH = "yolo_dataset/indoors_subset/filtered_indoors.json"
    IMAGES_DIR = "yolo_dataset/indoors_subset/images"
    OUTPUT_JSON = "yolo_dataset/filtered_indoors_LM.json"
    
    # 在这里指定 target_count，例如找到 50 张满足条件的就停下来
    run_llm(JSON_PATH, IMAGES_DIR, OUTPUT_JSON, target_count=500, max_workers=12)