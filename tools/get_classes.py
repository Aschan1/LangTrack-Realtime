import json
import ast
import time
import dspy
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from LLMs.get_prompt import PromptProcessor

# parse_dspy_to_dict 函数保持不变...
def parse_dspy_to_dict(llm_result, original_phrase):
    try:
        result_dict = llm_result.toDict()
        parsed_data = {
            "target": result_dict.get("target", original_phrase),
            "attributes": result_dict.get("attributes", []),
            "anchor_object": result_dict.get("anchor_object"),
            "spatial_relation": result_dict.get("spatial_relation")
        }
        for key in ['anchor_object', 'spatial_relation']:
            val = parsed_data.get(key)
            if isinstance(val, str) and val.strip().lower() in ['null', 'none', '']:
                parsed_data[key] = None
        attr_val = parsed_data["attributes"]
        if isinstance(attr_val, str):
            try:
                parsed_data["attributes"] = ast.literal_eval(attr_val)
            except (ValueError, SyntaxError):
                parsed_data["attributes"] = [attr_val]
        elif not isinstance(attr_val, list):
            parsed_data["attributes"] = []
        return parsed_data
    except Exception:
        return {"target": original_phrase, "attributes": [], "anchor_object": None, "spatial_relation": None}

def preprocess_json_with_llm(input_json_path, output_json_path, max_workers=24, retries=1, limit_images=None):
    print("Initializing DSPy and LLM...")
    main_lm = dspy.LM(model="openai/qwen3.5", api_base="http://127.0.0.1:8080/v1", api_key="unused", cache=False)
    dspy.configure(lm=main_lm)

    prompt_processor = dspy.Predict(PromptProcessor)
    try:
        prompt_processor.load("./LLMs/optimized_prompt.json")
        print("✅ Loaded optimized_prompt.json successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Using default prompt. Error: {e}")

    # 1. 加载数据并进行图片采样
    print(f"Loading data from {input_json_path}...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    # 找到所有的图片 ID
    all_image_ids = sorted(list(set(item["image_id"] for item in full_data)))
    
    if limit_images:
        # 随机抽取指定数量的图片 ID
        sampled_image_ids = set(random.sample(all_image_ids, min(limit_images, len(all_image_ids))))
        # 仅保留属于这些图片的条目
        data = [item for item in full_data if item["image_id"] in sampled_image_ids]
        print(f"📊 Subset Mode: Sampled {limit_images} images, resulting in {len(data)} bounding boxes.")
    else:
        data = full_data
        print(f"📊 Full Mode: Processing all {len(all_image_ids)} images.")

    # 2. 提取子集中的唯一 phrase
    unique_phrases = sorted(set(item["phrase"] for item in data))
    print(f"Unique phrases to extract: {len(unique_phrases)}")

    phrase_to_parsed_json = {}
    print(f"🚀 Concurrency: {max_workers} threads...")

    def process_one_phrase(phrase: str):
        for attempt in range(retries + 1):
            try:
                llm_result = prompt_processor(phrase=phrase)
                return phrase, parse_dspy_to_dict(llm_result, phrase), None
            except Exception as e:
                time.sleep(0.05 * (attempt + 1))
        return phrase, {"target": phrase, "attributes": [], "anchor_object": None, "spatial_relation": None}, "Error"

    # 3. 并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_phrase, p) for p in unique_phrases]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting knowledge"):
            phrase, parsed_json_dict, _ = fut.result()
            phrase_to_parsed_json[phrase] = parsed_json_dict

    # 4. 写回并保存
    for item in data:
        item["keywords"] = phrase_to_parsed_json.get(item["phrase"])

    print(f"Saving to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("🎉 Done!")

if __name__ == "__main__":
    INPUT_JSON = "yolo_dataset/home_ovd_filtered.json"
    OUTPUT_JSON = "yolo_dataset/home_ovd_keywords_json_full.json"

    # 设置 limit_images=30 即可只处理 30 张图片的数据
    preprocess_json_with_llm(INPUT_JSON, OUTPUT_JSON, max_workers=24, limit_images=None)