import dspy
import json
import os
from tqdm import tqdm

class Filter(dspy.Signature):
    """Determine whether the picture is related to the theme of family environment."""
    image_1: dspy.Image = dspy.InputField()
    answer: str = dspy.OutputField(desc="Determine whether the picture is related to the theme of family environment. Only output Yes or No.")

def run_llm(json_path, images_dir, output_json_path):
    # 1. Initialize DSPy and local VLM
    lm = dspy.LM(
        model="openai/qwen3.5", # Make sure your local model supports multimodal visual input
        api_base="http://127.0.0.1:8080/v1",
        api_key="unused",
        cache=False, 
    )
    dspy.configure(lm=lm)
    Judge = dspy.Predict(Filter)

    # 2. Read original JSON data
    print(f"Reading original JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. Extract unique image_id set to avoid duplicate inference
    unique_image_ids = set(str(item['image_id']) for item in data)
    valid_image_ids = set() # Used to store image IDs judged as "home scene"
    
    print(f"There are {len(unique_image_ids)} unique images that need to be judged by the large model.")

    # 4. Iterate through all images for inference
    for img_id in tqdm(unique_image_ids, desc="VLM is identifying images"):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        
        if not os.path.exists(img_path):
            continue
            
        try:
            # Load image for DSPy
            img_input = dspy.Image(img_path)
            
            # Let the model judge
            result = Judge(image_1=img_input)
            res = result.answer.strip()
            
            # To prevent the model from occasionally outputting "Yes." or case issues, add some tolerance
            if "yes" in res.lower():
                valid_image_ids.add(img_id)
                
        except Exception as e:
            print(f"\nError processing image {img_id}.jpg: {e}")
            
    # 5. Filter original JSON data based on whitelist (valid_image_ids)
    filtered_data = [item for item in data if str(item['image_id']) in valid_image_ids]
    
    # 6. Save the cleaned new JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*40)
    print("✅ Data cleaning completed!")
    print(f"Image cleaning status: Retained {len(valid_image_ids)} / Total {len(unique_image_ids)}")
    print(f"Annotation cleaning status: Retained {len(filtered_data)} / Total {len(data)}")
    print(f"Cleaned data saved to: {output_json_path}")
    print("="*40)

if __name__ == "__main__":
    # Please modify the following three paths according to your actual situation
    JSON_PATH = "yolo_dataset/home_ovd_samples.json"
    IMAGES_DIR = "yolo_dataset/images/val"
    OUTPUT_JSON = "yolo_dataset/home_ovd_filtered.json"
    
    run_llm(JSON_PATH, IMAGES_DIR, OUTPUT_JSON)
