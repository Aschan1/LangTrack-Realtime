from run_yolo import get_predictions_and_gts
import json
import os
from PIL import Image
import dspy
from tqdm import tqdm
from typing import List

# 1. 优化后的 DSPy Signature
class LLMFilter(dspy.Signature):
    """You are a visual AI assistant. Analyze the image and the provided candidate bounding boxes. 
    Find the one bounding box that best matches the described object and its attributes.
    """
    image: dspy.Image = dspy.InputField(desc="Image input with bounding boxes for evaluation.")
    target_description: str = dspy.InputField(desc="The object to look for.")
    candidate_boxes: str = dspy.InputField(desc="List of candidate bounding boxes in the format 'ID: [x1, y1, x2, y2]'.")
    best_box_id: int = dspy.OutputField(desc="Only output the integer ID of the best matching bounding box. If none match, output -1.")

def run_llm(json_path, images_dir):
    all_gts, all_preds = get_predictions_and_gts(json_path, images_dir)

    img_to_anns = {}
    for item in tqdm(all_preds, desc='Loading predictions'):
        img_id = str(item['image_id'])
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(item)
    
    # 2. 配置 DSPy
    main_lm = dspy.LM(
        model="openai/qwen3.5", 
        api_base="http://127.0.0.1:8080/v1",
        api_key="unused",
        cache=False, 
    )
    dspy.configure(lm=main_lm)
    
    # 使用 Predict 模块（也可以换成 dspy.ChainOfThought，但我们在 Signature 里已经写了 reasoning）
    Selector = dspy.Predict(LLMFilter)
    
    # 3. 循环调用 LLM
    for img_id, preds in tqdm(img_to_anns.items(), desc='Filtering with LLM'):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
            
        # 假设我们这轮要找的是 "blue shirt" (实际应用中你需要从你的 GT 里拿到这个词)
        # 这里仅作示例：
        current_target = "shirt"
        current_attribute = "blue"
        description = f"{current_attribute} {current_target}".strip()
        
        # 核心技巧：把 Bounding Boxes 格式化为带有 ID 的字符串列表
        # 比如： "0: [10, 20, 50, 60] \n 1: [100, 100, 200, 200]"
        box_str_list = []
        for idx, p in enumerate(preds):
            # 取整，VLM 对整数坐标更敏感
            box_int = [int(coord) for coord in p['box']]
            box_str_list.append(f"Box ID {idx}: {box_int}")
            
        candidate_boxes_str = "\n".join(box_str_list)
        
        try:
            # 加载图片并推理
            img_input = dspy.Image(img_path)
            result = Selector(
                image=img_input,
                target_description=description,
                candidate_boxes=candidate_boxes_str
            )
            
            # 解析结果
            best_id = int(result.best_box_id)
            print(f"\nImage: {img_id}")
            print(f"Looking for: {description}")
            print(f"LLM Reasoning: {result.reasoning}")
            print(f"LLM Selected Box ID: {best_id}")
            
            if best_id != -1 and best_id < len(preds):
                best_prediction = preds[best_id]
                # 接下来可以保存或处理这个 best_prediction
                
        except Exception as e:
            pass # 实际代码中记得打印错误