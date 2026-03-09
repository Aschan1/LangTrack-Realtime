import json
import random
import os
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModel

def siglip_get_score(crop_img, phrase, processor, model, device):
    """
    修改为直接返回匹配概率的得分函数
    """
    text = f"a photo of {phrase}"
    
    inputs = processor(text=[text], images=crop_img, padding="max_length", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits_per_image = outputs.logits_per_image
    prob = torch.sigmoid(logits_per_image).item() 
    
    print(f"    [SigLIP] '{phrase}' 的匹配概率: {prob:.4f}")
    return prob

def visualize_pipeline(img_path, original_phrase, keywords, yolo_model, siglip_processor, siglip_model, device, output_path, gt_box=None, conf_thresh=0.2, siglip_thresh=0.15):
    # 读取图像 (OpenCV 用于画图，PIL 用于裁剪给 SigLIP)
    cv_img = cv2.imread(img_path)
    if cv_img is None:
        print(f"  [错误] 无法读取图片: {img_path}")
        return
        
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    # ==========================================
    # 优先画出 Ground Truth (黄色)
    # ==========================================
    if gt_box is not None:
        gx, gy, gw, gh = gt_box
        gx1, gy1, gx2, gy2 = int(gx), int(gy), int(gx + gw), int(gy + gh)
        
        cv2.rectangle(cv_img, (gx1, gy1), (gx2, gy2), (0, 255, 255), 3)
        
        label_gt = "Ground Truth"
        (w, h), _ = cv2.getTextSize(label_gt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(cv_img, (gx1, max(0, gy1 - 25)), (gx1 + w, gy1), (0, 255, 255), -1)
        cv2.putText(cv_img, label_gt, (gx1, max(0, gy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 1. 喂给 YOLO 所有的 keywords
    yolo_model.set_classes(keywords)
    results = yolo_model.predict(img_path, conf=conf_thresh, verbose=False)[0]
    
    raw_pred_boxes = results.boxes.xyxy.cpu().numpy()
    raw_pred_classes = results.boxes.cls.cpu().numpy()
    raw_pred_confs = results.boxes.conf.cpu().numpy()

    print(f"  -> YOLO 共找到了 {len(raw_pred_boxes)} 个候选框。")

    # ==========================================
    # 2. [第一遍] 遍历所有候选框，获取得分并寻找最高分
    # ==========================================
    box_scores = []
    best_score = -1.0
    best_idx = -1

    for idx, (p_box, p_cls, p_conf) in enumerate(zip(raw_pred_boxes, raw_pred_classes, raw_pred_confs)):
        x1, y1, x2, y2 = map(int, p_box)
        
        # 防越界保护
        if x2 <= x1 or y2 <= y1:
            box_scores.append(-1.0)
            continue
            
        crop_img = pil_img.crop((x1, y1, x2, y2))
        
        # 获取绝对概率
        score = siglip_get_score(crop_img, original_phrase, siglip_processor, siglip_model, device)
        box_scores.append(score)
        
        # 记录最高分
        if score > best_score:
            best_score = score
            best_idx = idx

    # 判断是否有任何框及格（超过阈值）
    any_passed = any(score > siglip_thresh for score in box_scores)
    if not any_passed and best_score > 0:
        print(f"  -> [保底机制触发] 所有框均未达到阈值 {siglip_thresh}，放行最高分框 (得分: {best_score:.4f})")

    # ==========================================
    # 3. [第二遍] 根据得分和保底逻辑进行绘制
    # ==========================================
    for idx, (p_box, p_cls, p_conf) in enumerate(zip(raw_pred_boxes, raw_pred_classes, raw_pred_confs)):
        score = box_scores[idx]
        if score < 0:
            continue
            
        x1, y1, x2, y2 = map(int, p_box)
        detected_keyword = keywords[int(p_cls)]
        
        # --- 核心判断逻辑 ---
        if any_passed:
            # 正常模式：过阈值的都算 PASS
            is_valid = score > siglip_thresh
        else:
            # 保底模式：只让得分最高的那个 PASS
            is_valid = (idx == best_idx)
        
        # 设置颜色和文本
        if is_valid:
            color = (0, 255, 0) # 绿色
            status = f"PASS ({score:.2f})"
            thickness = 3 if not any_passed else 2 # 触发保底时把这根“独苗”画粗一点
        else:
            color = (0, 0, 255) # 红色
            status = f"FAIL ({score:.2f})"
            thickness = 2

        # 画框
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, thickness)
        
        # 贴标签
        label = f"{detected_keyword} {status}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(cv_img, (x1, max(0, y1 - 20)), (x1 + w, y1), color, -1)
        cv2.putText(cv_img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 4. 保存可视化结果
    cv2.imwrite(output_path, cv_img)
    print(f"  -> 图片已保存至: {output_path}")


if __name__ == "__main__":
    # 配置路径
    JSON_FILE = "yolo_dataset/home_ovd_subset_30.json" 
    IMAGES_DIR = "yolo_dataset/filtered_images"
    OUTPUT_DIR = "visualizations"
    
    # 如果输出文件夹不存在，则创建
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==========================================
    # 提前加载模型
    # ==========================================
    print("正在加载 YOLO 和 SigLIP 模型，请稍候...")
    yolo_model = YOLO("ultralytics/yolov8s-worldv2.pt").to(device)
    siglip_processor = AutoProcessor.from_pretrained("/home/chen/workplace/LangTrack-Realtime/ultralytics/SigLIP/siglip2-base-patch16-224")
    siglip_model = AutoModel.from_pretrained("/home/chen/workplace/LangTrack-Realtime/ultralytics/SigLIP/siglip2-base-patch16-224").to(device)
    siglip_model.eval()
    print("模型加载完成！")

    # ==========================================
    # 读取 JSON 并随机采样
    # ==========================================
    print(f"\n正在读取数据集: {JSON_FILE}")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机抽取 10 个样例
    num_samples = min(10, len(data))
    random_samples = random.sample(data, num_samples)
    
    print(f"成功抽取 {num_samples} 个样例，开始可视化...\n")

    for i, item in enumerate(random_samples):
        print(f"[{i+1}/{num_samples}] 正在处理 Region ID: {item.get('region_id')} | Phrase: {item.get('phrase')}")
        
        img_id = str(item['image_id'])
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        
        original_phrase = item['phrase']
        keywords = item.get('keywords', None) 
        keywords = [keywords['target']] if keywords else [original_phrase]
        gt_box = [item['x'], item['y'], item['width'], item['height']]
        
        output_path = os.path.join(OUTPUT_DIR, f"vis_sample_{i+1}_img{img_id}.jpg")
        
        if not os.path.exists(img_path):
            print(f"  [警告] 找不到图片文件: {img_path}，跳过该样例。")
            continue
            
        visualize_pipeline(
            img_path=img_path,
            original_phrase=original_phrase,
            keywords=keywords,
            yolo_model=yolo_model,
            siglip_processor=siglip_processor,
            siglip_model=siglip_model,
            device=device,
            output_path=output_path,
            gt_box=gt_box,
            conf_thresh=0.02,
            siglip_thresh=0.15  # 你可以在这里调整及格线阈值
        )
        print("-" * 50)
        
    print(f"\n全部完成！请前往 '{OUTPUT_DIR}' 文件夹查看结果。")