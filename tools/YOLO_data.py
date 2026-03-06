import json
import os
from PIL import Image

def convert_ovd_to_yolo(json_path, images_dir, output_dir):
    # 1. 读取 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 如果最外层是字典，可能是 {"annotations": [...]} 结构，适配一下
    if isinstance(data, dict):
        if 'annotations' in data:
            data = data['annotations']
        else:
            # 尝试获取字典里第一个列表类型的值
            for key, val in data.items():
                if isinstance(val, list):
                    data = val
                    break

    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    # 2. 收集所有独一无二的 phrase 作为类别 (YOLO-World 支持自然语言 Prompt)
    unique_phrases = list(set([item['phrase'] for item in data if 'phrase' in item]))
    phrase_to_id = {phrase: idx for idx, phrase in enumerate(unique_phrases)}
    
    # 3. 按 image_id 对标注进行分组
    img_to_anns = {}
    for item in data:
        img_id = str(item['image_id'])
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(item)
        
    # 4. 遍历处理每张图片
    processed_count = 0
    missing_images = 0
    
    for img_id, anns in img_to_anns.items():
        # 推测图片文件名，常见格式可能是 1.jpg 或 000001.jpg
        # 这里假设直接用 id + .jpg，如果你的图片名字是别的格式（比如 00001.jpg），请在这里修改 ↓
        img_filename = f"{img_id}.jpg" 
        img_path = os.path.join(images_dir, img_filename)
        
        if not os.path.exists(img_path):
            missing_images += 1
            continue
            
        # 获取整张图片的宽高
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            
        # 准备写入对应的 txt 标签文件
        txt_filename = f"{img_id}.txt"
        txt_filepath = os.path.join(labels_dir, txt_filename)
        
        with open(txt_filepath, 'w', encoding='utf-8') as txt_f:
            for ann in anns:
                x = ann['x']
                y = ann['y']
                w = ann['width']
                h = ann['height']
                
                # YOLO 需要中心点坐标，并进行归一化
                x_center = (x + w / 2.0) / img_w
                y_center = (y + h / 2.0) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # 防越界保护
                x_center, y_center = max(0, min(1, x_center)), max(0, min(1, y_center))
                norm_w, norm_h = max(0, min(1, norm_w)), max(0, min(1, norm_h))
                
                class_id = phrase_to_id[ann['phrase']]
                
                # 写入: class_id x_center y_center width height
                txt_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
        processed_count += 1
        
    # 5. 生成 YOLO-World 专用的 dataset.yaml
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write("path: ./yolo_dataset  # 请根据实际情况修改路径\n")
        f.write("train: images/train   # 训练集图片\n")
        f.write("val: images/val       # 验证集图片\n\n")
        f.write("names:\n")
        for phrase, class_id in phrase_to_id.items():
            # YOLO-World 会把这里的完整句子当作 Prompt
            # 处理一下单引号，避免 yaml 解析报错
            safe_phrase = phrase.replace("'", "''")
            f.write(f"  {class_id}: '{safe_phrase}'\n")
            
    print(f"\n✅ 转换完毕！")
    print(f"📊 共提取了 {len(unique_phrases)} 个独特的文本短语(类别)。")
    print(f"🖼️ 成功处理了 {processed_count} 张图片的标签。")
    if missing_images > 0:
        print(f"⚠️ 找不到 {missing_images} 张图片，已跳过。请检查 images_dir 路径或图片命名格式。")

if __name__ == "__main__":
    # === 请修改这里的路径 ===
    JSON_FILE = "home_ovd_samples.json"          # 你的 JSON 文件路径
    IMAGES_DIR = "/media/chen/study/VisualGenome/Home_data/" # 存放你原始图片的文件夹路径（必须有，为了读取图片尺寸）
    OUTPUT_DIR = "./yolo_dataset"                # 转换后结果存放的目录
    
    convert_ovd_to_yolo(JSON_FILE, IMAGES_DIR, OUTPUT_DIR)