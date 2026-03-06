import os
import cv2
import random
import glob

def load_yaml_names(yaml_path):
    """手动解析我们刚才生成的 dataset.yaml，提取 names 字典"""
    names_dict = {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        in_names_section = False
        for line in f:
            line = line.strip()
            if line.startswith('names:'):
                in_names_section = True
                continue
            if in_names_section and ':' in line:
                try:
                    # 解析类似 "  0: 'the clock is green in colour'"
                    cls_id_str, cls_name = line.split(':', 1)
                    cls_id = int(cls_id_str.strip())
                    # 去掉名字两边的空格和引号
                    cls_name = cls_name.strip().strip("'").strip('"')
                    names_dict[cls_id] = cls_name
                except ValueError:
                    continue
    return names_dict

def verify_yolo_format(yaml_path, images_dir, labels_dir, output_dir, sample_name="1"):
    """读取指定的 txt 和原图，反向计算坐标并画图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载类别字典
    names_dict = load_yaml_names(yaml_path)
    print(f"✅ 成功从 yaml 加载了 {len(names_dict)} 个类别。")
    
    # 2. 找到对应的原图和 txt 标签
    img_path = os.path.join(images_dir, f"{sample_name}.jpg")
    txt_path = os.path.join(labels_dir, f"{sample_name}.txt")
    
    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        print(f"❌ 找不到图片或标签文件：\n图片: {img_path}\n标签: {txt_path}")
        return

    # 3. 读取图片获取真实宽高
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法使用 OpenCV 读取图片: {img_path}")
        return
        
    img_h, img_w = img.shape[:2]
    
    # 4. 读取 txt 并反推坐标画图
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"📄 开始绘制 {sample_name}.jpg，共有 {len(lines)} 个目标框...")
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
            
        cls_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        w_norm = float(parts[3])
        h_norm = float(parts[4])
        
        # 核心：将 YOLO 归一化坐标反算为绝对像素坐标
        # abs_x = x_center_norm * img_w
        abs_w = w_norm * img_w
        abs_h = h_norm * img_h
        abs_x_center = x_center_norm * img_w
        abs_y_center = y_center_norm * img_h
        
        # 计算左上角和右下角坐标
        x_min = int(abs_x_center - abs_w / 2.0)
        y_min = int(abs_y_center - abs_h / 2.0)
        x_max = int(abs_x_center + abs_w / 2.0)
        y_max = int(abs_y_center + abs_h / 2.0)
        
        # 获取文本描述
        phrase = names_dict.get(cls_id, f"Unknown_{cls_id}")
        
        # 画框 (红色，B,G,R = 0,0,255)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        # 画底色和文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(phrase, font, font_scale, font_thickness)
        
        text_y = max(y_min, text_size[1] + 5)
        cv2.rectangle(img, (x_min, text_y - text_size[1] - 4), (x_min + text_size[0], text_y + 2), (0, 0, 0), -1)
        cv2.putText(img, phrase, (x_min, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
    # 5. 保存并输出
    save_path = os.path.join(output_dir, f"verified_{sample_name}.jpg")
    cv2.imwrite(save_path, img)
    print(f"🎉 验证图片已保存至: {save_path}")

if __name__ == "__main__":
    # --- 目录配置 ---
    YAML_PATH = "./yolo_dataset/dataset.yaml"      # 刚才生成的 yaml
    LABELS_DIR = "./yolo_dataset/labels"           # 刚才生成的 txt 文件夹
    IMAGES_DIR = "/media/chen/study/VisualGenome/Home_data/"                        # 原始图片文件夹
    OUTPUT_DIR = "."           # 验证结果保存位置
    
    # 你想检查的图片名（不带后缀），例如 "1"
    SAMPLE_NAME = "3"                              
    
    verify_yolo_format(YAML_PATH, IMAGES_DIR, LABELS_DIR, OUTPUT_DIR, SAMPLE_NAME)