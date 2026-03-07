import os
import cv2
import random
import glob

def load_yaml_names(yaml_path):
    """Manually parse the dataset.yaml we just generated, extract the names dictionary"""
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
                    # Parse lines like "  0: 'the clock is green in colour'"
                    cls_id_str, cls_name = line.split(':', 1)
                    cls_id = int(cls_id_str.strip())
                    # Remove spaces and quotes around the name
                    cls_name = cls_name.strip().strip("'").strip('"')
                    names_dict[cls_id] = cls_name
                except ValueError:
                    continue
    return names_dict

def verify_yolo_format(yaml_path, images_dir, labels_dir, output_dir, sample_name="1"):
    """Read the specified txt and original image, reverse calculate coordinates and draw"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load category dictionary
    names_dict = load_yaml_names(yaml_path)
    print(f"✅ Successfully loaded {len(names_dict)} categories from yaml.")
    
    # 2. Find the corresponding original image and txt label
    img_path = os.path.join(images_dir, f"{sample_name}.jpg")
    txt_path = os.path.join(labels_dir, f"{sample_name}.txt")
    
    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        print(f"❌ Cannot find image or label file:\nImage: {img_path}\nLabel: {txt_path}")
        return

    # 3. Read image to get real width and height
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Unable to read image with OpenCV: {img_path}")
        return
        
    img_h, img_w = img.shape[:2]
    
    # 4. Read txt and reverse calculate coordinates to draw
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"📄 Starting to draw {sample_name}.jpg, with {len(lines)} bounding boxes...")
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
            
        cls_id = int(parts[0])
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        w_norm = float(parts[3])
        h_norm = float(parts[4])
        
        # Core: Convert YOLO normalized coordinates back to absolute pixel coordinates
        # abs_x = x_center_norm * img_w
        abs_w = w_norm * img_w
        abs_h = h_norm * img_h
        abs_x_center = x_center_norm * img_w
        abs_y_center = y_center_norm * img_h
        
        # Calculate top-left and bottom-right coordinates
        x_min = int(abs_x_center - abs_w / 2.0)
        y_min = int(abs_y_center - abs_h / 2.0)
        x_max = int(abs_x_center + abs_w / 2.0)
        y_max = int(abs_y_center + abs_h / 2.0)
        
        # Get text description
        phrase = names_dict.get(cls_id, f"Unknown_{cls_id}")
        
        # Draw box (red, B,G,R = 0,0,255)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        # Draw background and text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size, _ = cv2.getTextSize(phrase, font, font_scale, font_thickness)
        
        text_y = max(y_min, text_size[1] + 5)
        cv2.rectangle(img, (x_min, text_y - text_size[1] - 4), (x_min + text_size[0], text_y + 2), (0, 0, 0), -1)
        cv2.putText(img, phrase, (x_min, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
    # 5. Save and output
    save_path = os.path.join(output_dir, f"verified_{sample_name}.jpg")
    cv2.imwrite(save_path, img)
    print(f"🎉 Verified image saved to: {save_path}")

if __name__ == "__main__":
    # --- Directory Configuration ---
    YAML_PATH = "./yolo_dataset/dataset.yaml"      # The yaml generated just now
    LABELS_DIR = "./yolo_dataset/labels"           # The txt folder generated just now
    IMAGES_DIR = "/media/chen/study/VisualGenome/Home_data/"                        # Original images folder
    OUTPUT_DIR = "."           # Verification results save location
    
    # The image name you want to check (without extension), e.g. "1"
    SAMPLE_NAME = "3"                              
    
    verify_yolo_format(YAML_PATH, IMAGES_DIR, LABELS_DIR, OUTPUT_DIR, SAMPLE_NAME)