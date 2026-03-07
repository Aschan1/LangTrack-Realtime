import json
import os
from PIL import Image

def convert_ovd_to_yolo(json_path, images_dir, output_dir):
    # 1. Read JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # If the outermost is a dict, it might be {"annotations": [...]} structure, adapt it
    if isinstance(data, dict):
        if 'annotations' in data:
            data = data['annotations']
        else:
            # Try to get the first list-type value in the dict
            for key, val in data.items():
                if isinstance(val, list):
                    data = val
                    break

    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    # 2. Collect all unique phrases as categories (YOLO-World supports natural language prompts)
    unique_phrases = list(set([item['phrase'] for item in data if 'phrase' in item]))
    phrase_to_id = {phrase: idx for idx, phrase in enumerate(unique_phrases)}
    
    # 3. Group annotations by image_id
    img_to_anns = {}
    for item in data:
        img_id = str(item['image_id'])
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(item)
        
    # 4. Iterate and process each image
    processed_count = 0
    missing_images = 0
    
    for img_id, anns in img_to_anns.items():
        # Infer image filename, common formats might be 1.jpg or 000001.jpg
        # Here assume directly id + .jpg, if your image names are in other formats (e.g. 00001.jpg), please modify here ↓
        img_filename = f"{img_id}.jpg" 
        img_path = os.path.join(images_dir, img_filename)
        
        if not os.path.exists(img_path):
            missing_images += 1
            continue
            
        # Get the width and height of the whole image
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            
        # Prepare to write the corresponding txt label file
        txt_filename = f"{img_id}.txt"
        txt_filepath = os.path.join(labels_dir, txt_filename)
        
        with open(txt_filepath, 'w', encoding='utf-8') as txt_f:
            for ann in anns:
                x = ann['x']
                y = ann['y']
                w = ann['width']
                h = ann['height']
                
                # YOLO needs center point coordinates and normalization
                x_center = (x + w / 2.0) / img_w
                y_center = (y + h / 2.0) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                # Anti-out-of-bounds protection
                x_center, y_center = max(0, min(1, x_center)), max(0, min(1, y_center))
                norm_w, norm_h = max(0, min(1, norm_w)), max(0, min(1, norm_h))
                
                class_id = phrase_to_id[ann['phrase']]
                
                # Write: class_id x_center y_center width height
                txt_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
        processed_count += 1
        
    # 5. Generate YOLO-World specific dataset.yaml
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write("path: ./yolo_dataset  # Please modify the path according to your actual situation\n")
        f.write("train: images/train   # Training set images\n")
        f.write("val: images/val       # Validation set images\n\n")
        f.write("names:\n")
        for phrase, class_id in phrase_to_id.items():
            # YOLO-World will use the complete sentence here as Prompt
            # Handle single quotes to avoid yaml parsing errors
            safe_phrase = phrase.replace("'", "''")
            f.write(f"  {class_id}: '{safe_phrase}'\n")
            
    print(f"\n✅ Conversion completed!")
    print(f"📊 Extracted {len(unique_phrases)} unique text phrases (categories).")
    print(f"🖼️ Successfully processed labels for {processed_count} images.")
    if missing_images > 0:
        print(f"⚠️ {missing_images} images not found, skipped. Please check images_dir path or image naming format.")

if __name__ == "__main__":
    # === Please modify the paths here ===
    JSON_FILE = "home_ovd_samples.json"          # Your JSON file path
    IMAGES_DIR = "/media/chen/study/VisualGenome/Home_data/" # Folder path containing your original images (must have, to read image sizes)
    OUTPUT_DIR = "./yolo_dataset"                # Directory to store the converted results
    
    convert_ovd_to_yolo(JSON_FILE, IMAGES_DIR, OUTPUT_DIR)