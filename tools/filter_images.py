import json
import os
import shutil

def filter_and_copy_images(json_path, images_dir, output_dir):
    """
    Filter images based on the JSON file and copy matching images to a new directory.
    """
    # 1. Read the JSON file and extract unique image_ids
    print(f"Reading JSON file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract unique image_ids
    unique_image_ids = set(str(item['image_id']) for item in data)
    print(f"Found {len(unique_image_ids)} unique image IDs in JSON.")

    # 2. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 3. Check and copy images
    copied_count = 0
    missing_count = 0

    for img_id in unique_image_ids:
        img_filename = f"{img_id}.jpg"
        src_path = os.path.join(images_dir, img_filename)
        dst_path = os.path.join(output_dir, img_filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            missing_count += 1
            print(f"Warning: Image {img_filename} not found in {images_dir}")

    # 4. Summary
    print("\n" + "="*50)
    print("Image filtering and copying completed!")
    print(f"Total unique images in JSON: {len(unique_image_ids)}")
    print(f"Images copied: {copied_count}")
    print(f"Images missing: {missing_count}")
    print(f"Output directory: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    # Define paths
    JSON_PATH = "yolo_dataset/home_ovd_filtered.json"
    IMAGES_DIR = "yolo_dataset/images/val"
    OUTPUT_DIR = "yolo_dataset/filtered_images"

    filter_and_copy_images(JSON_PATH, IMAGES_DIR, OUTPUT_DIR)