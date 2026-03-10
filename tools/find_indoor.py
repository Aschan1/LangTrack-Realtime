import json
import os
import shutil
from tqdm import tqdm

def filter_by_localization(json_path, images_dir, output_dir, target_count=50):
    """
    根据 localization 属性过滤图片，并在达到指定数量后停止。
    """
    # 1. 加载 JSON 数据
    print(f"Loading JSON data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 准备输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_images = {}
    count = 0

    # 3. 遍历并筛选
    # data.items() 获取 (image_id, info_dict)
    pbar = tqdm(data.items(), desc="Filtering indoors scenes")
    
    for img_id, info in pbar:
        # 使用 .get() 安全获取 localization，如果不存在则返回 None
        # 判断是否为 'indoors'
        print(img_id)
        if info.get('location') == 'indoors':
            img_filename = f"{img_id}.jpg"
            src_path = os.path.join(images_dir, img_filename)
            dst_path = os.path.join(output_dir, img_filename)

            # 检查本地是否有对应的图片文件
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                valid_images[img_id] = info
                count += 1
                
                # 更新进度条信息
                pbar.set_postfix({"Found": count})

                # 达到指定数量后停止
                if count >= target_count:
                    print(f"\n✅ 已经找到 {target_count} 张符合条件的室内图片。")
                    break

    # 4. 保存筛选后的新 JSON 文件（包含标注信息）
    output_json = os.path.join(output_dir, "filtered_indoors.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(valid_images, f, indent=2, ensure_ascii=False)

    print("\n" + "="*40)
    print(f"筛选完成！")
    print(f"找到图片总数: {len(valid_images)}")
    print(f"图片保存目录: {output_dir}")
    print(f"新标注文件: {output_json}")
    print("="*40)

if __name__ == "__main__":
    # 配置路径
    JSON_PATH = "yolo_dataset/indoors_subset/filtered_indoors.json" # GQA 原始大型 JSON
    IMAGES_DIR = "yolo_dataset/indoors_subset/images"         # 原始图片文件夹
    OUTPUT_DIR = "yolo_dataset/indoors_subset_filtered"     # 存放结果的文件夹
    
    # 运行脚本，找 100 张
    filter_by_localization(JSON_PATH, IMAGES_DIR, OUTPUT_DIR, target_count=1000)