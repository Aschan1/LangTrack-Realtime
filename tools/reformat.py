import json

def simplify_gqa_data(gqa_data):
    """
    将 GQA 的复杂场景图格式转换为扁平化的单关系区域描述格式。
    """
    result_list = []
    
    # 遍历每一张图片
    for image_id_str, image_data in gqa_data.items():
        image_id = int(image_id_str)
        objects_dict = image_data.get("objects", {})
        
        # 遍历图片中的每一个物体
        for obj_id_str, obj_info in objects_dict.items():
            region_id = int(obj_id_str)
            target_name = obj_info.get("name", "")
            attributes = obj_info.get("attributes", [])
            
            # --- 处理关系（只保留一个） ---
            relations = obj_info.get("relations", [])
            anchor_object_name = None
            spatial_relation = None
            
            if relations:
                # 取出第一个关系
                first_relation = relations[0]
                spatial_relation = first_relation.get("name")
                anchor_obj_id = first_relation.get("object")
                
                # 去 objects_dict 里面反查这个关联物体的真实名字 (比如把 1188513 变成 skillet)
                if anchor_obj_id in objects_dict:
                    anchor_object_name = objects_dict[anchor_obj_id].get("name")
            
            # --- 自动生成 phrase (描述短语) ---
            phrase_parts = []
            if attributes:
                phrase_parts.append(" ".join(attributes))  # 加入属性，如 "clear light blue"
            phrase_parts.append(target_name)               # 加入主体，如 "sky"
            
            if spatial_relation and anchor_object_name:
                phrase_parts.append(spatial_relation)      # 加入关系，如 "above"
                phrase_parts.append(anchor_object_name)    # 加入参考物，如 "building"
                
            # 拼合短语，清理多余空格
            phrase = " ".join(phrase_parts).strip()
            
            # --- 构建目标 JSON 格式 ---
            target_obj = {
                "region_id": region_id,
                "width": obj_info.get("w"),
                "height": obj_info.get("h"),
                "image_id": image_id,
                "phrase": phrase,
                "y": obj_info.get("y"),
                "x": obj_info.get("x"),
                "keywords": {
                    "target": target_name,
                    "attributes": attributes,
                    "anchor_object": anchor_object_name,
                    "spatial_relation": spatial_relation
                }
            }
            
            result_list.append(target_obj)
            
    return result_list

# ================= 使用示例 =================

if __name__ == "__main__":
    # 假设这是你读取的原始 JSON 数据
    input_file = "yolo_dataset/indoors_subset/filtered_indoors_LM.json" # 替换为你的文件路径
    output_file = "yolo_dataset/indoors_subset/filtered_indoors_LM_vg.json"
    
    # 测试代码 (模拟读取)
    with open(input_file, 'r', encoding='utf-8') as f:
        gqa_raw_data = json.load(f)
    
    # 这里用你发的数据做个小测试（只放了前两个物体演示）
    # sample_gqa_data = {
    #   "2397582": {
    #     "width": 500,
    #     "height": 333,
    #     "objects": {
    #       "1188535": {
    #         "name": "meat", "h": 22, "w": 32, "y": 231, "x": 176, "attributes": [],
    #         "relations": [{"object": "1188513", "name": "inside"}, {"object": "1188515", "name": "to the right of"}]
    #       },
    #       "1188513": {
    #         "name": "skillet", "h": 118, "w": 350, "y": 210, "x": 148, "attributes": ["black"],
    #         "relations": []
    #       }
    #     }
    #   }
    # }

    # 执行转换
    simplified_data = simplify_gqa_data(gqa_raw_data)
    
    # 打印结果看看
    print(json.dumps(simplified_data, indent=2, ensure_ascii=False))
    
    # 实际保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_data, f, indent=2, ensure_ascii=False)