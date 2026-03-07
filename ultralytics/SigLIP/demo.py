import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModel
from ultralytics import YOLO
from torchvision.transforms.functional import crop


image_path = "SigLIP/bubble_tea.jpg"
classes = ["a glass cup with buble tea inside", "glass cup", "buble tea", "cup"]

# Initialize a YOLO-World model
yolo = YOLO("yolov8x-worldv2.pt")  # or choose yolov8m/l-world.pt
vlm_model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

# # Define custom classes
yolo.set_classes(classes)

raw_img = Image.open(image_path)
# Execute prediction for specified categories on an image
results = yolo.predict(image_path, stream=True, save=True, show=False)
fine_grained_results = list()

for result in results:
    # xywh = result.boxes.xywh  # center-x, center-y, width, height
    # xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    # xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    number_of_boxes = len(result.boxes)
    print(f"检测到 {number_of_boxes} 个目标，正在进行细粒度分类...")
    for i in range(number_of_boxes):
        if confs[i] < 0.3:
            continue
        
        cropped_img = crop(raw_img, int(xyxy[i][1]), int(xyxy[i][0]), int(xyxy[i][3] - xyxy[i][1]), int(xyxy[i][2] - xyxy[i][0]))

        candidate_labels = ["a glass cup with buble tea inside", "cup", "background", "blur", "nothing"]

        # follows the pipeline prompt template to get same results
        texts = [f'This is a photo of {label}.' for label in candidate_labels]

        # IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
        inputs = processor(text=texts, images=cropped_img, padding="max_length", max_length=64, return_tensors="pt").to(vlm_model.device)

        with torch.no_grad():
            outputs = vlm_model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)
        # print(torch.argmax(probs, dim=1))

        # fine_grained_results.append(xyxy)

        # 获取目标类(index 0)和负类(index 1,2,3)的分数
        print(probs)
        score_target = probs[0][0].item()
        score_negatives = probs[0][1:].sum().item() # 取所有负类里最高的一个作为基准

        print(f"Label conf: {score_target:.4f} | Noise conf: {score_negatives:.4f}")

        if score_target > score_negatives: 
            print("Passed")
            fine_grained_results.append(xyxy[i])
        else:
            print("Missed")

print(f"\n正在绘图，共有 {len(fine_grained_results)} 个目标通过筛选...")
draw = ImageDraw.Draw(raw_img)

# 尝试加载字体，如果失败则使用默认
try:
    font = ImageFont.truetype("arial.ttf", size=40)
except:
    font = ImageFont.load_default(size=40)

for box in fine_grained_results:
    # 绘制矩形框 (红色, 宽度3)
    xyxy = ((box[0], box[1]), (box[2], box[3]))
    draw.rectangle(xyxy, outline="red", width=18)
    
    # 绘制标签背景和文字
    text = classes[0]
    text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
    draw.rectangle(text_bbox, fill="red")
    draw.text((box[0], box[1]), text, fill="white", font=font)

# 5. 保存并显示
raw_img.show()
        