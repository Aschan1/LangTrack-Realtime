from ultralytics import YOLOE
# Load the YOLO26 model
model = YOLOE("yoloe-26l-seg.pt")

# Export the model to TrT format
model.export(format="engine")  # creates 'yoloe26l.engine'