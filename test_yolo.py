if __name__ == "__main__":
    from ultralytics import YOLO
    #model = YOLO("yolov8n.pt")
    #results = model.train(data="coco8.yaml",epochs=2)
    model = YOLO("runs/detect/train3/weights/best.pt")
    results = model.predict(source="0",show=True)
