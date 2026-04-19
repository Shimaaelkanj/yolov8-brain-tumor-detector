from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        name="brain_tumor_detection"
    )

if __name__ == "__main__":
    main()