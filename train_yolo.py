from ultralytics import YOLO
import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU not found, training on CPU")

    model = YOLO('yolov8n.pt')

    model.train(
        data=r"D:/4TIE_JosepRonaldoFrancisSiregar_CV/Dataset/guns-knives-yolo/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        workers=2,  # or set to 0 to disable multiprocessing
        name="guns_knives_model"
    )
