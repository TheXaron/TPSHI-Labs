import sys
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["pizza", "steak", "sushi"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Не вдалося відкрити зображення: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_float = img_resized.astype(np.float32) / 255.0
    img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

def softmax(logits: np.ndarray) -> np.ndarray:
    exp_x = np.exp(logits - np.max(logits))
    return exp_x / np.sum(exp_x)

def main():
    model_path = "resnet18_food.onnx"
    # Якщо шлях до картинки не передано аргументом, беремо перше зображення з dataset
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = str(next(Path("dataset").rglob("*.jpg")))

    # Завантаження лише ONNX Runtime сесії (PyTorch повністю відсутній)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    input_tensor = preprocess(image_path)

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    probs = softmax(outputs[0])
    pred_idx = int(np.argmax(probs))

    print("=== Автономний Inference (ONNX Runtime без PyTorch) ===")
    print(f"Файл зображення: {image_path}")
    print(f"Передбачений клас: {CLASSES[pred_idx].upper()} (впевненість: {probs[pred_idx]*100:.2f}%)")
    print(f"Час виконання: {latency_ms:.2f} мс")

if __name__ == "__main__":
    main()