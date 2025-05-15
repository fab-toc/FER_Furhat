import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train.utils import get_data_transforms, get_model


# === Dataset en mémoire ===
class InMemoryFaceDataset(Dataset):
    def __init__(self, image_list, transform=None, label=0):
        self.image_list = image_list
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = self.image_list[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        if self.transform:
            pil = self.transform(pil)
        return pil, self.label


# === Initialisation caméra & visage ===
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Échec du chargement du CascadeClassifier")

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
pipeline.start(cfg)

cv2.namedWindow("Visages détectés", cv2.WINDOW_NORMAL)


# === Hyperparams & modèle ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

NUM_IMAGES = 60
BATCH_SIZE = 32
NUM_WORKERS = 4
PREFETCH = 2

MODEL_NAME = "convnext"
MODEL_VER = "large"
UNFREEZE_LAYER = 3
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "trained",
    MODEL_NAME,
    f"fine-tuned_{MODEL_NAME}_{MODEL_VER}_b{BATCH_SIZE}_l{UNFREEZE_LAYER}_end_e20.pt",
)

transforms_pipeline = get_data_transforms(
    input_format="rgb",
    target_channels=3,
    target_size=(224, 224),
    augmentation_level="none",
    custom_means=[0.485, 0.456, 0.406],
    custom_stds=[0.229, 0.224, 0.225],
)

# Charger le modèle une seule fois
base_model = get_model(
    model_name=MODEL_NAME,
    model_version=MODEL_VER,
    num_classes=4,
    unfreeze_layer_start=UNFREEZE_LAYER,
)
base_model.load_state_dict(torch.load(MODEL_PATH))
base_model.eval()
base_model.to(device, non_blocking=True)

# ThreadPool pour l’inférence
executor = ThreadPoolExecutor(max_workers=1)
last_prediction = "En attente..."


def inference_task(image_batch):
    """Tâche d'inférence, tourne en thread séparé."""
    ds = InMemoryFaceDataset(image_list=image_batch, transform=transforms_pipeline)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH,
    )
    all_preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            outputs = base_model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())

    counts = {"Angry": 0, "Fear": 0, "Happy": 0, "Sad": 0}
    for p in all_preds:
        if p == 0:
            counts["Angry"] += 1
        elif p == 1:
            counts["Fear"] += 1
        elif p == 2:
            counts["Happy"] += 1
        elif p == 3:
            counts["Sad"] += 1
    dominant = max(counts, key=counts.get)
    return dominant


# === Boucle principale ===
collected = []
last_capture = time.time()

print("✅ Caméra démarrée, appuyez sur 'q' pour quitter.")

try:
    while True:
        # 1) Lecture & détection visage
        frames = pipeline.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf:
            continue
        img = np.asanyarray(cf.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # 2) Capture en mémoire
        if faces is not None and len(collected) < NUM_IMAGES:
            for x, y, w, h in faces:
                now = time.time()
                if now - last_capture >= 1 / 60:
                    collected.append(img[y : y + h, x : x + w].copy())
                    last_capture = now
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 3) Lancer inférence quand batch complet
        if len(collected) >= NUM_IMAGES:
            # Submit sans bloquer la boucle
            future = executor.submit(inference_task, list(collected))
            future.add_done_callback(
                lambda f: globals().update(last_prediction=f.result())
            )
            collected.clear()

        # 4) Affichage prédiction sur l’image
        cv2.putText(
            img,
            f"Emotion: {last_prediction}",  # texte
            (10, 30),  # position
            cv2.FONT_HERSHEY_SIMPLEX,  # police :contentReference[oaicite:2]{index=2}
            1.0,  # échelle
            (0, 255, 0),  # couleur BGR :contentReference[oaicite:3]{index=3}
            2,  # épaisseur
            cv2.LINE_AA,
        )

        cv2.imshow("Visages détectés", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)
    print("🧼 Caméra arrêtée.")
