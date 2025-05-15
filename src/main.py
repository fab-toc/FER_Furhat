import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyrealsense2 as rs
import requests
import torch

# Import de l'API Furhat
from furhat_remote_api import FurhatRemoteAPI  # Ajout de LedColor
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train.utils import get_data_transforms, get_model


# === Dataset en mémoire ===
class InMemoryFaceDataset(Dataset):
    # Classe inchangée
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


# === Configuration Furhat ===
# Configuration du robot Furhat
FURHAT_IP = "192.168.10.14"  # Remplacez par l'IP du robot Furhat
FURHAT_PORT = 8080  # Port par défaut, ajustez si nécessaire

# Mappage des émotions vers les expressions Furhat
EMOTION_TO_FURHAT_GESTURE = {
    "Angry": "ExpressionAngry",
    "Fear": "ExpressionFear",
    "Happy": "ExpressionHappy",
    "Sad": "ExpressionSad",
}

# Phrases à prononcer selon l'émotion
EMOTION_PHRASES = {
    "Angry": [
        "Je vois que vous êtes en colère. Respirez profondément.",
        "La colère ne résout pas les problèmes. Prenez un moment pour vous calmer.",
        "Vous semblez frustré. Puis-je vous aider?",
    ],
    "Fear": [
        "Je détecte de la peur. N'ayez pas d'inquiétude, tout va bien.",
        "Vous semblez anxieux. Respirez lentement.",
        "La peur est normale, mais vous êtes en sécurité ici.",
    ],
    "Happy": [
        "Votre sourire est contagieux! Je suis content de vous voir heureux.",
        "Quelle belle journée quand on sourit comme ça!",
        "Votre bonheur illumine la pièce!",
    ],
    "Sad": [
        "Je vois que vous êtes triste. Je suis là si vous avez besoin de parler.",
        "La tristesse est temporaire. Gardez espoir.",
        "N'hésitez pas à partager ce qui vous préoccupe.",
    ],
}

# Couleurs LED pour chaque émotion
EMOTION_TO_LED_COLOR = {
    "Angry": (255, 0, 0),  # Rouge
    "Fear": (143, 0, 255),  # Violet
    "Happy": (255, 255, 0),  # Jaune
    "Sad": (0, 0, 255),  # Bleu
}

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
    unfreeze_feature_layer_start=UNFREEZE_LAYER,
)
base_model.load_state_dict(torch.load(MODEL_PATH))
base_model.eval()
base_model.to(device, non_blocking=True)

# Initialiser la connexion avec Furhat
furhat = FurhatRemoteAPI(FURHAT_IP)
furhat_is_speaking = False
last_emotion_time = 0
MIN_EMOTION_INTERVAL = (
    3  # Intervalle minimum entre deux changements d'émotion (secondes)
)

# ThreadPool pour l'inférence
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


def update_furhat_emotion(emotion):
    """Met à jour l'expression, la parole et la LED du robot Furhat."""
    global furhat_is_speaking, last_emotion_time

    # Vérifier si le robot est occupé et si suffisamment de temps s'est écoulé
    now = time.time()
    if furhat_is_speaking or (now - last_emotion_time < MIN_EMOTION_INTERVAL):
        return

    # Marquer le début du traitement
    furhat_is_speaking = True
    last_emotion_time = now

    try:
        # S'assurer que Furhat regarde l'utilisateur
        furhat.attend(attention_target="CAMERA")

        # Définir l'expression faciale
        furhat_gesture = EMOTION_TO_FURHAT_GESTURE.get(emotion, "ExpressionNeutral")
        furhat.gesture(name=furhat_gesture)

        # Changer la couleur de la LED via l'API REST
        led_color = EMOTION_TO_LED_COLOR.get(emotion)
        if led_color:
            r, g, b = led_color
            set_led_via_rest(FURHAT_IP, r, g, b)

        # Choisir et dire une phrase correspondant à l'émotion
        import random

        phrases = EMOTION_PHRASES.get(emotion, ["Je ne sais pas quoi dire."])
        selected_phrase = random.choice(phrases)

        # Faire parler Furhat et attendre la fin de la synthèse vocale
        result = furhat.say(text=selected_phrase, blocking=True)

        # Retourner à l'expression neutre après un court délai
        time.sleep(1)
        furhat.gesture(name="ExpressionNeutral")

    except Exception as e:
        print(f"Erreur lors de la communication avec Furhat: {e}")
    finally:
        # Marquer la fin du traitement
        furhat_is_speaking = False


# Fonction de callback pour gérer les résultats d'inférence
def handle_inference_result(future):
    emotion = future.result()
    globals().update(last_prediction=emotion)

    # Mettre à jour Furhat dans un thread séparé pour ne pas bloquer la boucle principale
    executor.submit(update_furhat_emotion, emotion)


# Fonction pour contrôler les LEDs via l'API REST
def set_led_via_rest(ip, r, g, b):
    """
    Contrôle les LEDs de Furhat via l'API REST.

    Args:
        ip (str): Adresse IP du robot Furhat
        r, g, b (int): Valeurs RGB (0-255) pour la couleur de la LED
    """
    url = f"http://{ip}:{FURHAT_PORT}/furhat/led"
    payload = {
        "red": r / 255.0,  # L'API REST attend des valeurs entre 0 et 1
        "green": g / 255.0,
        "blue": b / 255.0,
    }

    try:
        response = requests.put(url, json=payload)
        if response.status_code == 200:
            print(f"LED color set to RGB({r},{g},{b})")
        else:
            print(f"Failed to set LED color: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error setting LED color via REST API: {e}")


# === Boucle principale ===
collected = []
last_capture = time.time()

print("✅ Caméra démarrée, Furhat connecté. Appuyez sur 'q' pour quitter.")

try:
    # Initialiser Furhat à l'état neutre
    furhat.gesture(name="ExpressionNeutral")
    furhat.attend(attention_target="CAMERA")

    # Vérifier que la connexion est établie
    status = furhat.get_state()
    print(f"✅ Connecté à Furhat: {status.get('characterName', 'Unknown')}")
except Exception as e:
    print(f"❌ Échec de connexion à Furhat: {e}")
    print("⚠️ Le programme continuera sans contrôle du robot Furhat.")
    # Désactiver les fonctionnalités liées à Furhat
    furhat = None

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
        if len(faces) > 0 and len(collected) < NUM_IMAGES:
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
            future.add_done_callback(handle_inference_result)
            collected.clear()

        # 4) Affichage prédiction sur l'image
        cv2.putText(
            img,
            f"Emotion: {last_prediction}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Ajouter le statut de Furhat à l'affichage
        status = "Furhat: En train de parler" if furhat_is_speaking else "Furhat: Prêt"
        cv2.putText(
            img,
            status,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0) if furhat_is_speaking else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Visages détectés", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Nettoyer et fermer proprement
    pipeline.stop()
    cv2.destroyAllWindows()

    # Remettre Furhat en état neutre avant de terminer
    try:
        furhat.gesture(name="ExpressionNeutral")
        # Utiliser l'API REST pour éteindre les LEDs
        set_led_via_rest(FURHAT_IP, 0, 0, 0)
    except Exception as e:
        print(f"Erreur lors de la remise à zéro de Furhat: {e}")

    executor.shutdown(wait=True)
    print("🧼 Caméra arrêtée, connexion Furhat fermée.")
