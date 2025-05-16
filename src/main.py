import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyrealsense2 as rs
import requests
import torch
from furhat_remote_api import FurhatRemoteAPI
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train.utils import get_data_transforms, get_model


class FurhatController:
    def __init__(self, host="192.168.10.14"):
        self.host = host
        self.furhat = FurhatRemoteAPI(host=host)
        self.speaking = False
        self.emotion_queue = queue.Queue()
        self.current_emotion = None
        self.worker_thread = threading.Thread(
            target=self._process_emotions, daemon=True
        )
        self.worker_thread.start()

        # Messages par émotion
        self.messages = {
            "angry": [
                "Je ressens de la colère. C'est une émotion forte.",
                "La colère peut être difficile à gérer parfois.",
                "Quand on est en colère, il faut essayer de respirer profondément.",
                "La colère est parfois justifiée, mais il faut savoir la canaliser.",
            ],
            "fear": [
                "J'ai peur. C'est une émotion qui nous protège du danger.",
                "La peur peut parfois nous paralyser.",
                "Respirer calmement peut aider à surmonter la peur.",
                "La peur est naturelle, elle fait partie de nous.",
            ],
            "happy": [
                "Je suis heureux! C'est agréable de ressentir de la joie.",
                "Le bonheur est un état merveilleux à partager.",
                "Sourire est contagieux, essayez de sourire à quelqu'un aujourd'hui!",
                "La joie illumine notre journée et celle des autres.",
            ],
            "sad": [
                "Je me sens triste. C'est normal d'être triste parfois.",
                "La tristesse fait partie de la vie, comme la joie.",
                "Exprimer sa tristesse peut aider à se sentir mieux.",
                "Après la pluie vient le beau temps, la tristesse ne dure pas éternellement.",
            ],
        }

        # Couleurs LED par émotion (format RGB)
        self.colors = {
            "angry": {"r": 255, "g": 0, "b": 0},  # Rouge
            "fear": {"r": 128, "g": 0, "b": 128},  # Violet
            "happy": {"r": 255, "g": 255, "b": 0},  # Jaune
            "sad": {"r": 0, "g": 0, "b": 255},  # Bleu
        }

        self.expressions = {
            "angry": "ExpressAnger",
            "fear": "ExpressFear",
            "happy": "BigSmile",
            "sad": "ExpressSad",
        }

        try:
            self.furhat.get_voices()
            print("✅ Connexion réussie au robot Furhat")

            self.set_voice("Margaux", "French")
        except Exception as e:
            print(f"❌ Erreur lors de la connexion au robot Furhat: {e}")

    def set_voice(self, voice_name: str, language: str = "French"):
        try:
            # Utiliser l'API Python pour configurer la voix
            self.furhat.set_voice(name=voice_name, language=language)
            return True
        except Exception as e:
            print(f"Erreur lors de la configuration de la voix: {e}")
            return False

    def set_led(self, r: int, g: int, b: int):
        try:
            # Using query parameters for color values
            url = f"http://{self.host}/led"
            params = {"red": r, "green": g, "blue": b}
            response = requests.post(url, params=params)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Erreur lors du changement de la LED: {e}")
            return False

    def set_face(self, expression: str):
        try:
            # Utiliser l'API Python pour changer l'expression
            self.furhat.gesture(name=expression)
            return True
        except Exception as e:
            print(f"Erreur lors du changement d'expression: {e}")
            return False

    def say(self, text: str):
        self.speaking = True
        try:
            # Utiliser l'API Python pour faire parler Furhat
            self.furhat.say(text=text, blocking=True)
            self.speaking = False
            return True
        except Exception as e:
            print(f"Erreur lors de la parole: {e}")
            self.speaking = False
            return False

    def is_speaking(self):
        try:
            # L'API ne fournit pas directement de méthode pour vérifier si le robot parle
            # On s'appuie donc sur notre variable interne
            return self.speaking
        except Exception as e:
            print(f"Erreur lors de la vérification du statut: {e}")
            return self.speaking  # Fallback sur l'état interne

    def handle_emotion(self, emotion):
        if emotion != self.current_emotion:
            # Récupérer l'émotion en minuscule pour la correspondance avec les clés
            emotion_lower = emotion.lower()
            self.emotion_queue.put(emotion_lower)

    def _process_emotions(self):
        while True:
            if not self.emotion_queue.empty() and not self.is_speaking():
                emotion = self.emotion_queue.get()
                self.current_emotion = emotion

                # Changer expression et LED
                self.set_face(self.expressions.get(emotion, "Neutral"))
                color = self.colors.get(emotion, {"r": 255, "g": 255, "b": 255})
                self.set_led(color["r"], color["g"], color["b"])

                # Dire une phrase aléatoire correspondant à l'émotion
                # Et ajouter la mention de l'émotion détectée
                emotion_display = emotion.capitalize()
                intro = f"Je détecte que vous ressentez de la {emotion_display}. "

                messages = self.messages.get(
                    emotion, ["Je ne sais pas comment me sentir."]
                )
                message = intro + random.choice(messages)
                self.say(message)

            time.sleep(0.5)  # Vérifier toutes les 500ms


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


# Créer une instance de FurhatController
furhat_controller = FurhatController()

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

            def handle_emotion_result(future):
                global last_prediction
                result = future.result()
                last_prediction = result
                # Envoyer l'émotion détectée au contrôleur Furhat
                furhat_controller.handle_emotion(result.lower())

            future.add_done_callback(handle_emotion_result)
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

        cv2.imshow("Visages détectés", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)
    print("🧼 Caméra arrêtée.")
