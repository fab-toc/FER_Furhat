import pyrealsense2 as rs
import numpy as np
import cv2
import time
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import copy
import os
import shutil
from typing import Literal

import kagglehub
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
from train.utils import (
    eval_classifier,
    get_data_transforms,
    get_model,
    save_model,
    train_classifier_with_validation,
)

# class InMemoryFaceDataset(Dataset):
#     def __init__(self, image_list, transform=None):
#         self.image_list = image_list
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         image = self.image_list[idx]

#         # OpenCV -> PIL
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = transforms.ToPILImage()(image)

#         if self.transform:
#             image = self.transform(image)

#         return image



# === Chargement du classificateur de visage ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("❌ Échec du chargement du détecteur de visage.")
    exit(1)

# === Initialisation de la caméra RealSense ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
pipeline.start(config)

# === Créer UNE SEULE fenêtre ===
cv2.namedWindow("Visages détectés", cv2.WINDOW_NORMAL)

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test","fake_class")
if os.path.exists(output_dir):
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
os.makedirs(output_dir, exist_ok=True)

# === Initialisation du timer ===
last_capture_time = time.time()
i = 0
print("✅ Caméra en cours. Appuie sur 'q' pour quitter.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Image couleur
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Détection de visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        

        # Si des visages sont détectés
        current_time = time.time()
        if len(faces) > 0:
            for _, (x, y, w, h) in enumerate(faces):

                # Capturer toutes les 0.5 secondes
                if current_time - last_capture_time >= 0.2:
                    face_crop = color_image[y:y+h, x:x+w].copy()
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = os.path.join(output_dir, f"face_{timestamp}_{i}.jpg")
                    cv2.imwrite(filename, face_crop)
                    print(f"📸 Visage enregistré : {filename}")
                    last_capture_time = current_time
                    i += 1

                # Dessiner le rectangle vert
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Affichage
        cv2.imshow("Visages détectés", color_image)

        # Sortie avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("🧼 Caméra arrêtée.")



# # Set up the script
# args = parse_args()

# Set the random seed for reproducibility
torch.manual_seed(0)

# # Automatic configuration of PyTorch settings
# hw_info, optimal_params = setup_pytorch_optimal(verbose=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


######################## HYPER PARAMETERS ########################
emotions_to_exclude = ["surprise", "neutral", "disgust"]

# Use the optimal parameters
num_workers = 10
prefetch_factor = 4
augmentation_level = "none"  # Options: "none", "light", "medium", "heavy"

model_name: Literal["vgg", "convnext"] = "convnext"
model_version: Literal["11", "13", "16", "19", "tiny", "small", "base", "large"] = (
    "tiny"
)

batch_size = 16
num_epochs: int = 20
learning_rate: float = 1e-4
loss_fn: nn.Module = nn.CrossEntropyLoss()

unfreeze_feature_layer_start: int = (
    2  # Unfreeze the feature layers starting from this one
)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../trained/{model_name}_{model_version}_b{512}_l{unfreeze_feature_layer_start}_end_e{num_epochs}.pt")


data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

data_transforms = get_data_transforms(
    input_format="rgb",
    target_channels=3,  # Target channels for the model
    target_size=(224, 224),  # Target size of each image for the model used
    augmentation_level=augmentation_level,
    custom_means=[0.485, 0.456, 0.406],  # ImageNet stats by default
    custom_stds=[0.229, 0.224, 0.225],  # ImageNet stats by default
)



class_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test")

# Load the dataset
train_data = torchvision.datasets.ImageFolder(
    os.path.join(data_dir, class_dir),
    transform=data_transforms,
)


test_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=num_workers,pin_memory=True,persistent_workers=True,prefetch_factor=prefetch_factor,)


# Get the model based on the hyperparameters
model = get_model(
    model_name=model_name,
    model_version=model_version,
    num_classes=4,
    unfreeze_feature_layer_start=unfreeze_feature_layer_start,
)

# Print model structure to understand what we're working with
print("\nModel structure:")
print(model, "\n")


######################## TESTING ########################
model_test = copy.deepcopy(model)
model_test.load_state_dict(torch.load(model_path))

# Set the model in 'evaluation' mode (this disables some layers (batch norm, dropout...) which are not needed when testing)
model_test.eval()

model_test.to(device, non_blocking=True)

# initialize the total and correct number of labels to compute the accuracy
correct_labels = 0
total_labels = 0
total_loss = 0  # Pour accumuler la perte totale
labels_predicted = []

# In evaluation phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    # Iterate over the dataset using the dataloader
    for images, labels in test_dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Get the predicted labels
        y_predicted = model_test(images)


        _, labels_predicted = torch.max(y_predicted.data, 1)


angry = 0
fear = 0
happy = 0
sad = 0

for resultat in labels_predicted:
    if resultat == 0:
        angry +=1
    elif resultat == 1:
        fear +=1
    elif resultat == 2:
        happy +=1
    elif resultat == 3:
        sad +=1

emotions = {'Angry': angry, 'Fear': fear, 'Happy': happy, 'Sad': sad}
current_emotion = max(emotions, key=emotions.get)
print("Emotion :", current_emotion)


