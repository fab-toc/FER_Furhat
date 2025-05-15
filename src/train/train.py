import os.path as path
from copy import deepcopy
from shutil import rmtree

import torch
from kagglehub import dataset_download
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from utils import (
    eval_classifier,
    get_data_transforms,
    get_model,
    save_model,
    train_classifier_with_validation,
)

# args = parse_args()

# Set the random seed for reproducibility
torch.manual_seed(0)

# Some optimizations for CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


# ================ HYPER PARAMETERS ================
EMOTIONS_TO_EXCLUDE = ["surprise", "neutral", "disgust"]

NUM_WORKERS = 12
PREFETCH_FACTOR = 4
AUGMENTATION_LEVEL = "heavy"  # Options: "none", "light", "medium", "heavy"

MODEL_NAME = "convnext"  # Options: "convnext", "vgg"
MODEL_VERSION = (
    "large"  # Options: "11", "13", "16", "19", "tiny", "small", "base", "large"
)

BATCH_SIZE = 256
EPOCHS: int = 15
LR: float = 1e-4
LOSS_FN: Module = CrossEntropyLoss()

UNFREEZE_LAYER_START: int = 5  # Unfreeze the feature layers starting from this one


# ================ DATASET ================
data_dir = dataset_download("msambare/fer2013")

data_transforms = get_data_transforms(
    input_format="grayscale",
    target_channels=3,
    target_size=(224, 224),
    augmentation_level=AUGMENTATION_LEVEL,
    custom_means=[0.485, 0.456, 0.406],
    custom_stds=[0.229, 0.224, 0.225],
)

# Remove the directories of the excluded emotions
for split in ["train", "test"]:
    split_dir = path.join(data_dir, split)

    for emotion in EMOTIONS_TO_EXCLUDE:
        emotion_dir = path.join(split_dir, emotion)

        if path.exists(emotion_dir):
            print(f"Removing directory {emotion_dir}")
            rmtree(emotion_dir)

# Load the dataset
train_data = ImageFolder(
    path.join(data_dir, "train"),
    transform=data_transforms,
)

test_data = ImageFolder(
    path.join(data_dir, "test"),
    transform=data_transforms,
)

CLASSES = train_data.classes
print("Classes of the dataset:", CLASSES)

print("\nMapping of labels to emotions:")
for idx, emotion in enumerate(CLASSES):
    print(f"Label {idx} → {emotion}")

# Define the validation set by splitting the training data into 2 subsets (80% training and 20% validation)
n_train_examples = int(len(train_data) * 0.8)
n_valid_examples = len(train_data) - n_train_examples
train_data, valid_data = random_split(train_data, [n_train_examples, n_valid_examples])

print(f"\nNumber of train samples: {len(train_data)}")
print(f"Number of validation samples: {len(valid_data)}")
print(f"Number of test samples: {len(test_data)}")


# ================ DATALOADERS ================
train_dataloader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=PREFETCH_FACTOR,
)

valid_dataloader = DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=PREFETCH_FACTOR,
)

test_dataloader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=PREFETCH_FACTOR,
)

print("\nNumber of batches in the training subset:", len(train_dataloader))
print("Number of batches in the validation subset:", len(valid_dataloader))
print("Number of batches in the testing subset:", len(test_dataloader))


# ================ MODEL =================
model = get_model(
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
    num_classes=len(CLASSES),
    unfreeze_layer_start=UNFREEZE_LAYER_START,
)


# ================ TRAINING =================
model_trained, train_losses, val_accuracies = train_classifier_with_validation(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    batch_size=BATCH_SIZE,
    num_epochs=EPOCHS,
    loss_fn=LOSS_FN,
    optimizer_class=torch.optim.Adam,
    learning_rate=LR,
    device=device,
    verbose=True,
)

model_path = save_model(
    model=model_trained,
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
    batch_size=BATCH_SIZE,
    unfreeze_layer_start=UNFREEZE_LAYER_START,
    num_epochs=EPOCHS,
)


# ================ TESTING =================
model_test = deepcopy(model)
model_test.load_state_dict(torch.load(model_path))

# Apply the evaluation function using the test dataloader
test_accuracy, avg_loss = eval_classifier(
    model=model_test, eval_dataloader=test_dataloader, device=device, loss_fn=LOSS_FN
)

print("======== TEST RESULTS ========")
print("Test accuracy: {:.2f}%".format(test_accuracy))
print("Average loss: {:.4f}".format(avg_loss))
