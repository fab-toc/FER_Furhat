import copy
import os

import torch
import torchvision
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader

from train.utils import (
    eval_classifier,
    get_data_transforms,
    get_model,
)

# args = parse_args()

# Set the random seed for reproducibility
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


# ================ HYPER PARAMETERS ================
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
AUGMENTATION_LEVEL = "heavy"  # Options: "none", "light", "medium", "heavy"

MODEL_NAME = "convnext"  # Options: "convnext", "vgg"
MODEL_VERSION = (
    "large"  # Options: "11", "13", "16", "19", "tiny", "small", "base", "large"
)

BATCH_SIZE = 32
EPOCHS: int = 15
LR: float = 1e-4
LOSS_FN: Module = CrossEntropyLoss()

UNFREEZE_LAYER_START: int = 3  # Unfreeze the feature layers starting from this one

model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trained",
    MODEL_NAME,
    f"fine-tuned_{MODEL_NAME}_{MODEL_VERSION}_b{BATCH_SIZE}_l{UNFREEZE_LAYER_START}_end_e{EPOCHS}.pt",
)


# ================ DATASET ================
data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"
)

data_transforms = get_data_transforms(
    input_format="rgb",
    target_channels=3,
    target_size=(224, 224),
    augmentation_level=AUGMENTATION_LEVEL,
    custom_means=[0.485, 0.456, 0.406],
    custom_stds=[0.229, 0.224, 0.225],
)

# Load the dataset
test_data = torchvision.datasets.ImageFolder(
    data_dir,
    transform=data_transforms,
)

CLASSES = test_data.classes
print("Classes of the dataset:", CLASSES)

print("\nMapping of labels to emotions:")
for idx, emotion in enumerate(CLASSES):
    print(f"Label {idx} → {emotion}")

print(f"\nNumber of test samples: {len(test_data)}")


# ================ DATALOADERS ================
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

print("Number of batches in the testing subset:", len(test_dataloader))


# ================ MODEL =================
model = get_model(
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
    num_classes=len(CLASSES),
    unfreeze_layer_start=UNFREEZE_LAYER_START,
)


# ================ TRAINING =================
model_test = copy.deepcopy(model)
model_test.load_state_dict(torch.load(model_path))

# Apply the evaluation function using the test dataloader
test_accuracy, avg_loss = eval_classifier(
    model=model_test, eval_dataloader=test_dataloader, device=device, loss_fn=LOSS_FN
)

print("======== TEST RESULTS ========")
print("Test accuracy: {:.2f}%".format(test_accuracy))
print("Average loss: {:.4f}".format(avg_loss))
