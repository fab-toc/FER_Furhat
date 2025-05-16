import copy
import os
from typing import Literal

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
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

# # Automatic configuration of PyTorch settings
# hw_info, optimal_params = setup_pytorch_optimal(verbose=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


######################## HYPER PARAMETERS ########################
num_workers = 10
prefetch_factor = 4
augmentation_level = "heavy"  # Options: "none", "light", "medium", "heavy"

model_name: Literal["vgg", "convnext"] = "convnext"
model_version: Literal["11", "13", "16", "19", "tiny", "small", "base", "large"] = (
    "large"
)

batch_size = 32
num_epochs: int = 20
learning_rate: float = 1e-4
loss_fn: nn.Module = nn.CrossEntropyLoss()

unfreeze_feature_layer_start: int = (
    3  # Unfreeze the feature layers starting from this one
)

# Print all hyperparameters for verification
print("\n======== HYPERPARAMETERS ========")
print(f"Augmentation level: {augmentation_level}")
print(f"Model name: {model_name}")
print(f"Model version: {model_version}")
print(f"Batch size: {batch_size}")
print(f"Number of epochs: {num_epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Unfreeze feature layers starting from: {unfreeze_feature_layer_start}")
print(f"Number of workers: {num_workers}")
print(f"Prefetch factor: {prefetch_factor}")
print(f"Device: {device}")
print("================================\n")


######################## DATASET ########################
data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset",
)

data_transforms = get_data_transforms(
    input_format="rgb",
    target_channels=3,  # Target channels for the model
    target_size=(224, 224),  # Target size of each image for the model used
    augmentation_level=augmentation_level,
    custom_means=[0.485, 0.456, 0.406],  # ImageNet stats by default
    custom_stds=[0.229, 0.224, 0.225],  # ImageNet stats by default
)


# Load the dataset
train_data = torchvision.datasets.ImageFolder(
    data_dir,
    transform=data_transforms,
)

CLASSES = train_data.classes
print("Classes of the dataset:", CLASSES)

# Print the mapping of labels to emotions
print("\nMapping of labels to emotions:")
for idx, emotion in enumerate(CLASSES):
    print(f"Label {idx} → {emotion}")

n_train_examples = int(len(train_data) * 0.7)
n_test_examples = len(train_data) - n_train_examples
train_data, test_data = random_split(train_data, [n_train_examples, n_test_examples])

# Define the test set by splitting the training data into 2 subsets (80% training and 20% testing)
n_train_examples = int(len(train_data) * 0.8)
n_valid_examples = len(train_data) - n_train_examples
train_data, valid_data = random_split(train_data, [n_train_examples, n_valid_examples])

print(f"\nNumber of train samples: {len(train_data)}")
print(f"Number of validation samples: {len(valid_data)}")
print(f"Number of test samples: {len(test_data)}")


######################## DATALOADERS ########################
train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=prefetch_factor,
)

valid_dataloader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=prefetch_factor,
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=prefetch_factor,
)

num_batches = len(train_dataloader)
print("Number of batches in the training subset:", num_batches)

num_batches = len(valid_dataloader)
print("Number of batches in the validation subset:", num_batches)

num_batches = len(test_dataloader)
print("Number of batches in the testing subset:", num_batches)


######################## MODEL ########################
# Get the model based on the hyperparameters
model = get_model(
    model_name=model_name,
    model_version=model_version,
    num_classes=len(CLASSES),
    unfreeze_layer_start=unfreeze_feature_layer_start,
)

model.load_state_dict(
    torch.load(
        os.path.join(
            os.path.dirname(data_dir),
            "trained",
            model_name,
            f"{model_name}_{model_version}_b{256}_l{unfreeze_feature_layer_start}_end_e{num_epochs}.pt",
        )
    )
)


######################## TRAINING ########################
model_trained, train_losses, val_accuracies = train_classifier_with_validation(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    batch_size=batch_size,
    num_epochs=num_epochs,
    loss_fn=loss_fn,
    optimizer_class=torch.optim.Adam,
    learning_rate=learning_rate,
    device=device,
    transform_fn=None,
    verbose=True,
)

# Call the function with the trained model and parameters
model_path = save_model(
    model=model_trained,
    model_name=model_name,
    model_version=model_version,
    batch_size=batch_size,
    unfreeze_layer_start=unfreeze_feature_layer_start,
    num_epochs=num_epochs,
    fine_tuned=True,
)


######################## TESTING ########################
model_test = copy.deepcopy(model)
model_test.load_state_dict(torch.load(model_path))

# - Apply the evaluation function using the test dataloader
test_accuracy, avg_loss = eval_classifier(
    model=model_test, eval_dataloader=test_dataloader, device=device, loss_fn=loss_fn
)

print("======== TEST RESULTS ========")
print("Test accuracy: {:.2f}%".format(test_accuracy))
print("Average loss: {:.4f}".format(avg_loss))
