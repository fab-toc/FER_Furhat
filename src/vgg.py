import copy

import kagglehub
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from classifier import eval_classifier, train_classifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

# Download FER-2013 latest version
data_dir = kagglehub.dataset_download("msambare/fer2013")

data_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Grayscale(
            num_output_channels=3
        ),  # si modèles pré-entraînés ImageNet
        torchvision.transforms.Resize((224, 224)),  # ou autre résolution
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=mean, std=std),
    ]
)

# Load the FER-2013 dataset
train_data = torchvision.datasets.ImageFolder(
    data_dir + "/train", transform=data_transforms
)

test_data = torchvision.datasets.ImageFolder(
    data_dir + "/test", transform=data_transforms
)

print("Classes of the dataset:", train_data.classes)
print("Number of training samples:", len(train_data))
print("Number of test samples:", len(test_data))

# Pour afficher le mapping complet
for idx, emotion in enumerate(train_data.classes):
    print(f"Label {idx} → {emotion}")


batch_size = 512

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

# - print the number of batches in the training subset
num_batches = len(train_dataloader)
print("Number of batches in the training subset:", num_batches)

# - print the number of batches in the testing subset
num_batches = len(test_dataloader)
print("Number of batches in the testing subset:", num_batches)


weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
model = torchvision.models.vgg19(
    weights=weights
)  # charges les poids ImageNet pré-entraînés

# # Geler toutes les couches
# for param in model.parameters():
#     param.requires_grad = (
#         False  # figure l’intégralité des poids pour ne pas les recalculer
#     )

# Remplacer la couche de sortie (le dernier module du classifier)
num_classes = 7
model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)


num_epochs = 40
learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()

model_trained, train_losses = train_classifier(
    model=model,
    train_dataloader=train_dataloader,
    batch_size=batch_size,
    num_epochs=num_epochs,
    loss_fn=loss_fn,
    learning_rate=learning_rate,
    device=device,
    transform_fn=None,
    verbose=True,
)

torch.save(model_trained.state_dict(), "training/vgg-11_trained.pt")

model_test = copy.deepcopy(model)
model_test.load_state_dict(torch.load("training/vgg-11_trained.pt"))

# - Apply the evaluation function using the test dataloader
test_accuracy = eval_classifier(
    model=model_test, eval_dataloader=test_dataloader, device=device
)

# - Print the test accuracy
print("Test accuracy: {:.2f}%".format(test_accuracy))

# # - Plot the training loss over epochs
# plt.figure()
# plt.plot(train_losses)
# plt.title("Training loss over epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid()
# plt.show()
