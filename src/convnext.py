import copy

import kagglehub
import matplotlib
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split

from classifier import eval_classifier, train_classifier_with_validation, filter_dataset

matplotlib.use("Agg")  # Use a non-interactive backend

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
        torchvision.transforms.RandomHorizontalFlip(),  # Retournement aléatoire
        torchvision.transforms.RandomRotation(10),  # Rotation légère
        torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Translation
        torchvision.transforms.Resize((224, 224)),  # ou autre résolution
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
        torchvision.transforms.ToTensor(),
        # Utiliser les statistiques d'ImageNet
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Load the FER-2013 dataset
train_data = torchvision.datasets.ImageFolder(
    data_dir + "/train", transform=data_transforms
)

test_data = torchvision.datasets.ImageFolder(
    data_dir + "/test", transform=data_transforms
)

CLASSES = train_data.classes

# Émotions à exclure
emotions_to_exclude = ["surprise", "neutral", "disgust"]

# Identifier les indices correspondants
indices_to_exclude = [CLASSES.index(emotion) for emotion in emotions_to_exclude]
print("Indices des émotions à exclure :", indices_to_exclude)

# Filtrer les datasets
train_data = filter_dataset(train_data, indices_to_exclude)
test_data = filter_dataset(test_data, indices_to_exclude)

# Mettre à jour la liste des classes après filtrage
CLASSES = [emotion for emotion in CLASSES if emotion not in emotions_to_exclude]
print("Classes après filtrage :", CLASSES)

# Define the validation set by splitting the training data into 2 subsets (80% training and 20% validation)
n_train_examples = int(len(train_data) * 0.8)
n_valid_examples = len(train_data) - n_train_examples
train_data, valid_data = random_split(train_data, [n_train_examples, n_valid_examples])


print("Classes of the dataset:", CLASSES)
print("Number of training samples:", len(train_data))
print("Number of test samples:", len(test_data))

# Pour afficher le mapping complet
for idx, emotion in enumerate(CLASSES):
    print(f"Label {idx} → {emotion}")

batch_size = 64

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=10,
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=10,
)

valid_dataloader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=10,
)

# - print the number of batches in the training subset
num_batches = len(train_dataloader)
print("Number of batches in the training subset:", num_batches)

# - print the number of batches in the testing subset
num_batches = len(test_dataloader)
print("Number of batches in the testing subset:", num_batches)

# - print the number of batches in the validation subset
num_batches = len(valid_dataloader)
print("Number of batches in the validation subset:", num_batches)


weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
model = torchvision.models.convnext_large(
    weights=weights
)  # charges les poids ImageNet pré-entraînés


# 1. Geler uniquement les premiers blocs convolutifs (features)
for param in model.features.parameters():
    param.requires_grad = False

for param in model.features[2:].parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

print(model.classifier)
# Remplacer la couche de sortie (le dernier module du classifier)
num_classes = 7
model.classifier[-1] = nn.Linear(in_features=1536, out_features=num_classes)

# Définir les hyperparamètres
num_epochs = 50
learning_rate = 0.00005
loss_fn = nn.CrossEntropyLoss()

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

torch.save(model_trained.state_dict(), "training/convnext_large_234_50.pt")

model_test = copy.deepcopy(model)
model_test.load_state_dict(torch.load("training/convnext_large_234_50.pt"))

# - Apply the evaluation function using the test dataloader
test_accuracy, avg_loss = eval_classifier(
    model=model_test, eval_dataloader=test_dataloader, device=device, loss_fn=loss_fn
)

print("====TEST RESULTS====")

# - Print the test accuracy
print("Test accuracy: {:.2f}%".format(test_accuracy))

# - Print the average loss
print("Average loss: {:.4f}".format(avg_loss))

# # - Plot the training loss over epochs
# plt.figure()
# plt.plot(train_losses)
# plt.title("Training loss over epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid()
# plt.show()
