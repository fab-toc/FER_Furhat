import argparse
import copy
import os
from typing import Callable, Literal, Optional, Tuple, Type

import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a facial expression recognition model"
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="convnext",
        choices=["vgg", "convnext"],
        help="Type of model to use (vgg or convnext)",
    )

    parser.add_argument(
        "--model_version",
        type=str,
        default="tiny",
        choices=["11", "13", "16", "19", "tiny", "small", "base", "large"],
        help="Version of the model",
    )

    parser.add_argument(
        "--unfreeze_layer", type=int, default=6, help="Layer to start unfreezing from"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        choices=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="Batch size (default: auto-optimized)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        choices=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        help="Number of training epochs",
    )

    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")

    parser.add_argument(
        "--augmentation",
        type=str,
        default="medium",
        choices=["none", "light", "medium", "heavy"],
        help="Level of data augmentation",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of worker threads (default: auto-optimized)",
    )

    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Number of batches to prefetch (default: auto-optimized)",
    )

    return parser.parse_args()


def get_model(
    num_classes: int,
    unfreeze_layer_start: int,
    model_name: Literal["vgg", "convnext"] = "convnext",
    model_version: Literal[
        "11", "13", "16", "19", "tiny", "small", "base", "large"
    ] = "tiny",
):
    # VGG models
    if model_name.lower() == "vgg":
        if model_version == "11":
            weights = torchvision.models.VGG11_Weights.IMAGENET1K_V1
            model = torchvision.models.vgg11(weights=weights)
            in_features = 4096

        elif model_version == "13":
            weights = torchvision.models.VGG13_Weights.IMAGENET1K_V1
            model = torchvision.models.vgg13(weights=weights)
            in_features = 4096

        elif model_version == "16":
            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
            model = torchvision.models.vgg16(weights=weights)
            in_features = 4096

        elif model_version == "19":
            weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
            model = torchvision.models.vgg19(weights=weights)
            in_features = 4096

        else:
            raise ValueError(f"Unsupported VGG version: {model_version}")

    # ConvNeXt models
    elif model_name.lower() == "convnext":
        if model_version == "tiny":
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            model = torchvision.models.convnext_tiny(weights=weights)
            in_features = 768

        elif model_version == "small":
            weights = torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            model = torchvision.models.convnext_small(weights=weights)
            in_features = 768

        elif model_version == "base":
            weights = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            model = torchvision.models.convnext_base(weights=weights)
            in_features = 1024

        elif model_version == "large":
            weights = torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
            model = torchvision.models.convnext_large(weights=weights)
            in_features = 1536

        else:
            raise ValueError(f"Unsupported ConvNeXt version: {model_version}")

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Freeze all the convolutional layers of the model
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the convolutional layers starting from the specified index
    features = nn.Sequential(*list(model.features.children()))
    for i in range(unfreeze_layer_start, len(features)):
        for param in features[i].parameters():
            param.requires_grad = True

    # Reassign features to the model
    model.features = features

    # Replace the output layer (the last layer of the classifier)
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)

    # Unfreeze the classifier part of the model
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def save_model(
    model: nn.Module,
    model_name: str,
    model_version: str,
    batch_size: int,
    unfreeze_layer_start: int,
    num_epochs: int,
    base_dir: Optional[str] = None,
    fine_tuned: bool = False,
) -> str:
    if base_dir is None:
        # Get the project root directory (two levels up from the current file)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        base_dir = os.path.join(project_root, "trained")

    # Create a subdirectory with the model_name
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    if fine_tuned:
        model_filename = f"fine-tuned_{model_name}_{model_version}_b{batch_size}_l{unfreeze_layer_start}_end_e{num_epochs}.pt"
    else:
        model_filename = f"{model_name}_{model_version}_b{batch_size}_l{unfreeze_layer_start}_end_e{num_epochs}.pt"

    model_path = os.path.join(model_dir, model_filename)

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model_path


def get_data_transforms(
    input_format: Literal["grayscale", "rgb", "rgba"],
    target_channels: int,
    target_size: Tuple[int, int],
    augmentation_level: Optional[
        Literal["none", "light", "medium", "heavy", None]
    ] = "medium",
    custom_means: Optional[list[float]] = [
        0.485,
        0.456,
        0.406,
    ],  # Default to ImageNet stats if not specified
    custom_stds: Optional[list[float]] = [
        0.229,
        0.224,
        0.225,
    ],  # Default to ImageNet stats if not specified
) -> transforms.Compose:
    transform_list = []

    if input_format == "grayscale":
        transform_list.append(transforms.Grayscale(num_output_channels=target_channels))

    elif input_format == "rgb":
        if target_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        else:
            pass  # Custom handling would go here

    elif input_format == "rgba":
        if target_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        else:
            pass  # Custom handling would go here

    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    # Apply augmentation based on level
    if augmentation_level != "none":
        if augmentation_level in ["medium", "heavy"]:
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())
            transform_list.append(transforms.RandomRotation(10))
            transform_list.append(transforms.RandomAffine(0, translate=(0.1, 0.1)))

        elif augmentation_level == "light":
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomRotation(5))

        else:
            raise ValueError(
                f"Unsupported data augmentation level: {augmentation_level}"
            )

    # Always resize to target size
    transform_list.append(transforms.Resize(target_size))

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    if augmentation_level == "heavy":
        # Add more intense augmentations
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
        transform_list.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.5))
        transform_list.append(transforms.RandomErasing(p=0.2))

    # Normalize using custom means and stds
    transform_list.append(transforms.Normalize(mean=custom_means, std=custom_stds))

    return transforms.Compose(transform_list)


def train_classifier_with_validation(
    model: nn.Module,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    batch_size: int,
    num_epochs: int,
    loss_fn: nn.Module,
    optimizer_class: Type[torch.optim.Optimizer],
    learning_rate: float,
    device: torch.device,
    verbose: bool = True,
):
    # Make a copy of the model (avoid changing the model outside this function)
    model_tr = copy.deepcopy(model)
    model_tr = model_tr.to(device, non_blocking=True)

    # Set the model to training mode, this is important for models that have layers like dropout or batch normalization
    model_tr.train()

    optimizer = optimizer_class(model_tr.parameters(), **{"lr": learning_rate})  # type: ignore

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=1)

    train_losses = []
    accuracy_max = 0
    val_accuracies = []
    model_opt = copy.deepcopy(model_tr)  # Initialize model_opt with the initial model
    model_opt.to(device, non_blocking=True)

    for epoch in range(num_epochs):
        # Initialize the training loss for the current epoch
        tr_loss = 0

        # Iterate over batches using the dataloader
        for images, labels in train_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Calculate the predicted labels
            labels_pred = model_tr(images)

            # Using loss_fn, calculate the 'loss' between the predicted and true labels
            loss = loss_fn(labels_pred, labels)

            # Set the optimizer gradients at "None" for safety
            optimizer.zero_grad(set_to_none=True)

            # Compute the gradients (use the 'backward' method on 'loss')
            loss.backward()

            # Apply the gradient descent algorithm (perform a step of the optimizer)
            optimizer.step()

            # Update the current epoch loss
            # Note that 'loss.item()' is the loss averaged over the batch, so multiply it with the current batch size to get the total batch loss
            with torch.no_grad():
                tr_loss += loss.item() * batch_size

        # At the end of each epoch, get the average training loss and store it
        tr_loss = tr_loss / (len(train_dataloader) * batch_size)
        train_losses.append(tr_loss)

        if verbose:
            print(
                "Epoch [{}/{}], Training loss: {:.4f}".format(
                    epoch + 1, num_epochs, tr_loss
                )
            )

        # Evaluate the model on the validation set
        accuracy, avg_loss = eval_classifier(
            model_tr, valid_dataloader, device=device, loss_fn=loss_fn
        )
        val_accuracies.append(accuracy)

        # Store the validation accuracy
        if accuracy > accuracy_max:
            accuracy_max = accuracy
            model_opt = copy.deepcopy(model_tr)
            model_opt.to(device, non_blocking=True)

        if verbose:
            print(
                "Validation accuracy: {:.2f}%\nValidation loss: {:.4f}".format(
                    accuracy, avg_loss
                )
            )

        # Step the scheduler with the validation loss
        scheduler.step(avg_loss)

        # Display the current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        if verbose:
            print(f"LR after epoch {epoch + 1}: {current_lr:.3e}\n")

    return model_opt, train_losses, val_accuracies


def eval_classifier(
    model: nn.Module,
    eval_dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    transform_fn: Optional[Callable] = None,
):
    # Set the model in 'evaluation' mode (this disables some layers (batch norm, dropout...) which are not needed when testing)
    model.eval()

    model.to(device, non_blocking=True)

    # Initialize the total and correct number of labels to compute the accuracy
    correct_labels = 0
    total_labels = 0
    total_loss = 0

    # In evaluation phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        # Iterate over the dataset using the dataloader
        for images, labels in eval_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # If a transform function is provided, apply it to the images
            if transform_fn is not None:
                images = transform_fn(images)

            # Get the predicted labels
            y_predicted = model(images)

            loss = loss_fn(y_predicted, labels)
            total_loss += loss.item() * labels.size(0)  # Multiply by batch size

            # To get the predicted labels, we need to get the max over all possible classes
            _, labels_predicted = torch.max(y_predicted.data, 1)

            # Compute accuracy: count the total number of samples, and the correct labels (compare the true and predicted labels)
            total_labels += labels.size(0)
            correct_labels += (labels_predicted == labels).sum().item()

    accuracy = 100 * correct_labels / total_labels
    avg_loss = total_loss / total_labels

    return accuracy, avg_loss
