import argparse
import copy
import os
import platform
import subprocess
import time
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type

import numpy as np
import psutil
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
        default="vgg",
        choices=["vgg", "convnext"],
        help="Type of model to use (vgg or convnext)",
    )

    parser.add_argument(
        "--model_version",
        type=str,
        default="11",
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
        default=None,
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
        default="heavy",
        choices=["none", "light", "medium", "heavy"],
        help="Level of data augmentation",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker threads (default: auto-optimized)",
    )

    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="Number of batches to prefetch (default: auto-optimized)",
    )

    return parser.parse_args()


def get_hardware_info() -> Dict[str, Any]:
    """Analyse exhaustive de la configuration matérielle."""
    hw_info = {}

    # Informations système de base
    hw_info["platform"] = platform.system()
    hw_info["platform_release"] = platform.release()

    # CPU
    hw_info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    hw_info["cpu_count_logical"] = psutil.cpu_count(logical=True)

    # Informations CPU plus détaillées (Linux)
    if hw_info["platform"] == "Linux":
        try:
            cpu_info = (
                subprocess.check_output("lscpu", shell=True)
                .decode()
                .strip()
                .split("\n")
            )
            for line in cpu_info:
                if "Model name" in line:
                    hw_info["cpu_model"] = line.split(":")[1].strip()
                if "CPU MHz" in line:
                    hw_info["cpu_freq"] = float(line.split(":")[1].strip())
        except Exception:
            pass

    # Mémoire
    vm = psutil.virtual_memory()
    hw_info["ram_total"] = vm.total / (1024**3)  # GB
    hw_info["ram_available"] = vm.available / (1024**3)  # GB

    # GPU
    hw_info["cuda_available"] = torch.cuda.is_available()
    if hw_info["cuda_available"]:
        hw_info["cuda_device_count"] = torch.cuda.device_count()
        hw_info["cuda_devices"] = []

        for i in range(hw_info["cuda_device_count"]):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "name": torch.cuda.get_device_name(i),
                "compute_capability": f"{device_props.major}.{device_props.minor}",
                "total_memory": device_props.total_memory / (1024**3),  # GB
                "multi_processor_count": device_props.multi_processor_count,
                # Vérifier si clock_rate existe avant d'y accéder
                "clock_rate": device_props.clock_rate / 1000
                if hasattr(device_props, "clock_rate")
                else None,
                "pci_bus_id": device_props.pci_bus_id
                if hasattr(device_props, "pci_bus_id")
                else None,
                "pci_device_id": device_props.pci_device_id
                if hasattr(device_props, "pci_device_id")
                else None,
            }
            hw_info["cuda_devices"].append(device_info)

        try:
            # Différentes versions de PyTorch peuvent stocker la version CUDA différemment
            if hasattr(torch, "version") and hasattr(torch.version, "cuda"):  # type: ignore
                hw_info["cuda_version"] = torch.version.cuda  # type: ignore
            else:
                # Alternative pour obtenir la version CUDA
                hw_info["cuda_version"] = (
                    torch.cuda.get_device_properties(0).major
                    + "."
                    + str(torch.cuda.get_device_properties(0).minor)
                )
        except Exception:
            hw_info["cuda_version"] = "Unknown"

    # Stockage
    disk = psutil.disk_usage("/")
    hw_info["disk_total"] = disk.total / (1024**3)  # GB
    hw_info["disk_free"] = disk.free / (1024**3)  # GB

    # Tester la vitesse de lecture disque (approximative)
    try:
        test_file = "/tmp/pytorch_read_test"
        test_size_mb = 100

        # Créer un fichier test
        with open(test_file, "wb") as f:
            f.write(os.urandom(test_size_mb * 1024 * 1024))

        # Mesurer le temps de lecture
        start_time = time.time()
        with open(test_file, "rb") as f:
            _ = f.read()
        read_time = time.time() - start_time

        hw_info["disk_read_speed_mb_s"] = test_size_mb / read_time

        # Supprimer le fichier test
        os.remove(test_file)
    except Exception:
        hw_info["disk_read_speed_mb_s"] = None

    return hw_info


def get_optimal_params(hw_info: Dict[str, Any]) -> Dict[str, Any]:
    """Calcule les paramètres optimaux en fonction de l'analyse matérielle."""
    params = {}

    # num_workers optimal: généralement 4x le nombre de cœurs physiques
    # mais limité par la RAM disponible
    cpu_count = hw_info["cpu_count_physical"]
    ram_gb = hw_info["ram_available"]

    # Calculer num_workers
    # Formule: min(4 * cpu_count, ram_gb / 2)
    # Assume ~2GB par worker pour le chargement des données
    params["num_workers"] = min(4 * cpu_count, int(ram_gb / 2))
    params["num_workers"] = max(1, params["num_workers"])  # Au moins 1

    # prefetch_factor optimal: basé sur la vitesse de lecture disque et mémoire disponible
    if hw_info.get("disk_read_speed_mb_s"):
        disk_speed = hw_info["disk_read_speed_mb_s"]
        # Plus rapide le disque, plus petit peut être le prefetch_factor
        if disk_speed > 500:  # SSD rapide
            params["prefetch_factor"] = 2
        elif disk_speed > 100:  # SSD standard
            params["prefetch_factor"] = 4
        else:  # HDD
            params["prefetch_factor"] = 8
    else:
        # Valeur par défaut conservatrice
        params["prefetch_factor"] = 5

    # Paramètres CUDA / cuDNN
    if hw_info["cuda_available"]:
        # allow_tf32: activer sur les GPUs Ampere (compute capability >= 8.0)
        # Vérifier que cuda_devices existe et n'est pas vide avant d'accéder à compute_capability
        if hw_info.get("cuda_devices") and len(hw_info["cuda_devices"]) > 0:
            try:
                newest_cc = max(
                    [
                        float(device["compute_capability"])
                        for device in hw_info["cuda_devices"]
                        if "compute_capability" in device
                    ]
                )
                params["allow_tf32"] = newest_cc >= 8.0
            except (ValueError, KeyError):
                # Valeur par défaut si on ne peut pas déterminer la compute capability
                params["allow_tf32"] = False
        else:
            params["allow_tf32"] = False

        # cudnn.benchmark: activer pour des tailles d'entrées fixes
        params["cudnn_benchmark"] = True

        # Taille de batch optimale basée sur la mémoire GPU
        smallest_gpu_mem = min(
            [device["total_memory"] for device in hw_info["cuda_devices"]]
        )
        # Approximation: 1GB → batch_size=64 pour un ResNet50 standard
        params["batch_size"] = int((smallest_gpu_mem / 2) * 64)
        # Limiter à une plage raisonnable
        params["batch_size"] = max(min(params["batch_size"], 512), 16)
        # Arrondir à la puissance de 2 la plus proche
        params["batch_size"] = 2 ** int(np.log2(params["batch_size"]) + 0.5)

    return params


def setup_pytorch_optimal(verbose=True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Configure PyTorch de manière optimale en fonction du matériel.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: (hw_info, optimal_params)
    """
    # Analyser le matériel
    hw_info = get_hardware_info()

    # Déterminer les paramètres optimaux
    params = get_optimal_params(hw_info)

    # Configurer PyTorch
    if hw_info["cuda_available"]:
        if params["allow_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        torch.backends.cudnn.benchmark = params["cudnn_benchmark"]

    # Afficher les informations si demandé
    if verbose:
        print("\n===== HARDWARE CONFIGURATION =====")
        print(f"Platform: {hw_info['platform']} {hw_info['platform_release']}")
        print(f"CPU: {hw_info.get('cpu_model', 'Unknown')}")
        print(
            f"CPU Cores: {hw_info['cpu_count_physical']} physical, {hw_info['cpu_count_logical']} logical"
        )
        print(
            f"RAM: {hw_info['ram_total']:.2f} GB total, {hw_info['ram_available']:.2f} GB available"
        )

        if hw_info["cuda_available"]:
            print("\n===== GPU CONFIGURATION =====")
            print(f"CUDA Version: {hw_info['cuda_version']}")
            print(f"GPU Count: {hw_info['cuda_device_count']}")
            for i, device in enumerate(hw_info["cuda_devices"]):
                print(f"GPU {i}: {device['name']}")
                print(f"  Compute Capability: {device['compute_capability']}")
                print(f"  Memory: {device['total_memory']:.2f} GB")
                print(f"  SMs: {device['multi_processor_count']}")

        print("\n===== OPTIMAL PARAMETERS =====")
        print(f"num_workers: {params['num_workers']}")
        print(f"prefetch_factor: {params['prefetch_factor']}")
        if hw_info["cuda_available"]:
            print(f"allow_tf32: {params['allow_tf32']}")
            print(f"cudnn_benchmark: {params['cudnn_benchmark']}")
            print(f"recommended batch_size: {params['batch_size']}")

    return hw_info, params


def get_model(
    num_classes: int,
    unfreeze_feature_layer_start: int,
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

    # # ResNet models
    # elif model_name.lower() == "resnet":
    #     if model_version == "18":
    #         weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    #         model = torchvision.models.resnet18(weights=weights)
    #         in_features = 512

    #     elif model_version == "34":
    #         weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
    #         model = torchvision.models.resnet34(weights=weights)
    #         in_features = 512

    #     elif model_version == "50":
    #         weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    #         model = torchvision.models.resnet50(weights=weights)
    #         in_features = 2048

    #     else:
    #         raise ValueError(f"Unsupported ResNet version: {model_version}")

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Freeze all the convolutional layers of the model
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the convolutional layers starting from the specified index
    features = nn.Sequential(*list(model.features.children()))
    for i in range(unfreeze_feature_layer_start, len(features)):
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
) -> str:
    if base_dir is None:
        # Get the project root directory (two levels up from the current file)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        # Create training directory at the project root
        base_dir = os.path.join(project_root, "training")

    # Create a subdirectory with the model_name
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Create a dynamic filename based on model parameters
    model_filename = f"{model_name}_{model_version}_b{batch_size}_l{unfreeze_layer_start}:end_e{num_epochs}.pt"
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

    # Handle color mode conversion
    if input_format == "grayscale":
        transform_list.append(transforms.Grayscale(num_output_channels=target_channels))

    # Handle other color conversions
    if input_format == "rgb":
        if target_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        elif target_channels == 4:
            # Add alpha channel (convert RGB to RGBA)
            # This would typically be handled in a custom transform
            pass  # Custom handling would go here
    elif input_format == "rgba":
        if target_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        elif target_channels == 3:
            # Drop alpha channel (convert RGBA to RGB)
            # This would typically be handled in a custom transform
            pass  # Custom handling would go here

    # Apply augmentation based on level
    if augmentation_level != "none":
        if augmentation_level in ["medium", "heavy"]:
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())
            transform_list.append(transforms.RandomRotation(10))
            transform_list.append(transforms.RandomAffine(0, translate=(0.1, 0.1)))
            transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))

        if augmentation_level == "light":
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomRotation(5))

        if augmentation_level == "heavy":
            # Add more intense augmentations
            transform_list.append(
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
            )
            transform_list.append(transforms.RandomErasing(p=0.2))

    # Always resize to target size
    transform_list.append(transforms.Resize(target_size))

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

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
    transform_fn: Optional[Callable] = None,
    verbose: bool = True,
):
    # Make a copy of the model (avoid changing the model outside this function)
    model_tr = copy.deepcopy(model)
    model_tr = model_tr.to(device, non_blocking=True)

    # Set the model to training mode, this is important for models that have layers like dropout or batch normalization
    model_tr.train()

    optimizer = optimizer_class(model_tr.parameters(), **{"lr": learning_rate})  # type: ignore

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    # Initialize a list for storing the training loss over epochs
    train_losses = []

    accuracy_max = 0
    val_accuracies = []
    model_opt = copy.deepcopy(model_tr)  # Initialize model_opt with the initial model
    model_opt.to(device, non_blocking=True)

    # Training loop
    for epoch in range(num_epochs):
        # Initialize the training loss for the current epoch
        tr_loss = 0

        # Iterate over batches using the dataloader
        for images, labels in train_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # If a transform function is provided, apply it to the images
            if transform_fn is not None:
                images = transform_fn(images)

            # - calculate the predicted labels from the vectorized images using 'model_tr'
            labels_pred = model_tr(images)

            # - using loss_fn, calculate the 'loss' between the predicted and true labels
            loss = loss_fn(labels_pred, labels)

            # - set the optimizer gradients at 0 for safety
            optimizer.zero_grad(set_to_none=True)

            # - compute the gradients (use the 'backward' method on 'loss')
            loss.backward()

            # - apply the gradient descent algorithm (perform a step of the optimizer)
            optimizer.step()

            # Update the current epoch loss
            # Note that 'loss.item()' is the loss averaged over the batch, so multiply it with the current batch size to get the total batch loss
            with torch.no_grad():
                tr_loss += loss.item() * batch_size

        # At the end of each epoch, get the average training loss and store it
        tr_loss = tr_loss / (len(train_dataloader) * batch_size)
        train_losses.append(tr_loss)

        # Display the training loss
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


def train_classifier(
    model: nn.Module,
    train_dataloader: DataLoader,
    batch_size: int,
    num_epochs: int,
    loss_fn: nn.Module,
    optimizer_class: Type[torch.optim.Optimizer],
    learning_rate: float,
    device: torch.device,
    transform_fn: Optional[Callable] = None,
    verbose: bool = True,
):
    # Make a copy of the model (avoid changing the model outside this function)
    model_tr = copy.deepcopy(model)

    # Move the model to the specified device (GPU or CPU)
    model_tr = model_tr.to(device, non_blocking=True)

    # Set the model to training mode, this is important for models that have layers like dropout or batch normalization
    model_tr.train()

    optimizer = optimizer_class(model_tr.parameters(), **{"lr": learning_rate})  # type: ignore

    # Initialize a list for storing the training loss over epochs
    train_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Initialize the training loss for the current epoch
        tr_loss = 0

        # Iterate over batches using the dataloader
        for images, labels in train_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # If a transform function is provided, apply it to the images
            if transform_fn is not None:
                images = transform_fn(images)

            # - calculate the predicted labels from the images using 'model_tr'
            labels_pred = model_tr(images)

            # - using loss_fn, calculate the 'loss' between the predicted and true labels
            loss = loss_fn(labels_pred, labels)

            # - set the optimizer gradients at 0 for safety
            optimizer.zero_grad(set_to_none=True)

            # - compute the gradients (use the 'backward' method on 'loss')
            loss.backward()

            # - apply the gradient descent algorithm (perform a step of the optimizer)
            optimizer.step()

            # Update the current epoch loss
            # Note that 'loss.item()' is the loss averaged over the batch, so multiply it with the current batch size to get the total batch loss
            with torch.no_grad():
                tr_loss += loss.item() * batch_size

        # At the end of each epoch, get the average training loss and store it
        tr_loss = tr_loss / (len(train_dataloader) * batch_size)
        train_losses.append(tr_loss)

        # Display the training loss
        if verbose:
            print(
                "Epoch [{}/{}], Training loss: {:.4f}".format(
                    epoch + 1, num_epochs, tr_loss
                )
            )

    return model_tr, train_losses


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

    # initialize the total and correct number of labels to compute the accuracy
    correct_labels = 0
    total_labels = 0
    total_loss = 0  # Pour accumuler la perte totale

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
            total_loss += loss.item() * labels.size(
                0
            )  # Multiplier par la taille du batch

            # To get the predicted labels, we need to get the max over all possible classes
            _, labels_predicted = torch.max(y_predicted.data, 1)

            # Compute accuracy: count the total number of samples, and the correct labels (compare the true and predicted labels)
            total_labels += labels.size(0)
            correct_labels += (labels_predicted == labels).sum().item()

    accuracy = 100 * correct_labels / total_labels
    avg_loss = total_loss / total_labels

    return accuracy, avg_loss
