import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader


def train_classifier(
    model: nn.Module,
    train_dataloader: DataLoader,
    batch_size: int,
    num_epochs: int,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    device: torch.device,
    transform_fn: Optional[Callable] = None,
    verbose: bool = True,
):
    # Make a copy of the model (avoid changing the model outside this function)
    model_tr = copy.deepcopy(model)

    # Move the model to the specified device (GPU or CPU)
    model_tr = model_tr.to(device, non_blocking=True)

    # Create a GradScaler to handle dynamic scale of loss
    scaler = GradScaler(
        init_scale=2**10,  # Valeur d'échelle initiale
        growth_factor=2.0,  # Facteur de croissance si pas d'inf/nan
        backoff_factor=0.5,  # Facteur de réduction si inf/nan détecté
        growth_interval=2000,  # Nombre d'étapes réussies avant d'augmenter l'échelle
        enabled=True,
    )

    # Set the model to training mode, this is important for models that have layers like dropout or batch normalization
    model_tr.train()

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

            with autocast("cuda"):
                # - calculate the predicted labels from the images using 'model_tr'
                labels_pred = model_tr(images)

                # - using loss_fn, calculate the 'loss' between the predicted and true labels
                loss = loss_fn(labels_pred, labels)

            # - set the optimizer gradients at 0 for safety
            optimizer.zero_grad(set_to_none=True)

            # - compute the gradients (use the 'backward' method on 'loss')
            scaler.scale(loss).backward()

            # Désescalade pour le gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_tr.parameters(), max_norm=1.0)

            # - apply the gradient descent algorithm (perform a step of the optimizer)
            scaler.step(optimizer)

            # Gestion sécurisée de la mise à jour du scaler
            try:
                scaler.update()
            except RuntimeError as e:
                print(f"Warning: {e}. Continuing with training...")
                # Réinitialisation du scaler si nécessaire
                scaler = GradScaler(init_scale=2**10, enabled=True)

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
    transform_fn: Optional[Callable] = None,
):
    # Set the model in 'evaluation' mode (this disables some layers (batch norm, dropout...) which are not needed when testing)
    model.eval()

    model.to(device, non_blocking=True)

    # In evaluation phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        # initialize the total and correct number of labels to compute the accuracy
        correct_labels = 0
        total_labels = 0

        # Iterate over the dataset using the dataloader
        for images, labels in eval_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # If a transform function is provided, apply it to the images
            if transform_fn is not None:
                images = transform_fn(images)

            # Get the predicted labels
            y_predicted = model(images)

            # To get the predicted labels, we need to get the max over all possible classes
            _, labels_predicted = torch.max(y_predicted.data, 1)

            # Compute accuracy: count the total number of samples, and the correct labels (compare the true and predicted labels)
            total_labels += labels.size(0)
            correct_labels += (labels_predicted == labels).sum().item()

    accuracy = 100 * correct_labels / total_labels

    return accuracy
