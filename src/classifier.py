import copy
from typing import Callable, Optional, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

def filter_dataset(dataset, indices_to_exclude):
    # Déterminer les labels valides et leur mapping vers de nouveaux indices
    valid_labels = sorted(set(range(len(dataset.classes))) - set(indices_to_exclude))
    mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}
    
    # Filtrer les samples et mettre à jour les labels
    new_samples = [(path, mapping[label])
                   for path, label in dataset.samples
                   if label not in indices_to_exclude]
    
    # Mettre à jour le dataset directement
    dataset.samples = new_samples
    dataset.targets = [label for _, label in new_samples]
    dataset.classes = [dataset.classes[i] for i in valid_labels]
    
    return dataset


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

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

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
        current_lr = optimizer.param_groups[0]['lr']
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
    loss_fn: Optional[nn.Module] = None,
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

            # Calculer la perte si une fonction de perte est fournie
            if loss_fn is not None:
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
    avg_loss = total_loss / total_labels if loss_fn is not None else None

    return accuracy, avg_loss
