# instead of coding the training process, abstract the training process and testing process into two functions
import torch
import torch.nn as nn
import torchmetrics
import time
from tqdm.auto import tqdm

def train_model(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: torchmetrics.Accuracy,
               device: torch.device = 'mps',
               epoch: int = 1):
    """Train the model

    Args:
        model (nn.Module): The model to train
        data_loader (torch._utils.data.data_loader): The data loader for the training data
        loss_fn (nn.Module): The loss function to calculate the loss
        optimizer (torch.optim.Optimizer): The optimizer to update the model's weights
        accuracy_fn (torchmetrics.Accuracy): The accuracy function to calculate the accuracy
        device (torch.device, optional): The device used to train the model. Defaults to 'mps'
        epoch (int, optional): The number of epochs to train the model. Defaults to 1.
    """
    
    model.train()
    avg_loss, avg_acc = 0, 0
    
    with tqdm(data_loader, desc=f"Training | Epoch: {epoch}") as t:
        for X, y in t:
            X, y = X.to(device), y.to(device)
            
            y_logits = model(X)
            y_pred = y_logits.softmax(dim=1).argmax(dim=1)
            
            # Calculate loss
            loss = loss_fn(y_logits, y)
            avg_loss += loss
            
            # Calculate accuracy
            acc = accuracy_fn(y, y_pred)
            avg_acc += acc
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Gradient descent
            optimizer.step()
            
            # Update tqdm description
            t.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc.item():.4f}')
    
    # Divide total test loss by length of test dataloader (per batch)
    avg_loss /= len(data_loader)
    avg_acc /= len(data_loader)
    
    return {
        "model_name": model.__class__.__name__,
        "model_loss": avg_loss.item(),
        "model_accuracy": avg_acc.item(),
    }
    
    
def eval_model(model: nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              accuracy_fn: torchmetrics.Accuracy,
              device: torch.device = 'mps',
              epoch: int = 1):
    """eval the model

    Args:
        model (nn.Module): The model to test
        data_loader (torch.utils.data.DataLoader): The data loader for the testing data
        loss_fn (nn.Module): The loss function to calculate the loss
        accuracy_fn (torchmetrics.Accuracy): The accuracy function to calculate the accuracy
        device (torch.device, optional): The device used to train the model. Defaults to 'mps'
        epoch (int, optional): The number of epochs to train the model. Defaults to 1.
    """
    
    model.eval()
    avg_loss, avg_acc = 0, 0
    
    with torch.inference_mode():
        with tqdm(data_loader, desc=f"Evaluating | Epoch: {epoch}") as t:
            for X, y in t:
                X, y = X.to(device), y.to(device)
                y_logits = model(X)
                y_pred = y_logits.softmax(dim=1).argmax(dim=1)
                
                # Calculate loss and accuracy
                loss = loss_fn(y_logits, y)
                avg_loss += loss
                
                acc = accuracy_fn(y, y_pred)
                avg_acc += acc
                
                # Update tqdm description
                t.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc.item():.4f}')
    
        # Divide total test loss by length of test dataloader (per batch)
        avg_loss /= len(data_loader)
        avg_acc /= len(data_loader)
        
        return {
                "model_name": model.__class__.__name__,
                "model_loss": avg_loss.item(),
                "model_accuracy": avg_acc.item(),
            }


def train_model(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: torchmetrics.Accuracy,
               device: torch.device = 'mps',
               epoch: int = 1):
    """Train the model

    Args:
        model (nn.Module): The model to train
        data_loader (torch._utils.data.data_loader): The data loader for the training data
        loss_fn (nn.Module): The loss function to calculate the loss
        optimizer (torch.optim.Optimizer): The optimizer to update the model's weights
        accuracy_fn (torchmetrics.Accuracy): The accuracy function to calculate the accuracy
        device (torch.device, optional): The device used to train the model. Defaults to 'mps'
        epoch (int, optional): The number of epochs to train the model. Defaults to 1.
    """
    
    model.train()
    avg_loss, avg_acc = 0, 0
    
    with tqdm(data_loader, desc=f"Training | Epoch: {epoch}") as t:
        for X, y in t:
            X, y = X.to(device), y.to(device)
            
            y_logits = model(X)
            y_pred = y_logits.softmax(dim=1).argmax(dim=1)
            
            # Calculate loss
            loss = loss_fn(y_logits, y)
            avg_loss += loss
            
            # Calculate accuracy
            acc = accuracy_fn(y, y_pred)
            avg_acc += acc
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Gradient descent
            optimizer.step()
            
            # Update tqdm description
            t.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc.item():.4f}')
    
    # Divide total test loss by length of test dataloader (per batch)
    avg_loss /= len(data_loader)
    avg_acc /= len(data_loader)
    
    return {
        "model_name": model.__class__.__name__,
        "model_loss": avg_loss.item(),
        "model_accuracy": avg_acc.item(),
    }
    
    
def eval_model(model: nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              metrics: torchmetrics.MetricCollection,
              device: torch.device = 'mps',
              epoch: int = 1):
    """eval the model with torchmetrics

    Args:
        model (nn.Module): The model to test
        data_loader (torch.utils.data.DataLoader): The data loader for the testing data
        loss_fn (nn.Module): The loss function to calculate the loss
        accuracy_fn (torchmetrics.Accuracy): The accuracy function to calculate the accuracy
        device (torch.device, optional): The device used to train the model. Defaults to 'mps'
        epoch (int, optional): The number of epochs to train the model. Defaults to 1.
    """
    
    model.eval()
    avg_loss, avg_acc = 0, 0
    
    with torch.inference_mode():
        with tqdm(data_loader, desc=f"Evaluating | Epoch: {epoch}") as t:
            for X, y in t:
                X, y = X.to(device), y.to(device)
                y_logits = model(X)
                y_pred = y_logits.softmax(dim=1).argmax(dim=1)
                
                # Calculate loss and accuracy
                loss = loss_fn(y_logits, y)
                avg_loss += loss
                
                # Calculate metrics
                metric = metrics(y_pred, y)
                
                acc = accuracy_fn(y, y_pred)
                avg_acc += acc
                
                # Update tqdm description
                t.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc.item():.4f}')
    
        # Divide total test loss by length of test dataloader (per batch)
        avg_loss /= len(data_loader)
        avg_acc /= len(data_loader)
        
        return {
                "model_name": model.__class__.__name__,
                "model_loss": avg_loss.item(),
                "model_accuracy": avg_acc.item(),
            }


def timer():
    start_time = time.time()
    
    # Your code or function calls here
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

# Example usage:
timer()
 