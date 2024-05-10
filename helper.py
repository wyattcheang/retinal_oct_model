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
    total_loss, total_acc = 0, 0
    batch_result = []
    
    with tqdm(data_loader, desc=f"Epoch {epoch} | Training", leave=True) as t:
        for batch, (X, y) in enumerate(t):
            X, y = X.to(device), y.to(device)
            
            y_logits = model(X)
            y_pred = y_logits.softmax(dim=1).argmax(dim=1)
            
            # Calculate loss
            loss = loss_fn(y_logits, y)
            
            # Calculate accuracy
            acc = accuracy_fn(y, y_pred)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Gradient descent
            optimizer.step()
            
            # Update data
            total_loss += loss
            total_acc += acc
            
            cuurent_avg_loss = total_loss.item()/(batch + 1)
            current_avg_acc = total_acc.item()/(batch + 1)
            
            batch_result.append({
                "batch": batch + 1,
                "loss": cuurent_avg_loss,
                "accuracy": current_avg_acc
            })
            
            # Update tqdm description
            t.set_postfix(loss=f'{cuurent_avg_loss:.4f}', acc=f'{current_avg_acc:.4f}')
    
    return {
        "name": model.__class__.__name__,
        "result": batch_result,
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
    total_loss, total_acc = 0, 0
    batch_result = []
    
    with torch.inference_mode():
        with tqdm(data_loader, desc=f"Epoch {epoch} | Evaluating", leave=True) as t:
            for batch, (X, y) in enumerate(t):
                X, y = X.to(device), y.to(device)
                y_logits = model(X)
                y_pred = y_logits.softmax(dim=1).argmax(dim=1)
                
                # Calculate loss and accuracy
                loss = loss_fn(y_logits, y)
                
                acc = accuracy_fn(y, y_pred)
                
                # Update data
                total_loss += loss
                total_acc += acc
                
                cuurent_avg_loss = total_loss.item()/(batch + 1)
                current_avg_acc = total_acc.item()/(batch + 1)
                
                batch_result.append({
                    "batch": batch + 1,
                    "loss": cuurent_avg_loss,
                    "accuracy": current_avg_acc
                })
                
                # Update tqdm description
                t.set_postfix(loss=f'{cuurent_avg_loss:.4f}', acc=f'{current_avg_acc:.4f}')
    
        return {
            "name": model.__class__.__name__,
            "result": batch_result,
        }


def predict_model(model: nn.Module,
              data_loader: torch.utils.data.DataLoader,
              device: torch.device = 'mps'):
    """eval the model

    Args:
        model (nn.Module): The model to test
        data_loader (torch.utils.data.DataLoader): The data loader for the testing data
        device (torch.device, optional): The device used to train the model. Defaults to 'mps'
    """
    
    model.eval()
    y_label, y_preds = [], []
    
    with torch.inference_mode():
        with tqdm(data_loader, desc=f"Making Prediction", leave=True) as t:
            for batch, (X, y) in enumerate(t):
                X, y = X.to(device), y.to(device)
                y_logits = model(X)
                y_pred = y_logits.softmax(dim=1).argmax(dim=1)
                
                y_label.append(y.cpu())
                y_preds.append(y_pred.cpu())
        
        return {
            "name": model.__class__.__name__,
            "target": y_label,
            "preds": y_preds,
        }

def timer():
    start_time = time.time()
    
    # Your code or function calls here
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
 