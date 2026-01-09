import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Returns:
        dict: Dictionary with loss, accuracy, precision, recall, f1.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = batch_y.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train(model, train_loader, val_loader, epochs, lr, device):
    """
    Full training loop.
    
    Args:
        model: The DeepLOB model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: torch device.
    
    Returns:
        dict: Training history with losses and metrics.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1)
    criterion = nn.CrossEntropyLoss()
    
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    
    return history
