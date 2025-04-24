
import os
import json
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import yaml

from DataProcessor.gesture_dataset import GestureDataset, collate_fn

from Models.model_utils import get_model

def train_model(model, train_loader, val_loader, optimizer, criterion, device, label_map,
                epochs, save_path, plot_path, meta_path, scheduler=None, patience=15):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_acc, epochs_since_improvement = 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x_batch, lengths, y_batch in train_loader:
            x_batch, lengths, y_batch = x_batch.to(device), lengths.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch, lengths)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)
        train_acc = correct / total
        history['train_loss'].append(total_loss)
        history['train_acc'].append(train_acc)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_val, lengths, y_val in val_loader:
                x_val, lengths, y_val = x_val.to(device), lengths.to(device), y_val.to(device)
                outputs = model(x_val, lengths)
                pred_val = outputs.argmax(dim=1)
                val_correct += (pred_val == y_val).sum().item()
                val_total += y_val.size(0)
        val_acc = val_correct / val_total
        history['val_acc'].append(val_acc)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

        if scheduler: scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_map': label_map,
                'model_meta': {
                    'input_size': model.input_size,
                    'hidden_size': model.hidden_size,
                    'bidirectional': model.bidirectional,
                    'num_layers': model.num_layers,
                    'dropout': model.dropout.p if hasattr(model.dropout, 'p') else 0,
                    'type': model.__class__.__name__
                }
            }, save_path)
            print(f"✅ Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")
        else:
            epochs_since_improvement += 1
            print(f"⚠️ No improvement for {epochs_since_improvement} epochs")
        if epochs_since_improvement >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss"); plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend(); plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Overfitting View"); plt.ylim([0.5, 1.0]); plt.grid()
    plt.tight_layout(); plt.savefig(plot_path); plt.show()

    meta = {
        "epochs_trained": len(history['train_loss']),
        "best_val_acc": best_val_acc,
        "early_stopped": epochs_since_improvement >= patience
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def get_data_loader(data_config):
    dataset = GestureDataset(data_config['data_path'])
    label_map = dataset.label_map
    val_len = int(len(dataset) * data_config['val_ratio'])
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, label_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--exp_name', type=str, default='experiment')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config['output']['dir'], exist_ok=True)

    train_loader, val_loader, label_map = get_data_loader(config['data'])
    model = get_model(config, len(label_map)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    save_path = os.path.join(config['output']['dir'], f"{args.exp_name}.pt")
    plot_path = os.path.join(config['output']['dir'], f"{args.exp_name}_metrics.png")
    meta_path = os.path.join(config['output']['dir'], f"{args.exp_name}_meta.json")

    train_model(model, train_loader, val_loader, optimizer, criterion, device, label_map,
                config['train']['epochs'], save_path, plot_path, meta_path, scheduler, config['train']['patience'])

if __name__ == "__main__":
    main()
