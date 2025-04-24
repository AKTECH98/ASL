
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from DataProcessor.gesture_dataset import GestureDataset, collate_fn
from Models.attention_lstm import AttentionLSTM
from Models.gesture_lstm import GestureLSTM
from Models.gesture_gru import GestureGRU
from Models.attention_gru import AttentionGestureGRU
from Models.model_utils import load_model

def plot_confusion_matrix(cm, classes, save_path, normalize=False, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate(model_path, data_path, batch_size=32, val_ratio=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GestureDataset(data_path)
    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len
    _, val_dataset = random_split(dataset, [train_len, val_len])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model, LABEL_MAP = load_model(model_path, device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, lengths, y_batch in val_loader:
            x_batch, lengths = x_batch.to(device), lengths.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch, lengths)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"✅ Validation Accuracy: {acc:.4f}\n")

    report = classification_report(all_labels, all_preds, target_names=LABEL_MAP.values())
    print("📊 Classification Report:\n", report)

    report_path = os.path.splitext(model_path)[0] + "_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Validation Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"📝 Report saved to {report_path}")

    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.splitext(model_path)[0] + "_cmatrix.png"
    plot_confusion_matrix(cm, LABEL_MAP, save_path=cm_path)
    print(f"🧩 Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment name for model checkpoint")
    parser.add_argument('--data_path', type=str, default='Preprocessed/Data45')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    model_file = os.path.join("Output", f"{args.exp_name}.pt")
    evaluate(
        model_path=model_file,
        data_path=args.data_path,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio
    )
