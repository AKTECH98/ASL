import torch
# from attention_lstm import AttentionLSTM]

from .attention_lstm import AttentionLSTM
from .attention_gru import AttentionGestureGRU
from .gesture_gru import GestureGRU


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    label_map = {v: k for k, v in checkpoint['label_map'].items()}
    meta = checkpoint['model_meta']

    model_type = meta.get('type')
    model_cls = {
        'AttentionLSTM': AttentionLSTM,
        'GestureGRU': GestureGRU,
        'AttentionGestureGRU': AttentionGestureGRU
    }.get(model_type)

    if model_cls is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model_cls(
        input_size=meta['input_size'],
        hidden_size=meta['hidden_size'],
        num_classes=len(label_map),
        num_layers=meta.get('num_layers', 1),
        bidirectional=meta['bidirectional'],
        dropout=meta['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, label_map

def get_model(config, num_classes):
    model_type = config['model']['type']
    common_args = {
        "input_size": config['model']['input_size'],
        "hidden_size": config['model']['hidden_size'],
        "num_classes": num_classes,
        "num_layers": config['model']['num_layers'],
        "bidirectional": config['model']['bidirectional'],
        "dropout": config['model']['dropout']
    }
    if model_type == "attention":
        return AttentionLSTM(**common_args)
    elif model_type == "gru":
        return GestureGRU(**common_args)
    elif model_type == "attention_gru":
        return AttentionGestureGRU(**common_args)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")