import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle
from utils.training_utils import train_model as train_model_utils, accuracy
from utils.visualization_utils import plot_learning_curves, plot_gradient_flow, plot_feature_maps
from utils.comparison_utils import compare_accuracy, compare_time, compare_params, print_comparison_table

# =====================
# 2.1 Влияние размера ядра свертки
# =====================

class CNNKernel3x3(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels * 14 * 14, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNNKernel5x5(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels * 14 * 14, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNNKernel7x7(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels * 14 * 14, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNNKernelCombo(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels * 14 * 14, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =====================
# 2.2 Влияние глубины CNN
# =====================

class ShallowCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels * 14 * 14, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MediumCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels * 14 * 14, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DeepCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv6 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels * 14 * 14, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = F.relu(out)
        return out

class ResidualCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(out_channels * 14 * 14, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =====================
# Вспомогательные функции 
# =====================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_history(history, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(history, f)

def save_plot(fig, path):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path)
    plt.close(fig)

def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def analyze_receptive_field(model, save_path=None):
    """
    Аналитически вычисляет рецептивное поле для последовательности conv/pool слоёв.
    """
    layers = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            layers.append({'type': 'conv', 'kernel': m.kernel_size, 'stride': m.stride, 'padding': m.padding})
        elif isinstance(m, nn.MaxPool2d):
            layers.append({'type': 'pool', 'kernel': m.kernel_size, 'stride': m.stride, 'padding': m.padding})
    rf = 1
    jump = 1
    info = []
    for i, l in enumerate(layers):
        k, s, p = l['kernel'], l['stride'], l['padding']
        k = k[0] if isinstance(k, tuple) else k
        s = s if isinstance(s, int) else s[0]
        p = p if isinstance(p, int) else p[0]
        rf_new = rf + (k - 1) * jump
        info.append(f"Layer {i+1}: {l['type']} | kernel={k}, stride={s}, padding={p} | RF={rf_new}")
        jump = jump * s
        rf = rf_new
    result = '\n'.join(info)
    print("\nReceptive field analysis:")
    print(result)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(result)

def visualize_first_layer_activations(model, loader, device, save_path=None):
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            if hasattr(model, 'conv'):
                activations = model.conv(inputs)
            elif hasattr(model, 'conv1'):
                activations = model.conv1(inputs)
            else:
                return
            act = activations[0].detach().cpu().numpy()
            C = act.shape[0]
            fig, axes = plt.subplots(1, min(8, C), figsize=(2*min(8, C),2))
            for i in range(min(8, C)):
                axes[i].imshow(act[i], cmap='viridis')
                axes[i].axis('off')
                axes[i].set_title(f'Ch {i}')
            plt.tight_layout()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path)
                plt.close(fig)
            else:
                plt.show()
            break

def visualize_feature_maps(model, loader, device, save_path=None):
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            x = inputs
            maps = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    x = module(x)
                    maps.append(x[0].detach().cpu().numpy())
            if not maps:
                return
            fig, axes = plt.subplots(1, min(8, len(maps)), figsize=(2*min(8, len(maps)),2))
            for i in range(min(8, len(maps))):
                axes[i].imshow(maps[i][0], cmap='viridis')
                axes[i].axis('off')
                axes[i].set_title(f'Layer {i}')
            plt.tight_layout()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path)
                plt.close(fig)
            else:
                plt.show()
            break

# =====================
# Основной цикл анализа
# =====================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 8
    BATCH_SIZE = 128
    LR = 0.001
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    # 2.1 Влияние размера ядра свертки
    kernel_models = [
        ("3x3", CNNKernel3x3(1, 32)),
        ("5x5", CNNKernel5x5(1, 16)),
        ("7x7", CNNKernel7x7(1, 8)),
        ("combo", CNNKernelCombo(1, 16)),
    ]
    for name, model in kernel_models:
        print(f"\nTraining kernel {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        history = train_model_utils(model, train_loader, test_loader, criterion, optimizer, device, epochs=EPOCHS)
        if 'val_loss' in history:
            history['test_loss'] = history.pop('val_loss')
        if 'val_acc' in history:
            history['test_acc'] = history.pop('val_acc')
        print(f"{name} kernel parameters: {count_parameters(model)}")
        save_model(model, "models/cnn_models.pt")
        save_history(history, f"results/architecture_analysis/kernel_{name}_history.pkl")
        plot_learning_curves(history, f"Kernel {name}", save_path=f"plots/architecture_analysis/kernel_{name}_learning_curve.png")
        analyze_receptive_field(model, save_path=f"results/architecture_analysis/kernel_{name}_receptive_field.txt")
        visualize_first_layer_activations(model, train_loader, device, save_path=f"plots/architecture_analysis/kernel_{name}_first_layer_activations.png")
    # 2.2 Влияние глубины CNN
    depth_models = [
        ("shallow", ShallowCNN(1, 16)),
        ("medium", MediumCNN(1, 16)),
        ("deep", DeepCNN(1, 8)),
        ("residual", ResidualCNN(1, 16)),
    ]
    for name, model in depth_models:
        print(f"\nTraining depth {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        history = train_model_utils(model, train_loader, test_loader, criterion, optimizer, device, epochs=EPOCHS)
        if 'val_loss' in history:
            history['test_loss'] = history.pop('val_loss')
        if 'val_acc' in history:
            history['test_acc'] = history.pop('val_acc')
        print(f"{name} depth parameters: {count_parameters(model)}")
        save_model(model, "models/cnn_models.pt")
        save_history(history, f"results/architecture_analysis/depth_{name}_history.pkl")
        plot_learning_curves(history, f"Depth {name}", save_path=f"plots/architecture_analysis/depth_{name}_learning_curve.png")
        plot_gradient_flow(model, save_path=f"plots/architecture_analysis/depth_{name}_gradient_flow.png")
        analyze_receptive_field(model, save_path=f"results/architecture_analysis/depth_{name}_receptive_field.txt")
        visualize_feature_maps(model, train_loader, device, save_path=f"plots/architecture_analysis/depth_{name}_feature_maps.png") 