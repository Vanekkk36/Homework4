import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.comparison_utils import compare_accuracy, compare_time, compare_params, print_comparison_table
from utils.visualization_utils import plot_learning_curves, plot_gradient_flow

# =====================
# 3.1 Реализация кастомных слоев
# =====================

# Кастомный сверточный слой с дополнительной логикой
class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        out = self.conv(x)
        out = out * self.scale
        return out

# Attention механизм для CNN
class SimpleSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn

# Кастомная функция активации
class CustomActivation(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * torch.tanh(input)
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (torch.tanh(input) + input * (1 - torch.tanh(input) ** 2))
        return grad_input

def custom_activation(x):
    return CustomActivation.apply(x)

# Кастомный pooling слой
class CustomMaxAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
    def forward(self, x):
        max_pool = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        avg_pool = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return 0.5 * (max_pool + avg_pool)

# =====================
# Тесты и сравнения кастомных слоев
# =====================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_text(text, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(text)

def save_plot(fig, path):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path)
    plt.close(fig)

def save_model(model, path):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)

def test_custom_layers():
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    results = []
    # Custom Conv2d
    print('3.1 Кастомные слои:')
    print('Кастомный сверточный слой:')
    print('Цель: Реализовать сверточный слой с масштабированием выхода.')
    results.append('3.1 Кастомные слои:')
    results.append('Кастомный сверточный слой:')
    results.append(' Цель: Реализовать сверточный слой с масштабированием выхода.')
    custom_conv = CustomConv2d(3, 4, 3, padding=1)
    std_conv = nn.Conv2d(3, 4, 3, padding=1)
    # Forward
    t0 = time.time(); out1 = custom_conv(x); t1 = time.time()
    t2 = time.time(); out2 = std_conv(x); t3 = time.time()
    msg = f'    *  "Custom Conv2D: Forward pass successful"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Conv2D: Output shape matches torch.nn.Conv2d"' if out1.shape == out2.shape else f'    *  "Custom Conv2D: Output shape mismatch"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Conv2D: Forward time: {t1-t0:.6f}, Torch Conv2d: {t3-t2:.6f}"'
    print(msg); results.append(msg)
    # Backward
    grad = torch.randn_like(out1)
    t0 = time.time(); out1.backward(grad, retain_graph=True); t1 = time.time()
    t2 = time.time(); out2.backward(grad, retain_graph=True); t3 = time.time()
    msg = f'    *  "Custom Conv2D: Backward pass successful"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Conv2D: Backward time: {t1-t0:.6f}, Torch Conv2d: {t3-t2:.6f}"'
    print(msg); results.append(msg)
    save_model(custom_conv, 'models/custom_layers.pt')
    save_model(std_conv, 'models/custom_layers.pt')
    # Attention
    print('Attention-механизм:')
    print('  Цель: Реализовать spatial attention для CNN.')
    results.append('Attention-механизм:')
    results.append('  Цель: Реализовать spatial attention для CNN.')
    attn = SimpleSpatialAttention(3)
    t0 = time.time(); out_attn = attn(x); t1 = time.time()
    msg = f'    *  "Custom Spatial Attention: Forward pass successful"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Spatial Attention: Output shape matches input shape"' if out_attn.shape == x.shape else f'    *  "Custom Spatial Attention: Output shape mismatch"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Spatial Attention: Forward time: {t1-t0:.6f}"'
    print(msg); results.append(msg)
    grad = torch.randn_like(out_attn)
    t0 = time.time(); out_attn.backward(grad, retain_graph=True); t1 = time.time()
    msg = f'    *  "Custom Spatial Attention: Backward pass successful"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Spatial Attention: Backward time: {t1-t0:.6f}"'
    print(msg); results.append(msg)
    save_model(attn, 'models/custom_layers.pt')
    # Custom Activation
    print('Кастомная функция активации:')
    print('  Цель: Реализовать активацию через torch.autograd.Function.')
    results.append('Кастомная функция активации:')
    results.append('  Цель: Реализовать активацию через torch.autograd.Function.')
    y = torch.linspace(-2, 2, steps=10, requires_grad=True)
    t0 = time.time(); out_custom = custom_activation(y); t1 = time.time()
    t2 = time.time(); out_relu = F.relu(y); t3 = time.time()
    msg = f'    *  "Custom Activation: Forward pass successful"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Activation: Output shape matches torch.nn.ReLU"' if out_custom.shape == out_relu.shape else f'    *  "Custom Activation: Output shape mismatch"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Activation: Forward time: {t1-t0:.6f}, Torch ReLU: {t3-t2:.6f}"'
    print(msg); results.append(msg)
    t0 = time.time(); out_custom.sum().backward(retain_graph=True); t1 = time.time()
    t2 = time.time(); out_relu.sum().backward(retain_graph=True); t3 = time.time()
    msg = f'    *  "Custom Activation: Backward pass successful"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Activation: Backward time: {t1-t0:.6f}, Torch ReLU: {t3-t2:.6f}"'
    print(msg); results.append(msg)
    fig = plt.figure()
    plt.plot(y.detach().numpy(), out_custom.detach().numpy(), label='Custom')
    plt.plot(y.detach().numpy(), out_relu.detach().numpy(), label='ReLU')
    plt.legend(); plt.title('Custom vs ReLU activation')
    save_plot(fig, 'plots/custom_layers_experiments/custom_vs_relu_activation.png')
    # Custom Pooling
    print('Кастомный pooling слой:')
    print('  Цель: Реализовать среднее между max и avg pooling.')
    results.append('Кастомный pooling слой:')
    results.append('  Цель: Реализовать среднее между max и avg pooling.')
    pool = CustomMaxAvgPool2d(2)
    std_pool = nn.MaxPool2d(2)
    t0 = time.time(); out_pool = pool(x); t1 = time.time()
    t2 = time.time(); out_std_pool = std_pool(x); t3 = time.time()
    msg = f'    *  "Custom Pooling: Forward pass successful"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Pooling: Output shape matches torch.nn.MaxPool2d"' if out_pool.shape == out_std_pool.shape else f'    *  "Custom Pooling: Output shape mismatch"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Pooling: Forward time: {t1-t0:.6f}, Torch MaxPool2d: {t3-t2:.6f}"'
    print(msg); results.append(msg)
    grad = torch.randn_like(out_pool)
    t0 = time.time(); out_pool.backward(grad, retain_graph=True); t1 = time.time()
    t2 = time.time(); out_std_pool.backward(grad, retain_graph=True); t3 = time.time()
    msg = f'    *  "Custom Pooling: Backward pass successful"'
    print(msg); results.append(msg)
    msg = f'    *  "Custom Pooling: Backward time: {t1-t0:.6f}, Torch MaxPool2d: {t3-t2:.6f}"'
    print(msg); results.append(msg)
    save_model(pool, 'models/custom_layers.pt')
    save_model(std_pool, 'models/custom_layers.pt')
    save_text('\n'.join(results), 'results/custom_layers_experiments/custom_layers_test.txt')

def compare_custom_vs_standard():
    x = torch.randn(16, 3, 32, 32)
    custom_conv = CustomConv2d(3, 8, 3, padding=1)
    std_conv = nn.Conv2d(3, 8, 3, padding=1)
    t0 = time.time(); _ = custom_conv(x); t1 = time.time();
    t2 = time.time(); _ = std_conv(x); t3 = time.time();
    text = f'CustomConv2d time: {t1-t0:.6f}s, StdConv2d time: {t3-t2:.6f}s'
    save_text(text, 'results/custom_layers_experiments/custom_vs_std_conv_time.txt')

# =====================
# 3.2 Эксперименты с Residual блоками
# =====================

# Базовый Residual блок
class BasicResidualBlock(nn.Module):
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

# Bottleneck Residual блок
class BottleneckResidualBlock(nn.Module):
    def __init__(self, channels, bottleneck=4):
        super().__init__()
        mid = channels // bottleneck
        self.conv1 = nn.Conv2d(channels, mid, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, channels, 1)
        self.bn3 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        out = F.relu(out)
        return out

# Wide Residual блок
class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, widen_factor=2):
        super().__init__()
        out_channels = in_channels * widen_factor
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# =====================
# Эксперименты с Residual блоками
# =====================

def test_residual_blocks():
    x = torch.randn(2, 8, 16, 16)
    results = []
    print('3.2 Эксперименты с Residual блоками:')
    # Basic
    print('Базовый Residual блок:')
    print('  Цель: Реализовать базовый residual блок.')
    results.append('3.2 Эксперименты с Residual блоками:')
    results.append('Базовый Residual блок:')
    results.append('  Цель: Реализовать базовый residual блок.')
    basic = BasicResidualBlock(8)
    t0 = time.time(); out_basic = basic(x); t1 = time.time()
    msg = f'    "Basic Residual Block: Forward pass successful"'
    print(msg); results.append(msg)
    msg = f'    "Basic Residual Block: Parameters: {sum(p.numel() for p in basic.parameters())}"'
    print(msg); results.append(msg)
    t2 = time.time(); out_basic.mean().backward(retain_graph=True); t3 = time.time()
    msg = f'    "Basic Residual Block: Backward pass successful"'
    print(msg); results.append(msg)
    # Bottleneck
    print('Bottleneck Residual блок:')
    print('  Цель: Реализовать bottleneck residual блок.')
    results.append('Bottleneck Residual блок:')
    results.append('  Цель: Реализовать bottleneck residual блок.')
    bottleneck = BottleneckResidualBlock(8)
    t0 = time.time(); out_bottleneck = bottleneck(x); t1 = time.time()
    msg = f'    "Bottleneck Residual Block: Forward pass successful"'
    print(msg); results.append(msg)
    msg = f'    "Bottleneck Residual Block: Parameters: {sum(p.numel() for p in bottleneck.parameters())}"'
    print(msg); results.append(msg)
    t2 = time.time(); out_bottleneck.mean().backward(retain_graph=True); t3 = time.time()
    msg = f'    "Bottleneck Residual Block: Backward pass successful"'
    print(msg); results.append(msg)
    # Wide
    print('Wide Residual блок:')
    print('  Цель: Реализовать wide residual блок.')
    results.append('Wide Residual блок:')
    results.append('  Цель: Реализовать wide residual блок.')
    wide = WideResidualBlock(8, widen_factor=2)
    t0 = time.time(); out_wide = wide(x); t1 = time.time()
    msg = f'    "Wide Residual Block: Forward pass successful"'
    print(msg); results.append(msg)
    msg = f'    "Wide Residual Block: Parameters: {sum(p.numel() for p in wide.parameters())}"'
    print(msg); results.append(msg)
    t2 = time.time(); out_wide.mean().backward(); t3 = time.time()
    msg = f'    "Wide Residual Block: Backward pass successful"'
    print(msg); results.append(msg)
    save_model(basic, 'models/custom_layers.pt')
    save_model(bottleneck, 'models/custom_layers.pt')
    save_model(wide, 'models/custom_layers.pt')
    save_text('\n'.join(results), 'results/custom_layers_experiments/residual_blocks_test.txt')

# =====================
# Основной цикл
# =====================

def get_cifar10_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CustomConv2d(3, 32, 3, padding=1)
        self.attn = SimpleSpatialAttention(32)
        self.conv2 = CustomConv2d(32, 64, 3, padding=1)
        self.pool = CustomMaxAvgPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.attn(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.res1 = BasicResidualBlock(32)
        self.res2 = BasicResidualBlock(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.res1(x)
        x = self.pool(x)
        x = self.res2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Сравнение производительности и стабильности
def train_and_analyze(model, train_loader, test_loader, device, name, results_dir, plots_dir, epochs=5):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    grad_steps = [0, 100, 200, 300, 400, 500]
    batch_count = 0
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            if batch_count in grad_steps:
                plot_gradient_flow(model, save_path=f"{plots_dir}/{name}_gradient_flow_epoch{epoch+1}_batch{batch_count}.png", title=f"{name} Gradient Flow Epoch {epoch+1} Batch {batch_count}")
            batch_count += 1
        train_loss = running_loss / total
        train_acc = correct / total
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(targets).sum().item()
                test_total += targets.size(0)
        test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        print(f"{name} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    # Сохранение истории и графиков
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    with open(f"{results_dir}/{name}_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    plot_learning_curves(history, f"{name} CIFAR-10", save_path=f"{plots_dir}/{name}_learning_curve.png", model_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, history

if __name__ == "__main__":
    print("\n=== 3.1 Тесты кастомных слоев ===")
    test_custom_layers()
    compare_custom_vs_standard()
    print("\n=== 3.2 Тесты Residual блоков ===")
    test_residual_blocks()
    print("\n=== 4. Сравнение производительности на CIFAR-10 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    models_to_compare = [
        ("StandardCNN", StandardCNN()),
        ("CustomCNN", CustomCNN()),
        ("ResidualCNN", ResidualCNN()),
    ]
    histories = {}
    for name, model in models_to_compare:
        model, history = train_and_analyze(model, train_loader, test_loader, device, name,
                                           results_dir="results/custom_layers_experiments",
                                           plots_dir="plots/custom_layers_experiments",
                                           epochs=5)
        histories[name] = history
    # Сравнение точности и времени инференса
    accs = []
    times = []
    for name, model in models_to_compare:
        model = model.to(device)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        accs.append({'model': name, 'test_acc': correct / total})
        import time
        start = time.time()
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                _ = model(x)
        elapsed = time.time() - start
        times.append({'model': name, 'inference_time': elapsed})
    import pandas as pd
    acc_df = pd.DataFrame(accs)
    time_df = pd.DataFrame(times)
    print_comparison_table(acc_df, title="Test Accuracy Comparison (CIFAR-10)")
    print_comparison_table(time_df, title="Inference Time Comparison (CIFAR-10)")
    acc_df.to_csv("results/custom_layers_experiments/cifar10_accuracy_comparison.csv", index=False)
    time_df.to_csv("results/custom_layers_experiments/cifar10_inference_time_comparison.csv", index=False) 