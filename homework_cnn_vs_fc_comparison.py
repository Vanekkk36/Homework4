import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
import os
import pickle
from utils.training_utils import train_model as train_model_utils, accuracy
from utils.visualization_utils import plot_learning_curves, plot_confusion_matrix, plot_gradient_flow
from utils.comparison_utils import compare_accuracy, compare_time, compare_params, print_comparison_table

# =====================
# 1.1 Сравнение на MNIST
# =====================

# 1.1.1 Полносвязная сеть (3-4 слоя)
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# 1.1.2 Простая CNN (2-3 conv слоя)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 1.1.3 CNN с Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.resblock = ResidualBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.resblock(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =====================
# 1.2 Сравнение на CIFAR-10
# =====================

# 1.2.1 Полносвязная сеть (глубокая)
class DeepFCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# 1.2.2 CNN с Residual блоками
class CIFARResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.resblock1 = ResidualBlock(32, 64, stride=2)
        self.resblock2 = ResidualBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 1.2.3 CNN с регуляризацией и Residual блоками
class CIFARResNetReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.resblock1 = ResidualBlock(32, 64, stride=2)
        self.resblock2 = ResidualBlock(64, 128, stride=2)
        self.dropout = nn.Dropout2d(0.3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.resblock1(x)
        x = self.dropout(x)
        x = self.resblock2(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =====================
# Вспомогательные функции
# =====================

def get_data_loaders(dataset_name, batch_size=128):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def save_model(model, path):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)

def save_history(history, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(history, f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_overfitting(history, name, save_dir):
    train_acc = np.array(history['train_acc'])
    test_acc = np.array(history['test_acc'])
    train_loss = np.array(history['train_loss'])
    test_loss = np.array(history['test_loss'])
    gap_acc = train_acc - test_acc
    gap_loss = test_loss - train_loss
    overfit_epochs = np.where(gap_acc > 0.05)[0]
    msg = f"Model: {name}\nMax acc gap: {gap_acc.max():.4f} at epoch {gap_acc.argmax()+1}\nMax loss gap: {gap_loss.max():.4f} at epoch {gap_loss.argmax()+1}\n"
    if len(overfit_epochs) > 0:
        msg += f"Potential overfitting detected at epochs: {overfit_epochs+1}\n"
    else:
        msg += "No strong overfitting detected.\n"
    print(msg)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{name}_overfitting_analysis.txt"), 'w') as f:
        f.write(msg)

def measure_inference_time(model, loader, device):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
    elapsed = time.time() - start
    return elapsed

# =====================
# Основной цикл сравнения
# =====================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 10
    BATCH_SIZE = 128
    LR = 0.001
    print("\n===== MNIST Experiments =====")
    train_loader, test_loader = get_data_loaders('MNIST', BATCH_SIZE)
    mnist_models = [
        ("FCNet", FCNet()),
        ("SimpleCNN", SimpleCNN()),
        ("ResNetLike", ResNetLike())
    ]
    for name, model in mnist_models:
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
        gradient_flow_steps = [0, 100, 200, 300, 400, 500]
        batch_count = 0
        for epoch in range(EPOCHS):
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
                if batch_count in gradient_flow_steps:
                    plot_gradient_flow(model, save_path=f"plots/mnist_comparison/{name}_gradient_flow_epoch{epoch+1}_batch{batch_count}.png", title=f"{name} Gradient Flow Epoch {epoch+1} Batch {batch_count}")
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
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"{name} parameters: {count_parameters(model)}")
        if name == "FCNet":
            save_model(model, "models/fc_models.pt")
        else:
            save_model(model, "models/cnn_models.pt")
        save_history(history, f"results/mnist_comparison/{name}_history.pkl")
        plot_learning_curves(history, f"MNIST {name}", save_path=f"plots/mnist_comparison/{name}_learning_curve.png", model_params=count_parameters(model))
        acc_train = history['train_acc'][-1]
        acc_test = history['test_acc'][-1]
        print(f"Final Train Acc: {acc_train:.4f}, Test Acc: {acc_test:.4f}")
        analyze_overfitting(history, name, "results/mnist_comparison/")
        inf_time = measure_inference_time(model, test_loader, device)
        print(f"Inference time on test set: {inf_time:.2f} sec")
        with open(f"results/mnist_comparison/{name}_inference_time.txt", 'w') as f:
            f.write(f"Inference time (test set): {inf_time:.2f} sec\n")
    print("\n===== CIFAR-10 Experiments =====")
    train_loader, test_loader = get_data_loaders('CIFAR10', BATCH_SIZE)
    cifar_models = [
        ("DeepFCNet", DeepFCNet()),
        ("CIFARResNetLike", CIFARResNetLike()),
        ("CIFARResNetReg", CIFARResNetReg())
    ]
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    for name, model in cifar_models:
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
        gradient_flow_steps = [0, 100, 200, 300, 400, 500]
        batch_count = 0
        for epoch in range(EPOCHS):
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
                if batch_count in gradient_flow_steps:
                    plot_gradient_flow(model, save_path=f"plots/cifar_comparison/{name}_gradient_flow_epoch{epoch+1}_batch{batch_count}.png", title=f"{name} Gradient Flow Epoch {epoch+1} Batch {batch_count}")
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
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"{name} parameters: {count_parameters(model)}")
        if name == "DeepFCNet":
            save_model(model, "models/fc_models.pt")
        else:
            save_model(model, "models/cnn_models.pt")
        save_history(history, f"results/cifar_comparison/{name}_history.pkl")
        plot_learning_curves(history, f"CIFAR-10 {name}", save_path=f"plots/cifar_comparison/{name}_learning_curve.png", model_params=count_parameters(model))
        acc_train = history['train_acc'][-1]
        acc_test = history['test_acc'][-1]
        print(f"Final Train Acc: {acc_train:.4f}, Test Acc: {acc_test:.4f}")
        analyze_overfitting(history, name, "results/cifar_comparison/")
        inf_time = measure_inference_time(model, test_loader, device)
        print(f"Inference time on test set: {inf_time:.2f} sec")
        with open(f"results/cifar_comparison/{name}_inference_time.txt", 'w') as f:
            f.write(f"Inference time (test set): {inf_time:.2f} sec\n")
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(targets.numpy())
        plot_confusion_matrix(y_true, y_pred, class_names, f"CIFAR-10 {name} Confusion Matrix", save_path=f"plots/cifar_comparison/{name}_confusion_matrix.png")
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            break
        plot_gradient_flow(model, save_path=f"plots/cifar_comparison/{name}_gradient_flow.png") 