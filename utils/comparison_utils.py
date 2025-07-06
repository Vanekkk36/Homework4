import time
import torch
import pandas as pd

def compare_accuracy(models, loaders, device):
    """
    Сравнивает точность моделей на заданных данных.
    models: список (имя, модель)
    loaders: dict с 'train' и 'test' DataLoader
    Возвращает DataFrame с результатами.
    """
    results = []
    for name, model in models:
        model.eval()
        accs = {}
        for split, loader in loaders.items():
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    pred = out.argmax(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            accs[split] = correct / total
        results.append({'model': name, 'train_acc': accs.get('train', None), 'test_acc': accs.get('test', None)})
    return pd.DataFrame(results)

def compare_time(models, loaders, device):
    """
    Сравнивает время инференса моделей на тестовых данных.
    Возвращает DataFrame с результатами.
    """
    results = []
    for name, model in models:
        model.eval()
        start = time.time()
        with torch.no_grad():
            for x, _ in loaders['test']:
                x = x.to(device)
                _ = model(x)
        elapsed = time.time() - start
        results.append({'model': name, 'inference_time': elapsed})
    return pd.DataFrame(results)

def compare_params(models):
    """
    Сравнивает количество обучаемых параметров моделей.
    Возвращает DataFrame с результатами.
    """
    results = []
    for name, model in models:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results.append({'model': name, 'params': n_params})
    return pd.DataFrame(results)

def print_comparison_table(df, title=None):
    if title:
        print(f'\n{title}')
    print(df.to_string(index=False)) 