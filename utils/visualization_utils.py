import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_learning_curves(history, title=None, save_path=None, model_params=None):
    fig = plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss', color='tab:blue')
    plt.plot(history['test_loss'], label='Test Loss', color='tab:orange')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    best_epoch = np.argmin(history['test_loss'])
    plt.scatter(best_epoch, history['test_loss'][best_epoch], color='red', zorder=5)
    plt.annotate(f"Best: {history['test_loss'][best_epoch]:.4f}", (best_epoch, history['test_loss'][best_epoch]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
    plt.title(f'{title} Loss' if title else 'Loss')
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc', color='tab:green')
    plt.plot(history['test_acc'], label='Test Acc', color='tab:purple')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    best_epoch = np.argmax(history['test_acc'])
    plt.scatter(best_epoch, history['test_acc'][best_epoch], color='red', zorder=5)
    plt.annotate(f"Best: {history['test_acc'][best_epoch]:.4f}", (best_epoch, history['test_acc'][best_epoch]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
    acc_title = f'{title} Accuracy' if title else 'Accuracy'
    if model_params:
        acc_title += f"\nParams: {model_params}"
    plt.title(acc_title)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title=None, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.grid(False)
    plt.title(title or 'Confusion Matrix')
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def plot_feature_maps(feature_maps, save_path=None, title=None):
    C = feature_maps.shape[0]
    fig, axes = plt.subplots(1, min(8, C), figsize=(2*min(8, C),2))
    for i in range(min(8, C)):
        axes[i].imshow(feature_maps[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Ch {i}')
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def plot_gradient_flow(model, save_path=None, title=None):
    ave_grads = []
    layers = []
    for n, p in model.named_parameters():
        if p.requires_grad and "bias" not in n:
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
    fig = plt.figure(figsize=(10,4))
    plt.plot(ave_grads, alpha=0.7, color="b", marker='o')
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads)), layers, rotation="vertical", fontsize=8)
    plt.xlim(xmin=0, xmax=len(ave_grads)-1)
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title(title or "Gradient flow")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show() 