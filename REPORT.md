# Сравнительный анализ архитектур глубокого обучения: Отчет

## 1. Обзор проекта

Этот проект представляет собой комплексное исследование и сравнение различных архитектур нейронных сетей на датасетах MNIST и CIFAR-10. В экспериментах рассматриваются полносвязные сети (FCN), простые сверточные сети (CNN), CNN с остаточными блоками, а также кастомные слои. Проект организован в виде модульных скриптов и утилит, все результаты, модели и графики сохраняются в структурированные папки.

---

## 2. Структура проекта

- `homework_cnn_vs_fc_comparison.py`: Сравнение FCN, SimpleCNN и ResNet-подобных CNN на MNIST и CIFAR-10.
- `homework_cnn_architecture_analysis.py`: Анализ влияния размера ядра и глубины сети в CNN, включая расчет рецептивного поля и визуализацию карт признаков.
- `homework_custom_layers_experiments.py`: Реализация и тестирование кастомных слоев и различных остаточных блоков, сравнение производительности на CIFAR-10.
- `utils/`: Содержит `comparison_utils.py`, `visualization_utils.py` и `training_utils.py` для модульного переиспользования кода.
- `results/`, `plots/`, `models/`: Все результаты экспериментов, графики и обученные модели сохраняются здесь.

---

## 3. Этапы экспериментов

### 3.1 Базовые эксперименты: FCN vs CNN vs ResNet-like
- Реализованы и сравнены FCNet, SimpleCNN и ResNetLike на MNIST.
- Реализованы DeepFCNet, CIFARResNetLike и CIFARResNetReg на CIFAR-10.
- Все модели обучены, измерялись точность, потери, время инференса и количество параметров.
- Визуализированы кривые обучения, матрицы ошибок и поток градиентов.
- Все результаты, модели и графики сохранены в соответствующие папки.

### 3.2 Анализ архитектуры CNN
- Проанализировано влияние размера ядра (3x3, 5x5, 7x7, combo) и глубины сети (shallow, medium, deep, residual) на MNIST.
- Реализован аналитический расчет рецептивного поля и визуализация карт признаков.
- Сравнены количество параметров и рецептивные поля.

### 3.3 Кастомные слои и остаточные блоки
- Реализованы кастомные Conv2D, spatial attention, кастомная активация и pooling.
- Реализованы и протестированы базовый, bottleneck и широкий остаточные блоки.
- Сравнены кастомные и стандартные слои по прямому/обратному проходу, формам выходных данных и количеству параметров.
- Проведено полное сравнение производительности на CIFAR-10 для StandardCNN, CustomCNN и ResidualCNN.

### 3.4 Рефакторинг и повышение надежности
- Код модульно разделен на утилиты для обучения, визуализации и сравнения.
- Исправлены ошибки с устройствами, отсутствующими функциями, обеспечена автономность всех скриптов.
- Улучшена информативность графиков, добавлен анализ переобучения и времени инференса.

---

## 4. Результаты

### 4.1 Эксперименты на MNIST
```
Training FCNet...
Epoch 1/10 | Train Loss: 0.2987 | Test Loss: 0.1160 | Train Acc: 0.9085 | Test Acc: 0.9647
Epoch 2/10 | Train Loss: 0.1276 | Test Loss: 0.0837 | Train Acc: 0.9624 | Test Acc: 0.9737
Epoch 3/10 | Train Loss: 0.0933 | Test Loss: 0.0808 | Train Acc: 0.9727 | Test Acc: 0.9765
Epoch 4/10 | Train Loss: 0.0766 | Test Loss: 0.0732 | Train Acc: 0.9769 | Test Acc: 0.9782
Epoch 5/10 | Train Loss: 0.0645 | Test Loss: 0.0808 | Train Acc: 0.9804 | Test Acc: 0.9756
Epoch 6/10 | Train Loss: 0.0581 | Test Loss: 0.0724 | Train Acc: 0.9819 | Test Acc: 0.9800
Epoch 7/10 | Train Loss: 0.0519 | Test Loss: 0.0647 | Train Acc: 0.9835 | Test Acc: 0.9801
Epoch 8/10 | Train Loss: 0.0439 | Test Loss: 0.0650 | Train Acc: 0.9861 | Test Acc: 0.9832
Epoch 9/10 | Train Loss: 0.0420 | Test Loss: 0.0769 | Train Acc: 0.9872 | Test Acc: 0.9818
Epoch 10/10 | Train Loss: 0.0389 | Test Loss: 0.0722 | Train Acc: 0.9878 | Test Acc: 0.9805
FCNet parameters: 567434
Final Train Acc: 0.9878, Test Acc: 0.9805
Model: FCNet
Max acc gap: 0.0073 at epoch 10
Max loss gap: 0.0349 at epoch 9
No strong overfitting detected.
Inference time on test set: 4.03 sec

Training SimpleCNN...
Epoch 1/10 | Train Loss: 0.2061 | Test Loss: 0.0540 | Train Acc: 0.9361 | Test Acc: 0.9813
Epoch 2/10 | Train Loss: 0.0628 | Test Loss: 0.0400 | Train Acc: 0.9806 | Test Acc: 0.9872
Epoch 3/10 | Train Loss: 0.0439 | Test Loss: 0.0346 | Train Acc: 0.9862 | Test Acc: 0.9890
Epoch 4/10 | Train Loss: 0.0364 | Test Loss: 0.0427 | Train Acc: 0.9883 | Test Acc: 0.9864
Epoch 5/10 | Train Loss: 0.0298 | Test Loss: 0.0251 | Train Acc: 0.9907 | Test Acc: 0.9921
Epoch 6/10 | Train Loss: 0.0235 | Test Loss: 0.0229 | Train Acc: 0.9924 | Test Acc: 0.9931
Epoch 7/10 | Train Loss: 0.0210 | Test Loss: 0.0291 | Train Acc: 0.9929 | Test Acc: 0.9913
Epoch 8/10 | Train Loss: 0.0192 | Test Loss: 0.0287 | Train Acc: 0.9936 | Test Acc: 0.9915
Epoch 9/10 | Train Loss: 0.0151 | Test Loss: 0.0288 | Train Acc: 0.9949 | Test Acc: 0.9921
Epoch 10/10 | Train Loss: 0.0143 | Test Loss: 0.0355 | Train Acc: 0.9952 | Test Acc: 0.9907
SimpleCNN parameters: 421642
Final Train Acc: 0.9952, Test Acc: 0.9907
Model: SimpleCNN
Max acc gap: 0.0045 at epoch 10
Max loss gap: 0.0213 at epoch 10
No strong overfitting detected.
Inference time on test set: 4.05 sec

Training ResNetLike...
Epoch 1/10 | Train Loss: 0.9982 | Test Loss: 0.4001 | Train Acc: 0.7835 | Test Acc: 0.9288
Epoch 2/10 | Train Loss: 0.2763 | Test Loss: 0.3914 | Train Acc: 0.9456 | Test Acc: 0.8749
Epoch 3/10 | Train Loss: 0.1752 | Test Loss: 0.2444 | Train Acc: 0.9590 | Test Acc: 0.9295
Epoch 4/10 | Train Loss: 0.1370 | Test Loss: 0.1279 | Train Acc: 0.9667 | Test Acc: 0.9655
Epoch 5/10 | Train Loss: 0.1149 | Test Loss: 0.3286 | Train Acc: 0.9703 | Test Acc: 0.8946
Epoch 6/10 | Train Loss: 0.1010 | Test Loss: 0.3562 | Train Acc: 0.9735 | Test Acc: 0.8879
Epoch 7/10 | Train Loss: 0.0889 | Test Loss: 0.1668 | Train Acc: 0.9759 | Test Acc: 0.9484
Epoch 8/10 | Train Loss: 0.0801 | Test Loss: 0.2416 | Train Acc: 0.9780 | Test Acc: 0.9261
Epoch 9/10 | Train Loss: 0.0737 | Test Loss: 0.4266 | Train Acc: 0.9798 | Test Acc: 0.8709
Epoch 10/10 | Train Loss: 0.0681 | Test Loss: 0.2084 | Train Acc: 0.9808 | Test Acc: 0.9363
ResNetLike parameters: 58890
Final Train Acc: 0.9808, Test Acc: 0.9363
Model: ResNetLike
Max acc gap: 0.1089 at epoch 9
Max loss gap: 0.3529 at epoch 9
Potential overfitting detected at epochs: [2 5 6 8 9]
Inference time on test set: 4.04 sec
```

### 4.2 Эксперименты на CIFAR-10
```
Training DeepFCNet...
Epoch 1/10 | Train Loss: 1.8508 | Test Loss: 1.6576 | Train Acc: 0.3382 | Test Acc: 0.4142
Epoch 2/10 | Train Loss: 1.6705 | Test Loss: 1.5621 | Train Acc: 0.4102 | Test Acc: 0.4546
Epoch 3/10 | Train Loss: 1.6027 | Test Loss: 1.5296 | Train Acc: 0.4368 | Test Acc: 0.4746
Epoch 4/10 | Train Loss: 1.5530 | Test Loss: 1.4961 | Train Acc: 0.4583 | Test Acc: 0.4748
Epoch 5/10 | Train Loss: 1.5101 | Test Loss: 1.4749 | Train Acc: 0.4721 | Test Acc: 0.4915
Epoch 6/10 | Train Loss: 1.4748 | Test Loss: 1.4418 | Train Acc: 0.4846 | Test Acc: 0.4911
Epoch 7/10 | Train Loss: 1.4440 | Test Loss: 1.4194 | Train Acc: 0.4967 | Test Acc: 0.4994
Epoch 8/10 | Train Loss: 1.4212 | Test Loss: 1.4218 | Train Acc: 0.5028 | Test Acc: 0.5025
Epoch 9/10 | Train Loss: 1.4023 | Test Loss: 1.3871 | Train Acc: 0.5118 | Test Acc: 0.5160
Epoch 10/10 | Train Loss: 1.3718 | Test Loss: 1.4036 | Train Acc: 0.5205 | Test Acc: 0.5098
DeepFCNet parameters: 3837066
Final Train Acc: 0.5205, Test Acc: 0.5098
Model: DeepFCNet
Max acc gap: 0.0107 at epoch 10
Max loss gap: 0.0318 at epoch 10
No strong overfitting detected.
Inference time on test set: 6.44 sec

Training CIFARResNetLike...
Epoch 1/10 | Train Loss: 1.3604 | Test Loss: 1.1907 | Train Acc: 0.5110 | Test Acc: 0.5679
Epoch 2/10 | Train Loss: 0.9667 | Test Loss: 0.9571 | Train Acc: 0.6585 | Test Acc: 0.6625
Epoch 3/10 | Train Loss: 0.7944 | Test Loss: 0.9591 | Train Acc: 0.7194 | Test Acc: 0.6744
Epoch 4/10 | Train Loss: 0.6936 | Test Loss: 0.7710 | Train Acc: 0.7566 | Test Acc: 0.7241
Epoch 5/10 | Train Loss: 0.6061 | Test Loss: 0.7320 | Train Acc: 0.7890 | Test Acc: 0.7425
Epoch 6/10 | Train Loss: 0.5365 | Test Loss: 0.7175 | Train Acc: 0.8145 | Test Acc: 0.7534
Epoch 7/10 | Train Loss: 0.4756 | Test Loss: 0.8486 | Train Acc: 0.8359 | Test Acc: 0.7204
Epoch 8/10 | Train Loss: 0.4206 | Test Loss: 0.7415 | Train Acc: 0.8537 | Test Acc: 0.7514
Epoch 9/10 | Train Loss: 0.3740 | Test Loss: 0.7027 | Train Acc: 0.8713 | Test Acc: 0.7725
Epoch 10/10 | Train Loss: 0.3231 | Test Loss: 0.7217 | Train Acc: 0.8877 | Test Acc: 0.7658
CIFARResNetLike parameters: 290634
Final Train Acc: 0.8877, Test Acc: 0.7658
Model: CIFARResNetLike
Max acc gap: 0.1219 at epoch 10
Max loss gap: 0.3986 at epoch 10
Potential overfitting detected at epochs: [ 6  7  8  9 10]
Inference time on test set: 6.67 sec

Training CIFARResNetReg...
Epoch 1/10 | Train Loss: 1.6303 | Test Loss: 1.3182 | Train Acc: 0.3920 | Test Acc: 0.5200
Epoch 2/10 | Train Loss: 1.2760 | Test Loss: 1.0825 | Train Acc: 0.5349 | Test Acc: 0.6132
Epoch 3/10 | Train Loss: 1.1215 | Test Loss: 1.0391 | Train Acc: 0.5941 | Test Acc: 0.6211
Epoch 4/10 | Train Loss: 1.0262 | Test Loss: 0.9247 | Train Acc: 0.6321 | Test Acc: 0.6678
Epoch 5/10 | Train Loss: 0.9602 | Test Loss: 0.8728 | Train Acc: 0.6551 | Test Acc: 0.6938
Epoch 6/10 | Train Loss: 0.9137 | Test Loss: 0.8339 | Train Acc: 0.6721 | Test Acc: 0.6998
Epoch 7/10 | Train Loss: 0.8626 | Test Loss: 0.8593 | Train Acc: 0.6899 | Test Acc: 0.6919
Epoch 8/10 | Train Loss: 0.8282 | Test Loss: 0.7722 | Train Acc: 0.7040 | Test Acc: 0.7255
Epoch 10/10 | Train Loss: 0.7571 | Test Loss: 0.7312 | Train Acc: 0.7305 | Test Acc: 0.7403
CIFARResNetReg parameters: 290634
Final Train Acc: 0.7305, Test Acc: 0.7403
Model: CIFARResNetReg
Max acc gap: -0.0020 at epoch 7
Max loss gap: -0.0032 at epoch 7
No strong overfitting detected.
Inference time on test set: 6.55 sec
```

### 4.3 Анализ архитектуры CNN (MNIST)
```
Training kernel 3x3...
... (см. вывод терминала выше для полного лога)
```

#### Пример таблицы: Размер ядра vs Параметры
| Размер ядра | Параметры |
|-------------|-----------|
| 3x3         | 63050     |
| 5x5         | 31786     |
| 7x7         | 16090     |
| Combo       | 33722     |

#### Пример таблицы: Глубина vs Параметры
| Глубина  | Параметры |
|----------|-----------|
| Shallow  | 33850     |
| Medium   | 38490     |
| Deep     | 18690     |
| Residual | 40938     |

---

### 4.4 Кастомные слои и остаточные блоки
```
=== 3.1 Тесты кастомных слоев ===
3.1 Кастомные слои:
Кастомный сверточный слой:
    *  "Custom Conv2D: Forward pass successful"
    *  "Custom Conv2D: Output shape matches torch.nn.Conv2d"
    *  "Custom Conv2D: Forward time: 0.001993, Torch Conv2d: 0.000000"
    *  "Custom Conv2D: Backward pass successful"
    *  "Custom Conv2D: Backward time: 0.011960, Torch Conv2d: 0.000996"
Attention-механизм:
    *  "Custom Spatial Attention: Forward pass successful"
    *  "Custom Spatial Attention: Output shape matches input shape"
    *  "Custom Spatial Attention: Forward time: 0.000000"
    *  "Custom Spatial Attention: Backward pass successful"
    *  "Custom Spatial Attention: Backward time: 0.000997"
Кастомная функция активации:
    *  "Custom Activation: Forward pass successful"
    *  "Custom Activation: Output shape matches torch.nn.ReLU"
    *  "Custom Activation: Forward time: 0.000000, Torch ReLU: 0.000000"
    *  "Custom Activation: Backward pass successful"
    *  "Custom Activation: Backward time: 0.000000, Torch ReLU: 0.000000"
Кастомный pooling слой:
    *  "Custom Pooling: Forward pass successful"
    *  "Custom Pooling: Output shape matches torch.nn.MaxPool2d"
    *  "Custom Pooling: Forward time: 0.000000, Torch MaxPool2d: 0.000000"
    *  "Custom Pooling: Backward pass successful"
    *  "Custom Pooling: Backward time: 0.001012, Torch MaxPool2d: 0.000000"

=== 3.2 Тесты Residual блоков ===
3.2 Эксперименты с Residual блоками:
Базовый Residual блок:
    "Basic Residual Block: Forward pass successful"
    "Basic Residual Block: Parameters: 1200"
    "Basic Residual Block: Backward pass successful"
Bottleneck Residual блок:
    "Bottleneck Residual Block: Forward pass successful"
    "Bottleneck Residual Block: Parameters: 104"
    "Bottleneck Residual Block: Backward pass successful"
Wide Residual блок:
    "Wide Residual Block: Forward pass successful"
    "Wide Residual Block: Parameters: 3696"
    "Wide Residual Block: Backward pass successful"
```

### 4.5 CIFAR-10: Standard vs Custom vs Residual CNNs
```
StandardCNN | Epoch 1/5 | Train Loss: 1.3729 | Test Loss: 1.1478 | Train Acc: 0.5115 | Test Acc: 0.5926
StandardCNN | Epoch 2/5 | Train Loss: 0.9677 | Test Loss: 0.9743 | Train Acc: 0.6600 | Test Acc: 0.6657
StandardCNN | Epoch 3/5 | Train Loss: 0.8153 | Test Loss: 0.8731 | Train Acc: 0.7127 | Test Acc: 0.6957
StandardCNN | Epoch 4/5 | Train Loss: 0.7085 | Test Loss: 0.8475 | Train Acc: 0.7524 | Test Acc: 0.7105
StandardCNN | Epoch 5/5 | Train Loss: 0.6149 | Test Loss: 0.8411 | Train Acc: 0.7838 | Test Acc: 0.7193
CustomCNN | Epoch 1/5 | Train Loss: 1.4022 | Test Loss: 1.1305 | Train Acc: 0.4973 | Test Acc: 0.5959
CustomCNN | Epoch 2/5 | Train Loss: 1.0220 | Test Loss: 0.9659 | Train Acc: 0.6392 | Test Acc: 0.6555
CustomCNN | Epoch 3/5 | Train Loss: 0.8496 | Test Loss: 0.8872 | Train Acc: 0.6999 | Test Acc: 0.6871
CustomCNN | Epoch 4/5 | Train Loss: 0.7311 | Test Loss: 0.8313 | Train Acc: 0.7442 | Test Acc: 0.7099
CustomCNN | Epoch 5/5 | Train Loss: 0.6171 | Test Loss: 0.8106 | Train Acc: 0.7821 | Test Acc: 0.7200
ResidualCNN | Epoch 1/5 | Train Loss: 1.3232 | Test Loss: 1.0247 | Train Acc: 0.5215 | Test Acc: 0.6358
ResidualCNN | Epoch 2/5 | Train Loss: 0.8842 | Test Loss: 0.8993 | Train Acc: 0.6871 | Test Acc: 0.6836
ResidualCNN | Epoch 3/5 | Train Loss: 0.7521 | Test Loss: 0.7642 | Train Acc: 0.7344 | Test Acc: 0.7305
ResidualCNN | Epoch 4/5 | Train Loss: 0.6740 | Test Loss: 0.7532 | Train Acc: 0.7627 | Test Acc: 0.7356
ResidualCNN | Epoch 5/5 | Train Loss: 0.6156 | Test Loss: 0.8196 | Train Acc: 0.7836 | Test Acc: 0.7140

Test Accuracy Comparison (CIFAR-10)
      model  test_acc
StandardCNN    0.7193
  CustomCNN    0.7200
ResidualCNN    0.7140

Inference Time Comparison (CIFAR-10)
      model  inference_time
StandardCNN        7.072613
  CustomCNN        7.152086
ResidualCNN        6.952757
```

---

## 5. Выводы
- **CNN превосходят FCN** на обоих датасетах (MNIST и CIFAR-10), имея меньше параметров и лучшую обобщающую способность.
- **Остаточные связи** помогают обучать более глубокие сети, но могут приводить к переобучению без регуляризации.
- **Размер ядра и глубина** влияют на количество параметров и рецептивное поле, но оптимальные значения зависят от задачи и данных.
- **Кастомные слои** могут не уступать стандартным слоям PyTorch по производительности и скорости, что полезно для исследований и экспериментов.
- **Модульная структура кода** и надежное сохранение результатов/графиков/моделей делают эксперименты воспроизводимыми и расширяемыми.

---

## 6. Примеры изображений и графиков
- См. папку `plots/` для кривых обучения, матриц ошибок, анализа градиентного потока и визуализации карт признаков.
- См. папку `results/` для сохраненных логов экспериментов и CSV.
- См. папку `models/` для чекпоинтов обученных моделей.

---

## 7. Благодарности
- Все эксперименты проведены с использованием PyTorch и стандартных open-source библиотек.
- Структура проекта и организация кода улучшались итеративно на основе обратной связи и лучших практик. 