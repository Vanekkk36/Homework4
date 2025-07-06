###Homework4


## 1. Структура проекта

- `homework_cnn_vs_fc_comparison.py`: сравнение FCN, SimpleCNN и ResNet-подобных CNN на MNIST и CIFAR-10.
- `homework_cnn_architecture_analysis.py`: анализ влияния размера ядра и глубины сети в CNN, включая расчет рецептивного поля и визуализацию карт признаков.
- `homework_custom_layers_experiments.py`: реализация и тестирование кастомных слоев и различных остаточных блоков, сравнение производительности на CIFAR-10.
- `utils/`: содержит `comparison_utils.py`, `visualization_utils.py` и `training_utils.py` для модульного переиспользования кода.
- `results/`, `plots/`, `models/`: все результаты экспериментов, графики и обученные модели сохраняются здесь.

---

## 2. Этапы экспериментов

### 2.1 Базовые эксперименты: FCN vs CNN vs ResNet-like
- Реализованы и сравнены FCNet, SimpleCNN и ResNetLike на MNIST.
- Реализованы DeepFCNet, CIFARResNetLike и CIFARResNetReg на CIFAR-10.
- Все модели обучены, измерялись точность, потери, время инференса и количество параметров.
- Визуализированы кривые обучения, матрицы ошибок и поток градиентов.

### 2.2 Анализ архитектуры CNN
- Проанализировано влияние размера ядра (3x3, 5x5, 7x7, combo) и глубины сети (shallow, medium, deep, residual) на MNIST.
- Реализован аналитический расчет рецептивного поля и визуализация карт признаков.
- Сравнены количество параметров и рецептивные поля.

### 2.3 Кастомные слои и остаточные блоки
- Реализованы кастомные Conv2D, spatial attention, кастомная активация и pooling.
- Реализованы и протестированы базовый, bottleneck и широкий остаточные блоки.
- Сравнены кастомные и стандартные слои по прямому/обратному проходу, формам выходных данных и количеству параметров.
- Проведено полное сравнение производительности на CIFAR-10 для StandardCNN, CustomCNN и ResidualCNN.


---

## 3. Результаты

### 3.1 Эксперименты на MNIST
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
### Анализ и выводы
Точность:
  SimpleCNN показала наилучшую точность на тестовом наборе (0.9907) при сопоставимом времени инференса.
  FCNet достигла высокой точности (0.9805), но немного уступила SimpleCNN.
  ResNetLike значительно хуже всех по точности на тестовом наборе (0.9363).

Переобучение:
  ResNetLike демонстрирует признаки сильного переобучения, так как разница между точностью на обучающем и тестовом наборах наиболее высока (особенно заметно в динамике обучения).
  FCNet показывает слабое переобучение.
  SimpleCNN не выявила признаков переобучения.

Количество параметров:
  ResNetLike имеет значительно меньше параметров, чем FCNet и SimpleCNN, что обычно является преимуществом с точки зрения вычислительных ресурсов, но в данном случае, вероятно, является причиной недостаточной способности к обучению (низкая точность на тесте и переобучение).
  SimpleCNN имеет меньше параметров, чем FCNet, и показывает лучшую точность.

Время инференса:
  Время инференса практически одинаковое для всех трех моделей.

Вывод:

Для задачи классификации MNIST, SimpleCNN является оптимальным выбором. Она обеспечивает высокую точность, не переобучается и имеет разумное количество параметров. FCNet также показывает хорошие результаты, но требует больше параметров. ResNetLike, несмотря на наименьшее количество параметров, показывает неприемлемо низкую точность на тестовом наборе и явно переобучается, что говорит о неоптимальной архитектуре или необходимости применения регуляризации. Необходимо отметить, что время инференса всех моделей сопоставимо. Таким образом, переход к более сложной архитектуре ResNetLike без должной настройки не привел к улучшению результатов.


### 3.2 Эксперименты на CIFAR-10
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
### Анализ и выводы
```

DeepFCNet:

•  Достигает точности на тестовом наборе около 51%.
•  Имеет большое количество параметров (3.8M), что может указывать на избыточность.
•  Признаков переобучения не выявлено.
•  Время инференса: 6.44 сек.

CIFARResNetLike:

•  Значительно превосходит полносвязную сеть по точности на тестовом наборе (76.58%).
•  Имеет существенно меньше параметров (290k).
•  Признаки переобучения выявлены начиная с 6 эпохи, что видно по увеличивающемуся разрыву между train и test accuracy.
•  Время инференса: 6.67 сек.

CIFARResNetReg:

•  Близкая точность к ResNet без регуляризации (74.03%).
•  Тоже имеет небольшое количество параметров (290k).
•  Регуляризация помогла избежать переобучения, максимальный разрыв между train и test accuracy минимален.
•  Время инференса: 6.55 сек.

Выводы:

1. CNN (ResNet) значительно превосходит полносвязную сеть по точности на CIFAR-10, что связано с лучшей обработкой пространственных особенностей изображений.
2. ResNet склонен к переобучению на CIFAR-10 при отсутствии регуляризации.
3. Добавление регуляризации в ResNet помогает предотвратить переобучение, сохраняя при этом высокую точность.
4. При примерно одинаковом времени инференса ResNet показывает значительно лучшие результаты по точности, чем полносвязная сеть.
5. Количество параметров напрямую не влияет на качество модели (полносвязная сеть имеет в 10 раз больше параметров, но показывает худший результат).
```

### 3.3 Анализ архитектуры CNN (MNIST)
```
Training kernel 3x3...
Epoch 1/8 | Train Loss: 0.2155 | Val Loss: 0.0952 | Train Acc: 0.9369 | Val Acc: 0.9724
Epoch 2/8 | Train Loss: 0.0752 | Val Loss: 0.0639 | Train Acc: 0.9781 | Val Acc: 0.9800
Epoch 3/8 | Train Loss: 0.0562 | Val Loss: 0.0633 | Train Acc: 0.9832 | Val Acc: 0.9790
Epoch 4/8 | Train Loss: 0.0474 | Val Loss: 0.0565 | Train Acc: 0.9856 | Val Acc: 0.9821
Epoch 5/8 | Train Loss: 0.0393 | Val Loss: 0.0581 | Train Acc: 0.9879 | Val Acc: 0.9818
Epoch 6/8 | Train Loss: 0.0345 | Val Loss: 0.0552 | Train Acc: 0.9892 | Val Acc: 0.9818
Epoch 7/8 | Train Loss: 0.0297 | Val Loss: 0.0592 | Train Acc: 0.9909 | Val Acc: 0.9812
Epoch 8/8 | Train Loss: 0.0265 | Val Loss: 0.0574 | Train Acc: 0.9917 | Val Acc: 0.9824
3x3 kernel parameters: 63050

Receptive field analysis:
Layer 1: conv | kernel=3, stride=1, padding=1 | RF=3
Layer 2: pool | kernel=2, stride=2, padding=0 | RF=4

Training kernel 5x5...
Epoch 1/8 | Train Loss: 0.2276 | Val Loss: 0.0760 | Train Acc: 0.9339 | Val Acc: 0.9775
Epoch 2/8 | Train Loss: 0.0726 | Val Loss: 0.0553 | Train Acc: 0.9785 | Val Acc: 0.9824
Epoch 3/8 | Train Loss: 0.0555 | Val Loss: 0.0543 | Train Acc: 0.9829 | Val Acc: 0.9822
Epoch 4/8 | Train Loss: 0.0462 | Val Loss: 0.0503 | Train Acc: 0.9849 | Val Acc: 0.9834
Epoch 5/8 | Train Loss: 0.0403 | Val Loss: 0.0468 | Train Acc: 0.9876 | Val Acc: 0.9851
Epoch 6/8 | Train Loss: 0.0350 | Val Loss: 0.0473 | Train Acc: 0.9891 | Val Acc: 0.9845
Epoch 7/8 | Train Loss: 0.0307 | Val Loss: 0.0513 | Train Acc: 0.9904 | Val Acc: 0.9842
Epoch 8/8 | Train Loss: 0.0281 | Val Loss: 0.0544 | Train Acc: 0.9912 | Val Acc: 0.9833
5x5 kernel parameters: 31786

Receptive field analysis:
Layer 1: conv | kernel=5, stride=1, padding=2 | RF=5
Layer 2: pool | kernel=2, stride=2, padding=0 | RF=6

Training kernel 7x7...
Epoch 1/8 | Train Loss: 0.2669 | Val Loss: 0.1100 | Train Acc: 0.9250 | Val Acc: 0.9664
Epoch 2/8 | Train Loss: 0.0910 | Val Loss: 0.0647 | Train Acc: 0.9732 | Val Acc: 0.9787
Epoch 3/8 | Train Loss: 0.0693 | Val Loss: 0.0638 | Train Acc: 0.9788 | Val Acc: 0.9795
Epoch 4/8 | Train Loss: 0.0586 | Val Loss: 0.0539 | Train Acc: 0.9821 | Val Acc: 0.9821
Epoch 5/8 | Train Loss: 0.0507 | Val Loss: 0.0503 | Train Acc: 0.9842 | Val Acc: 0.9836
Epoch 6/8 | Train Loss: 0.0446 | Val Loss: 0.0452 | Train Acc: 0.9863 | Val Acc: 0.9841
Epoch 7/8 | Train Loss: 0.0403 | Val Loss: 0.0544 | Train Acc: 0.9877 | Val Acc: 0.9804
Epoch 8/8 | Train Loss: 0.0378 | Val Loss: 0.0567 | Train Acc: 0.9880 | Val Acc: 0.9822
7x7 kernel parameters: 16090

Receptive field analysis:
Layer 1: conv | kernel=7, stride=1, padding=3 | RF=7
Layer 2: pool | kernel=2, stride=2, padding=0 | RF=8

Training kernel combo...
Epoch 1/8 | Train Loss: 0.2854 | Val Loss: 0.0886 | Train Acc: 0.9187 | Val Acc: 0.9718
Epoch 2/8 | Train Loss: 0.0842 | Val Loss: 0.0690 | Train Acc: 0.9748 | Val Acc: 0.9776
Epoch 3/8 | Train Loss: 0.0638 | Val Loss: 0.0707 | Train Acc: 0.9809 | Val Acc: 0.9783
Epoch 4/8 | Train Loss: 0.0530 | Val Loss: 0.0582 | Train Acc: 0.9839 | Val Acc: 0.9818
Epoch 5/8 | Train Loss: 0.0442 | Val Loss: 0.0590 | Train Acc: 0.9865 | Val Acc: 0.9825
Epoch 6/8 | Train Loss: 0.0394 | Val Loss: 0.0595 | Train Acc: 0.9878 | Val Acc: 0.9811
Epoch 7/8 | Train Loss: 0.0343 | Val Loss: 0.0627 | Train Acc: 0.9888 | Val Acc: 0.9813
Epoch 8/8 | Train Loss: 0.0295 | Val Loss: 0.0571 | Train Acc: 0.9905 | Val Acc: 0.9825
combo kernel parameters: 33722

Receptive field analysis:
Layer 1: conv | kernel=1, stride=1, padding=0 | RF=1
Layer 2: conv | kernel=3, stride=1, padding=1 | RF=3
Layer 3: pool | kernel=2, stride=2, padding=0 | RF=4

Training depth shallow...
Epoch 1/8 | Train Loss: 0.2276 | Val Loss: 0.0685 | Train Acc: 0.9346 | Val Acc: 0.9786
Epoch 2/8 | Train Loss: 0.0664 | Val Loss: 0.0574 | Train Acc: 0.9800 | Val Acc: 0.9816
Epoch 3/8 | Train Loss: 0.0490 | Val Loss: 0.0535 | Train Acc: 0.9848 | Val Acc: 0.9824
Epoch 4/8 | Train Loss: 0.0386 | Val Loss: 0.0569 | Train Acc: 0.9878 | Val Acc: 0.9823
Epoch 5/8 | Train Loss: 0.0324 | Val Loss: 0.0472 | Train Acc: 0.9900 | Val Acc: 0.9849
Epoch 6/8 | Train Loss: 0.0283 | Val Loss: 0.0431 | Train Acc: 0.9915 | Val Acc: 0.9855
Epoch 7/8 | Train Loss: 0.0238 | Val Loss: 0.0466 | Train Acc: 0.9925 | Val Acc: 0.9861
Epoch 8/8 | Train Loss: 0.0208 | Val Loss: 0.0459 | Train Acc: 0.9934 | Val Acc: 0.9870
shallow depth parameters: 33850

Receptive field analysis:
Layer 1: conv | kernel=3, stride=1, padding=1 | RF=3
Layer 2: conv | kernel=3, stride=1, padding=1 | RF=5
Layer 3: pool | kernel=2, stride=2, padding=0 | RF=6

Training depth medium...
Epoch 1/8 | Train Loss: 0.2249 | Val Loss: 0.0619 | Train Acc: 0.9300 | Val Acc: 0.9807
Epoch 2/8 | Train Loss: 0.0629 | Val Loss: 0.0468 | Train Acc: 0.9808 | Val Acc: 0.9855
Epoch 3/8 | Train Loss: 0.0482 | Val Loss: 0.0488 | Train Acc: 0.9856 | Val Acc: 0.9836
Epoch 4/8 | Train Loss: 0.0386 | Val Loss: 0.0352 | Train Acc: 0.9880 | Val Acc: 0.9885
Epoch 5/8 | Train Loss: 0.0329 | Val Loss: 0.0399 | Train Acc: 0.9896 | Val Acc: 0.9881
Epoch 6/8 | Train Loss: 0.0274 | Val Loss: 0.0368 | Train Acc: 0.9914 | Val Acc: 0.9886
Epoch 7/8 | Train Loss: 0.0250 | Val Loss: 0.0352 | Train Acc: 0.9921 | Val Acc: 0.9893
Epoch 8/8 | Train Loss: 0.0207 | Val Loss: 0.0447 | Train Acc: 0.9935 | Val Acc: 0.9859
medium depth parameters: 38490

Receptive field analysis:
Layer 1: conv | kernel=3, stride=1, padding=1 | RF=3
Layer 2: conv | kernel=3, stride=1, padding=1 | RF=5
Layer 3: conv | kernel=3, stride=1, padding=1 | RF=7
Layer 4: conv | kernel=3, stride=1, padding=1 | RF=9
Layer 5: pool | kernel=2, stride=2, padding=0 | RF=10

Training depth deep...
Epoch 1/8 | Train Loss: 0.3477 | Val Loss: 0.0846 | Train Acc: 0.8875 | Val Acc: 0.9748
Epoch 2/8 | Train Loss: 0.0803 | Val Loss: 0.0647 | Train Acc: 0.9758 | Val Acc: 0.9784
Epoch 3/8 | Train Loss: 0.0592 | Val Loss: 0.0483 | Train Acc: 0.9816 | Val Acc: 0.9838
Epoch 4/8 | Train Loss: 0.0474 | Val Loss: 0.0453 | Train Acc: 0.9856 | Val Acc: 0.9842
Epoch 5/8 | Train Loss: 0.0413 | Val Loss: 0.0418 | Train Acc: 0.9868 | Val Acc: 0.9843
Epoch 6/8 | Train Loss: 0.0365 | Val Loss: 0.0382 | Train Acc: 0.9885 | Val Acc: 0.9873
Epoch 7/8 | Train Loss: 0.0317 | Val Loss: 0.0442 | Train Acc: 0.9897 | Val Acc: 0.9832
Epoch 8/8 | Train Loss: 0.0283 | Val Loss: 0.0476 | Train Acc: 0.9909 | Val Acc: 0.9849
deep depth parameters: 18690

Receptive field analysis:
Layer 1: conv | kernel=3, stride=1, padding=1 | RF=3
Layer 2: conv | kernel=3, stride=1, padding=1 | RF=5
Layer 3: conv | kernel=3, stride=1, padding=1 | RF=7
Layer 4: conv | kernel=3, stride=1, padding=1 | RF=9
Layer 5: conv | kernel=3, stride=1, padding=1 | RF=11
Layer 6: conv | kernel=3, stride=1, padding=1 | RF=13
Layer 7: pool | kernel=2, stride=2, padding=0 | RF=14

Training depth residual...
Epoch 1/8 | Train Loss: 0.1816 | Val Loss: 0.0599 | Train Acc: 0.9446 | Val Acc: 0.9804
Epoch 2/8 | Train Loss: 0.0565 | Val Loss: 0.0483 | Train Acc: 0.9826 | Val Acc: 0.9839
Epoch 3/8 | Train Loss: 0.0439 | Val Loss: 0.0525 | Train Acc: 0.9862 | Val Acc: 0.9827
Epoch 4/8 | Train Loss: 0.0366 | Val Loss: 0.0510 | Train Acc: 0.9883 | Val Acc: 0.9839
Epoch 5/8 | Train Loss: 0.0318 | Val Loss: 0.0409 | Train Acc: 0.9899 | Val Acc: 0.9875
Epoch 6/8 | Train Loss: 0.0261 | Val Loss: 0.0455 | Train Acc: 0.9916 | Val Acc: 0.9869
Epoch 7/8 | Train Loss: 0.0237 | Val Loss: 0.0315 | Train Acc: 0.9923 | Val Acc: 0.9892
residual depth parameters: 40938

Receptive field analysis:
Layer 1: conv | kernel=3, stride=1, padding=1 | RF=3
Layer 2: conv | kernel=3, stride=1, padding=1 | RF=5
Layer 3: conv | kernel=3, stride=1, padding=1 | RF=7
Layer 4: conv | kernel=3, stride=1, padding=1 | RF=9
Layer 5: conv | kernel=3, stride=1, padding=1 | RF=11
Layer 6: pool | kernel=2, stride=2, padding=0 | RF=12
```
### 3.4 Анализ
```
Анализ результатов для ядра 3x3:

•  Достигнута точность на валидационном наборе 98.24% после 8 эпох.
•  Наблюдается стабильное улучшение точности в процессе обучения, но после 4 эпохи прирост незначительный.
•  Количество параметров: 63050.
•  Переобучение незначительное.
•  Рецептивное поле (RF) первого сверточного слоя: 3. После pooling RF = 4. Это означает, что каждый нейрон в feature map первого слоя "видит" область 3x3 в исходном изображении, а после pooling эта область расширяется до 4x4.

Анализ результатов для ядра 5x5:

•  Достигнута точность на валидационном наборе 98.33% после 8 эпох.
•  Наблюдается стабильное улучшение точности в процессе обучения, но после 5 эпохи прирост незначительный.
•  Количество параметров: 31786 (значительно меньше, чем у 3x3).
•  Переобучение незначительное.
•  Рецептивное поле (RF) первого сверточного слоя: 5. После pooling RF = 6. Каждый нейрон в feature map первого слоя "видит" область 5x5 в исходном изображении, а после pooling эта область расширяется до 6x6.

Анализ результатов для ядра 7x7:

•  Достигнута точность на валидационном наборе 98.22% после 8 эпох.
•  Наблюдается стабильное улучшение точности в процессе обучения, но после 6 эпохи прирост незначительный.
•  Количество параметров: 16090 (наименьшее из всех трех вариантов).
•  Переобучение незначительное.
•  Рецептивное поле (RF) первого сверточного слоя: 7. После pooling RF = 8. Каждый нейрон в feature map первого слоя "видит" область 7x7 в исходном изображении, а после pooling эта область расширяется до 8x8.

Анализ результатов для комбинации ядер:

•  Достигнута точность на валидационном наборе 98.25% после 8 эпох.
•  Наблюдается стабильное улучшение точности в процессе обучения, но после 5 эпохи прирост незначительный.
•  Количество параметров: 33722.
•  Переобучение незначительное.
•  Рецептивное поле (RF) после первого сверточного слоя (1x1): 1. После второго сверточного слоя (3x3) RF = 3. После pooling RF = 4. Использование ядра 1х1 не увеличило рецептивное поле.

Анализ результатов для неглубокой CNN (2 сверточных слоя):

•  Достигнута точность на валидационном наборе 98.70% после 8 эпох.
•  Наблюдается стабильное улучшение точности в процессе обучения.
•  Количество параметров: 33850.
•  RF после первого сверточного слоя: 3, после второго: 5, после pooling: 6.
•  Переобучение незначительное.

Анализ результатов для средней CNN (4 сверточных слоя):

•  Достигнута точность на валидационном наборе 98.59% после 8 эпох.
•  Наблюдается стабильное улучшение точности в процессе обучения.
•  Количество параметров: 38490.
•  RF после первого сверточного слоя: 3, после второго: 5, после третьего: 7, после четвертого: 9, после pooling: 10.
•  Переобучение незначительное.

Анализ результатов для глубокой CNN (6 сверточных слоев):

•  Достигнута точность на валидационном наборе 98.49% после 8 эпох.
•  Наблюдается стабильное улучшение точности в процессе обучения.
•  Количество параметров: 18690.
•  RF после первого сверточного слоя: 3, после второго: 5, после третьего: 7, после четвертого: 9, после пятого: 11, после шестого: 13, после pooling: 14.
•  Переобучение незначительное.

Анализ результатов для CNN с Residual связями:

•  Достигнута точность на валидационном наборе 98.92% после 7 эпох.
•  Наблюдается стабильное улучшение точности в процессе обучения.
•  Количество параметров: 40938.
•  RF после первого сверточного слоя: 3, после второго: 5, после третьего: 7, после четвертого: 9, после пятого: 11, после pooling: 12.
•  Переобучение незначительное.
•  Достигнута самая высокая точность на валидационном наборе среди всех протестированных архитектур.
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
###4.6 Анализ и выводы
```
Анализ результатов на CIFAR-10:

•  StandardCNN: Test Acc = 0.7193
•  CustomCNN: Test Acc = 0.7200
•  ResidualCNN: Test Acc = 0.7140

Анализ результатов по точности и времени инференса на CIFAR-10:

Test Accuracy:
  •  StandardCNN: 0.7193
  •  CustomCNN: 0.7200
  •  ResidualCNN: 0.7140
Inference Time:
  •  StandardCNN: 7.07 сек
  •  CustomCNN: 7.15 сек
  •  ResidualCNN: 6.95 сек

Вывод:

•  Все три модели показали схожую точность на тестовом наборе.
•  CustomCNN и StandardCNN достигли немного более высокой точности, чем ResidualCNN.
•  За 5 эпох ни одна из моделей не продемонстрировала значительного преимущества в производительности.
•  CustomCNN показала незначительно лучшую точность, чем StandardCNN и ResidualCNN.
•  ResidualCNN имеет наименьшее время инференса, а CustomCNN - наибольшее.
•  Различия в точности и времени инференса между моделями незначительны.


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
