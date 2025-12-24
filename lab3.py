import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

se_length = 20
# Вертикальная линия; для угла используйте cv2.getStructuringElement(cv2.MORPH_RECT, (se_length, 1)) и поворот
se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, se_length))

# Применение открытия
opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)

# Вычитание для выделения проблемных зубьев
problematic_teeth = cv2.subtract(image, opened)

# Дилатация
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
refined = cv2.dilate(problematic_teeth, kernel, iterations=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Исходник')
axes[0].axis('off')

axes[1].imshow(opened, cmap='gray')
axes[1].set_title('Открытое изображение (Opening)')
axes[1].axis('off')

axes[2].imshow(refined, cmap='gray')
axes[2].set_title('Проблемные зубья (маска, дилатация)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

se_lengths = [10, 15, 100, 180, 210, 250]
angle = 0

results = []

for se_length in se_lengths:
    # Создание SE: горизонтальная линия
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_length, 1))
    # Поворот SE, если угол не 0°
    if angle != 0:
        center = (se_length // 2, 0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        se = cv2.warpAffine(se.astype(np.uint8), rotation_matrix, (se_length, 1))
    # opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, se)
    opened = cv2.erode(binary_image, se)
    #  Вычитание для маски проблемных зубьев
    problematic_teeth = cv2.subtract(binary_image, opened)
    # дилатация
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    refined = cv2.dilate(problematic_teeth, kernel, iterations=1)

    results.append((se_length, opened, refined))

fig, axes = plt.subplots(2, len(se_lengths) + 1, figsize=(15, 8))

axes[0, 0].imshow(binary_image, cmap='gray')
axes[0, 0].set_title('Исходник')
axes[0, 0].axis('off')


# Результаты для разных длин SE
for i, (se_length, opened, refined) in enumerate(results):
    axes[0, i+1].imshow(opened, cmap='gray')
    axes[0, i+1].set_title(f'Eroded Image (длина SE={se_length})')
    axes[0, i+1].axis('off')

    axes[1, i+1].imshow(refined, cmap='gray')
    axes[1, i+1].set_title(f'Проблемные зубья (длина SE={se_length})')
    axes[1, i+1].axis('off')

plt.tight_layout()
plt.show()