import cv2
import matplotlib.pyplot as plt

image_path = '2.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold_value = 128

_, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Исходник')
axes[0].axis('off')

axes[1].imshow(gray_image, cmap='gray')
axes[1].set_title('полутон')
axes[1].axis('off')

axes[2].imshow(binary_image, cmap='gray')
axes[2].set_title(f'бинарное (порог: {threshold_value})')
axes[2].axis('off')

plt.tight_layout()
plt.show()

"""# Нижний, Верхний, Диапозон"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


image_path = '2.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold = 128
low_threshold = 100
high_threshold = 200

_, lower_binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)

_, upper_binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

range_binary = np.where((gray_image >= low_threshold) & (gray_image <= high_threshold), 255, 0).astype(np.uint8)

otsu_threshold, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Исходник')
axes[0, 0].axis('off')

axes[0, 1].imshow(lower_binary, cmap='gray')
axes[0, 1].set_title(f'Нижняя {threshold}')
axes[0, 1].axis('off')

axes[0, 2].imshow(upper_binary, cmap='gray')
axes[0, 2].set_title(f'Верхняя {threshold}')
axes[0, 2].axis('off')

axes[1, 0].imshow(range_binary, cmap='gray')
axes[1, 0].set_title(f'Диапозон ({low_threshold} - {high_threshold})')
axes[1, 0].axis('off')

axes[1, 1].imshow(binary_image, cmap='gray')
axes[1, 1].set_title('Оцу глобальный')
axes[1, 1].axis('off')


plt.tight_layout()
plt.show()

"""# Локальный Оцу"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = '2.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Параметры для локального Оцу
block_size = 50

height, width = gray_image.shape

# Создаем пустое изображение для результата
local_otsu_binary = np.zeros_like(gray_image)

# Разбиваем изображение на фрагменты и применяем Оцу к каждому
for y in range(0, height, block_size):
    for x in range(0, width, block_size):
        # Вырезаем фрагмент
        block = gray_image[y:y+block_size, x:x+block_size]
        # Применяем Оцу к фрагменту
        if block.size > 0:
            _, block_binary = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Вставляем  фрагмент обратно
            local_otsu_binary[y:y+block_size, x:x+block_size] = block_binary


fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Исходник')
axes[0].axis('off')

axes[1].imshow(local_otsu_binary, cmap='gray')
axes[1].set_title('Локальный оцу')
axes[1].axis('off')

plt.tight_layout()
plt.show()

"""# Иерархический Оцу"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def hierarchical_otsu(image, min_size):
    height, width = image.shape

    # Базовый случай: если область слишком маленькая, применяем глобальный Оцу
    if height <= min_size or width <= min_size:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # Разбиваем на 4 квадранта
    mid_h = height // 2
    mid_w = width // 2

    # Верхний левый
    quad1 = hierarchical_otsu(image[:mid_h, :mid_w], min_size)
    # Верхний правый
    quad2 = hierarchical_otsu(image[:mid_h, mid_w:], min_size)
    # Нижний левый
    quad3 = hierarchical_otsu(image[mid_h:, :mid_w], min_size)
    # Нижний правый
    quad4 = hierarchical_otsu(image[mid_h:, mid_w:], min_size)

    top = np.hstack((quad1, quad2))
    bottom = np.hstack((quad3, quad4))
    return np.vstack((top, bottom))

image_path = '2.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

min_size = 50

hierarchical_binary = hierarchical_otsu(gray_image, min_size)


fig, axes = plt.subplots(1,2, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Исходник')
axes[0].axis('off')

axes[1].imshow(hierarchical_binary, cmap='gray')
axes[1].set_title('Иерархический оцу')
axes[1].axis('off')

plt.tight_layout()
plt.show()

"""# Квантование Яркости"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def quantize_brightness(image, num_levels):
    # Диапазон яркостей: 0-255
    step = 256 // num_levels
    # Создаем уровни: 0, step, 2*step, ..., (num_levels-1)*step
    levels = np.arange(0, 256, step)

    # Для каждого пикселя находим ближайший уровень
    quantized = np.zeros_like(image, dtype=np.uint8)
    for i in range(num_levels):
        lower = i * step
        upper = (i + 1) * step if i < num_levels - 1 else 256
        mask = (image >= lower) & (image < upper)
        quantized[mask] = levels[i]
    return quantized

image_path = '2.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

num_levels = 3
quantized_image = quantize_brightness(gray_image, num_levels)
num_levels_2 = 9
quantized_image_2 = quantize_brightness(gray_image, num_levels_2)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Исходник')
axes[0].axis('off')

axes[1].imshow(quantized_image, cmap='gray')
axes[1].set_title(f'Квантование {num_levels}')
axes[1].axis('off')

axes[2].imshow(quantized_image_2, cmap='gray')
axes[2].set_title(f'Квантование {num_levels_2}')
axes[2].axis('off')

plt.tight_layout()
plt.show()