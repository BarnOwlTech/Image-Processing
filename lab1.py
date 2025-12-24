import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

img = cv2.imread('img2.jpg') # Загрузка изображения
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Перевод из сломанного РГБ в нормальное РГБ (кривая библиотека)

original_image = Image.open('img2.jpg')
gray_image = None
if original_image.mode != 'L':
    gray_image = original_image.convert('L')
else:
    gray_image = original_image
gray_array = np.array(gray_image)

# 1 Коррекция с опорным цветом
def reference_color_correction(img, reference_color, target_color):
    ratio = target_color / reference_color
    corrected = np.clip(img * ratio, 0, 255).astype(np.uint8)
    return corrected

# 2 Серый мир (+)
def gray_world(img):
    return gray_image

# 3 Гамма-коррекция
def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

# 4 Кусочно-линейная функция
def piecewise_linear(img, points):
    x, y = zip(*points)
    lut = np.interp(np.arange(256), x, y).astype('uint8')
    lut = np.clip(lut, 0, 255).astype('uint8')
    return cv2.LUT(img, lut)

# 5 Сплайн-интерполяция
def spline_correction(img, control_points):
    control_points.sort()
    x, y = zip(*control_points)
    spline = UnivariateSpline(x, y, k=3)
    lut = spline(np.arange(256)).clip(0, 255).astype('uint8')
    return cv2.LUT(img, lut)

# 6 Нормализация гистограммы
def hist_normalize(img):
    img_np = np.array(img)
    L = 256
    hist, _ = np.histogram(img_np.flatten(), bins=L, range=(0, L-1), density=False)
    p_r = hist / hist.sum()  # нормализация гистограммы
    # (ФР) T(r)
    cdf = np.cumsum(p_r)
    T = np.floor((L - 1) * cdf).astype(np.uint8)

    img_eq_np = T[img_np]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = cv2.normalize(hsv[:,:,2], None, 0, 255, cv2.NORM_MINMAX)
    return img_eq_np # cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# 2 Эквализация гистограммы
def hist_equalize(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

refColor = np.array([50, 50, 50])  # Опорный цвет
targetColor = np.array([155, 155, 255])  # Целевой цвет
correctedRef = reference_color_correction(img_RGB, refColor, targetColor)

img_gray_world = gray_world(img_RGB)

gammaV = 0.6
img_gamma_correction = gamma_correction(img_RGB, gamma=gammaV)

points = [(0, 0), (25, 25), (155, 155), (255, 255)]
corrected_linear = piecewiseLinear(img_RGB, points)

control_points = [(0, 0), (64, 80), (128, 160), (192, 220), (255, 255)]
corrected_spline = spline_correction(img_RGB, control_points)

normalized = hist_normalize(img_RGB)

equalized = hist_equalize(img_RGB)


fig, axes = plt.subplots(2, 4, figsize=(20, 15))
images = [
    img_RGB, correctedRef, img_gray_world, img_gamma_correction,
    corrected_linear, corrected_spline, normalized, equalized
]
titles = [
    'Исходник', 'Опорный цвет', 'Серый мир', f'Гамма-коррекция g={gammaV}',
    'Линейно кусочная', 'Сплайн', 'Нормализация гистограммы', 'Эквализация гистограммы'
]

for i, (ax, image, title) in enumerate(zip(axes.flat, images, titles)):
    print(ax)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
axes[0][2].imshow(gray_array, cmap='gray')
axes[0][2].set_title('Серый мир')
axes[0][2].axis('off')

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title('Гистограмма исходника')
L = 256
plt.hist(img_RGB.flatten(), bins=L, range=(0, L-1), color='gray')
plt.xlabel('Яркость')
plt.ylabel('Частота')

plt.subplot(1,3,2)
plt.title('Гистограмма норм.')
plt.hist(normalized.flatten(), bins=L, range=(0, L-1), color='gray')
plt.xlabel('Яркость')
plt.ylabel('Частота')

plt.subplot(1,3,3)
plt.title('Гистограмма экв.')
L = 256
plt.hist(equalized.flatten(), bins=L, range=(0, L-1), color='gray')
plt.xlabel('Яркость')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()

