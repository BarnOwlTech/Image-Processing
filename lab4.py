import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загружаем изображение
image = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Сглаживание (медианный фильтр для удаления шума)
smoothed = cv2.medianBlur(image, 5)

# 2. Повышение резкости (фильтр Лапласа)
laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
sharpened = smoothed - 0.5 * laplacian
sharpened = np.uint8(np.clip(sharpened, 0, 255))

# 3. Эквализация гистограммы для улучшения контраста
enhanced = cv2.equalizeHist(sharpened)

# Визуализация результатов
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Исходное изображение')
axes[0, 0].axis('off')

axes[0, 1].imshow(smoothed, cmap='gray')
axes[0, 1].set_title('После сглаживания')
axes[0, 1].axis('off')

axes[1, 0].imshow(sharpened, cmap='gray')
axes[1, 0].set_title('После повышения резкости')
axes[1, 0].axis('off')

axes[1, 1].imshow(enhanced, cmap='gray')
axes[1, 1].set_title('После эквализации гистограммы')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# Сохранение результата
cv2.imwrite('enhanced_image.jpg', enhanced)

# @title
import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import partial

# --------------------------------------------------
# 1. Вспомогательные функции
# --------------------------------------------------
def create_test_image(size=256):
    image = np.zeros((size, size), dtype=np.float32)
    cv2.circle(image, (size // 2, size // 2), 60, 180, -1)
    for i in range(0, size, 10):
        for j in range(0, size, 10):
            if (i // 10 + j // 10) % 2 == 0:
                image[i:i+3, j:j+3] = 255
    return image

def add_periodic_noise(image, freq_x=20, freq_y=15, strength=60):
    rows, cols = image.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    noise = strength * np.sin(2 * np.pi * (freq_x * X / cols + freq_y * Y / rows))
    return image + noise

def add_illumination_gradient(image):
    rows, cols = image.shape
    gradient = np.linspace(0.3, 1.0, rows).reshape(-1, 1)
    return (image * gradient).astype(np.float32)

# --------------------------------------------------
# 2. Фильтры
# --------------------------------------------------
def low_pass_filter(shape, cutoff_radius):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - center_row
    v = np.arange(cols).reshape(1, -1) - center_col
    distance = np.sqrt(u**2 + v**2)
    return np.exp(-(distance**2) / (2 * cutoff_radius**2))

def high_pass_filter(shape, cutoff_radius):
    return 1 - low_pass_filter(shape, cutoff_radius)

def laplacian_sharpen(shape):
    rows, cols = shape
    u = np.fft.fftfreq(rows).reshape(-1, 1) * rows
    v = np.fft.fftfreq(cols).reshape(1, -1) * cols
    return -4 * (np.pi ** 2) * (u**2 + v**2)

def remove_periodic_noise(image, noise_frequencies, radius=5):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    filter_mask = np.ones_like(image)
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    for freq in noise_frequencies:
        u, v = freq
        for du, dv in [(u, v), (-u, v), (u, -v), (-u, -v)]:
            y = center_row + du
            x = center_col + dv
            if 0 <= y < rows and 0 <= x < cols:
                y_start, y_end = y - radius, y + radius + 1
                x_start, x_end = x - radius, x + radius + 1
                y_clip = slice(max(0, y_start), min(rows, y_end))
                x_clip = slice(max(0, x_start), min(cols, x_end))
                if y_clip.start < y_clip.stop and x_clip.start < x_clip.stop:
                    yy, xx = np.ogrid[y_clip.start - y:y_clip.stop - y,
                                      x_clip.start - x:x_clip.stop - x]
                    mask_local = (yy**2 + xx**2) <= radius**2
                    if mask_local.shape == filter_mask[y_clip, x_clip].shape:
                        filter_mask[y_clip, x_clip][mask_local] = 0
    filtered_dft = dft_shift * filter_mask
    result = np.fft.ifft2(np.fft.ifftshift(filtered_dft))
    return np.abs(result), filter_mask

def homomorphic_filter(image, cutoff, gamma_low=0.5, gamma_high=1.5):
    log_image = np.log1p(image.astype(np.float64))
    dft = np.fft.fft2(log_image)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image.shape
    u = np.fft.fftfreq(rows).reshape(-1, 1) * rows
    v = np.fft.fftfreq(cols).reshape(1, -1) * cols
    D = np.sqrt(u**2 + v**2)
    H = (gamma_high - gamma_low) * (1 - np.exp(-D**2 / (2 * cutoff**2))) + gamma_low
    filtered_dft = dft_shift * H
    filtered_log = np.fft.ifft2(np.fft.ifftshift(filtered_dft))
    result = np.expm1(np.real(filtered_log))
    return np.clip(result, 0, 255).astype(np.uint8)

def texture_analysis(image, region_size=64):
    textures = []
    h, w = image.shape
    step = region_size // 2
    for i in range(0, h - region_size + 1, step):
        for j in range(0, w - region_size + 1, step):
            region = image[i:i+region_size, j:j+region_size]
            dft = np.fft.fft2(region)
            dft_shift = np.fft.fftshift(dft)
            spectrum = np.abs(dft_shift)
            center = region_size // 2
            margin = region_size // 4
            low_freq = spectrum[center - margin:center + margin,
                                center - margin:center + margin]
            low_freq_energy = np.sum(low_freq)
            total_energy = np.sum(spectrum)
            texture_feature = low_freq_energy / (total_energy + 1e-8)
            textures.append({'position': (i, j), 'texture_feature': texture_feature})
    return textures

# --------------------------------------------------
# 3. Конвейеры
# --------------------------------------------------
def frequency_processing_pipeline(image, filter_function):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    filter_mask = filter_function(dft_shift.shape)
    filtered_dft = dft_shift * filter_mask
    result = np.fft.ifft2(np.fft.ifftshift(filtered_dft))
    return np.abs(result), filter_mask

def apply_laplacian_in_frequency(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    H = laplacian_sharpen(dft_shift.shape)
    filtered_dft = dft_shift * H
    result = np.fft.ifft2(np.fft.ifftshift(filtered_dft))
    return np.real(result)

# --------------------------------------------------
# 4. Основной запуск — раздельная визуализация
# --------------------------------------------------
if __name__ == "__main__":
    size = 256
    clean_image = create_test_image(size)

    # =================================================================
    # 1. Исходное изображение
    # =================================================================
    plt.figure(figsize=(6, 6))
    plt.imshow(clean_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # =================================================================
    # 2. Периодический шум и его удаление
    # =================================================================
    noisy = add_periodic_noise(clean_image, freq_x=30, freq_y=25, strength=60)
    denoised, mask = remove_periodic_noise(noisy, noise_frequencies=[(25, 30)], radius=6)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(noisy, cmap='gray')
    plt.title('С периодическим шумом')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Режекторная маска')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised, cmap='gray')
    plt.title('После удаления шума')
    plt.axis('off')
    plt.suptitle('Удаление периодического шума (муара)', fontsize=14)
    plt.tight_layout()
    plt.show()

    # =================================================================
    # 3. Гомоморфная фильтрация
    # =================================================================
    ill_image = add_illumination_gradient(clean_image)
    homo_image = homomorphic_filter(ill_image, cutoff=15, gamma_low=0.4, gamma_high=2.0)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(ill_image, cmap='gray')
    plt.title('С градиентом освещения')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(homo_image, cmap='gray')
    plt.title('После гомоморфной фильтрации')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(ill_image - homo_image, cmap='seismic', vmin=-50, vmax=50)
    plt.title('Разность (до – после)')
    plt.axis('off')
    plt.colorbar(shrink=0.6)
    plt.suptitle('Гомоморфная фильтрация (улучшение освещения)', fontsize=14)
    plt.tight_layout()
    plt.show()

    # =================================================================
    # 4. Анализ текстур
    # =================================================================
    texture_data = texture_analysis(clean_image, region_size=64)
    texture_map = np.zeros_like(clean_image, dtype=np.float32)
    for t in texture_data:
        i, j = t['position']
        texture_map[i:i+64, j:j+64] = t['texture_feature']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(clean_image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    im = plt.imshow(texture_map, cmap='viridis')
    plt.colorbar(im, shrink=0.8)
    plt.title('Карта текстур\n(низкие значения = детализированные области)')
    plt.axis('off')
    plt.suptitle('Анализ текстур через частотный спектр', fontsize=14)
    plt.tight_layout()
    plt.show()

    # =================================================================
    # 5. Низкочастотный фильтр (ФНЧ)
    # =================================================================
    lpf_image, lpf_mask = frequency_processing_pipeline(clean_image, partial(low_pass_filter, cutoff_radius=20))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(clean_image, cmap='gray')
    plt.title('Исходное')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(lpf_mask, cmap='gray')
    plt.title('Маска ФНЧ')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(lpf_image, cmap='gray')
    plt.title('После ФНЧ (сглаживание)')
    plt.axis('off')
    plt.suptitle('Низкочастотная фильтрация', fontsize=14)
    plt.tight_layout()
    plt.show()

    # =================================================================
    # 6. Высокочастотный фильтр (ФВЧ)
    # =================================================================
    hpf_image, hpf_mask = frequency_processing_pipeline(clean_image, partial(high_pass_filter, cutoff_radius=10))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(clean_image, cmap='gray')
    plt.title('Исходное')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(hpf_mask, cmap='gray')
    plt.title('Маска ФВЧ')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hpf_image, cmap='gray')
    plt.title('После ФВЧ (края и детали)')
    plt.axis('off')
    plt.suptitle('Высокочастотная фильтрация', fontsize=14)
    plt.tight_layout()
    plt.show()

    # =================================================================
    # 7. Лапласиан и усиление резкости
    # =================================================================
    laplacian = apply_laplacian_in_frequency(clean_image)
    sharpened = np.clip(clean_image - 0.0005 * laplacian, 0, 255)

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(clean_image, cmap='gray')
    plt.title('Исходное')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(np.clip(laplacian, -50, 50), cmap='seismic')
    plt.title('Лапласиан (частотная область)')
    plt.axis('off')
    plt.colorbar(shrink=0.6)

    plt.subplot(1, 4, 3)
    plt.imshow(sharpened, cmap='gray')
    plt.title('Усиление резкости')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    diff = sharpened - clean_image
    plt.imshow(diff, cmap='seismic', vmin=-20, vmax=20)
    plt.title('Разность (резкость – оригинал)')
    plt.axis('off')
    plt.colorbar(shrink=0.6)
    plt.suptitle('Лапласиан и усиление чёткости', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("✅ Все результаты отображены по отдельности.")

# @title
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Простая версия только с ДПФ и ОДПФ
def simple_dft_idft():
    # Загружаем существующее изображение
    image = cv2.imread('img2.jpg', 0)

    if image is None:
        print("Ошибка: файл img1.jpg не найден!")
        return

    print("Загружено изображение:", image.shape)

    # Прямое ДПФ
    dft = np.fft.fft2(image.astype(float))
    print("ДПФ выполнено")

    # Обратное ДПФ
    idft = np.fft.ifft2(dft)
    reconstructed = np.real(idft).astype(np.uint8)
    print("ОДПФ выполнено")

    # Визуализация
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Исходное img1.jpg')

    plt.subplot(1, 3, 2)
    plt.imshow(np.log(np.abs(np.fft.fftshift(dft)) + 1), cmap='gray')
    plt.title('Спектр ДПФ')

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed, cmap='gray')
    plt.title('После ОДПФ')

    plt.show()

# Запуск
simple_dft_idft()

# @title
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def ideal_lowpass_filter(shape, cutoff):
    """Идеальный ФНЧ"""
    rows, cols = shape
    u = np.fft.fftfreq(rows).reshape(-1, 1)
    v = np.fft.fftfreq(cols).reshape(1, -1)
    D = np.sqrt(u**2 + v**2)
    H = np.zeros(shape)
    H[D <= cutoff] = 1
    return H

def butterworth_lowpass_filter(shape, cutoff, order=2):
    """ФНЧ Баттерворта"""
    rows, cols = shape
    u = np.fft.fftfreq(rows).reshape(-1, 1)
    v = np.fft.fftfreq(cols).reshape(1, -1)
    D = np.sqrt(u**2 + v**2)
    H = 1 / (1 + (D / cutoff) ** (2 * order))
    return H

def gaussian_lowpass_filter(shape, sigma):
    """Гауссов ФНЧ"""
    rows, cols = shape
    u = np.fft.fftfreq(rows).reshape(-1, 1)
    v = np.fft.fftfreq(cols).reshape(1, -1)
    D_squared = u**2 + v**2
    H = np.exp(-D_squared / (2 * sigma**2))
    return H

def apply_frequency_filter(image, H_filter):
    """Применение частотного фильтра"""
    # Преобразование Фурье
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Применение фильтра
    filtered_dft = dft_shift * H_filter

    # Обратное преобразование
    filtered_shift = np.fft.ifftshift(filtered_dft)
    filtered_image = np.fft.ifft2(filtered_shift)

    return np.abs(filtered_image)

# Пример использования сглаживающих фильтров
def demo_smoothing_filters():
    # Загрузка изображения
    image = cv2.imread('img3.jpg', 0)  # в оттенках серого
    if image is None:
        # Создаем тестовое изображение если файла нет
        image = np.random.rand(256, 256) * 255
        image = image.astype(np.uint8)

    shape = image.shape

    # Параметры фильтров
    cutoff_freq = 0.1  # нормированная частота среза
    sigma = 0.05       # параметр Гаусса
    order = 2          # порядок Баттерворта

    # Создание фильтров
    H_ideal = ideal_lowpass_filter(shape, cutoff_freq)
    H_butterworth = butterworth_lowpass_filter(shape, cutoff_freq, order)
    H_gaussian = gaussian_lowpass_filter(shape, sigma)

    # Применение фильтров
    ideal_filtered = apply_frequency_filter(image, H_ideal)
    butterworth_filtered = apply_frequency_filter(image, H_butterworth)
    gaussian_filtered = apply_frequency_filter(image, H_gaussian)

    # Визуализация
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    axes[0,0].imshow(image, cmap='gray')
    axes[0,0].set_title('Исходное изображение')

    axes[0,1].imshow(H_ideal, cmap='hot')
    axes[0,1].set_title('Идеальный ФНЧ')

    axes[0,2].imshow(H_butterworth, cmap='hot')
    axes[0,2].set_title('Баттерворт ФНЧ')

    axes[0,3].imshow(H_gaussian, cmap='hot')
    axes[0,3].set_title('Гауссов ФНЧ')

    axes[1,0].imshow(image, cmap='gray')
    axes[1,0].set_title('Исходное')

    axes[1,1].imshow(ideal_filtered, cmap='gray')
    axes[1,1].set_title('Идеальный ФНЧ')

    axes[1,2].imshow(butterworth_filtered, cmap='gray')
    axes[1,2].set_title('Баттерворт ФНЧ')

    axes[1,3].imshow(gaussian_filtered, cmap='gray')
    axes[1,3].set_title('Гауссов ФНЧ')

    plt.tight_layout()
    plt.show()

demo_smoothing_filters()

# @title
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

# Загружаем изображение в оттенках серого
# Замените 'image.jpg' на путь к вашему файлу
img = cv2.imread('img3.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    # Генерируем тестовое изображение, если файл не найден
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 30, 255, -1)
    cv2.rectangle(img, (50, 50), (100, 100), 200, -1)
    cv2.line(img, (0, 128), (256, 128), 150, 3)

plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Оригинал'), plt.axis('off')

def fft_shifted(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def ifft_shifted(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

F = fft_shifted(img)
magnitude_spectrum = np.log(1 + np.abs(F))

def ideal_lowpass(shape, D0):
    P, Q = shape
    u, v = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((u - Q/2)**2 + (v - P/2)**2)
    H = np.where(D <= D0, 1, 0)
    return H

def butterworth_lowpass(shape, D0, n=2):
    P, Q = shape
    u, v = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((u - Q/2)**2 + (v - P/2)**2)
    H = 1 / (1 + (D / D0)**(2*n))
    return H

def gaussian_lowpass(shape, D0):
    P, Q = shape
    u, v = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((u - Q/2)**2 + (v - P/2)**2)
    H = np.exp(-(D**2) / (2 * (D0**2)))
    return H

D0 = 30  # пороговое расстояние (можно подбирать)
H_ideal = ideal_lowpass(img.shape, D0)
H_butter = butterworth_lowpass(img.shape, D0, n=2)
H_gauss = gaussian_lowpass(img.shape, D0)

G_ideal = F * H_ideal
G_butter = F * H_butter
G_gauss = F * H_gauss

img_ideal_lp = ifft_shifted(G_ideal)
img_butter_lp = ifft_shifted(G_butter)
img_gauss_lp = ifft_shifted(G_gauss)

# Визуализация (пример — гаусс)
plt.subplot(133), plt.imshow(img_gauss_lp, cmap='gray'), plt.title('Гаусс ФНЧ'), plt.axis('off')
plt.show()


def ideal_highpass(shape, D0):
    return 1 - ideal_lowpass(shape, D0)

def butterworth_highpass(shape, D0, n=2):
    return 1 - butterworth_lowpass(shape, D0, n)

def gaussian_highpass(shape, D0):
    return 1 - gaussian_lowpass(shape, D0)

# Лапласиан в частотной области: H(u,v) = -4π²(u² + v²)
def laplacian_filter(shape):
    P, Q = shape
    u = np.arange(-Q//2, Q//2)
    v = np.arange(-P//2, P//2)
    u, v = np.meshgrid(u, v)
    H = -4 * (np.pi**2) * (u**2 + v**2)
    return H

# Нерезкое маскирование: H = 1 + k * (1 - H_lowpass)
def unsharp_masking(shape, D0, k=1.0, filter_type='gaussian'):
    if filter_type == 'gaussian':
        H_low = gaussian_lowpass(shape, D0)
    elif filter_type == 'butterworth':
        H_low = butterworth_lowpass(shape, D0, n=2)
    else:
        H_low = ideal_lowpass(shape, D0)
    H = 1 + k * (1 - H_low)
    return H

D0 = 30
H_ideal_hp = ideal_highpass(img.shape, D0)
H_butter_hp = butterworth_highpass(img.shape, D0)
H_gauss_hp = gaussian_highpass(img.shape, D0)
H_laplacian = laplacian_filter(img.shape)
H_unsharp = unsharp_masking(img.shape, D0, k=1.5, filter_type='gaussian')

# Применяем
G_hp = F * H_gauss_hp
G_lap = F * H_laplacian
G_unsharp = F * H_unsharp

img_hp = ifft_shifted(G_hp)
img_lap = ifft_shifted(G_lap)
img_unsharp = ifft_shifted(G_unsharp)

# Приводим к диапазону [0,255]
img_hp = np.clip(img_hp, 0, 255)
img_unsharp = np.clip(img_unsharp, 0, 255)

plt.figure(figsize=(15, 5))
plt.subplot(141), plt.imshow(img, cmap='gray'), plt.title('Оригинал'), plt.axis('off')
plt.subplot(142), plt.imshow(img_hp, cmap='gray'), plt.title('Гаусс ФВЧ'), plt.axis('off')
plt.subplot(143), plt.imshow(img_lap, cmap='gray'), plt.title('Лапласиан'), plt.axis('off')
plt.subplot(144), plt.imshow(img_unsharp, cmap='gray'), plt.title('Нерезкое маскирование'), plt.axis('off')
plt.show()


def notch_reject_filter(shape, centers, D0):
    """Подавляет частоты в окрестности точек centers (список (u,v))"""
    H = np.ones(shape)
    P, Q = shape
    for (u0, v0) in centers:
        # Делаем 4 симметричные точки (из-за центрирования FFT)
        for du, dv in [(u0, v0), (-u0, v0), (u0, -v0), (-u0, -v0)]:
            u = np.arange(Q) - Q//2 + du
            v = np.arange(P) - P//2 + dv
            U, V = np.meshgrid(u, v)
            D = np.sqrt(U**2 + V**2)
            H *= 1 - np.exp(-(D**2) / (2 * D0**2))  # гауссово "вырезание"
    return H

# Пример: подавим "шум" на частотах (±40, ±30)
centers = [(40, 30)]
H_notch = notch_reject_filter(img.shape, centers, D0=10)
G_notch = F * H_notch
img_notch = ifft_shifted(G_notch)

plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('С шумом (условно)'), plt.axis('off')
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('АЧХ с "пиками"'), plt.axis('off')
plt.subplot(133), plt.imshow(img_notch, cmap='gray'), plt.title('После режекторного фильтра'), plt.axis('off')
plt.show()


def bandpass_filter(shape, D0_low, D0_high, filter_type='gaussian'):
    if filter_type == 'gaussian':
        H_low = gaussian_lowpass(shape, D0_high)
        H_high = 1 - gaussian_lowpass(shape, D0_low)
    else:
        H_low = ideal_lowpass(shape, D0_high)
        H_high = 1 - ideal_lowpass(shape, D0_low)
    return H_low * H_high

H_bp = bandpass_filter(img.shape, D0_low=20, D0_high=50)
G_bp = F * H_bp
img_bp = ifft_shifted(G_bp)

plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Оригинал'), plt.axis('off')
plt.subplot(132), plt.imshow(np.log(1 + np.abs(G_bp)), cmap='gray'), plt.title('АЧХ после ПФ'), plt.axis('off')
plt.subplot(133), plt.imshow(img_bp, cmap='gray'), plt.title('Полосовой фильтр'), plt.axis('off')
plt.show()

def homomorphic_filter(img, d0=10, gamma_l=0.5, gamma_h=2.0, c=1):
    # Логарифмирование
    img_log = np.log1p(img.astype(np.float64))
    # FFT
    F_log = fft_shifted(img_log)
    # ФВЧ (Баттерворт)
    H = butterworth_highpass(img.shape, d0, n=1)
    H = gamma_l + (gamma_h - gamma_l) * H
    # Применяем
    G = F_log * H
    g = ifft_shifted(G)
    # Экспонента
    img_hom = np.expm1(g)
    return np.clip(img_hom, 0, 255).astype(np.uint8)

img_hom = homomorphic_filter(img, d0=15, gamma_l=0.3, gamma_h=1.8)

plt.figure(figsize=(20, 8))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Оригинал'), plt.axis('off')
plt.subplot(132), plt.imshow(img_hom, cmap='gray'), plt.title('Гомоморфная фильтрация'), plt.axis('off')
# Добавим резкую границу для демонстрации
if img.shape[0] > 100:
    cv2.rectangle(img_hom, (20,20), (80,80), 255, 2)
plt.subplot(133), plt.imshow(img_hom, cmap='gray'), plt.title('После резкого контрастирования'), plt.axis('off')
plt.show()

# @title
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d

class FrequencySpatialComparison:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, 0)
        if self.image is None:
            # Создаем тестовое изображение если файл не найден
            self.image = self.create_test_image()
        self.image = cv2.resize(self.image, (256, 256))
        self.image_float = self.image.astype(np.float32)

    def create_test_image(self):
        """Создает тестовое изображение с различными features"""
        img = np.ones((300, 300)) * 128
        cv2.rectangle(img, (50, 50), (100, 100), 200, -1)
        cv2.rectangle(img, (150, 150), (200, 200), 50, -1)
        cv2.circle(img, (200, 80), 25, 180, -1)

        # Добавляем шум и текстуру
        noise = np.random.normal(0, 15, img.shape)
        img = img + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    # 1. СГЛАЖИВАНИЕ (НИЗКОЧАСТОТНАЯ ФИЛЬТРАЦИЯ)

    def ideal_lowpass_frequency(self, cutoff):
        """Идеальный ФНЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        # Создаем маску идеального ФНЧ
        mask = np.zeros((rows, cols))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
        mask[mask_area] = 1

        # Применяем в частотной области
        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def ideal_lowpass_spatial(self, size):
        """Идеальный ФНЧ в пространственной области (усредняющий фильтр)"""
        kernel = np.ones((size, size)) / (size * size)
        return convolve2d(self.image_float, kernel, mode='same', boundary='symm')

    def butterworth_lowpass_frequency(self, cutoff, order=2):
        """ФНЧ Баттерворта в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Фильтр Баттерворта
        mask = 1 / (1 + (d / cutoff)**(2 * order))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def butterworth_lowpass_spatial(self, sigma):
        """Аналог ФНЧ Баттерворта в пространственной области (Гауссов фильтр)"""
        return ndimage.gaussian_filter(self.image_float, sigma=sigma)

    def gaussian_lowpass_frequency(self, sigma):
        """Гауссов ФНЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Гауссов фильтр
        mask = np.exp(-(d**2) / (2 * sigma**2))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def gaussian_lowpass_spatial(self, sigma):
        """Гауссов ФНЧ в пространственной области"""
        return ndimage.gaussian_filter(self.image_float, sigma=sigma)

    # 2. ПОВЫШЕНИЕ РЕЗКОСТИ (ВЫСОКОЧАСТОТНАЯ ФИЛЬТРАЦИЯ)

    def ideal_highpass_frequency(self, cutoff):
        """Идеальный ФВЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        mask = np.ones((rows, cols))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
        mask[mask_area] = 0

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def ideal_highpass_spatial(self):
        """Идеальный ФВЧ в пространственной области (лапласиан)"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        filtered = convolve2d(self.image_float, kernel, mode='same', boundary='symm')
        return self.image_float + filtered

    def butterworth_highpass_frequency(self, cutoff, order=2):
        """ФВЧ Баттерворта в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        mask = 1 / (1 + (cutoff / d)**(2 * order))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def gaussian_highpass_frequency(self, sigma):
        """Гауссов ФВЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        mask = 1 - np.exp(-(d**2) / (2 * sigma**2))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def laplacian_frequency(self):
        """Лапласиан в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)

        # Лапласиан в частотной области: -(u^2 + v^2)
        mask = -(u**2 + v**2)

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def laplacian_spatial(self):
        """Лапласиан в пространственной области"""
        kernel = np.array([[0, -1, 0],
                          [-1,  4, -1],
                          [0, -1, 0]])
        return convolve2d(self.image_float, kernel, mode='same', boundary='symm')

    def unsharp_masking_frequency(self, sigma, alpha=1.5):
        """Нерезкое маскирование в частотной области"""
        # Размытая версия (ФНЧ)
        blurred = self.gaussian_lowpass_frequency(sigma)

        # Маска = оригинал - размытая версия
        mask = self.image_float - blurred

        # Усиленная версия = оригинал + alpha * маска
        sharpened = self.image_float + alpha * mask

        return sharpened

    def unsharp_masking_spatial(self, sigma, alpha=1.5):
        """Нерезкое маскирование в пространственной области"""
        blurred = ndimage.gaussian_filter(self.image_float, sigma=sigma)
        mask = self.image_float - blurred
        sharpened = self.image_float + alpha * mask
        return sharpened

    def homomorphic_filtering(self, gamma_low=0.5, gamma_high=2.0, cutoff=30, order=1):
        """Гомоморфная фильтрация"""
        # Логарифм изображения
        img_log = np.log(self.image_float + 1)

        # ДПФ
        dft = np.fft.fft2(img_log)
        dft_shift = np.fft.fftshift(dft)

        # Создаем гомоморфный фильтр
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2
        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Фильтр гомоморфной фильтрации
        H = (gamma_high - gamma_low) * (1 - np.exp(-(d**2) / (2 * cutoff**2))) + gamma_low

        # Применяем фильтр
        filtered_dft = dft_shift * H
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        # Экспонента для возврата к исходному диапазону
        result = np.exp(np.real(idft)) - 1

        return np.clip(result, 0, 255)

    # 3. ИЗБИРАТЕЛЬНАЯ ФИЛЬТРАЦИЯ

    def notch_reject_filter_frequency(self, frequencies, bandwidth=5):
        """Режекторный фильтр в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        mask = np.ones((rows, cols))

        for freq in frequencies:
            u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
            d = np.sqrt((u - freq[0])**2 + (v - freq[1])**2)
            d2 = np.sqrt((u + freq[0])**2 + (v + freq[1])**2)

            # Подавляем частоты в заданной полосе
            mask[d <= bandwidth] = 0
            mask[d2 <= bandwidth] = 0

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def bandpass_filter_frequency(self, low_cutoff, high_cutoff):
        """Полосовой фильтр в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Полосовой фильтр
        mask = (d >= low_cutoff) & (d <= high_cutoff)
        mask = mask.astype(float)

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def narrowband_filter_frequency(self, center_freq, bandwidth=2):
        """Узкополосный фильтр в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Узкополосный фильтр
        mask = (d >= center_freq - bandwidth) & (d <= center_freq + bandwidth)
        mask = mask.astype(float)

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def visualize_comparison(self):
        """Визуализация сравнения всех методов"""
        results = []
        titles = []

        # Исходное изображение
        results.append(self.image)
        titles.append('Исходное изображение')
        results.append(self.image)
        titles.append('Исходное изображение')

        # 1. СГЛАЖИВАНИЕ
        # Идеальный ФНЧ
        results.append(self.ideal_lowpass_frequency(30))
        titles.append('Идеальный ФНЧ (частотный) 30')
        results.append(self.ideal_lowpass_frequency(60))
        titles.append('Идеальный ФНЧ (частотный) 60')
        results.append(self.ideal_lowpass_frequency(120))
        titles.append('Идеальный ФНЧ (частотный) 120')
        results.append(self.ideal_lowpass_spatial(7))
        titles.append('Идеальный ФНЧ (пространственный)')

        # ФНЧ Баттерворта
        results.append(self.butterworth_lowpass_frequency(30, 2))
        titles.append('ФНЧ Баттерворта (частотный)')
        results.append(self.butterworth_lowpass_spatial(2))
        titles.append('ФНЧ Баттерворта (пространственный)')

        # Гауссов ФНЧ
        results.append(self.gaussian_lowpass_frequency(30))
        titles.append('Гауссов ФНЧ (частотный)')
        results.append(self.gaussian_lowpass_spatial(2))
        titles.append('Гауссов ФНЧ (пространственный)')

        # 2. ПОВЫШЕНИЕ РЕЗКОСТИ
        # Идеальный ФВЧ
        results.append(self.ideal_highpass_frequency(15))
        titles.append('Идеальный ФВЧ (частотный)')
        results.append(self.ideal_highpass_spatial())
        titles.append('Идеальный ФВЧ (пространственный)')

        # ФВЧ Баттерворта
        results.append(self.butterworth_highpass_frequency(15, 2))
        titles.append('ФВЧ Баттерворта (частотный)')

        # Гауссов ФВЧ
        results.append(self.gaussian_highpass_frequency(15))
        titles.append('Гауссов ФВЧ (частотный)')

        # Лапласиан
        results.append(self.laplacian_frequency())
        titles.append('Лапласиан (частотный)')
        results.append(self.laplacian_spatial())
        titles.append('Лапласиан (пространственный)')

        # Нерезкое маскирование
        results.append(self.unsharp_masking_frequency(3, 1.5))
        titles.append('Нерезкое маскирование (частотный)')
        results.append(self.unsharp_masking_spatial(3, 1.5))
        titles.append('Нерезкое маскирование (пространственный)')

        # Гомоморфная фильтрация
        results.append(self.homomorphic_filtering())
        titles.append('Гомоморфная фильтрация')

        # 3. ИЗБИРАТЕЛЬНАЯ ФИЛЬТРАЦИЯ
        # Режекторный фильтр
        results.append(self.notch_reject_filter_frequency([(30, 30), (-30, -30)]))
        titles.append('Режекторный фильтр (частотный)')

        # Полосовой фильтр
        results.append(self.bandpass_filter_frequency(20, 60))
        titles.append('Полосовой фильтр (частотный)')

        # Узкополосный фильтр
        results.append(self.narrowband_filter_frequency(40, 5))
        titles.append('Узкополосный фильтр (частотный)')

        # Визуализация
        fig, axes = plt.subplots(6, 4, figsize=(60, 80))
        axes = axes.ravel()

        for i, (result, title) in enumerate(zip(results, titles)):
            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(title, fontsize=25)
            axes[i].axis('off')

        # Скрываем пустые subplots
        for i in range(len(results), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

# Запуск сравнения
if __name__ == "__main__":
    comparator = FrequencySpatialComparison('img4.jpg')
    comparator.visualize_comparison()

# @title
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def fft_shifted(img):
    """Прямое FFT с центрированием"""
    f = np.fft.fft2(img.astype(np.float64))
    return np.fft.fftshift(f)

def ifft_shifted(fshift):
    """Обратное FFT с нормализацией"""
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

def normalize_image(img):
    """Нормализация к [0, 255] и uint8"""
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# ========== ЗАГРУЗКА ИЗОБРАЖЕНИЯ ==========
img = cv2.imread('img2.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Файл 'img3.jpg' не найден → создаём тестовое изображение")
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 200, -1)
    cv2.rectangle(img, (60, 60), (120, 120), 150, -1)
    cv2.line(img, (20, 200), (230, 200), 100, 3)
    cv2.putText(img, 'TEST', (90, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

# ========== ПРЕОБРАЗОВАНИЕ В ЧАСТОТНУЮ ОБЛАСТЬ ==========
F = fft_shifted(img)
magnitude_spectrum = np.log(1 + np.abs(F))

# ========== A. НИЗКОЧАСТОТНАЯ ФИЛЬТРАЦИЯ ==========
def ideal_lowpass(shape, D0):
    P, Q = shape
    u, v = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((u - Q/2)**2 + (v - P/2)**2)
    return (D <= D0).astype(float)

def butterworth_lowpass(shape, D0, n=2):
    P, Q = shape
    u, v = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((u - Q/2)**2 + (v - P/2)**2)
    return 1 / (1 + (D / D0)**(2*n))

def gaussian_lowpass(shape, D0):
    P, Q = shape
    u, v = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((u - Q/2)**2 + (v - P/2)**2)
    return np.exp(-(D**2) / (2 * D0**2))

print("✅ Применяем ФНЧ фильтры...")
D0 = 30
H_ideal = ideal_lowpass(img.shape, D0)
H_butter = butterworth_lowpass(img.shape, D0, n=2)
H_gauss = gaussian_lowpass(img.shape, D0)

G_ideal = F * H_ideal
G_butter = F * H_butter
G_gauss = F * H_gauss

img_ideal_lp = normalize_image(ifft_shifted(G_ideal))
img_butter_lp = normalize_image(ifft_shifted(G_butter))
img_gauss_lp = normalize_image(ifft_shifted(G_gauss))

plt.figure(figsize=(15, 5))
plt.subplot(151), plt.imshow(img, cmap='gray'), plt.title('Оригинал')
plt.subplot(152), plt.imshow(img_ideal_lp, cmap='gray'), plt.title('Идеальный ФНЧ')
plt.subplot(153), plt.imshow(img_butter_lp, cmap='gray'), plt.title('Баттерворт ФНЧ')
plt.subplot(154), plt.imshow(img_gauss_lp, cmap='gray'), plt.title('Гаусс ФНЧ')
plt.subplot(155), plt.imshow(H_gauss, cmap='hot'), plt.title('Маска Гаусса')
plt.tight_layout()
plt.show()

# ========== B. ВЫСОКОЧАСТОТНАЯ ФИЛЬТРАЦИЯ ==========
def ideal_highpass(shape, D0):
    return 1 - ideal_lowpass(shape, D0)

def butterworth_highpass(shape, D0, n=2):
    return 1 - butterworth_lowpass(shape, D0, n)

def gaussian_highpass(shape, D0):
    return 1 - gaussian_lowpass(shape, D0)

def laplacian_filter(shape):
    P, Q = shape
    u = np.fft.fftfreq(Q).reshape(1, -1) * Q
    v = np.fft.fftfreq(P).reshape(-1, 1) * P
    H = -4 * np.pi**2 * (u**2 + v**2)
    H = H / (np.max(np.abs(H)) + 1e-8)  # нормализация
    return H

def unsharp_masking(shape, D0, k=1.0, filter_type='gaussian'):
    if filter_type == 'gaussian':
        H_low = gaussian_lowpass(shape, D0)
    elif filter_type == 'butterworth':
        H_low = butterworth_lowpass(shape, D0, n=2)
    else:
        H_low = ideal_lowpass(shape, D0)
    return 1 + k * (1 - H_low)

print("✅ Применяем ФВЧ фильтры...")
D0 = 30
H_ideal_hp = ideal_highpass(img.shape, D0)
H_butter_hp = butterworth_highpass(img.shape, D0)
H_gauss_hp = gaussian_highpass(img.shape, D0)
H_laplacian = laplacian_filter(img.shape)
H_unsharp = unsharp_masking(img.shape, D0, k=1.2, filter_type='gaussian')

# Применяем все фильтры
G_ideal_hp = F * H_ideal_hp
G_butter_hp = F * H_butter_hp
G_gauss_hp = F * H_gauss_hp
G_lap = F * H_laplacian
G_unsharp = F * H_unsharp

img_ideal_hp = normalize_image(ifft_shifted(G_ideal_hp))
img_butter_hp = normalize_image(ifft_shifted(G_butter_hp))
img_gauss_hp = normalize_image(ifft_shifted(G_gauss_hp))
img_lap = normalize_image(ifft_shifted(G_lap))
img_unsharp = normalize_image(ifft_shifted(G_unsharp))

# Визуализация: 2 строки × 5 столбцов = 10 изображений
plt.figure(figsize=(20, 8))
filters = [
    ("Идеальный ФВЧ", img_ideal_hp, H_ideal_hp),
    ("Баттерворт ФВЧ", img_butter_hp, H_butter_hp),
    ("Гаусс ФВЧ", img_gauss_hp, H_gauss_hp),
    ("Лапласиан", img_lap, H_laplacian),
    ("Нерезкое маск.", img_unsharp, H_unsharp),
]

for i, (title, img_out, mask) in enumerate(filters):
    # Результат фильтрации
    plt.subplot(2, 5, i + 1)
    plt.imshow(img_out, cmap='gray')
    plt.title(f'{title}\n(результат)')
    plt.axis('off')

    # Маска фильтра
    plt.subplot(2, 5, i + 6)
    plt.imshow(mask, cmap='hot')
    plt.title(f'{title}\n(маска)')
    plt.axis('off')

plt.suptitle('Высокочастотные фильтры: результаты и частотные маски', fontsize=14)
plt.tight_layout()
plt.show()

# ========== C. УЗКОПОЛОСНЫЕ ФИЛЬТРЫ (ИСПРАВЛЕНО!) ==========
def narrow_band_filter(shape, center_freq, bandwidth=0.05, filter_type='gaussian'):
    P, Q = shape
    u = np.fft.fftfreq(Q).reshape(1, -1)
    v = np.fft.fftfreq(P).reshape(-1, 1)
    u0, v0 = center_freq

    if filter_type == 'gaussian':
        D_pos = np.sqrt((u - u0)**2 + (v - v0)**2)
        D_neg = np.sqrt((u + u0)**2 + (v + v0)**2)
        H = np.exp(-(D_pos**2)/(2*(bandwidth/2)**2)) + np.exp(-(D_neg**2)/(2*(bandwidth/2)**2))
    elif filter_type == 'ideal':
        D_pos = np.sqrt((u - u0)**2 + (v - v0)**2)
        D_neg = np.sqrt((u + u0)**2 + (v + v0)**2)
        H = ((D_pos <= bandwidth) | (D_neg <= bandwidth)).astype(float)
    elif filter_type == 'butterworth':
        D_pos = np.sqrt((u - u0)**2 + (v - v0)**2)
        D_neg = np.sqrt((u + u0)**2 + (v + v0)**2)
        H_pos = 1 / (1 + (D_pos/(bandwidth/2))**4)
        H_neg = 1 / (1 + (D_neg/(bandwidth/2))**4)
        H = (H_pos + H_neg) / 2
    else:
        H = np.zeros(shape)
    return H

print("✅ Применяем УЗКОПОЛОСНЫЕ фильтры (исправлено!)...")
center_freqs = [(0.1, 0.1), (0.3, 0.2)]
bandwidth = 0.02
filter_types = ['gaussian', 'ideal', 'butterworth']

# Сетка: 4 строки × 3 столбца = 12 subplot'ов (2 центра × 2: маска+результат × 3 типа)
plt.figure(figsize=(16, 10))
rows, cols = 4, 3

for i, center_freq in enumerate(center_freqs):
    for j, filter_type in enumerate(filter_types):
        H_narrow = narrow_band_filter(img.shape, center_freq, bandwidth, filter_type)
        G_narrow = F * H_narrow
        img_narrow = normalize_image(ifft_shifted(G_narrow))

        # Маска: строки 0 и 2 (i=0 → строка 0; i=1 → строка 2)
        plt.subplot(rows, cols, (2*i) * cols + j + 1)
        plt.imshow(H_narrow, cmap='hot')
        plt.title(f'Центр: ({center_freq[0]:.1f}, {center_freq[1]:.1f})\n{filter_type}')
        plt.axis('off')

        # Результат: строки 1 и 3
        plt.subplot(rows, cols, (2*i + 1) * cols + j + 1)
        plt.imshow(img_narrow, cmap='gray')
        plt.title(f'Результат {filter_type}')
        plt.axis('off')

plt.suptitle('Узкополосные фильтры: маски и результаты', fontsize=14)
plt.tight_layout()
plt.show()

# ========== C2. ВЫДЕЛЕНИЕ ТЕКСТУР С ПОМОЩЬЮ УЗКОПОЛОСНЫХ ФИЛЬТРОВ ==========
print("✅ Выделение текстур...")
x, y = np.meshgrid(np.arange(256), np.arange(256))
pattern1 = 100 * (np.sin(2*np.pi*0.05*x) > 0)
pattern2 = 80 * (np.sin(2*np.pi*0.1*(x+y)) > 0)
pattern3 = 60 * (np.sin(2*np.pi*0.2*x) * np.sin(2*np.pi*0.15*y) > 0)
test_img = np.clip(pattern1 + pattern2 + pattern3, 0, 255).astype(np.uint8)

F_test = fft_shifted(test_img)
freqs = [(0.05, 0.0), (0.07, 0.07), (0.2, 0.15)]
descs = ['Вертикальные\nполосы', 'Диагональные\nполосы', 'Мелкая\nтекстура']

# Сетка: 2 строки × (1 + 3×2) = 2×7 = 14 → слишком много
# Лучше: 3 строки × 3 столбца = 9 ячеек
plt.figure(figsize=(14, 10))
rows, cols = 3, 3

# 1. Исходное изображение
plt.subplot(rows, cols, 1)
plt.imshow(test_img, cmap='gray')
plt.title('Тестовое изображение')
plt.axis('off')

# 2. Амплитудный спектр
plt.subplot(rows, cols, 2)
plt.imshow(np.log(1 + np.abs(F_test)), cmap='hot')
plt.title('Амплитудный спектр')
plt.axis('off')

# 3–8. Маски и результаты (по 2 на текстуру)
for i, (freq, desc) in enumerate(zip(freqs, descs)):
    H = narrow_band_filter(test_img.shape, freq, bandwidth=0.015, filter_type='gaussian')
    G = F_test * H
    img_out = normalize_image(ifft_shifted(G))

    # Маска — 3,4,5
    plt.subplot(rows, cols, 3 + i)
    plt.imshow(H, cmap='hot')
    plt.title(f'Маска: {desc}')
    plt.axis('off')

    # Результат — 6,7,8
    plt.subplot(rows, cols, 6 + i)
    plt.imshow(img_out, cmap='gray')
    plt.title(f'Выделено: {desc}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# ========== D. РЕЖЕКТОРНЫЙ И ПОЛОСОВОЙ ФИЛЬТРЫ ==========
def notch_reject_filter(shape, centers, D0):
    H = np.ones(shape)
    P, Q = shape
    u = np.arange(Q) - Q//2
    v = np.arange(P) - P//2
    U, V = np.meshgrid(u, v)

    for (u0, v0) in centers:
        for du, dv in [(u0, v0), (-u0, v0), (u0, -v0), (-u0, -v0)]:
            D = np.sqrt((U - du)**2 + (V - dv)**2)
            H *= (1 - np.exp(-(D**2)/(2*D0**2)))
    return H

def bandpass_filter(shape, D0_low, D0_high, filter_type='gaussian'):
    if filter_type == 'gaussian':
        return gaussian_lowpass(shape, D0_high) * gaussian_highpass(shape, D0_low)
    else:
        return ideal_lowpass(shape, D0_high) * ideal_highpass(shape, D0_low)

print("✅ Применяем режекторный и полосовой фильтры...")
H_notch = notch_reject_filter(img.shape, centers=[(40, 30)], D0=8)
G_notch = F * H_notch
img_notch = normalize_image(ifft_shifted(G_notch))

H_bp = bandpass_filter(img.shape, D0_low=20, D0_high=50)
G_bp = F * H_bp
img_bp = normalize_image(ifft_shifted(G_bp))

plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Оригинал')
plt.subplot(232), plt.imshow(H_notch, cmap='hot'), plt.title('Режекторная маска')
plt.subplot(233), plt.imshow(img_notch, cmap='gray'), plt.title('После режекторного')
plt.subplot(234), plt.imshow(H_bp, cmap='hot'), plt.title('Полосовая маска')
plt.subplot(235), plt.imshow(img_bp, cmap='gray'), plt.title('После полосового')
plt.subplot(236), plt.imshow(magnitude_spectrum, cmap='hot'), plt.title('АЧХ (оригинал)')
plt.tight_layout()
plt.show()

# ========== E. ГОМОМОРФНАЯ ФИЛЬТРАЦИЯ ==========
def homomorphic_filter(img, d0=30, gamma_l=0.3, gamma_h=1.8):
    img_f = img.astype(np.float64) + 1e-6
    img_log = np.log(img_f)
    F_log = fft_shifted(img_log)

    P, Q = img.shape
    u, v = np.meshgrid(np.arange(Q), np.arange(P))
    D = np.sqrt((u - Q/2)**2 + (v - P/2)**2)
    H = gamma_l + (gamma_h - gamma_l) * (1 - np.exp(-0.05 * (D**2 / d0**2)))

    G = F_log * H
    g = ifft_shifted(G)
    img_out = np.exp(g)
    return normalize_image(img_out)

print("✅ Применяем гомоморфную фильтрацию...")
img_hom = homomorphic_filter(img, d0=25, gamma_l=0.4, gamma_h=2.0)

plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Оригинал')
plt.subplot(132), plt.imshow(img_hom, cmap='gray'), plt.title('Гомоморфная')
plt.subplot(133), plt.imshow(cv2.equalizeHist(img_hom), cmap='gray'), plt.title('Гомоморфная + эквализация')
plt.tight_layout()
plt.show()

print("🎉 Все фильтры успешно применены!")

# @title
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. Создаём тестовое изображение с разными периодическими структурами ==========
H, W = 256, 256
x, y = np.meshgrid(np.arange(W), np.arange(H))

# Три компоненты:
img_vert   = 100 * (np.sin(2 * np.pi * 0.08 * x) > 0).astype(np.float32)      # вертикальные полосы (частота ~0.08 по x)
img_diag   = 80 * (np.sin(2 * np.pi * 0.06 * (x + y)) > 0).astype(np.float32) # диагональные (~0.06 по (x+y))
img_noise  = 30 * np.random.randn(H, W)                                      # шум

img = np.clip(img_vert + img_diag + img_noise, 0, 255).astype(np.uint8)

# ========== 2. Частотная область ==========
def fft_shifted(img):
    f = np.fft.fft2(img.astype(np.float64))
    return np.fft.fftshift(f)

def ifft_shifted(F):
    f = np.fft.ifftshift(F)
    img = np.fft.ifft2(f)
    return np.abs(img)

F = fft_shifted(img)
mag = np.log(1 + np.abs(F))

# ========== 3. Узкополосный фильтр (гауссов) ==========
def narrow_bandpass_gaussian(shape, center_freq, bandwidth):
    """
    Гауссов узкополосный фильтр (band-pass) для выделения частоты center_freq.
    Учитывает симметрию спектра (±f).
    """
    P, Q = shape
    # Частотные координаты в нормированных единицах [-0.5, 0.5)
    u = np.fft.fftfreq(Q).reshape(1, -1)   # [0, 1/Q, ..., 0.5, -0.5+1/Q, ..., -1/Q]
    v = np.fft.fftfreq(P).reshape(-1, 1)

    u0, v0 = center_freq  # например, (0.08, 0.0) — вертикальные полосы

    # Расстояние до +f и -f
    D_plus  = np.sqrt((u - u0)**2 + (v - v0)**2)
    D_minus = np.sqrt((u + u0)**2 + (v + v0)**2)

    # Гауссовы "колокола" вокруг +f и -f
    sigma = bandwidth / 2
    H_plus  = np.exp(-D_plus**2 / (2 * sigma**2))
    H_minus = np.exp(-D_minus**2 / (2 * sigma**2))

    return H_plus + H_minus

# Частоты, соответствующие нашим структурам:
freqs = [
    (0.08, 0.00),   # вертикальные полосы → пик на (±0.08, 0)
    (0.042, 0.042), # диагональные (~0.06 по sqrt(2)) → (±0.042, ±0.042)
]

bandwidth = 0.02  # узкая полоса

# ========== 4. Применяем фильтры и визуализируем ==========
plt.figure(figsize=(16, 12))

# Исходное изображение и спектр
plt.subplot(3, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Исходное изображение\n(полосы + шум)')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(mag, cmap='hot')
plt.title('Амплитудный спектр\n(видны пики на частотах)')
plt.axis('off')

# Для каждой частоты — маска и результат
for i, (u0, v0) in enumerate(freqs):
    # Создаём фильтр
    H = narrow_bandpass_gaussian(img.shape, (u0, v0), bandwidth)

    # Применяем
    G = F * H
    img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(G)))
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)

    # Маска фильтра (увеличенный фрагмент для наглядности)
    plt.subplot(3, 4, 3 + i)
    # Увеличим центр спектра
    cx, cy = H.shape[1]//2, H.shape[0]//2
    zoom = 40
    H_zoom = H[cy-zoom:cy+zoom, cx-zoom:cx+zoom]
    plt.imshow(H_zoom, cmap='hot', extent=[-zoom, zoom, -zoom, zoom])
    plt.title(f'Маска узкополосного\nфильтра\nцентр: ({u0:.2f}, {v0:.2f})')
    plt.xlabel('u (частота x)')
    plt.ylabel('v (частота y)')

    # Результат фильтрации
    plt.subplot(3, 4, 7 + i)
    plt.imshow(img_filtered, cmap='gray')
    desc = "вертикальные полосы" if i == 0 else "диагональные полосы"
    plt.title(f'Выделено: {desc}')
    plt.axis('off')

# Дополнительно: суммарный фильтр (обе частоты)
H_total = narrow_bandpass_gaussian(img.shape, freqs[0], bandwidth) + \
          narrow_bandpass_gaussian(img.shape, freqs[1], bandwidth)
G_total = F * H_total
img_total = np.clip(np.abs(np.fft.ifft2(np.fft.ifftshift(G_total))), 0, 255).astype(np.uint8)

plt.subplot(3, 4, 5)
H_zoom_total = H_total[cy-zoom:cy+zoom, cx-zoom:cx+zoom]
plt.imshow(H_zoom_total, cmap='hot', extent=[-zoom, zoom, -zoom, zoom])
plt.title('Маска: обе частоты')
plt.xlabel('u')
plt.ylabel('v')

plt.subplot(3, 4, 9)
plt.imshow(img_total, cmap='gray')
plt.title('Выделены обе структуры')
plt.axis('off')

# Шум, выделенный как высокочастотный (для контраста)
H_high = 1 - narrow_bandpass_gaussian(img.shape, (0.0, 0.0), 0.1)  # подавляем НЧ
G_high = F * H_high
img_high = np.clip(np.abs(np.fft.ifft2(np.fft.ifftshift(G_high))), 0, 255).astype(np.uint8)
plt.subplot(3, 4, 10)
plt.imshow(img_high, cmap='gray')
plt.title('Высокочастотный шум')
plt.axis('off')

plt.tight_layout()
plt.suptitle('Узкополосные фильтры: выделение конкретных частот\n'
             '→ Видно, как каждая структура соответствует пику в спектре', fontsize=14, y=0.98)
plt.show()

print("✅ Узкополосные фильтры успешно применены!")
print("💡 Совет: попробуйте изменить 'bandwidth' (0.01 — у́же, 0.05 — шире) и посмотрите на результат!")

# @title
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d

class FrequencySpatialComparison:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, 0)
        if self.image is None:
            # Создаем тестовое изображение если файл не найден
            self.image = self.create_test_image()
        self.image = cv2.resize(self.image, (256, 256))
        self.image_float = self.image.astype(np.float32)

    def create_test_image(self):
        """Создает тестовое изображение с различными features"""
        img = np.ones((300, 300)) * 128
        cv2.rectangle(img, (50, 50), (100, 100), 200, -1)
        cv2.rectangle(img, (150, 150), (200, 200), 50, -1)
        cv2.circle(img, (200, 80), 25, 180, -1)

        # Добавляем шум и текстуру
        noise = np.random.normal(0, 15, img.shape)
        img = img + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    # 1. СГЛАЖИВАНИЕ (НИЗКОЧАСТОТНАЯ ФИЛЬТРАЦИЯ)

    def ideal_lowpass_frequency(self, cutoff):
        """Идеальный ФНЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        # Создаем маску идеального ФНЧ
        mask = np.zeros((rows, cols))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
        mask[mask_area] = 1

        # Применяем в частотной области
        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def ideal_lowpass_spatial(self, size):
        """Идеальный ФНЧ в пространственной области (усредняющий фильтр)"""
        kernel = np.ones((size, size)) / (size * size)
        return convolve2d(self.image_float, kernel, mode='same', boundary='symm')

    def butterworth_lowpass_frequency(self, cutoff, order=2):
        """ФНЧ Баттерворта в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Фильтр Баттерворта
        mask = 1 / (1 + (d / cutoff)**(2 * order))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def butterworth_lowpass_spatial(self, sigma):
        """Аналог ФНЧ Баттерворта в пространственной области (Гауссов фильтр)"""
        return ndimage.gaussian_filter(self.image_float, sigma=sigma)

    def gaussian_lowpass_frequency(self, sigma):
        """Гауссов ФНЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Гауссов фильтр
        mask = np.exp(-(d**2) / (2 * sigma**2))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def gaussian_lowpass_spatial(self, sigma):
        """Гауссов ФНЧ в пространственной области"""
        return ndimage.gaussian_filter(self.image_float, sigma=sigma)

    # 2. ПОВЫШЕНИЕ РЕЗКОСТИ (ВЫСОКОЧАСТОТНАЯ ФИЛЬТРАЦИЯ)

    def ideal_highpass_frequency(self, cutoff):
        """Идеальный ФВЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        mask = np.ones((rows, cols))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
        mask[mask_area] = 0

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def ideal_highpass_spatial(self):
        """Идеальный ФВЧ в пространственной области (лапласиан)"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        filtered = convolve2d(self.image_float, kernel, mode='same', boundary='symm')
        return self.image_float + filtered

    def butterworth_highpass_frequency(self, cutoff, order=2):
        """ФВЧ Баттерворта в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        mask = 1 / (1 + (cutoff / d)**(2 * order))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def gaussian_highpass_frequency(self, sigma):
        """Гауссов ФВЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        mask = 1 - np.exp(-(d**2) / (2 * sigma**2))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def laplacian_frequency(self):
        """Лапласиан в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)

        # Лапласиан в частотной области: -(u^2 + v^2)
        mask = -(u**2 + v**2)

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def laplacian_spatial(self):
        """Лапласиан в пространственной области"""
        kernel = np.array([[0, -1, 0],
                          [-1,  4, -1],
                          [0, -1, 0]])
        return convolve2d(self.image_float, kernel, mode='same', boundary='symm')

    def unsharp_masking_frequency(self, sigma, alpha=1.5):
        """Нерезкое маскирование в частотной области"""
        # Размытая версия (ФНЧ)
        blurred = self.gaussian_lowpass_frequency(sigma)

        # Маска = оригинал - размытая версия
        mask = self.image_float - blurred

        # Усиленная версия = оригинал + alpha * маска
        sharpened = self.image_float + alpha * mask

        return sharpened

    def unsharp_masking_spatial(self, sigma, alpha=1.5):
        """Нерезкое маскирование в пространственной области"""
        blurred = ndimage.gaussian_filter(self.image_float, sigma=sigma)
        mask = self.image_float - blurred
        sharpened = self.image_float + alpha * mask
        return sharpened

    def homomorphic_filtering(self, gamma_low=0.5, gamma_high=2.0, cutoff=30, order=1):
        """Гомоморфная фильтрация"""
        # Логарифм изображения
        img_log = np.log(self.image_float + 1)

        # ДПФ
        dft = np.fft.fft2(img_log)
        dft_shift = np.fft.fftshift(dft)

        # Создаем гомоморфный фильтр
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2
        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Фильтр гомоморфной фильтрации
        H = (gamma_high - gamma_low) * (1 - np.exp(-(d**2) / (2 * cutoff**2))) + gamma_low

        # Применяем фильтр
        filtered_dft = dft_shift * H
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        # Экспонента для возврата к исходному диапазону
        result = np.exp(np.real(idft)) - 1

        return np.clip(result, 0, 255)

    # 3. ИЗБИРАТЕЛЬНАЯ ФИЛЬТРАЦИЯ

    def notch_reject_filter_frequency(self, frequencies, bandwidth=5):
        """Режекторный фильтр в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        mask = np.ones((rows, cols))

        for freq in frequencies:
            u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
            d = np.sqrt((u - freq[0])**2 + (v - freq[1])**2)
            d2 = np.sqrt((u + freq[0])**2 + (v + freq[1])**2)

            # Подавляем частоты в заданной полосе
            mask[d <= bandwidth] = 0
            mask[d2 <= bandwidth] = 0

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def bandpass_filter_frequency(self, low_cutoff, high_cutoff):
        """Полосовой фильтр в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Полосовой фильтр
        mask = (d >= low_cutoff) & (d <= high_cutoff)
        mask = mask.astype(float)

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def narrowband_filter_frequency(self, center_freq, bandwidth=2):
        """Узкополосный фильтр в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Узкополосный фильтр
        mask = (d >= center_freq - bandwidth) & (d <= center_freq + bandwidth)
        mask = mask.astype(float)

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        return np.real(idft)

    def visualize_comparison(self):
        """Визуализация сравнения всех методов"""
        results = []
        titles = []

        # Исходное изображение
        results.append(self.image)
        titles.append('Исходное изображение')
        results.append(self.image)
        titles.append('Исходное изображение')

        # 1. СГЛАЖИВАНИЕ
        # Идеальный ФНЧ
        results.append(self.ideal_lowpass_frequency(30))
        titles.append('Идеальный ФНЧ (частотный) 30')
        results.append(self.ideal_lowpass_frequency(60))
        titles.append('Идеальный ФНЧ (частотный) 60')
        results.append(self.ideal_lowpass_frequency(120))
        titles.append('Идеальный ФНЧ (частотный) 120')
        results.append(self.ideal_lowpass_spatial(7))
        titles.append('Идеальный ФНЧ (пространственный)')

        # ФНЧ Баттерворта
        results.append(self.butterworth_lowpass_frequency(30, 2))
        titles.append('ФНЧ Баттерворта (частотный)')
        results.append(self.butterworth_lowpass_spatial(2))
        titles.append('ФНЧ Баттерворта (пространственный)')

        # Гауссов ФНЧ
        results.append(self.gaussian_lowpass_frequency(30))
        titles.append('Гауссов ФНЧ (частотный)')
        results.append(self.gaussian_lowpass_spatial(2))
        titles.append('Гауссов ФНЧ (пространственный)')

        # 2. ПОВЫШЕНИЕ РЕЗКОСТИ
        # Идеальный ФВЧ
        results.append(self.ideal_highpass_frequency(15))
        titles.append('Идеальный ФВЧ (частотный)')
        results.append(self.ideal_highpass_spatial())
        titles.append('Идеальный ФВЧ (пространственный)')

        # ФВЧ Баттерворта
        results.append(self.butterworth_highpass_frequency(15, 2))
        titles.append('ФВЧ Баттерворта (частотный)')

        # Гауссов ФВЧ
        results.append(self.gaussian_highpass_frequency(15))
        titles.append('Гауссов ФВЧ (частотный)')

        # Лапласиан
        results.append(self.laplacian_frequency())
        titles.append('Лапласиан (частотный)')
        results.append(self.laplacian_spatial())
        titles.append('Лапласиан (пространственный)')

        # Нерезкое маскирование
        results.append(self.unsharp_masking_frequency(3, 1.5))
        titles.append('Нерезкое маскирование (частотный)')
        results.append(self.unsharp_masking_spatial(3, 1.5))
        titles.append('Нерезкое маскирование (пространственный)')

        # Гомоморфная фильтрация
        results.append(self.homomorphic_filtering())
        titles.append('Гомоморфная фильтрация')

        # 3. ИЗБИРАТЕЛЬНАЯ ФИЛЬТРАЦИЯ
        # Режекторный фильтр
        results.append(self.notch_reject_filter_frequency([(30, 30), (-30, -30)]))
        titles.append('Режекторный фильтр (частотный)')

        # Полосовой фильтр
        results.append(self.bandpass_filter_frequency(20, 60))
        titles.append('Полосовой фильтр (частотный)')

        # Узкополосный фильтр
        results.append(self.narrowband_filter_frequency(40, 5))
        titles.append('Узкополосный фильтр (частотный)')

        # Визуализация
        fig, axes = plt.subplots(6, 4, figsize=(60, 80))
        axes = axes.ravel()

        for i, (result, title) in enumerate(zip(results, titles)):
            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(title, fontsize=25)
            axes[i].axis('off')

        # Скрываем пустые subplots
        for i in range(len(results), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

# Запуск сравнения
if __name__ == "__main__":
    comparator = FrequencySpatialComparison('img4.jpg')
    comparator.visualize_comparison()

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d

class FrequencySpatialComparison:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, 0)
        if self.image is None:
            # Создаем тестовое изображение если файл не найден
            self.image = self.create_test_image()
        self.image = cv2.resize(self.image, (256, 256))
        self.image_float = self.image.astype(np.float32)

    def create_test_image(self):
        """Создает тестовое изображение с различными features"""
        img = np.ones((300, 300)) * 128
        cv2.rectangle(img, (50, 50), (100, 100), 200, -1)
        cv2.rectangle(img, (150, 150), (200, 200), 50, -1)
        cv2.circle(img, (200, 80), 25, 180, -1)

        # Добавляем шум и текстуру
        noise = np.random.normal(0, 15, img.shape)
        img = img + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    def get_spectrum(self, image):
        """Вычисляет и возвращает спектр изображения"""
        dft = np.fft.fft2(image)
        dft_shift = np.fft.fftshift(dft)
        spectrum = np.log(np.abs(dft_shift) + 1)
        return spectrum

    def visualize_spectrum(self, spectrum, title, ax):
        """Визуализирует спектр на заданной оси"""
        im = ax.imshow(spectrum, cmap='jet')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        return im

    # 1. СГЛАЖИВАНИЕ (НИЗКОЧАСТОТНАЯ ФИЛЬТРАЦИЯ)

    def ideal_lowpass_frequency(self, cutoff, return_spectrum=False):
        """Идеальный ФНЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        # Создаем маску идеального ФНЧ
        mask = np.zeros((rows, cols))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
        mask[mask_area] = 1

        # Применяем в частотной области
        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.real(idft), spectrum, mask
        return np.real(idft)

    def ideal_lowpass_spatial(self, size):
        """Идеальный ФНЧ в пространственной области (усредняющий фильтр)"""
        kernel = np.ones((size, size)) / (size * size)
        return convolve2d(self.image_float, kernel, mode='same', boundary='symm')

    def butterworth_lowpass_frequency(self, cutoff, order=2, return_spectrum=False):
        """ФНЧ Баттерворта в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Фильтр Баттерворта
        mask = 1 / (1 + (d / cutoff)**(2 * order))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.real(idft), spectrum, mask
        return np.real(idft)

    def butterworth_lowpass_spatial(self, sigma):
        """Аналог ФНЧ Баттерворта в пространственной области (Гауссов фильтр)"""
        return ndimage.gaussian_filter(self.image_float, sigma=sigma)

    def gaussian_lowpass_frequency(self, sigma, return_spectrum=False):
        """Гауссов ФНЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Гауссов фильтр
        mask = np.exp(-(d**2) / (2 * sigma**2))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.real(idft), spectrum, mask
        return np.real(idft)

    def gaussian_lowpass_spatial(self, sigma):
        """Гауссов ФНЧ в пространственной области"""
        return ndimage.gaussian_filter(self.image_float, sigma=sigma)

    # 2. ПОВЫШЕНИЕ РЕЗКОСТИ (ВЫСОКОЧАСТОТНАЯ ФИЛЬТРАЦИЯ)

    def ideal_highpass_frequency(self, cutoff, return_spectrum=False):
        """Идеальный ФВЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        mask = np.ones((rows, cols))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= cutoff**2
        mask[mask_area] = 0

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.real(idft), spectrum, mask
        return np.real(idft)

    def ideal_highpass_spatial(self):
        """Идеальный ФВЧ в пространственной области (лапласиан)"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        filtered = convolve2d(self.image_float, kernel, mode='same', boundary='symm')
        return self.image_float + filtered

    def butterworth_highpass_frequency(self, cutoff, order=2, return_spectrum=False):
        """ФВЧ Баттерворта в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        mask = 1 / (1 + (cutoff / d)**(2 * order))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.real(idft), spectrum, mask
        return np.real(idft)

    def gaussian_highpass_frequency(self, sigma, return_spectrum=False):
        """Гауссов ФВЧ в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        mask = 1 - np.exp(-(d**2) / (2 * sigma**2))

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.real(idft), spectrum, mask
        return np.real(idft)

    def laplacian_frequency(self, return_spectrum=False):
        """Лапласиан в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)

        # Лапласиан в частотной области: -(u^2 + v^2)
        mask = -(u**2 + v**2)

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.real(idft), spectrum, mask
        return np.real(idft)

    def laplacian_spatial(self):
        """Лапласиан в пространственной области"""
        kernel = np.array([[0, -1, 0],
                          [-1,  4, -1],
                          [0, -1, 0]])
        return convolve2d(self.image_float, kernel, mode='same', boundary='symm')

    def unsharp_masking_frequency(self, sigma, alpha=1.5, return_spectrum=False):
        """Нерезкое маскирование в частотной области"""
        # Размытая версия (ФНЧ)
        blurred = self.gaussian_lowpass_frequency(sigma)

        # Маска = оригинал - размытая версия
        mask = self.image_float - blurred

        # Усиленная версия = оригинал + alpha * маска
        sharpened = self.image_float + alpha * mask

        return sharpened

    def unsharp_masking_spatial(self, sigma, alpha=1.5):
        """Нерезкое маскирование в пространственной области"""
        blurred = ndimage.gaussian_filter(self.image_float, sigma=sigma)
        mask = self.image_float - blurred
        sharpened = self.image_float + alpha * mask
        return sharpened

    def homomorphic_filtering(self, gamma_low=0.5, gamma_high=2.0, cutoff=30, order=1, return_spectrum=False):
        """Гомоморфная фильтрация"""
        # Логарифм изображения
        img_log = np.log(self.image_float + 1)

        # ДПФ
        dft = np.fft.fft2(img_log)
        dft_shift = np.fft.fftshift(dft)

        # Создаем гомоморфный фильтр
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2
        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Фильтр гомоморфной фильтрации
        H = (gamma_high - gamma_low) * (1 - np.exp(-(d**2) / (2 * cutoff**2))) + gamma_low

        # Применяем фильтр
        filtered_dft = dft_shift * H
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        # Экспонента для возврата к исходному диапазону
        result = np.exp(np.real(idft)) - 1

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.clip(result, 0, 255), spectrum, H
        return np.clip(result, 0, 255)

    # 3. ИЗБИРАТЕЛЬНАЯ ФИЛЬТРАЦИЯ

    def notch_reject_filter_frequency(self, frequencies, bandwidth=5, return_spectrum=False):
        """Режекторный фильтр в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        mask = np.ones((rows, cols))

        for freq in frequencies:
            u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
            d = np.sqrt((u - freq[0])**2 + (v - freq[1])**2)
            d2 = np.sqrt((u + freq[0])**2 + (v + freq[1])**2)

            # Подавляем частоты в заданной полосе
            mask[d <= bandwidth] = 0
            mask[d2 <= bandwidth] = 0

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.real(idft), spectrum, mask
        return np.real(idft)

    def bandpass_filter_frequency(self, low_cutoff, high_cutoff, return_spectrum=False):
        """Полосовой фильтр в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Полосовой фильтр
        mask = (d >= low_cutoff) & (d <= high_cutoff)
        mask = mask.astype(float)

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.real(idft), spectrum, mask
        return np.real(idft)

    def narrowband_filter_frequency(self, center_freq, bandwidth=2, return_spectrum=False):
        """Узкополосный фильтр в частотной области"""
        rows, cols = self.image.shape
        crow, ccol = rows//2, cols//2

        u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        d = np.sqrt(u**2 + v**2)

        # Узкополосный фильтр
        mask = (d >= center_freq - bandwidth) & (d <= center_freq + bandwidth)
        mask = mask.astype(float)

        dft = np.fft.fft2(self.image_float)
        dft_shift = np.fft.fftshift(dft)
        filtered_dft = dft_shift * mask
        idft = np.fft.ifft2(np.fft.ifftshift(filtered_dft))

        if return_spectrum:
            spectrum = np.log(np.abs(filtered_dft) + 1)
            return np.real(idft), spectrum, mask
        return np.real(idft)

    def visualize_spectra_comparison(self):
        """Визуализация спектров для частотных методов"""
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.ravel()

        # Спектр исходного изображения
        original_spectrum = self.get_spectrum(self.image_float)
        im = self.visualize_spectrum(original_spectrum, 'Исходный спектр', axes[0])

        # 1. ФНЧ фильтры
        # Идеальный ФНЧ
        _, spectrum_ideal, mask_ideal = self.ideal_lowpass_frequency(30, return_spectrum=True)
        self.visualize_spectrum(spectrum_ideal, 'Идеальный ФНЧ спектр', axes[1])
        self.visualize_spectrum(mask_ideal, 'Маска идеального ФНЧ', axes[2])

        # ФНЧ Баттерворта
        _, spectrum_butter, mask_butter = self.butterworth_lowpass_frequency(30, 2, return_spectrum=True)
        self.visualize_spectrum(spectrum_butter, 'ФНЧ Баттерворта спектр', axes[3])
        self.visualize_spectrum(mask_butter, 'Маска Баттерворта', axes[4])

        # Гауссов ФНЧ
        _, spectrum_gauss, mask_gauss = self.gaussian_lowpass_frequency(30, return_spectrum=True)
        self.visualize_spectrum(spectrum_gauss, 'Гауссов ФНЧ спектр', axes[5])
        self.visualize_spectrum(mask_gauss, 'Маска Гауссова ФНЧ', axes[6])

        # 2. ФВЧ фильтры
        # Идеальный ФВЧ
        _, spectrum_ideal_hp, mask_ideal_hp = self.ideal_highpass_frequency(15, return_spectrum=True)
        self.visualize_spectrum(spectrum_ideal_hp, 'Идеальный ФВЧ спектр', axes[7])
        self.visualize_spectrum(mask_ideal_hp, 'Маска идеального ФВЧ', axes[8])

        # ФВЧ Баттерворта
        _, spectrum_butter_hp, mask_butter_hp = self.butterworth_highpass_frequency(15, 2, return_spectrum=True)
        self.visualize_spectrum(spectrum_butter_hp, 'ФВЧ Баттерворта спектр', axes[9])
        self.visualize_spectrum(mask_butter_hp, 'Маска Баттерворта ФВЧ', axes[10])

        # Лапласиан
        _, spectrum_laplace, mask_laplace = self.laplacian_frequency(return_spectrum=True)
        self.visualize_spectrum(spectrum_laplace, 'Лапласиан спектр', axes[11])
        self.visualize_spectrum(mask_laplace, 'Маска лапласиана', axes[12])

        # Гомоморфная фильтрация
        _, spectrum_homo, mask_homo = self.homomorphic_filtering(return_spectrum=True)
        self.visualize_spectrum(spectrum_homo, 'Гомоморфный спектр', axes[13])
        self.visualize_spectrum(mask_homo, 'Гомоморфный фильтр', axes[14])

        # Скрываем пустые subplots
        for i in range(15, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_comparison(self):
        """Визуализация сравнения всех методов"""
        results = []
        titles = []

        # Исходное изображение
        results.append(self.image)
        titles.append('Исходное изображение')
        results.append(self.image)
        titles.append('Исходное изображение')

        # 1. СГЛАЖИВАНИЕ
        # Идеальный ФНЧ
        results.append(self.ideal_lowpass_frequency(30))
        titles.append('Идеальный ФНЧ (частотный) 30')
        results.append(self.ideal_lowpass_frequency(60))
        titles.append('Идеальный ФНЧ (частотный) 60')
        results.append(self.ideal_lowpass_frequency(120))
        titles.append('Идеальный ФНЧ (частотный) 120')
        results.append(self.ideal_lowpass_spatial(7))
        titles.append('Идеальный ФНЧ (пространственный)')

        # ФНЧ Баттерворта
        results.append(self.butterworth_lowpass_frequency(120, 2))
        titles.append('ФНЧ Баттерворта (частотный) 120')
        results.append(self.butterworth_lowpass_spatial(2))
        titles.append('ФНЧ Баттерворта (пространственный)')

        # Гауссов ФНЧ
        results.append(self.gaussian_lowpass_frequency(120))
        titles.append('Гауссов ФНЧ (частотный) 120')
        results.append(self.gaussian_lowpass_spatial(2))
        titles.append('Гауссов ФНЧ (пространственный)')

        # 2. ПОВЫШЕНИЕ РЕЗКОСТИ
        # Идеальный ФВЧ
        results.append(self.ideal_highpass_frequency(15))
        titles.append('Идеальный ФВЧ (частотный) 15')
        results.append(self.ideal_highpass_spatial())
        titles.append('Идеальный ФВЧ (пространственный)')

        # ФВЧ Баттерворта
        results.append(self.butterworth_highpass_frequency(15, 2))
        titles.append('ФВЧ Баттерворта (частотный) 15')

        # Гауссов ФВЧ
        results.append(self.gaussian_highpass_frequency(15))
        titles.append('Гауссов ФВЧ (частотный) 15')

        # Лапласиан
        results.append(self.laplacian_frequency())
        titles.append('Лапласиан (частотный)')
        results.append(self.laplacian_spatial())
        titles.append('Лапласиан (пространственный)')

        # Нерезкое маскирование
        results.append(self.unsharp_masking_frequency(3, 1.5))
        titles.append('Нерезкое маскирование (частотный)')
        results.append(self.unsharp_masking_spatial(3, 1.5))
        titles.append('Нерезкое маскирование (пространственный)')

        # Гомоморфная фильтрация
        results.append(self.homomorphic_filtering())
        titles.append('Гомоморфная фильтрация')

        # 3. ИЗБИРАТЕЛЬНАЯ ФИЛЬТРАЦИЯ
        # Режекторный фильтр
        results.append(self.notch_reject_filter_frequency([(30, 30), (-30, -30)]))
        titles.append('Режекторный фильтр (частотный)')

        # Полосовой фильтр
        results.append(self.bandpass_filter_frequency(20, 60))
        titles.append('Полосовой фильтр (частотный)')

        # Узкополосный фильтр
        results.append(self.narrowband_filter_frequency(40, 5))
        titles.append('Узкополосный фильтр (частотный)')

        # Визуализация
        fig, axes = plt.subplots(6, 4, figsize=(60, 80))
        axes = axes.ravel()

        for i, (result, title) in enumerate(zip(results, titles)):
            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(title, fontsize=25)
            axes[i].axis('off')

        # Скрываем пустые subplots
        for i in range(len(results), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

# Запуск сравнения
if __name__ == "__main__":
    comparator = FrequencySpatialComparison('img4.jpg')

    # Визуализация спектров
    print("Визуализация спектров...")
    comparator.visualize_spectra_comparison()

    # Визуализация сравнения фильтров
    print("Визуализация сравнения фильтров...")
    comparator.visualize_comparison()