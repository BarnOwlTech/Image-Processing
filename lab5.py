import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2  # Центр для сдвига

# Вычисление DFT и сдвиг
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Вычисление спектра
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

def create_filters(rows, cols, crow, ccol, D0=120, n=2):
    # Создание сетки частот
    u = np.arange(cols)
    v = np.arange(rows)
    u, v = np.meshgrid(u, v)
    D = np.sqrt((u - ccol)**2 + (v - crow)**2)

    # Идеальный low-pass фильтр
    ideal_lp = np.zeros((rows, cols), np.float32)
    ideal_lp[D <= D0] = 1

    # Баттерворт low-pass фильтр
    butterworth_lp = 1 / (1 + (D / D0)**(2*n))

    # Гауссиан low-pass фильтр
    gaussian_lp = np.exp(-(D**2) / (2 * D0**2))

    # High-pass ф
    ideal_hp = 1 - ideal_lp
    butterworth_hp = 1 - butterworth_lp
    gaussian_hp = 1 - gaussian_lp

    return ideal_lp, butterworth_lp, gaussian_lp, ideal_hp, butterworth_hp, gaussian_hp

ideal_lp, butterworth_lp, gaussian_lp, ideal_hp, butterworth_hp, gaussian_hp = create_filters(rows, cols, crow, ccol)

def apply_and_show_filter(dft_shift, filter_mask, title_filter, title_spectrum, title_image):
    filtered_dft = dft_shift * filter_mask[:, :, np.newaxis]

    # Спектр после фильтрации
    filtered_magnitude = 20 * np.log(cv2.magnitude(filtered_dft[:, :, 0], filtered_dft[:, :, 1]))

    # Обратный DFT
    idft_shift = np.fft.ifftshift(filtered_dft)
    img_back = cv2.idft(idft_shift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])


    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(filter_mask, cmap='gray'), plt.title(title_filter)
    plt.subplot(132), plt.imshow(filtered_magnitude, cmap='gray'), plt.title(title_spectrum)
    plt.subplot(133), plt.imshow(img_back, cmap='gray'), plt.title(title_image)
    plt.show()


plt.figure(figsize=(8, 4))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Исходное изображение')
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Спектр изображения')
plt.show()


print("Low-pass фильтры (сглаживание):")
apply_and_show_filter(dft_shift, ideal_lp, 'Идеальный LP фильтр', 'Спектр после идеального LP', 'Изображение после идеального LP')
apply_and_show_filter(dft_shift, butterworth_lp, 'Баттерворт LP фильтр', 'Спектр после Баттерворт LP', 'Изображение после Баттерворт LP')
apply_and_show_filter(dft_shift, gaussian_lp, 'Гауссиан LP фильтр', 'Спектр после Гауссиан LP', 'Изображение после Гауссиан LP')


print("High-pass фильтры (повышение резкости):")
apply_and_show_filter(dft_shift, ideal_hp, 'Идеальный HP фильтр', 'Спектр после идеального HP', 'Изображение после идеального HP')
apply_and_show_filter(dft_shift, butterworth_hp, 'Баттерворт HP фильтр', 'Спектр после Баттерворт HP', 'Изображение после Баттерворт HP')
apply_and_show_filter(dft_shift, gaussian_hp, 'Гауссиан HP фильтр', 'Спектр после Гауссиан HP', 'Изображение после Гауссиан HP')