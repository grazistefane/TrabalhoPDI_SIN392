import numpy as np
from PIL import Image
from scipy.ndimage import (uniform_filter, median_filter, gaussian_filter,
                          maximum_filter, minimum_filter, convolve)
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
from skimage import exposure, morphology, filters
from skimage.morphology import erosion, dilation, square
from skimage.filters import threshold_otsu


class ImageOperations:
    @staticmethod
    def apply_otsu(image): #aplica limiarização de Otsu para binarização da imagem
        img_array = np.array(image)
        threshold = threshold_otsu(img_array)
        binary_img = img_array > threshold
        return Image.fromarray((binary_img * 255).astype(np.uint8)), threshold

    @staticmethod
    def contrast_stretching(image): #realiza estiramento de contraste usando percentis 2% e 98%
        img_array = np.array(image)
        p2, p98 = np.percentile(img_array, (2, 98))
        img_rescale = exposure.rescale_intensity(img_array, in_range=(p2, p98))
        return Image.fromarray(img_rescale)

    @staticmethod
    def histogram_equalization(image): #equaliza o histograma da imagem para melhorar contraste
        img_array = np.array(image)
        img_eq = exposure.equalize_hist(img_array)
        img_eq = (255 * img_eq).astype(np.uint8)
        return Image.fromarray(img_eq)

    @staticmethod
    def apply_filter(image, filter_type): #aplica filtros espaciais (passa-baixa ou passa-alta)
        img_array = np.array(image)
        
        if filter_type in ['mean', 'median', 'gaussian', 'max', 'min']:
            filtered_img = ImageOperations._apply_lowpass_filter(img_array, filter_type)
        elif filter_type in ['laplacian', 'roberts', 'prewitt', 'sobel']:
            filtered_img = ImageOperations._apply_highpass_filter(img_array, filter_type)
        else:
            raise ValueError("Filtro desconhecido")
        
        filtered_img = ImageOperations._normalize_image(filtered_img)
        return Image.fromarray(filtered_img)

    @staticmethod
    def _apply_lowpass_filter(img_array, filter_type): #aplica filtros passa-baixa (suavização)
        if filter_type == 'mean':
            return uniform_filter(img_array, size=3)
        elif filter_type == 'median':
            return median_filter(img_array, size=3)
        elif filter_type == 'gaussian':
            return gaussian_filter(img_array, sigma=1)
        elif filter_type == 'max':
            return maximum_filter(img_array, size=3)
        elif filter_type == 'min':
            return minimum_filter(img_array, size=3)

    @staticmethod
    def _apply_highpass_filter(img_array, filter_type): #aplica filtros passa-alta (detecção de bordas)
        if filter_type == 'laplacian':
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            return convolve(img_array, kernel)
        
        elif filter_type == 'roberts':
            kernel_x = np.array([[1, 0], [0, -1]])
            kernel_y = np.array([[0, 1], [-1, 0]])
            gx = convolve(img_array, kernel_x)
            gy = convolve(img_array, kernel_y)
            return np.sqrt(gx**2 + gy**2)
        
        elif filter_type == 'prewitt':
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            gx = convolve(img_array, kernel_x)
            gy = convolve(img_array, kernel_y)
            return np.sqrt(gx**2 + gy**2)
        
        elif filter_type == 'sobel':
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            gx = convolve(img_array, kernel_x)
            gy = convolve(img_array, kernel_y)
            return np.sqrt(gx**2 + gy**2)

    @staticmethod
    def frequency_filter(image, filter_type): #aplica filtros no domínio da frequência (ideal ou gaussiano)
        img_array = np.array(image)
        
        #transformada de Fourier
        f_transform = fft2(img_array)
        f_shift = fftshift(f_transform) #desloca as baixas frequências para o centro
        
        rows, cols = img_array.shape
        crow, ccol = rows // 2, cols // 2 #centro da imagem
        mask = np.zeros((rows, cols), np.float32) #máscara de filtro
        
        if 'low' in filter_type: #filtro passa-baixa
            if 'ideal' in filter_type: #filtro ideal
                radius = min(rows, cols) // 4
                cv2.circle(mask, (ccol, crow), radius, 1, -1)
            else: #filtro gaussiano
                sigma = min(rows, cols) // 6
                y, x = np.ogrid[:rows, :cols]
                mask = np.exp(-((x - ccol)**2 + (y - crow)**2) / (2 * sigma**2)) #cria mascara circular
        else: #filtro passa-alta
            if 'ideal' in filter_type:
                radius = min(rows, cols) // 4
                cv2.circle(mask, (ccol, crow), radius, 1, -1) #inverte a mascara
                mask = 1 - mask
            else: 
                sigma = min(rows, cols) // 6
                y, x = np.ogrid[:rows, :cols]
                mask = 1 - np.exp(-((x - ccol)**2 + (y - crow)**2) / (2 * sigma**2)) 
        
        #aplica o filtro e transformada inversa
        f_filtered = f_shift * mask
        f_ishift = ifftshift(f_filtered)
        img_back = np.abs(ifft2(f_ishift)) #transformada inversa
        
        return Image.fromarray(ImageOperations._normalize_image(img_back))

    @staticmethod
    def apply_morphology(image, operation): #aplica operações morfológicas em imagens binárias
        img_array = np.array(image)
        
        if img_array.dtype != bool:
            threshold = threshold_otsu(img_array)
            binary_img = img_array > threshold
        else:
            binary_img = img_array
            
        footprint = square(3) #elemento estruturante 3x3 (embora o VS falou que square tá desatualizado (?))
        
        if operation == 'erosion':
            result = erosion(binary_img, footprint)
        elif operation == 'dilation':
            result = dilation(binary_img, footprint)
        elif operation == 'opening':
            result = morphology.opening(binary_img, footprint)
        elif operation == 'closing':
            result = morphology.closing(binary_img, footprint)
        else:
            raise ValueError("Operação desconhecida")
        
        return Image.fromarray((result * 255).astype(np.uint8))

    @staticmethod
    def _normalize_image(img_array):
        img_array = img_array - img_array.min()
        if img_array.max() > 0:
            img_array = img_array / img_array.max() * 255
        return img_array.astype(np.uint8)

    @staticmethod
    def calculate_histogram(image):
        img_array = np.array(image)
        
        hist, _ = np.histogram(img_array.flatten(), bins=256, range=[0,256])
        return hist

    @staticmethod
    def calculate_fourier_spectrum(image): #calcula o espectro de Fourier da imagem (magnitude logarítmica)
        img_array = np.array(image)
        f = np.fft.fft2(img_array)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        magnitude_spectrum = ImageOperations._normalize_image(magnitude_spectrum)
        return Image.fromarray(magnitude_spectrum)