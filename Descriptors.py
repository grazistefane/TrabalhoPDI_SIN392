import numpy as np
import cv2
import skimage.feature
from skimage import measure

class Descriptors:
    @staticmethod
    def calculate_intensity_stats(image): #calcula estatísticas de intensidade da imagem
        img_array = np.array(image)
        return {
            'mean': np.mean(img_array),
            'std': np.std(img_array),
            'median': np.median(img_array),
            'min': np.min(img_array),
            'max': np.max(img_array),
            'energy': np.sum(img_array**2),
            'entropy': measure.shannon_entropy(img_array)
        }

    @staticmethod
    def calculate_haralick_features(image): #calcula características de textura de Haralick usando GLCM
        img_array = np.array(image)
        glcm = skimage.feature.graycomatrix(
            img_array, 
            distances=[1], 
            angles=[0], 
            levels=256,
            symmetric=True, 
            normed=True
        )
        
        return {
            'contrast': skimage.feature.graycoprops(glcm, 'contrast')[0, 0],
            'dissimilarity': skimage.feature.graycoprops(glcm, 'dissimilarity')[0, 0],
            'homogeneity': skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0],
            'energy': skimage.feature.graycoprops(glcm, 'energy')[0, 0],
            'correlation': skimage.feature.graycoprops(glcm, 'correlation')[0, 0],
            'asm': skimage.feature.graycoprops(glcm, 'ASM')[0, 0]
        }

    @staticmethod
    def calculate_shape_moments(image): #calcula momentos de forma da imagem (binarizada)
        img_array = np.array(image)
        _, binary_img = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        moments = cv2.moments(binary_img)
        hu_moments = cv2.HuMoments(moments)
        
        return {
            'spatial_moments': {
                'm00': moments['m00'],
                'm10': moments['m10'],
                'm01': moments['m01'],
                'm20': moments['m20'],
                'm11': moments['m11'],
                'm02': moments['m02']
            },
            'central_moments': {
                'mu20': moments['mu20'],
                'mu11': moments['mu11'],
                'mu02': moments['mu02'],
                'mu30': moments['mu30'],
                'mu21': moments['mu21'],
                'mu12': moments['mu12'],
                'mu03': moments['mu03']
            },
            'hu_moments': [hu_moments[i][0] for i in range(7)]
        }