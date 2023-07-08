from ..datasets.dataset import DatasetLoader
from .pre_processor import PreProcessor

import cv2
import numpy as np

class ImagePreProcessor(PreProcessor):

    def __init__(self, dataset: DatasetLoader, image_shape = (64, 64)):
        """Inicializa o PreProcessador de imagem
        Args:
            dataset (DatasetLoader): Classe responsável por carregar os dados
            image_shape (tuple): Shape das imagens de saída
        """
        super().__init__(dataset)
        self.x = self.__transform_x(dataset.get_x(), image_shape)
    
    def __transform_x(self, x, image_shape):
        """Aplica uma transformação nas imagens em X. Redimensiona as imagens e normaliza os canais
        Args:
            x (ArrayLike): Valor original de x
        Return:
            x (ArrayLike): Valores transformados
        """
        resized_x = []
        for image in x:
            resized_x.append(cv2.resize(image, image_shape, interpolation = cv2.INTER_AREA))
        resized_x = np.array(resized_x)

        normalized_x = (resized_x - resized_x.min()) / (resized_x.max() - resized_x.min())
        return normalized_x