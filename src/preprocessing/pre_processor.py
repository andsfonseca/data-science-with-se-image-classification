import numpy as np
from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod
from ..datasets.dataset import DatasetLoader

class PreProcessor(ABC):
    
    def __init__(self, dataset: DatasetLoader):
        """Inicializa o PreProcessador tendo como entrada um DatasetLoader
        Args:
            dataset (DatasetLoader): Classe responsável por carregar os dados
        """
        self.x = self.__transform_x(dataset.get_x())
        self.y = self.__transform_y(dataset.get_y())
        pass
        
    def __transform_x(self, x):
        """Aplica uma transformação em x
        Args:
            x (ArrayLike): Valor original de x
        Return:
            x (ArrayLike): Valores transformados
        """
        return x
    
    def __transform_y(self, y):
        """Aplica uma transformação em y
        Args:
            y (ArrayLike): Valor original de y
        Return:
            y (ArrayLike): Valores transformados
        """
        return y
    
    def get_x(self):
        """Retorna as features transformadas
        Return:
            x (ArrayLike): features
        """
        return self.x

    def get_y(self):
        """Retorna as labels transformadas
        Return:
            y (ArrayLike): labels
        """
        return self.y

    def train_test_valid_split(self, test_size=0.2, valid_size=0.1, random_state = 42):
        """Realiza a divisão de treino/teste/validação
        Args:
            test_size (float): Porcentagem de divisão do teste sobre o dataset completo. Valor entre 0 e 1
            valid_size (float): Porcentagem de divisão da validação sobre o dataset de treino. Valor entre 0 e 1
            random_state (int): Estado randômico do dataset
        Return:
            dataset (tuple): Retorna x_train, x_test, x_valid, y_train, y_test, y_valid respectivamente
        """
        indexes = np.arange(self.x.shape[0])

        indexes_train, indexes_test = train_test_split(
            indexes, test_size=test_size, random_state=random_state)
        indexes_train, indexes_valid = train_test_split(
            indexes_train, test_size=valid_size, random_state=random_state)
        
        return self.x[indexes_train], self.x[indexes_test], self.x[indexes_valid], self.y[indexes_train], self.y[indexes_test], self.y[indexes_valid]