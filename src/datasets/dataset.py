from abc import ABC, abstractmethod

class DatasetLoader (ABC):

    def __init__(self):
        """Inicializa um DatasetLoader

        """
        self.raw = None
        self.label_names = []
        pass
    
    def get_raw_dataset(self):
        """Retorna o dataset completo e sem tratamento

        Return:
            raw (any): dataset

        """
        return self.raw
    
    def get_label_names(self):
        """Retorna as labels do dataset

        Return:
            label_names (any): as labels do dataset

        """
        return self.label_names
    
    @abstractmethod
    def get_x(self):
        """Retorna as features do dataset

        Return:
            x (ArrayLike): Array de features

        """
        return []

    @abstractmethod
    def get_y(self):
        """Retorna as labels do dataset

        Return:
            y (ArrayLike): Array de labels

        """
        return []
    