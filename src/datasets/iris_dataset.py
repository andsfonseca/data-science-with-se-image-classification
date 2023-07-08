from .dataset import DatasetLoader
import os
import cv2
import numpy as np

class IrisDataset(DatasetLoader):

    def __init__(self, path):
        """Carrega o dataset da Iris a partir de um caminho informado

        Args:
            path (str): Caminho do dataset

        """
        super().__init__()

        self.path = path
        self.label_names = ["setosa", "versicolour", "virginica"]
        self.raw = self.__load_raw_dataset(self.label_names)
    
    def get_x(self):
        """Retorna as features do dataset

        Return:
            x (ArrayLike): Array de features

        """
        return np.concatenate(list(self.raw.values()))

    def get_y(self):
        """Retorna as labels do dataset

        Return:
            y (ArrayLike): Array de labels

        """
        y = []
        index = 0
        for label in self.raw:
            y.append(np.full(self.raw[label].shape[0], index))
            index += 1
        return np.concatenate(y)
    
    def __load_raw_dataset(self, label_names):
        """Possui os métodos para carregar as imagens do dataset da Iris no formato apropriado.

        Args:
            label_names (list[str]): Conjunto de labels

        Return:
            raw (ArrayLike): Dataset de imagens

        """
        raw = {}

        for label_name in label_names:
            folder = f"{self.path}/iris-{label_name}"

            raw[label_name] = []

            for filename in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if img is not None:
                    raw[label_name].append(img)

            raw[label_name] = np.array(raw[label_name])

        return raw
    
