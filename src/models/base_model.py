from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

from keras.callbacks import TensorBoard, ModelCheckpoint

class BaseModel(ABC):
    """ Classe abstrata de um modelo. Possui uma interface mínima para construir, compilar, treinar e prever um modelo.
    """

    def __init__(self):
        """Inicializa um BaseModel
        """
        super().__init__()

        self.name = "BaseModel"
        self.training_dir = ""

    @abstractmethod
    def build(self, input_shape, summary=False):
        """Compila um modelo de BaseModel.

        Args:
            input_shape (tupla): Shape da amostra 
            summary (bool): Mostra o sumário do modelo

        Return:
            model (keras.Model): The compiled model

        """
        return None

    def load(self, run_name: str, partial=True):
        """Carrega os pesos do modelo a partir de um treinamento salvo.

        Args:
            run_name (str): Nome do treinamento
            partial (bool): Permite carregamento parcial

        """
        if self.model is not None:
            self.training_dir = f"models/{self.model.name}/{run_name}"
            status = self.model.load_weights(
                Path(f"{self.training_dir}/weights_best.h5").absolute())
            if partial:
                status.expect_partial()

    @abstractmethod
    def compile(self, **kwargs):
        """Compila o  modelo
        Args:
            kwargs (**kwargs): Qualquer argumento do modelo
        """
        return

    @abstractmethod
    def train(self, *args, **kwargs):
        """Treina um modelo de BaseModel.

        Args:
            enable_tensorboard (bool): Ativa o tensoboard interno (Default: True)
            enable_save_best (bool): Ativa o model checkpoint interno (Default: True)
            args (*args): Argumentos do modelo
            kwargs (**kwargs): Qualquer argumento do modelo

        Return:
            history (Any): Retorna o histórico do treinamento
        """
        if 'enable_tensorboard' in kwargs:
            enable_tensorboard = kwargs['enable_tensorboard']
        else:
            enable_tensorboard = True
        
        if 'enable_save_best' in kwargs:
            enable_save_best = kwargs['enable_save_best']
        else:
            enable_save_best = True

        if self.model is not None:
            
            now = datetime.now()
            self.training_dir = f"models/{self.model.name}/{now.strftime('run_%Y-%m-%d_%H-%M-%S')}"

            # Already callbacks
            if 'callbacks' in kwargs:
                callbacks = kwargs["callbacks"]
            else:
                callbacks = []

            if enable_tensorboard:

                tensoboard_dir = Path(f"{self.training_dir}/logs")

                callback = TensorBoard(log_dir=tensoboard_dir.absolute(), histogram_freq=1, write_graph=True,
                                       write_images=False, update_freq='epoch')

                if not any(type(c) == TensorBoard for c in callbacks):
                    callbacks.append(callback)

            if enable_save_best:
                # Best Only
                callback = ModelCheckpoint(f"{self.training_dir}/weights_best.h5",
                                        monitor='val_loss',
                                        save_best_only=True,
                                        mode='max',
                                        save_weights_only=True)

                callbacks.append(callback)

            kwargs = dict(kwargs, callbacks=callbacks)

            kwargs.pop("enable_save_best", None)

            return self.model.fit(*args, **kwargs)
        return None

    @abstractmethod
    def predict(self, x, **kwargs):
        """Prevê os valores usando o BaseModel

        Args:
            x (ndarray): O dado de teste
            kwargs (**kwargs): Qualquer argumento do modelo

        Return:
            pred (ArrayLike): A predição do modelo
            
        """
        return self.model.predict(x, **kwargs)

    @abstractmethod
    def get_recommended_hyperparameters(self):
        """Retorna as recomendações de hiperparâmetros do modelo

        Return:
            param (dict): parâmetros
            
        """
        return {}
