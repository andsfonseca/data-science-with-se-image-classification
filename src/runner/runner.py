from ..datasets.dataset import DatasetLoader
from ..preprocessing.pre_processor import PreProcessor
from ..models.base_model import BaseModel
from ..models.unet import UNet
from ..evaluation.evaluator import Evaluator

from ..tensorflow_utils import TensorflowUtils
from ..plots import Plot

import itertools
import tensorflow as tf
import numpy as np
import json
import pandas as pd

class Runner:

    def __init__(self, dataset : DatasetLoader, preprocessor : PreProcessor, model: BaseModel, evaluator: Evaluator):

        self.dataset = dataset
        self.preprocessor = preprocessor
        self.model = model
        self.evaluator = evaluator

        x_train, x_test, x_valid, y_train, y_test, y_valid = preprocessor.train_test_valid_split()

        self.label_names = dataset.get_label_names()
        self.n_classes = len (self.label_names)

        self.sample_shape = x_train[0].shape

        self.x_test = x_test
        self.y_test = y_test

        self.train_dataset = TensorflowUtils.create_tfdataset(
                    x=x_train,
                    y=y_train,
                    n_classes=self.n_classes,
                    batch_size=16)

        self.valid_dataset = TensorflowUtils.create_tfdataset(
                            x=x_valid,
                            y=y_valid,
                            n_classes=self.n_classes,
                            batch_size=16)

    def run(self):
        """ Possui todos os conjuntos de métodos para excecutar o pipeline de treinamento do modelo.
            Responsável por gerar as variações de hiperparâmetros e a execução de cada uma delas.
        """

        all_results = []
        model_parameters = self.model.get_recommended_hyperparameters()

        for parameters in self.generate_parameters(model_parameters):

            if(type(self.model) == UNet):
                self.model.build(self.sample_shape, n_filters=parameters["n_filters"], n_classes=self.n_classes, summary=False)

            if parameters["optimizer"] == "Adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=parameters["learning_rate"])
            elif parameters["optimizer"] == "SGD":
                optimizer = tf.keras.optimizers.SGD(learning_rate=parameters["learning_rate"])
            else:
                continue

            loss = tf.keras.losses.CategoricalCrossentropy()

            callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=30,
                        verbose=2,
                        mode='min',
                        baseline=None,
                        restore_best_weights=True)
                ]

            self.model.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=['accuracy'])

            # Treinamento (Internamente, estamos salvando o melhor modelo do dataset e criando uma instância do tensorboard)
            history = self.model.train(self.train_dataset,
                        batch_size=16,
                        epochs=parameters["epochs"],
                        validation_data=self.valid_dataset,
                        callbacks=callbacks)
            
            # with tf.device('/cpu:0'):
            y_pred = self.model.predict(self.x_test)
            y_pred = np.argmax(y_pred, axis=1)
            
            # Salvamos os gráficos de treinamento
            Plot.save_linear_chart(history.history["loss"], history.history["val_loss"], "loss", f"{self.model.training_dir}/loss.png")
            Plot.save_linear_chart(history.history["accuracy"], history.history["val_accuracy"], "accuracy", f"{self.model.training_dir}/accuracy.png")

            # Salvamos a matrix de confusão
            cm = self.evaluator.get_confusion_matrix(self.y_test, y_pred)
            Plot.save_confusion_matrix(cm, self.label_names, f"{self.model.training_dir}/confusion_matrix.png")

            # Salvamos as metricas
            self.evaluator.save(self.y_test, y_pred, f"{self.model.training_dir}/evaluation.txt")

            # Salvamos os parametros
            with open(f"{self.model.training_dir}/parameters.json", 'a') as file:
                file.write(json.dumps(parameters))

            # Criamos um lista de dicionários para um csv com os resultados
            results = parameters.copy()

            results["name"] = self.model.name
            results["accuracy"] = self.evaluator.get_accuracy(self.y_test, y_pred)
            results["precision"] = self.evaluator.get_precision(self.y_test, y_pred)
            results["recall"] = self.evaluator.get_recall(self.y_test, y_pred)
            results["f1_score"] = self.evaluator.get_f1_score(self.y_test, y_pred)

            all_results.append(results)

        return pd.DataFrame(all_results)

    
    def generate_parameters(self, model_parameters):
        """Gera uma combinação de parâmetros a partir de uma lista de parâmetros.

        Args:
            model_parameters (dict): Substitua os parâmetros padrão pelos padrões enviados

        Returns:
            all_params (list[dict]): Uma lista de combinações de cada parâmetro
        """

        compile_parameters = {
            "optimizer" : ["Adam", "SGD"],
            "learning_rate" : [0.001, 0.0001], 
            "epochs" : [150]
        }

        parameters = model_parameters.copy()
        parameters.update(compile_parameters)

        all_params = []

        keys = list(parameters)
        for values in itertools.product(*map(parameters.get, keys)):
            all_params.append(dict(zip(keys, values)))

        return all_params


        