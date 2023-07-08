from .base_model import BaseModel

import tensorflow as tf
from keras import Model
from keras.layers import (
    Input,
    BatchNormalization,
    concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense)


class UNet(BaseModel):
    def __init__(self, batch_normalization=True):
        """Inicializamos o modelo da Unet
        """
        super().__init__()
        self.batch_normalization = batch_normalization
        self.name = "U-Net"

    def __encoderLayers(self, inputs, n_filters, dropout_prob=0, max_pooling=True):
        # Add 2 Conv2D Layers with relu activation and HeNormal
        layer = Conv2D(n_filters, kernel_size=3, activation='relu',
                       padding='same', kernel_initializer='HeNormal')(inputs)
        layer = Conv2D(n_filters, kernel_size=3, activation='relu',
                       padding='same', kernel_initializer='HeNormal')(layer)

        # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
        if self.batch_normalization:
            layer = BatchNormalization()(layer)

        # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
        if dropout_prob > 0:
            layer = Dropout(dropout_prob)(layer)

        # Pooling reduces the size of the image while keeping the number of channels same
        # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
        # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
        if max_pooling:
            next_layer = MaxPooling2D(pool_size=(2, 2))(layer)
        else:
            next_layer = layer

        # Skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions
        skip_connection = layer

        return next_layer, skip_connection

    def __decoderLayers(self, input, skip_connection, n_filters):

        # Start with a transpose convolution layer to first increase the size of the image
        layer = Conv2DTranspose(n_filters, kernel_size=(
            3, 3), strides=(2, 2), padding='same')(input)

        # Merge the skip connection from previous block to prevent information loss
        layer = concatenate([layer, skip_connection], axis=3)

        # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
        layer = Conv2D(n_filters, kernel_size=3, activation='relu',
                       padding='same', kernel_initializer='HeNormal')(layer)
        layer = Conv2D(n_filters, kernel_size=3, activation='relu',
                       padding='same', kernel_initializer='HeNormal')(layer)

        return layer

    def __finalLayers(self, input, n_filters, n_classes):

        layer = Conv2D(n_filters, kernel_size=3, activation='relu',
                       padding='same', kernel_initializer='he_normal')(input)

        # Aqui diferimos da Unet original, usamos uma Global Average Pooling 2D para gerar uma classificar a imagem como um todo
        final_layer = GlobalAveragePooling2D()(layer)
        final_layer = Dense(n_classes, activation='softmax')(final_layer)

        return final_layer

    def build(self, input_shape, n_filters, n_classes, summary=False):
        """Gera um modelo da UNet

        Args:
            input_shape (tuple): O formato dos dados de entrada
            n_filters (int): O número de filtros das camadas convolucionais
            n_classes (ints): Número de classes
            summary (bool): Mostrar descrição ao finalizar

        Return:
            model (keras.Model): O modelo gerado
        """
        inputs = Input(input_shape)

        # Encoder Section
        layer, skip_connection1 = self.__encoderLayers(inputs, n_filters)
        layer, skip_connection2 = self.__encoderLayers(layer, n_filters*2)
        layer, skip_connection3 = self.__encoderLayers(layer, n_filters*4)
        layer, skip_connection4 = self.__encoderLayers(
            layer, n_filters*8, dropout_prob=0.3)
        layer, _ = self.__encoderLayers(
            layer, n_filters*16, dropout_prob=0.3, max_pooling=False)

        # Decoder Section
        layer = self.__decoderLayers(layer, skip_connection4, n_filters*8)
        layer = self.__decoderLayers(layer, skip_connection3, n_filters*4)
        layer = self.__decoderLayers(layer, skip_connection2, n_filters*2)
        layer = self.__decoderLayers(layer, skip_connection1, n_filters)

        # Final Section
        layer = self.__finalLayers(layer, n_filters, n_classes)

        self.model = Model(inputs=inputs, outputs=layer, name=self.name)

        if (summary):
            self.model.summary()

        return self.model

    def compile(self, **kwargs):
        """Compila o  modelo
        Args:
            kwargs (**kwargs): Qualquer argumento do modelo
        """
        self.model.compile(**kwargs)

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
        return super().train(*args, **kwargs)

    def predict(self, x, **kwargs):
        """Prevê os valores usando o BaseModel

        Args:
            x (ndarray): O dado de teste
            kwargs (**kwargs): Qualquer argumento do modelo

        Return:
            pred (ArrayLike): A predição do modelo
            
        """
        return self.model.predict(x, **kwargs)
    
    def get_recommended_hyperparameters(self):
        """Retorna as recomendações de hiperparâmetros do modelo

        Return:
            param (dict): parâmetros
            
        """
        return {"n_filters" : [32, 64]}

