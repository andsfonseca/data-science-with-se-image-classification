import tensorflow as tf

class TensorflowUtils():

    @staticmethod
    def create_tfdataset(x, y, n_classes, batch_size):
        """Cria um objeto tf.data.Dataset para treinamento do dado com one-hot encoded labels.

        Args:
            x (ndarray): Um tensor contendo os dados de treinamento
            y (ndarray): Um tensor contendo os rótulos de treinamento
            n_classes (int): Um inteiro especificando o número de classes
            batch_size (int): Um inteiro especificando o tamanho do lote

        Returns:
            dataset (tf.data.Dataset): Um objeto que produz lotes de dados de treinamento com rótulos codificados one-hot
        
        """

        # Definimos uma função geradora para converter rótulos em vetores one-hot
        def generator():
            for sample, label in zip(x, y):
                yield sample, tf.one_hot(label, depth=n_classes)

        # Definimos as formas de resultado de cada conjunto de dados
        output_shapes = (
            x.shape[1:],
            (n_classes)
        )

        # Defina os tipos de resultado de cada conjunto de dados
        output_types = (
            x.dtype,
            tf.float32
        )

        # Criamos conjunto de dados a partir do gerador
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=output_shapes[0], dtype=output_types[0]),
                tf.TensorSpec(shape=output_shapes[1], dtype=output_types[1])
            )
        )

        # Aplicamos o batch
        dataset = dataset.batch(batch_size)
        return dataset