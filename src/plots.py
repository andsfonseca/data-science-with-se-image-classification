from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Plot:

    @staticmethod
    def save_confusion_matrix(confusion_matrix, labels, path):
        """Salva a matriz de confusão

        Args:
            confusion_matrix (ArrayLike): Matriz de confusão
            labels (list[str]): Nome das labels
            path (str): Defina o caminho do arquivo de saída
        """
        _, ax = plt.subplots(figsize=(11, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
        disp.plot(cmap="Blues", ax=ax, colorbar=False,
            xticks_rotation=90, values_format='.2f')

        plt.title("Normalized confusion Matrix")
        plt.savefig(path)
        plt.close()

    @staticmethod
    def save_linear_chart(train_values, val_values, metric_text: str, path="."):
        """Gera um gráfico de linha a partir de duas métricas

        Args:
            train_values (ArrayLike): Um array de métricas do treino
            val_values (ArrayLike): Um array de métricas de validação
            metric_text (str): A label da métrica
            path (str): Defina o caminho do arquivo de saída
        """
        plt.plot(train_values, color='r', label=f'train {metric_text}')
        plt.plot(val_values, color='b', label=f'val {metric_text}')
        plt.title(f'Comparison: {metric_text}')
        plt.legend()
        plt.savefig(path)
        plt.close()
