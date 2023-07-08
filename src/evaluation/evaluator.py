from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score

class Evaluator():
    def __init__(self,):
        pass

    def get_confusion_matrix(self, y_true, y_pred):
        """Retorna a matriz de confusão

        Args:
            y_true (ArrayLike): Label verdadeira
            y_pred (ArrayLike): Label predita
        """
        return confusion_matrix(y_true, y_pred, normalize="true")

    def get_accuracy(self, y_true, y_pred):
        """Retorna a acurácia 

        Args:
            y_true (ArrayLike): Label verdadeira
            y_pred (ArrayLike): Label predita
        """
        return accuracy_score(y_true, y_pred)

    def get_precision(self, y_true, y_pred):
        """Retorna a métrica de precisão

        Args:
            y_true (ArrayLike): Label verdadeira
            y_pred (ArrayLike): Label predita
        """
        return precision_score(y_true, y_pred, average='weighted')

    def get_recall(self, y_true, y_pred):
        """Retorna a métrica recall 

        Args:
            y_true (ArrayLike): Label verdadeira
            y_pred (ArrayLike): Label predita
        """     
        return recall_score(y_true, y_pred, average='weighted')
        

    def get_f1_score(self, y_true, y_pred):
        """Retorna a F1-score 

        Args:
            y_true (ArrayLike): Label verdadeira
            y_pred (ArrayLike): Label predita
        """ 
        return f1_score(y_true, y_pred, average='weighted')

    def get_all_metrics(self, y_true, y_pred):
        """Retorna todas as métricas

        Args:
            y_true (ArrayLike): Label verdadeira
            y_pred (ArrayLike): Label predita
        """
        metrics = [
            f"Accuracy: {self.get_accuracy( y_true, y_pred)}",
            f"Precision: {self.get_precision( y_true, y_pred)}",
            f"Recall: {self.get_recall( y_true, y_pred)}",
            f"F1 Score: {self.get_f1_score( y_true, y_pred)}",
        ]

        return "\n".join(metrics)

    def save(self, y_true, y_pred, path):
        """Salva as métricas em um arquivo

        Args:
            y_true (ArrayLike): Label verdadeira
            y_pred (ArrayLike): Label predita
            path (str): Caminho para salvar o arquivo
        """
        with open(path, 'a') as file:
            file.write(self.get_all_metrics( y_true, y_pred))
        pass