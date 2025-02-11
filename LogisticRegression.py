import numpy as np


class LogisticRegression:

    """
    Класс для обучения модели логистической регрессии.

    При создании объекта необходимо передать в конструктор параметр num_features - количество признаков при обучении.

    Инициализируем веса и свиг в конструкторе значениями 1.0
    Модель может выдавать какой-либо результат сразу после создания объекта класса.

    """  

    def __init__(self, num_features:int = None):
        """ Конструктор
        Инициализирует веса и сдвиг модели значениями 1.0
        Args:
            num_features (int, optional): Количество признаков при обучении модели. Defaults to None.
        """        
        self.weights = np.array([1.0] * num_features)  # [1,1,1,..] shape: 1 x num_features
        self.bias = 1

    def predict(self, X):  # X shape:  n_clients x num_features
        """
        Метод predict выполняет предсказание для переданной матрицы признаков X
        Args:
            X : Матрица признаков X  (X shape:  n_clients x num_features)
        Returns:
            np.array с предсказаниями (X shape:  n_clients, )
        """        
        sigmoid = lambda z: 1/(1+np.exp(-z))
        return sigmoid(np.dot(X, self.weights) + self.bias)

    def train(self, X, y, iterations_cnt = 500, lr = 0.01):
        """
        Метод для обучения модели.
        Выполняет iterations_cnt итераций градиентного спуска с learning_rate = lr

        Args:
            X (list/tuple/array):  Матрица признаков для обучения (X shape:  n_clients x num_features)
            y (list/tuple/array): Вектор признаков (y shape:  n_clients,)
            iterations_cnt (int, optional): Количество итераций градентного спуска. Defaults to 500.
            lr (float, optional): Скорость обучения в градиентном спуске. Defaults to 0.01.
        """        
        for _ in range(iterations_cnt):
            dw = 2 * np.dot(X.T, (np.array(y)-self.predict(X))) / X.shape[0]
            db = 2 * sum(np.array(y)-self.predict(X)) / X.shape[0]
            self.weights += dw * lr
            self.bias += db * lr

    def show_weights(self):
        """ Метод, который выводит веса и сдвиговый коэффициент логистической регрессии
        """        
        print(list(self.weights) + [self.bias])





