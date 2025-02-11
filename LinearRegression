import numpy as np

class LinearRegression:

    """
    loss = MSE:  sum((y-y_pred)**2)/N_clients
    y_pred = w1*x1+w2*x2 + ... + 1 * bias

    X.shape : N_clients x M_features
    y.shape: N_clients

    dloss/dw1   = sum( - 2 * x1 * (y-y_pred)) / N_clients  # сумма по всем клиентам
    dloss/bias  = sum( - 2 * 1 *  (y-y_pred)) / N_clients  # сумма по всем клиентам

    dw = 2 * np.dot(X.T, (y-self.predict(X))) / X.shape[0] # сумма учтена в матричном произведении
    db = 2 * sum(y-self.predict(X)) / X.shape[0]   # сумма по всем клиентам
    """  

    def __init__(self, num_features:int = None):
        self.weights = np.array([1.0] * num_features)  # [1,1,1,..] shape: 1 x num_features
        self.bias = 1

    def predict(self, X):  # X shape:  n_clients x num_features
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y, iterations_cnt = 500, lr = 0.01):
        # X[:, 0]:  (N_clients x 1);
        # (y-self.predict(X)): (N_clients,);
        for _ in range(iterations_cnt):
            dw = 2 * np.dot(X.T, (y-self.predict(X))) / X.shape[0]
            db = 2 * sum(y-self.predict(X)) / X.shape[0]
            self.weights += dw * lr
            self.bias += db * lr

    def show_weights(self):
        print(list(self.weights) + [self.bias])



if __name__ == '__main__':
    #### Проверка на примере (модель с параметрами [2,3,-100])
    
    def myfunc(x1,x2):
        return 2*x1 + 3*x2 - 100 + np.random.normal(loc = 0, scale = 0.3, size = 1)[0]

    x1_val = np.linspace(-3,3, 101)
    x2_val = np.linspace(-3,3,101)

    X = np.array([(x_1, x_2, myfunc(x_1,x_2))
                    for x_1 in x1_val
                    for x_2 in x2_val]
    )

    X, y = X[:, :2], X[:, 2]
    linreg = LinearRegression(num_features = 2)
    linreg.train(X,y)
    linreg.show_weights()




