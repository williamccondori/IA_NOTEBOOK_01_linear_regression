import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


def main():
    # Inicializacion del conjunto de datos.
    boston = load_boston()
    x, y = np.array(boston.data[:, 5]), np.array(boston.target)

    # Visualización de datos.
    plt.scatter(x, y, alpha=.3)

    # Añadimos una columna de 1 para representar el término independiente.
    x = np.array([np.ones(len(x)), x]).T

    # Regresión lineal: W = (XT*X)^-1*XT*Y
    w = np.linalg.inv(x.T @ x) @ x.T @ y
    w0, w1 = w[0], w[1]

    # Muestra de resultados.
    print(f'Valor independiente: {w0}')
    print(f'Valor de la pendiente: {w1}')

    x_start, x_end = 4, 10
    plt.plot([x_start, x_end], [w0 + w1 * x_start, w0 + w1 * x_end], color='red')

    plt.show()


if __name__ == '__main__':
    main()
