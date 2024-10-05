import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Título de la aplicación
st.title("API de clasificacion")

with st.container():
    
    if 'grafico1' not in st.session_state:
        st.session_state.grafico1 = None 
    
    # Definir las dos columnas
    col1, col2 = st.columns(2)


    with col1:
        st.header("Conjunto de datos simulado")
        num1 = st.number_input("Total de simulaciones", value=100)
        num2 = st.number_input("Total de clases", value=3)

        if st.button("Generar"):
            x, gt = make_blobs(n_samples=num1, centers=num2, n_features=2, random_state=42)
            gt = gt.reshape((len(gt), 1))
            plt.scatter(x[:,0], x[:,1], c=gt, cmap=plt.cm.Paired, edgecolors='k', marker='o')
            fig, ax = plt.subplots()
            ax.scatter(x[:, 0], x[:, 1], c=gt, cmap=plt.cm.Paired, edgecolors='k', marker='o')
            st.session_state.x = x
            st.session_state.gt = gt
            
            st.pyplot(fig)
            st.session_state.grafico1 = fig
 
    with col2:
        st.header("Perceptrón")
        num3 = st.number_input("Cantidad de iteraciones", value=num1)
        num4 = st.number_input("Cantidad de clases", value=num2)
        num5 = st.number_input("Tasa de aprendizaje", value=0.01)

        if st.button("Estimate Neural Network"):
            # Definición de la clase Perceptron
            class Perceptron:
                def __init__(self, learning_rate, n_iters):
                    self.lr = learning_rate
                    self.n_iters = n_iters
                    self.activation_func = self._unit_step_func
                    self.weights = None
                    self.bias = None
                    self.classes = None

                def fit(self, X, y):
                    self.classes = np.unique(y)
                    n_samples, n_features = X.shape
                    n_classes = len(self.classes)

                    self.weights = np.zeros((n_classes, n_features))
                    self.bias = np.zeros(n_classes)

                    for clases, c in enumerate(self.classes):
                        y_2 = np.where(y == c, 1, 0)

                        for _ in range(self.n_iters):
                            for i, x_i in enumerate(X):
                                linear_output = np.dot(x_i, self.weights[clases]) + self.bias[clases]
                                y_predicted = self.activation_func(linear_output)

                                update = self.lr * (y_2[i] - y_predicted)
                                self.weights[clases] += update * x_i
                                self.bias[clases] += update.item() if np.ndim(update) > 0 else update

                def predict(self, X):
                    linear_outputs = [np.dot(X, w) + b for w, b in zip(self.weights, self.bias)]
                    y_predicted = np.argmax(linear_outputs, axis=0)
                    return self.classes[y_predicted]

                def _unit_step_func(self, x):
                    return np.where(x >= 0, 1, 0)
                
                def save_model(self, file_path):
                    model_data = {
                        "weights": self.weights.tolist(),
                        "bias": self.bias.tolist(),
                        "classes": self.classes.tolist()
                    }

                    with open(file_path, 'w') as json_file:
                        json.dump(model_data, json_file)

                @classmethod
                def load_model(cls, file_path):
                    with open(file_path, 'r') as json_file:
                        model_data = json.load(json_file)

                def plot_decision_boundary(self, X, y, title="Perceptron Decision Boundary"):
                    fig, ax = plt.subplots()
                    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')

                    x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
                    y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1

                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

                    Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)

                    ax.set_title(title)
                    ax.set_xlabel('Class 1')
                    ax.set_ylabel('Class 2')

                    st.pyplot(fig)

            perceptron = Perceptron(learning_rate=num5, n_iters=num3)
            perceptron.fit(st.session_state.x, st.session_state.gt)
            perceptron.plot_decision_boundary(st.session_state.x, st.session_state.gt)
                
if st.session_state.grafico1:
    st.pyplot(st.session_state.grafico1)