import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Definir la función de pérdida
def loss_func(theta):
    x, y = theta
    R = np.sqrt(x**2 + y**2)
    return -np.sin(R)

# Definir el gradiente de la función de pérdida
def evaluate_gradient(theta):
    x, y = theta
    R = np.sqrt(x**2 + y**2)
    if R == 0:
        return np.zeros_like(theta)  # Evitar división por cero
    grad_x = -np.cos(R) * (x / R)
    grad_y = -np.cos(R) * (y / R)
    return np.array([grad_x, grad_y])

# Gradiente descendente
def gd(theta, epochs, eta):
    for i in range(epochs):
        gradient = evaluate_gradient(theta)
        theta -= eta * gradient
    dist = np.linalg.norm(theta)
    return theta, dist

# Gradiente descendente estocástico
def sgd(theta, data_train, epochs, eta):
    for i in range(epochs):
        np.random.shuffle(data_train)
        for example in data_train:
            gradient = evaluate_gradient(example)  # Usar el ejemplo
            theta -= eta * gradient
    dist = np.linalg.norm(theta)
    return theta, dist

# RMSprop
def rmsprop(theta, data_train, epochs, eta=0.001, decay=0.9, epsilon=1e-8):
    E_g2 = np.zeros_like(theta)
    for epoch in range(epochs):
        np.random.shuffle(data_train)
        for example in data_train:
            gradient = evaluate_gradient(example)  # Usar el ejemplo
            E_g2 = decay * E_g2 + (1 - decay) * gradient**2
            theta -= eta / (np.sqrt(E_g2) + epsilon) * gradient
    dist = np.linalg.norm(theta)
    return theta, dist

# Algoritmo Adam
def adam(theta, data_train, epochs, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0
    for epoch in range(epochs):
        np.random.shuffle(data_train)
        for example in data_train:
            t += 1
            gradient = evaluate_gradient(example)  # Usar el ejemplo
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            theta -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    dist = np.linalg.norm(theta)
    return theta, dist

# Inicialización de datos
np.random.seed(1006093739)
x_train = np.random.uniform(-6.5, 6.5, 100)
y_train = np.random.uniform(-6.5, 6.5, 100)
data_train = list(zip(x_train, y_train))

# Título de la aplicación
st.title("API MÉTODOS DE OPTIMIZACIÓN")

if 'grafico1' not in st.session_state:
    st.session_state.grafico1 = None 

# Crear columnas para graficar y calcular
col1, col2 = st.columns(2)

# Variable para almacenar el gráfico
fig = None

# Columna para graficar
with col1:
    st.subheader("Gráfica de la función")
    # Entradas del usuario para los límites de X e Y
    x_min = st.number_input("Límite inferior de X", value=-6.5)
    x_max = st.number_input("Límite superior de X", value=6.5)
    y_min = st.number_input("Límite inferior de Y", value=-6.5)
    y_max = st.number_input("Límite superior de Y", value=6.5)

    # Botón para graficar
    if st.button("Graficar"):
        # Verificar que los límites sean válidos
        if x_min < x_max and y_min < y_max:
            # Crear el gráfico
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Generar datos
            X = np.arange(x_min, x_max, 0.25)
            Y = np.arange(y_min, y_max, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X**2 + Y**2)
            Z = -np.sin(R)

            # Graficar la superficie
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)

            # Personalizar el eje z
            ax.set_zlim(-1.01, 1.01)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter('{x:.02f}')

            # Agregar una barra de colores
            fig.colorbar(surf, shrink=0.5, aspect=5)

            st.pyplot(fig)
            st.session_state.grafico1 = fig
            
        else:
            st.warning("Los límites inferiores deben ser menores que los superiores.")
            


# Columna para cálculo
with col2:
    st.subheader("Métodos de Optimización")
    
    # Seleccionar método de optimización
    method = st.selectbox("Selecciona el método de optimización:", ["Gradiente Descendente", "SGD", "RMSprop", "Adam"])

    # Configuración de parámetros según el método seleccionado
    if method == "Gradiente Descendente":
        epochs = st.number_input("Número de iteraciones:", min_value=1, value=1000)
        eta = st.number_input("Tasa de aprendizaje:", min_value=0.0, value=0.1, step=0.01)

    elif method == "SGD":
        epochs = st.number_input("Número de iteraciones:", min_value=1, value=100)
        eta = st.number_input("Tasa de aprendizaje:", min_value=0.0, value=0.01, step=0.01)

    elif method == "RMSprop":
        epochs = st.number_input("Número de iteraciones:", min_value=1, value=100)
        eta = st.number_input("Tasa de aprendizaje:", min_value=0.0, value=0.001, step=0.001)
        decay = st.number_input("Decaimiento:", min_value=0.0, value=0.9, step=0.01)
        epsilon = st.number_input("Epsilon:", min_value=0.0, value=1e-8)

    elif method == "Adam":
        epochs = st.number_input("Número de iteraciones:", min_value=1, value=100)
        alpha = st.number_input("Tasa de aprendizaje:", min_value=0.0, value=0.001, step=0.001)
        beta1 = st.number_input("Beta1:", min_value=0.0, value=0.9, step=0.01)
        beta2 = st.number_input("Beta2:", min_value=0.0, value=0.999, step=0.01)
        epsilon = st.number_input("Epsilon:", min_value=0.0, value=1e-8)

    # Botón para calcular
    if st.button("Calcular"):
        # Inicializar theta
        theta_init = np.random.uniform(-6.5, 6.5, 2)

        if method == "Gradiente Descendente":
            theta_min, dist = gd(theta_init.copy(), int(epochs), eta)
        
        elif method == "SGD":
            theta_min, dist = sgd(theta_init.copy(), data_train, int(epochs), eta)
        
        elif method == "RMSprop":
            theta_min, dist = rmsprop(theta_init.copy(), data_train, int(epochs), eta, decay, epsilon)
        
        elif method == "Adam":
            theta_min, dist = adam(theta_init.copy(), data_train, int(epochs), alpha, beta1, beta2, epsilon)

        # Mostrar el resultado
        st.success(f"Punto mínimo estimado: {theta_min}, Distancia al mínimo: {dist:.4f}")
if st.session_state.grafico1:
    st.pyplot(st.session_state.grafico1)
