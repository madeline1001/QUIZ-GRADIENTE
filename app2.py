import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Definir la función de pérdida
def loss_func(theta):
    x, y = theta
    R = np.sqrt(x**2 + y**2)
    return -np.sin(R)

# Definir el gradiente de la función de pérdida
def evaluate_gradient(loss_func, x_train, y_train, theta):
    x, y = theta
    R = np.sqrt(x**2 + y**2)
    grad_x = -np.cos(R) * (x / R)
    grad_y = -np.cos(R) * (y / R)
    return np.array([grad_x, grad_y])

# Gradiente descendente
def gd(theta, x_train, y_train, loss_func, epochs, eta):
    for i in range(epochs):
        gradient = evaluate_gradient(loss_func, x_train, y_train, theta)
        theta -= eta * gradient
        # theta = theta - eta * gradient
    return theta, gradient

# Gradiente descendente estocástico
def sgd(theta, data_train, loss_func, epochs, eta):
    for i in range(epochs):
        np.random.shuffle(data_train)  # Barajar los datos en cada época
        for example in data_train:
            x, y = example
            gradient = evaluate_gradient(loss_func, x, y, theta)
            theta = theta - eta * gradient  # Actualizar los parámetros con el gradiente
    return theta, gradient

# RMSprop
def rmsprop(theta, data_train, loss_func, epochs, eta=0.001, decay=0.9, epsilon=1e-8):
    E_g2 = np.zeros_like(theta)  # Inicializar E[g^2] en cero
    for epoch in range(epochs):
        np.random.shuffle(data_train)  # Barajar los datos
        for example in data_train:
            x, y = example
            gradient = evaluate_gradient(loss_func, x, y, theta)
            
            # Actualizar el promedio del cuadrado del gradiente
            E_g2 = decay * E_g2 + (1 - decay) * gradient**2
            
            # Actualizar los parámetros usando RMSprop
            theta -= eta / (np.sqrt(E_g2) + epsilon) * gradient
            
    return theta


# Algoritmo Adam
def adam(theta, data_train, loss_func, epochs, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = np.zeros_like(theta)  # Inicializar el momento de primer orden
    v = np.zeros_like(theta)  # Inicializar el momento de segundo orden
    t = 0  # Inicializar el contador de iteraciones

    for epoch in range(epochs):
        np.random.shuffle(data_train)  # Barajar los datos
        for example in data_train:
            x, y = example
            t += 1  # Incrementar el contador
            gradient = evaluate_gradient(loss_func, x, y, theta)

            # Actualizar los momentos de primer y segundo orden
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)

            # Corrección de sesgo para momentos de primer y segundo orden
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # Actualización de los parámetros
            theta -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    return theta

# Streamlit 
st.title('Optimización con diferentes métodos')

# Contenedores
with st.container():
    col1, col2 = st.columns(2)

# Input del usuario para los límites
    with col1:
        st.header('Parámetros de la función')
        x_min = st.slider('Límite inferior de x', -10.0, 0.0, -6.5)
        x_max = st.slider('Límite superior de x', 0.0, 10.0, 6.5)
        y_min = st.slider('Límite inferior de y', -10.0, 0.0, -6.5)
        y_max = st.slider('Límite superior de y', 0.0, 10.0, 6.5)


    with col2:
        # Parámetros para optimización
        st.header('Métodos de optimización')
        method = st.selectbox('Selecciona un método', ['GD', 'SGD', 'RMSprop', 'Adam'])

        # Configuración del método
        eta = st.slider('Tasa de aprendizaje (eta)', 0.001, 0.1, 0.01)
        epochs = st.slider('Número de épocas', 1000, 20000, 10000)

        # Parámetros adicionales para RMSprop y Adam
        if method == 'RMSprop':
            decay = st.slider('Factor de decaimiento (decay)', 0.1, 1.0, 0.9)
        elif method == 'Adam':
            beta1 = st.slider('Beta1 (Adam)', 0.1, 1.0, 0.9)
            beta2 = st.slider('Beta2 (Adam)', 0.1, 1.0, 0.999)
            alpha = st.slider('Alpha (Adam)', 0.001, 0.1, 0.001)

        # Botón para calcular
        if st.button('Calcular'):
            # Generar datos de entrenamiento
            x_train = np.random.uniform(x_min, x_max, 100)
            y_train = np.random.uniform(y_min, y_max, 100)
            data_train = list(zip(x_train, y_train))
            theta_init = np.array([2.0, 2.0])

            # Ejecutar el método seleccionado
            if method == 'GD':
                theta_final = gd(theta_init.copy(), x_train, y_train, loss_func, epochs, eta)
            elif method == 'SGD':
                theta_final = sgd(theta_init.copy(), data_train, loss_func, epochs, eta)
            elif method == 'RMSprop':
                theta_final = rmsprop(theta_init.copy(), data_train, loss_func, epochs, eta, decay)
            elif method == 'Adam':
                theta_final = adam(theta_init.copy(), data_train, loss_func, epochs, alpha, beta1, beta2)

            # Mostrar los resultados
            st.write(f"Punto mínimo estimado con {method}: {theta_final}")

            # Visualización de la función y la frontera de decisión
            x_vals = np.linspace(x_min, x_max, 100)
            y_vals = np.linspace(y_min, y_max, 100)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = np.zeros_like(X)

            # Rellenar la matriz Z con los valores de la función de pérdida en cada punto de la cuadrícula
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = loss_func([X[i, j], Y[i, j]])

            # Graficar la función y el punto mínimo estimado
            fig, ax = plt.subplots()
            contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
            fig.colorbar(contour)
            ax.scatter(theta_final[0], theta_final[1], color='red', label='Punto mínimo estimado')
            ax.legend()
            ax.set_title(f'Función y punto mínimo ({method})')
            st.pyplot(fig)