import random

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


# Entrenamiento y prueba de una pequeña red neuronal.
def mi_primera_red_neuronal():
    # Vamos a crear una red neuronal, que dados unos datos de entrada, y unos datos de salida de ejemplo, aprenda la relación
    # que hay entre dichos datos de entrada y dichos datos de salida, para que cuando meta nuevos valores, sea capáz de
    # predecir el resultado basandose en los ejemplos proporcionados.
    datos_entrada_entrenamiento = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=int)
    datos_salida_entrenamiento = np.array([3, 5, 7, 9], dtype=int)

    # Ahora vamos a definir la arquitectura de la red neuronal
    neural_network = Sequential()
    # Una vez creada le añadimos una capa oculta de 3 neuronas
    neural_network.add(Dense(units=3, activation='relu', input_shape=(2,)))
    # Y la neurona de salida
    neural_network.add(Dense(units=1))

    # Nos compilamos dicho modelo
    neural_network.compile(optimizer='adam', loss='mean_squared_error')
    # Procedemos con el entrenamiento, el entrenamiento vera los datos durante 500 veces
    neural_network.fit(datos_entrada_entrenamiento, datos_salida_entrenamiento, epochs=50000)

    # Ponemos a prueba dicho entrenamiento. El resultado esperado seria 11 si ha logrado de entender la relacion entre los
    # datos de entrada y de salida de ejemplo.
    print(neural_network.predict(np.array([[5, 6]])))


def entrenar_y_guardar_modelo():
    # to.do esto igual que antes
    datos_entrada_entrenamiento = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=int)
    datos_salida_entrenamiento = np.array([3, 5, 7, 9], dtype=int)

    neural_network = Sequential()
    neural_network.add(Dense(units=3, activation='relu', input_shape=(2,)))
    neural_network.add(Dense(units=1))

    neural_network.compile(optimizer='adam', loss='mean_squared_error')
    neural_network.fit(datos_entrada_entrenamiento, datos_salida_entrenamiento, epochs=50000)

    # Solo que ahora en lugar de probarlo, nos lo guardamos en un fichero para no estar gastando
    # cpu constantemente.
    neural_network.save("modelo_entrenado.keras")


def entrenar_y_guardar_modelo_providing_dataset(
        datos_entrada_entrenamiento: np.ndarray,
        datos_salida_entrenamiento: np.ndarray
):
    neural_network = Sequential()
    neural_network.add(Dense(units=3, activation='relu', input_shape=(2,)))
    neural_network.add(Dense(units=1))

    neural_network.compile(optimizer='adam', loss='mean_squared_error')
    neural_network.fit(datos_entrada_entrenamiento, datos_salida_entrenamiento, epochs=50000)

    # Solo que ahora en lugar de probarlo, nos lo guardamos en un fichero para no estar gastando
    # cpu constantemente.
    neural_network.save("modelo_entrenado.keras")


def cargar_modelo_entrenado_y_hacer_predicciones(x: int, y: int):
    # Nos cargamos un modelo previamente entrenado para comenzar a cacharrear con el
    neural_network: Sequential = load_model("modelo_entrenado.keras")
    prediccion = neural_network.predict(np.array([[x, y]]))
    print("El resultado de la predicción dados los parámetros " + str(x) + " y " + str(y) + " es -> " + str(prediccion))


def generar_un_dataset_como_es_debido(tamano_array: int):
    # Para que mi dataset no sea de literalmente de 4 elementos, vamos a automatizar la generacion de datos que usaremos
    # para entrenar nuestra red (no siempre dichos datos seran tan facil de generar, este es un ejemplo MUY tonto y
    # crear una IA para resolver este problema es matar un mosquito a cañonazos)

    # Definimos los 2 arrays
    datos_entrenamiento_operandos = []
    datos_entrenamiento_resultado = []

    # Generamos los datos que contendra
    for i in range(tamano_array):
        # Generamos los datos del array de operaciones (para no freir este primer intento, con que sean numeros positivos del 0 al 100 podemos ir jugando.)
        sumador = random.randint(0, 100)
        sumando = random.randint(0, 100)
        datos_entrenamiento_operandos.append([sumador, sumando])

        # generamos el resultado esperado y nos lo guardamos.
        resultado = sumador + sumando
        datos_entrenamiento_resultado.append(resultado)

    # Devolvemos ambos arrays
    return np.array(datos_entrenamiento_operandos), np.array(datos_entrenamiento_resultado)

# TODO def test_neural_network():
