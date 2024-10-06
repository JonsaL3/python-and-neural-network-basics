import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# Ahora vamos a crear una red neuronal que aprenda a identificar elementos en imagenes, como pueden
# ser aviones, coches, barcos etc etc, esta ia dada una imagen de un objeto será capaz de decirte
# que es lo que aparece (si ha sido entrenada para reconocerlo)
# Para lograrlo, las redes neuronales convolucionales son la mejor opcion actualmente para
# ello, las cuales iran "desglosando" poco a poco la imagen para que pueda sacar conclusiones de
# manera mas sencilla.

def mi_primera_red_neuronal_convolucional():
    # Para lograrlo, tensorflow nos facilita un dataset de ejemplo que es un mapa de imágenes con su etiqueta de 32x32.
    # y otro mapa para probar que dicho entrenamiento se ha realizado correctamente.
    (imagenes_para_entrenamiento, etiquetas_para_entrenamiento), (imagenes_para_probar_modelo, etiquetas_para_probar_modelo) = datasets.cifar10.load_data()

    # Definimos las etiquetas que irán asociadas a las imágenes, y por tanto las que nuestra red aprenderá a identificar.
    class_names = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

    # Normalizamos pixeles
    imagenes_para_entrenamiento = imagenes_para_entrenamiento / 255
    imagenes_para_probar_modelo = imagenes_para_probar_modelo / 255

    # Visualizamos algunas de las imagenes antes de nada para ver algunas de las imágenes que se van a usar durante el
    # entrenamiento.
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imagenes_para_entrenamiento[i])
        # Las etiquetas están en el formato [n], por eso usamos el índice 0
        plt.xlabel(class_names[etiquetas_para_entrenamiento[i][0]])
    plt.show()

    # Ahora si, nos procedemos a crear una red neuronal que aprenda a relacionar los conceptos con sus respectivas
    # imágenes, la "arquitectura" de la red, como he ido adelantando previamente, sera convolucional, que es una "arquitectura"
    # especialmente efectiva en lo que a procesamiento, entendimiento y generación de imágenes respecta.
    # Pero en esencia, es lo mismo que hice ayer en neural_network_basics, solo que con mas neuronas, mas capas etc etc...
    neural_network = models.Sequential()
    # Primera capa convolucional y de pooling
    neural_network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    neural_network.add(layers.MaxPooling2D((2, 2)))
    # Segunda capa convolucional y de pooling
    neural_network.add(layers.Conv2D(64, (3, 3), activation='relu'))
    neural_network.add(layers.MaxPooling2D((2, 2)))
    # Tercera capa convolucional y de pooling
    neural_network.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Aplanar la salida y añadir capas densas
    neural_network.add(layers.Flatten())
    neural_network.add(layers.Dense(64, activation='relu'))
    neural_network.add(layers.Dense(len(class_names)))  # 10 clases para la clasificación (tamaño del array de clases.)

    # Compilamos el modelo indicandole un par de parametros mas que en neural_network_bascis, que aun no entiendo
    # pero que espero entender en algún punto.
    neural_network.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Finalmente procedemos con el entrenamiento, además añadimos una cosa nueva, que es que aparte del entrenamiento, le
    # pasamos la lista de validation_datas imagenes_para_probar_modelo para que durante ese mismo proceso valide que tmodo este ok y se
    # "retroalimente" (?)

    # Otra cosa nueva respecto a "neural network basics" es que la funcion FIT devuelve el historico de la evolución del
    # modelo a lo largo del entrenamiento (a lo largo de las épocas (epochs))
    history = neural_network.fit(imagenes_para_entrenamiento, etiquetas_para_entrenamiento, epochs=10,validation_data=(imagenes_para_probar_modelo, etiquetas_para_probar_modelo))

    # Dicho historico podemos representarlo en una grafica ayudandonos de matplotlib (el que vimos con Mario Santos en DAM)
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend(loc='lower right')
    plt.show()

    # Como el entrenamiento es bastante demandante, nos guardamos el modelo ya entrenado...
    neural_network.save("modelo_entrenado_reconocimiento_imagenes.keras")

def pruebo_mi_primera_red_neuronal_convolucional():
    # Cargamos el modelo que he entrenado antes:
    neural_network = models.load_model("modelo_entrenado_reconocimiento_imagenes.keras")

    # Cargamos una imagen de prueba, la pillamos del dataset...
    (imagenes_para_entrenamiento, etiquetas_para_entrenamiento), (imagenes_para_probar_modelo, etiquetas_para_probar_modelo) = datasets.cifar10.load_data()
    imagen = imagenes_para_probar_modelo[12]

    # La mostramos antes de predecir nada para entender lo que está ocurriendo...
    plt.imshow(imagen)
    plt.show()

    # Las predicciones se hacen con lotes, así que necesitamos agregar una dimensión al array
    imagen = np.expand_dims(imagen, axis=0)

    # Lanzamos una predicción
    predicciones = neural_network.predict(imagen)

    # Mostrar la predicción
    class_names = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']
    prediccion_clase = np.argmax(predicciones)
    print(f'Clase predicha: {class_names[prediccion_clase]}')








