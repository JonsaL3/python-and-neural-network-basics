from transformers import BertTokenizer, TFBertModel

import neural_network_basics as nnb
import convolutional_neural_network_basics as cnnb
import tensorflow as tf
import mi_primera_gan as mpg
import tensorflow_datasets as tfds

# Mi primera red neuronal, aprendiendo el concepto de SUMA.
# nnb.mi_primera_red_neuronal()
# nnb.entrenar_y_guardar_modelo()
# nnb.cargar_modelo_entrenado_y_hacer_predicciones(20, 20)

# print("Generando dataset...")
# dataset = nnb.generar_un_dataset_como_es_debido(10)
#
# print("Generando y guardando modelo...")
# nnb.entrenar_y_guardar_modelo_providing_dataset(dataset[0], dataset[1])

# nnb.cargar_modelo_entrenado_y_hacer_predicciones(512, 23444)

# Un paso mas alla, mi primer red neuronal convolucional, aprende a identificar elementos en una imagen.
# cnnb.mi_primera_red_neuronal_convolucional()
# cnnb.pruebo_mi_primera_red_neuronal_convolucional()

# Intento de crear mi primera GAN
# Carga el dataset COCO
def preprocess(image, caption):
    image = tf.image.resize(image, (64, 64)) / 255.0  # Normaliza las imágenes
    return image, caption

# Modelo de BERT para procesar el texto
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")
dataset, info = tfds.load("coco", split="train", with_info=True, as_supervised=True)
dataset = dataset.map(preprocess).batch(32).shuffle(1000)
mpg.train(dataset, epochs=50)