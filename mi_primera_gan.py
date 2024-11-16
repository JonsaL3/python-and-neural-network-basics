import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, TFBertModel

def text_to_embedding(texts):
    inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = bert_model(inputs.input_ids).last_hidden_state
    return tf.reduce_mean(outputs, axis=1)  # Usa el promedio como embedding


# 3. Construcción del Generador
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(228,)),  # Texto (128) + Ruido (100)
        layers.Dense(16 * 16 * 128, activation="relu"),
        layers.Reshape((16, 16, 128)),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", activation="relu"),
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding="same", activation="tanh"),  # Imagen RGB
    ])
    return model


generator = build_generator()


# 4. Construcción del Discriminador
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=(64, 64, 3)),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),  # Salida: real o falso
    ])
    return model


discriminator = build_discriminator()

# 5. Funciones de pérdida
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# 6. Optimizadores
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# 7. Función de entrenamiento
@tf.function
def train_step(images, captions):
    # Convierte las captions en embeddings
    text_embeddings = text_to_embedding(captions)

    # Genera ruido aleatorio
    noise = tf.random.normal([images.shape[0], 100])

    # Concatenar embeddings de texto con ruido
    generator_input = tf.concat([noise, text_embeddings], axis=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(generator_input, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


# 8. Entrenamiento del Modelo
def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}")
        for image_batch, caption_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, caption_batch)

        # Mostrar ejemplo después de cada epoch
        generate_and_save_images(generator, epoch + 1, ["a cat sitting on a chair"])


# 9. Función para generar y guardar imágenes
def generate_and_save_images(model, epoch, captions):
    text_embeddings = text_to_embedding(captions)
    noise = tf.random.normal([len(captions), 100])
    generator_input = tf.concat([noise, text_embeddings], axis=1)
    predictions = model(generator_input, training=False)

    plt.figure(figsize=(8, 8))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2)  # Desnormalizar
        plt.axis("off")
    plt.savefig(f"image_at_epoch_{epoch}.png")
    plt.show()
