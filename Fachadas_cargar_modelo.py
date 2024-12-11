from tensorflow import keras
import tensorflow as tf
import os
import pathlib
import time
import datetime
import pickle
from matplotlib import pyplot as plt
from IPython import display
# espacio para funciones:

def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image
  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, w:, :]
  real_image = image[:, :w, :]

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

#funcion que renderiza las imagenes
def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

#funcion que cogé una parte de las imagenes pasadas como parametros
def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

#funcion utilizada para aumentar los datos
@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  test_debug = test_input[0]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


if __name__ == "__main__":
    # step 1: Load the entire model
    loaded_generator = keras.models.load_model('.\content\generator_model.keras')
    print(type(loaded_generator))

    # step 2 : load dataset
    dataset_name = "facades"
      
    
    path_train = ".\\dataset\\facades\\train"
    path_test = ".\\dataset\\facades\\path_test"
    path_val = ".\\dataset\\facades\\val"
    
   # .\dataset\facades\train\
    # step 3: cargar una imagen para demostracion
    inp, re = load(f"{path_train}\\400.jpg")

    # step 4: parametros de imagen. nuestro trabajo es identificar que hacen

    # The facade training set consist of 400 images
    BUFFER_SIZE = 400
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 1
    # Each image is 256x256 in size
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    try:
        test_dataset = tf.data.Dataset.list_files(f"{path_test}\\*.jpg")
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(f"{path_val}\\*.jpg")
        test_dataset = test_dataset.map(load_image_test)
        test_dataset = test_dataset.batch(BATCH_SIZE)

    # step 5: prediccion
    for inp, tar in test_dataset.take(10):
        generate_images(loaded_generator, inp, tar)
            

