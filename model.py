from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import Convolution2D as Conv2D , UpSampling2D
from keras.models import Sequential, Model


# generator = Sequential()
# discriminator = Sequential()

def build_generator():
  
  generator = Sequential()
  
  generator.add(Dense(6272, activation="relu", input_dim=100)) # Add dense layer
  generator.add(Reshape((7, 7, 128)))  # reshape the image
  generator.add(UpSampling2D()) # Upsampling layer to double the size of the image
  generator.add(Conv2D(128, kernel_size=3, padding="same", activation="relu"))
  generator.add(BatchNormalization(momentum=0.8))
  generator.add(UpSampling2D())

  # convolutional + batch normalization layers
  generator.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
  generator.add(BatchNormalization(momentum=0.8))
  
  # convolutional layer with filters = 1
  generator.add(Conv2D(1, kernel_size=3, padding="same", activation="relu"))
  generator.summary() # prints the model summary
  
  """
  We don't add upsampling here because the image size of 28 × 28 is 
  equal to the image size in the MNIST dataset. 
  You can adjust this for your own problem.
  """

  noise = Input(shape=(100,))
  fake_image = generator(noise)
  
  # Returns a model that takes the noise vector as an input and outputs the fake image
  return Model(inputs=noise, outputs=fake_image)


def build_discriminator():

  discriminator = Sequential()
  discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28,28,1), padding="same"))
  discriminator.add(LeakyReLU(alpha=0.2))
  discriminator.add(Dropout(0.25))
  
  discriminator.add(Conv2D(64, kernel_size=3, strides=2,padding="same"))
  discriminator.add(ZeroPadding2D(padding=((0,1),(0,1))))
  discriminator.add(BatchNormalization(momentum=0.8))
  
  discriminator.add(LeakyReLU(alpha=0.2))
  discriminator.add(Dropout(0.25))
  
  discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
  discriminator.add(BatchNormalization(momentum=0.8))
  discriminator.add(LeakyReLU(alpha=0.2))
  discriminator.add(Dropout(0.25))
  
  discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
  discriminator.add(BatchNormalization(momentum=0.8))
  discriminator.add(LeakyReLU(alpha=0.2))
  discriminator.add(Dropout(0.25))
  
  discriminator.add(Flatten())
  discriminator.add(Dense(1, activation='sigmoid'))
  
  img = Input(shape=(28,28,1))
  probability = discriminator(img)
 
  return Model(inputs=img, outputs=probability)

