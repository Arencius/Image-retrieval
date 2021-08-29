import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, BatchNormalization, ReLU, Reshape, Add, multiply, LeakyReLU
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.losses import CategoricalCrossentropy
from PIL import Image, ImageEnhance

# additional data augmentation techniques
def gaussian_noise(img, max_mean = 0.2, max_var = 0.0125):
  w,h,c = img.shape

  mean = np.random.uniform(0, max_mean)
  var = np.random.uniform(0, max_var)
  sigma = var ** 0.5

  noise = np.random.normal(mean, sigma, (w,h,c)).reshape((w,h,c))

  return img + noise


def random_cover(img, max_h = 0.45, max_w = 0.45):
  width, height, _ = img.shape

  # shape of the covering box
  cover_width = np.int32(np.random.uniform(0.1, max_w) * width)
  cover_height = np.int32(np.random.uniform(0.1, max_h) * height)

  # top left pixel of the covering box
  top = np.random.randint(0.1, height * (1 - max_h))
  left = np.random.randint(0.1, width * (1 - max_w))

  # distances from top-left corner to the other side of the box
  box_h = top + cover_height
  box_w = left + cover_width

  if box_h <= height and box_w <= width:
    img[top:box_h, left:box_w, :] = 0

  return img


def saturation(img, min_sat = 1.25, max_sat = 3):
  img = Image.fromarray(np.uint8(img * 255))
  converter = ImageEnhance.Color(img)

  saturation_factor = np.random.uniform(min_sat, max_sat)

  return np.asarray(converter.enhance(saturation_factor)) / 255.0


def crop_image(img):
    width, height, _ = img.shape
    crop_ratio = width // 2
    
    max_top = max_left = width - crop_ratio  # assuming that width == height
    
    top_corner = np.random.randint(0, max_top)
    left_corner = np.random.randint(0, max_left)
    
    cropped_image = img[top_corner: , left_corner: , :]
    cropped_image = Image.fromarray(np.uint8(cropped_image * 255)).resize((width, height))
    
    return np.asarray(cropped_image) / 250.0


def preprocessing_func(img):
  functions = [gaussian_noise, random_cover, saturation, crop_image]
  processed_img = np.random.choice(functions)(img)
    
  return processed_img

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# normalize images and set the precision to FP16
train_images = np.array(train_images).astype(np.float16) / 255.0
test_images = np.array(test_images).astype(np.float16) / 255.0

# data augmentation
train_datagen = ImageDataGenerator(rotation_range = 10,
                                   width_shift_range = 0.15,
                                   height_shift_range = 0.15,
                                   horizontal_flip = True,
                                   zoom_range = 0.2,
                                   preprocessing_function = preprocessing_func)


INPUT_SHAPE = train_images[0].shape
INITIAL_LR = np.float16(0.02)       
OPT = SGD(INITIAL_LR, momentum = 0.9)
WARMUP_EPOCHS = 20
EPOCHS = 300

def schedule(epoch, lr): 
  # exponential learning rate increase until it reaches the initial_lr value
  if epoch <= WARMUP_EPOCHS:
    return INITIAL_LR * epoch / WARMUP_EPOCHS
  
  # when warmed up, cosine lr decay is applied
  else:
    cos = np.cos((epoch * np.pi) / EPOCHS)
    return INITIAL_LR * (0.5 * (1 + cos))

  
early_stopping = EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience = 20, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(schedule, verbose = 1)


def squeeze_and_excitation_block(input_block, ch, ratio = 16):
    x = GlobalAveragePooling2D()(input_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    x = Reshape((1,1,ch))(x)
    
    return multiply([input_block, x])


def res_block(input_block, filters, kernel_size = 3, downsample = False, padding = 'same'):
  output = Conv2D(filters, kernel_size, strides = (1 if not downsample else 2), padding = padding)(input_block)
  output = BatchNormalization()(output)
  output = LeakyReLU()(output)
  output = Conv2D(filters, kernel_size, strides = 1, padding = 'same')(output)
  output = BatchNormalization()(output)
  output = LeakyReLU()(output)  
  
  se_block = squeeze_and_excitation_block(output, filters)

  if downsample:
    input_block = Conv2D(filters=filters, kernel_size=1, strides=2, padding='same')(input_block)
    input_block = BatchNormalization()(input_block)
    input_block = ReLU()(input_block)

  output = Add()([input_block, se_block])
  output = ReLU()(output)
  
  return output


def resnet():
  no_blocks = [3,4,4,3]
  no_filters = [96,128,256,512]

  model_input = Input(shape = INPUT_SHAPE)
  x = Conv2D(filters = 96, kernel_size = 3, padding = 'same')(model_input)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Conv2D(filters = 96, kernel_size = 3, padding = 'same')(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)

  for i in range(len(no_blocks)):
    nb = no_blocks[i]
    for j in range(nb):
      x = res_block(x, filters = no_filters[i], downsample=(j==0 and i!=0))

  gap = GlobalAveragePooling2D()(x)
  output = Dense(units = 10, activation = 'softmax')(gap)

  return Model(model_input, output)

model = resnet()
model.summary()

model.compile(optimizer=OPT, loss = CategoricalCrossentropy(label_smoothing = 0.2), metrics = ['accuracy'])

history = model.fit(train_datagen.flow(train_images, train_labels, batch_size=192),
          epochs = EPOCHS,
          validation_data = (test_images, test_labels),
          callbacks = [early_stopping, lr_scheduler])

model.evaluate(test_images, test_labels)
model.save('cifar_model.h5')