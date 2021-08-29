import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import cosine
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

datagen = ImageDataGenerator(rescale = 1.0/255.0)

base_model = load_model('/content/drive/MyDrive/cifar_model.h5')

class FeatureExtractor:
  def __init__(self):
    # get the feauture map as an output of the network
    gap = base_model.get_layer('global_average_pooling2d_14')
    self.extractor = Model(base_model.input, gap.output)    

  def extract_features(self, image):
    image = np.expand_dims(image, axis = 0)
    image = datagen.flow(image).next()

    features = self.extractor.predict(image)[0]
    return features

model = FeatureExtractor()

def dataset_imgs_features():
  return [model.extract_features(img) for img in train_images]

def most_similar_photos(query_img, features, n = 5):
  distances = []
  query_features = model.extract_features(query_img)

  # retrieving most similar images
  for f in features:
    distance = cosine(query_features, f)
    distances.append(distance)

  closest = np.argsort(distances)[:n+1]
  
  # visualizing the results
  axes = []
  fig = plt.figure(figsize = (10,10))

  for i, index in enumerate(closest):
    axes.append(fig.add_subplot(1, n+1, i+1))
    plt.imshow(query_img)
    plt.imshow(train_images[index])
    
  fig.tight_layout()
  fig.savefig('result.png')
  plt.show()


images_to_predict=[]
features = dataset_imgs_features()

for filename in glob.glob('/content/drive/MyDrive/image_retrieval/*.jpg'):
  img = np.asarray(Image.open(filename))
  images_to_predict.append(img)

for img in images_to_predict:
  most_similar_photos(img, features)