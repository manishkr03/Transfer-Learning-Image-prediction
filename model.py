import numpy as np
import keras, tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow. keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline


#resnet50
model = tensorflow.keras.applications.resnet50.ResNet50()

#model = ResNet50(include_top=True, weights='imagenet')

img_path = r'C:\Users\hp\Desktop\image\Image_1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = keras.applications.resnet50.preprocess_input(x)
print('Input image shape:', x.shape)

preds = model.predict(x)

print('Predicted:', imagenet_utils.decode_predictions(preds))

n = 3
vgg16_top_n = imagenet_utils.decode_predictions(preds, top=n)

ls_top3 ={}
for n, d in enumerate(vgg16_top_n[0]):
    print(d[1])
    ls_top3[n] = d[1]

print(ls_top3)
# =============================================================================
# #VGG16
# 
# import tensorflow as tf
# model = tf.keras.applications.vgg16.VGG16()
# #model = VGG16(include_top=True, weights='imagenet')
# img_path = r'img\1.png'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = keras.applications.vgg16.preprocess_input(x)
# print('Input image shape:', x.shape)
# 
# preds = model.predict(x)
# print('Predicted:', imagenet_utils.decode_predictions(preds))
# =============================================================================



