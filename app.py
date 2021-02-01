import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
#pytessearct ocr to extract text from image
import cv2
import imutils
import numpy as np
import io, os
import json

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
import itertools
from PIL import Image
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.resnet50 import ResNet50

#resnet50
#model = tensorflow.keras.applications.resnet50.ResNet50()
#MODEL_PATH =r"E:\softweb\ML\Heroku-Project\image_prediction_pretrained\resnet50_weights_tf_dim_ordering_tf_kernels.h5"   
#model = model.load_weights(MODEL_PATH)
model = ResNet50()

#model = ResNet50(weights = MODEL_PATH)

MEDIA = 'static/media/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = MEDIA

# check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def image_prediction(im):
    
    #img = image.load_img(im, target_size=(224, 224))
    #img = im.resize((224, 224), Image.ANTIALIAS)
    x = image.img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.resnet50.preprocess_input(x)
    print('Input image shape:', x.shape)
    
    preds = model.predict(x)
    
    print('Predicted:', imagenet_utils.decode_predictions(preds))
    
    n = 3

    img_pred =imagenet_utils.decode_predictions(preds, top =n)

    return img_pred
    
# route and function to handle the upload page
@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        # if no file is selected
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')

        if file and allowed_file(file.filename):
            file.save(MEDIA+file.filename)

            # call the OCR function on it
            #extracted_text = get_attendence(file)
            #extracted_text = get_text_from_api(file)
            #extracted_text = use_google_vision(file)
            print("111111111111", file)
            print("222222222", file.filename)
            
            #read image file
            #image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            with io.open('static/media/{}'.format(file.filename), 'rb') as image_file:
                #content = image_file.read()
          
                image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
            image = cv2.resize(image, (224,224))
            predicted_image = image_prediction(image)
            print(predicted_image)
            
            
            predicted_image_top3 ={}
            for n, pimg in enumerate(predicted_image[0]):
                print(pimg[1])
                predicted_image_top3[n] = pimg[1]
                
            print(predicted_image_top3)
            # extract the text and display it
            return render_template('upload.html',
                                   msg='Successfully processed',
                                   extracted_text = predicted_image_top3,
                                   img_src=MEDIA + file.filename)
    elif request.method == 'GET':
        return render_template('upload.html')

#@app.route('/')
#def home_page():
#    return render_template('index.html') 

if __name__ == '__main__':
    app.run()
