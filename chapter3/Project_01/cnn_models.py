import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import numpy as np
from keras.preprocessing.image import img_to_array # type: ignore
from keras.applications.resnet50 import preprocess_input, decode_predictions    # type: ignore

class cnnModels:
    def __init__(self):
        self.models = {'ResNet50': self.resnet(), 'VGGNet16': self.vggnet(), 
                       'InceptionV3': self.inception(), 'ConvNeXt': self.convnet(), 
                       'EfficientNet': self.efficientnet()}

    def resnet(self):       
        model = keras.applications.ResNet50(weights='imagenet')
        return model
    
    def vggnet(self):
        model = keras.applications.VGG16(weights='imagenet')
        return model
    
    def inception(self):
        model = keras.applications.InceptionV3(weights='imagenet')
        return model
    
    def convnet(self):
        model = keras.applications.ConvNeXtTiny(weights='imagenet')
        return model   

    
    def efficientnet(self):
        model = keras.applications.EfficientNetB7(weights='imagenet')
        return model

    def get_model(self, name):
        if name in self.models:
            return self.models[name]
        else:
            raise ValueError(f"Model '{name}' does not exist.")        
    
        
    def classify_image(self, name, img):
        model = self.get_model(name)      
        img = img.resize((model.input_shape[1], model.input_shape[2]))  
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x, verbose=0)
        return decode_predictions(preds, top=1)

