
import warnings
warnings.filterwarnings("ignore")
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.saving import load_model

import cv2
import numpy as np
from imutils import paths
import re



# loaded_model=load_model("model\CNN_model_2-16-150.keras")
mpath="model\CNN_Layer_22_numclasses_5-16-120.h5"
mpath="model\CNN_model_1-16-20-numclasses-36.h5"
mpath="D:\Programming\Django\hhcr\models_predict\CNN_model_2-32-30-numclasses-36.h5"
loaded_model=load_model(mpath)
img_height,img_width=32,32
CHANNELS=1
def checkfile(str):
    print(str)
    regex = "([^\s]+(\.(?i)(jpe?g|png|gif|bmp))$)"
    regex= ""
    p = re.compile(regex)
    # print(bool((re.search(p, str))))
    # if (str == None):
    #     return False
    # Return if the string 
    # matched the ReGex
    if (re.search(p, str)):
        return True
    else:
        return False

def predict_images(image,verbose=False):
    
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray=cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
    image = cv2.resize(image_gray, (img_height, img_width))

    img=np.array(image)
    img = img.reshape(1, img_height, img_width, CHANNELS)
    img = img.astype('float32')
    img /= 255
    result=loaded_model.predict(img,verbose=0)
    res=loaded_model(img)
    predicted_class = int(np.argmax(result, axis=1))
    return predicted_class
    # os.rename(
    #     imagePath,
    #     os.path.join(directory,f"{predicted_class}-{fileName}")
    # )
    # except FileNotFoundError:
    #     print("No file found")
    # except:
    #     print("Unknown Issue occured terminating!!!")



