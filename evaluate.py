import tensorflow as tf
import keras
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os
import emnist
import numpy as np
import cv2
# %matplotlib inline

#Loading Trained Model with 91.34% Accuracy on Test Data and 90.77% on training data using byMerge EMNIST DataSet
def load(path):
    model=tf.keras.models.load_model(path)
    return model

'''
Loading TEST and TRAINING DATASET
By Merge dataset used here
'''
def load_data():
    test_data,test_labels=emnist.extract_test_samples('bymerge')
    return test_data,test_labels

'''
Data Preprocessing
Reshaping and Normalizing
One Hot Encoding
'''

def preprocessing(test_data,test_labels):
    test_data=test_data.reshape(test_data.shape[0],28,28,1).astype('float32')
    test_data=test_data/255
    test_labels=np_utils.to_categorical(test_labels)
    return test_data,test_labels

'''
Predicting the Identified classes in our model
'''

def predictor(model,test_data,batch):
    prediction1=model.predict_classes(test_data,batch_size=batch)
    return prediction1

'''
Testing theaccuracy- could have been done with classification report or evaluate function
'''

def test_accuracy(prediction1,test_l):
    acc=np.sum(prediction1==test_l)
    print(acc/prediction1.shape)

'''
Creating 2 zeros array to predict which Character is best predicted, 
check having if test label ==predicted, occur having 
how many times a character occured in test label
'''

def best_accuracy(prediction1,test_l):
    check=np.zeros(47)
    occur=np.zeros(47)
    for i in range(prediction1.shape[0]):
        occur[test_l[i]]+=1
        if prediction1[i]==test_l[i]:
            check[test_l[i]]+=1
    character=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K',
               'L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','e','f','g','h','n','q','r','t']
    char_acc=[]
    predicted=check/occur
    for i in range(47):
        char_acc.append([predicted[i]*100,character[i]])
    '''
    Reverse Sorting in Descending order
    '''
    char_acc.sort(reverse=True)
    return char_acc,character

'''
Function to Crop the Image to get a good picture containing only the captcha
'''

def cropping(thresh,image):
    rows=thresh.shape[0]
    columns=thresh.shape[1]
    side = []
    g=1
    for i in range(rows):
        for j in range(columns):
            if thresh[i][j] == 255 : 
                side.append([i,j])
                g=0
                break
        if g==0 :
          break
    g=1
    for i in range(columns):
        for j in range(rows):
            if thresh[j][i] == 255 : 
                side.append([j,i])
                g=0
                break
        if g==0 :
          break
    g=1
    for i in range(rows-1,0,-1):
        for j in range(columns):
            if thresh[i][j] == 255 : 
                side.append([i,j])
                g=0
                break
        if g==0 :
          break
    g=1
    for i in range(columns-1,0,-1):
        for j in range(rows):
            if thresh[j][i] == 255 : 
                side.append([j,i])
                g=0
                break

        if g==0 :
          break
    

    thresh= thresh[max(0, side[0][0]-50): min(rows-1, side[2][0]+50) , max(0,side[1][1]-50):min(columns-1,side[3][1]+50)]
    image= image[max(0, side[0][0]-50): min(rows-1, side[2][0]+50) , max(0,side[1][1]-50):min(columns-1,side[3][1]+50)]
    return thresh,image

def thresholding(w):
    image = w
    #Expanding to make clearer spaces
    height, width, depth = image.shape

    #resizing the image to find spaces better

    image = cv2.resize(image, dsize=(width*5,height*4), interpolation=cv2.INTER_CUBIC)
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
    thresh,image = cropping(thresh,image)
    return thresh,image



#dilation and gaussian filter to enlarge gaps in the images and correct eroded images

def cleaning(thresh):
    kernel = np.ones((8,8), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=2)
    #adding GaussianBlur
    gsblur=cv2.GaussianBlur(img_dilation,(7,7),0)
    t=cv2.resize(gsblur, (480, 320)) 
    return gsblur

#find contours

def extract(model,gsblur,image,character):
    count=0
    answer=""
    height, width = gsblur.shape
    ctrs, hier = cv2.findContours(gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    dp = image.copy()
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        count+=1
    if w*h >(0.004*width*height) and w*h<(0.25*width*height):
        cv2.rectangle(dp,(x-10,y-10),( x + w + 10, y + h + 10 ),(90,0,255),9)

    if(count!=0):
        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            if w*h >(0.004*width*height) and w*h<(0.25*width*height):
        
                try:
                    roi=gsblur[y-5:y+h+5,x-5:x+w+5]
                    roi=cv2.resize(roi,dsize=(28,28),interpolation=cv2.INTER_CUBIC)
                    kernel = np.ones((2,2), np.uint8)
                    dilate = cv2.dilate(roi, kernel, iterations=1)
                    blur=cv2.GaussianBlur(dilate,(3,3),0)
                    roi=blur
                except cv2.error as e:
                    continue
                roi=np.array(roi)
                t=np.copy(roi)
                t=t/255
                t=t.reshape(1,28,28,1)
                
                pred=model.predict_classes(t)
                
                answer+=character[pred[0]]
    plt.figure(figsize=(10,10))
    plt.imshow(dp)

    return answer


