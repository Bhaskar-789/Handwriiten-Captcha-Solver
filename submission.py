'''
This will be main file which the co-ordinaters of the event will be using to test your
code. This file contains two functions:

1. predict: You will be given an rgb image which you will use to predict the output 
which will be a string. For the prediction you can use/import code,models from other files or
libraries. More detailes given above the function defination.

2. test: This will be used by the co-ordinators to test your code by giving sample 
inputs to the 'predict' function mentioned above. A sample test function is given for your
reference but it is subject to minor changes during the evaluation. However, note that
there won't be any changes in the input format given to the predict function.

Make sure all the necessary functions etc. you import are done from the same directory. And in 
the final submission make sure you provide them also along with this script.
'''

# Essential libraries can be imported here
import os
import cv2
import numpy as np 

'''
function: predict
input: image - A numpy array which is an rgb image
output: answer - A string which is the full captcha
'''
def predict(image):
    '''
    Write your code for prediction here.
    '''
    import evaluate
    answer = 'xyzabc' # sample needs to be modified
    img=image
    model = evaluate.load('./merge4.h5')
    test_data,test_labels = evaluate.load_data()
    
    test=test_data
    #train_l=train_labels
    test_l=test_labels
    
    test_data,test_labels= evaluate.preprocessing(test_data,test_labels)
    prediction=evaluate.predictor(model,test_data,250)

    accuracy=evaluate.test_accuracy(prediction,test_l)

    best_acc,character=evaluate.best_accuracy(prediction,test_l)

    print(best_acc)

    thresh,im=evaluate.thresholding(image)

    gsblur=evaluate.cleaning(thresh)

    answer=evaluate.extract(model,gsblur,im,character)

    print(answer)

    return answer


'''
function: test
input: None
output: None

This is a sample test function which the co-ordinaors will use to test your code. This is
subject to change but the imput to predict function and the output expected from the predict
function will not change. 
You can use this to test your code before submission: Some details are given below:
image_paths : A list that will store the paths of all the images that will be tested.
correct_answers: A list that holds the correct answers
score : holds the total score. Keep in mind that scoring is subject to change during testing.

You can play with these variables and test before final submission.
'''
def test():
    '''
    We will be using a similar template to test your code
    '''
    image_paths = ['./test_image/test1.jpg']
    correct_answers = ['49518']   
    #provide the correct answer for the captcha here to test the model
    score = 0

    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predict(image) # a string is expected

        if correct_answers[i] == answer:
            score += 10
    
    print('The final score of the participant is',score)


if __name__ == "__main__":
    test()