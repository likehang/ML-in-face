import cv2
import numpy as np
import os
from sklearn import preprocessing
import time
from pylab import *

# Class to handle tasks related to label encoding
class LabelEncoder(object):
    # Method to encode labels from words to numbers
    def encode_labels(self, label_words):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label_words)

    # Convert input label from word to number
    def word_to_num(self, label_word):
        return int(self.le.transform([label_word])[0])

    # Convert input label from number to word
    def num_to_word(self, label_num):
        return self.le.inverse_transform([label_num])[0]

# Extract images and labels from input path
def get_images_and_labels(input_path):
    label_words = []

    # Iterate through the input path and append files
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            label_words.append(filepath.split('\\')[-2]) 
            
    # Initialize variables
    images = []
    le = LabelEncoder()
    le.encode_labels(label_words)
    labels = []

    # Parse the input directory
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)

            # Read the image in grayscale format
            image = cv2.imread(filepath, 0) 

            # Extract the label
            name = filepath.split('\\')[-2]
                
            # Perform face detection
            faces = face_cascade.detectMultiScale(image, 1.1, 2, minSize=(100,100))

            # Iterate through face rectangles
            for (x, y, w, h) in faces:
                images.append(image[y:y+h, x:x+w])
                labels.append(le.word_to_num(name))

    return images, labels, le

def login(path_test):
    cap = cv2.VideoCapture(0)
    filepath = os.path.join(path_test)
    start=time.time()
    im=[]
    grayfile = 'test.jpg'
    testfile=os.path.join(filepath,grayfile)
    for i in range(0,30):
        ret, frame = cap.read()
        cv2.imshow('Face Detector',frame)
        ignore = cv2.waitKey(1)
    while True:
        ret, frame = cap.read()
        cv2.imwrite(testfile,frame)
        frame = cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,
                               interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.1, 2,minSize=(100,100))
        for (x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.imshow('Face Detector',frame)
        ignore = cv2.waitKey(1)
        try:
            if face.all() :
                cap.release()
                cv2.destroyAllWindows()
                predicted_person=-1
                for root, dirs, files in os.walk(path_test):
                    for filename in (x for x in files if x.endswith('.jpg')):
                        filepath = os.path.join(root, filename)
                            
                        predict_image = cv2.imread(filepath, 0)
                            
                        faces = face_cascade.detectMultiScale(predict_image, 1.1, 2, minSize=(100,100))
                            
                        for (x, y, w, h) in faces:
                            predicted_index, conf = recognizer.predict(predict_image[y:y+h, x:x+w])
                                
                            predicted_person = le.num_to_word(predicted_index)
                                            
                if predicted_person != -1 :
                    print(predicted_person)
                    face = True
                    return
        except AttributeError :
            pass
        end = time.time()
        if end-start>=10:
            print('Cancel...')
            face = False
            break
    cap.release()
    cv2.destroyAllWindows()
    return 
            
def create_new():
    new_number = input('number: ')
    new_path = os.path.join(path_train,new_number)
    isExist = os.path.exists(new_path)
    if not isExist:
        os.makedirs(new_path)
        cap = cv2.VideoCapture(0)
        for j in range(0,50):
            ret, frame = cap.read()
            cv2.imshow('Please move face in cam...',frame)
            ignore = cv2.waitKey(1)
        i=0
        start = time.time()
        while True:
            for j in range(0,10):
                ret, frame = cap.read()
                frame = cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,
                           interpolation=cv2.INTER_AREA)
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                saveface = face_cascade.detectMultiScale(gray, 1.3, 2,minSize=(100,100))
                for (x,y,w,h) in saveface:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.imshow('Please move face in cam...',frame)
                ignore = cv2.waitKey(1)
            try:
                if saveface.all():
                    ret, frame = cap.read()
                    cv2.imwrite(new_path+'\\'+str(i)+'.jpg',frame)
                    i+=1
                    if i>=5 :
                        print('create new number:',new_number)
                        cap.release()
                        cv2.destroyAllWindows()
                        boolean=True
                        return boolean
            except AttributeError :
                pass
            end = time.time()
            if end-start>=10:
                print('超时。。。取消！')
                os.removedirs(new_path)
                boolean=False
                return boolean
    else:
        print(new_number+'have exist!')
        boolean=False
    return boolean

if __name__=='__main__':
    face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
    path_train = 'faces_dataset/train'
    path_test = 'faces_dataset/test'
    scaling_factor = 1

    print('ready')
    if face_cascade.empty():
        raise IOError('Not find file')

    print('\nTaining...\n')
    recognizer = cv2.face.LBPHFaceRecognizer_create()    
    images, labels, le = get_images_and_labels(path_train)
    recognizer.train(images, np.array(labels))
    
    while True:        
        n = input('\n1:新顾客\n2:付账\n0:退出\n')
        if n is '1':
            boolean= create_new()
            if boolean:
                print("retrain...")
                images, labels, le = get_images_and_labels(path_train)
                recognizer.train(images, np.array(labels))
                print("OK")
        elif n is '2':
            number = login(path_test)
        elif n is '0':
            exit()
