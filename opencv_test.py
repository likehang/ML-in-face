import cv2
import numpy as np
import os
from sklearn import preprocessing
import time
from pylab import *
import csv

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

def print_csv(csv_file):
    csv_open = open(csv_file,'r') 
    guess = csv.reader(csv_open)
    for i in guess:
        print(i)
    csv_open.close()

def init_train_to_guess(csv_file,path_train,password,balance):
    with open(csv_file,'w') as save:
        fileheader = ['name','password','balance']
        dict_writer = csv.DictWriter(save,fileheader)
        dict_writer.writerow(dict(zip(fileheader,fileheader)))
        for root, dirs, files in os.walk(path_train):
            for file in dirs:
                filepath = os.path.join(root, file)
                name = filepath.split('\\')[-1]
                dict_writer.writerow({'name':name,'password':password,'balance':balance})
    print('Init Guess Success')
            
def remove_dir(dir):
    if(os.path.isdir(dir)):
        for p in os.listdir(dir):
            remove_dir(os.path.join(dir,p))
        if(os.path.exists(dir)):
            os.rmdir(dir)
    else:
        if(os.path.exists(dir)):
            os.remove(dir)
            
def login(path_test,csv_file,dec):
    cap = cv2.VideoCapture(0)
    filepath = os.path.join(path_test)
    start=time.time()
    grayfile = 'test.jpg'
    testfile=os.path.join(filepath,grayfile)
    for i in range(0,30):
        ret, frame = cap.read()
        cv2.imshow('Face Detector',frame)
        ignore = cv2.waitKey(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
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
                    break
        except AttributeError :
            pass
        end = time.time()
        if end-start>=10:
            print('Cancel...')
            face = False
            cap.release()
            cv2.destroyAllWindows()
            return number
    number =  str(predicted_person)
    password = input('输入密码：')
    data = []
    with open(csv_file,'r') as f:
        reader = csv.DictReader(f)
        boolean = True
        for guess in reader:
            csv_name ,csv_pw= guess['name'],guess['password']
            if (csv_name == number) and (str(csv_pw) == password):
                new_balance = float(guess['balance']) - float(dec)
                print(new_balance)
                while new_balance < 0:
                    print('余额不足，请交纳现金 '+str(-new_balance)+'元')
                    add = input('money')
                    new_balance = float(add) + new_balance
                boolean = False
                guess['balance']= new_balance
            data.append(guess)
        if boolean:
            print('error face or password ')
    with open(csv_file,'w') as f:
        fileheader = ['name','password','balance']
        writer = csv.DictWriter(f,fileheader)
        writer.writerow(dict(zip(fileheader,fileheader)))
        for i in data:
            #print(i)
            writer.writerow(i)
    return number
            
def create_new(password,balance):
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
            for j in range(0,5):
                ret, frame = cap.read()
                if ret:
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
                        break
            except AttributeError :
                pass
            end = time.time()
            if end-start>=10:
                print('超时。。。取消！')
                remove_dir(new_path)
                boolean=False
                return boolean
    else:
        print(new_number+'have exist!')
        boolean=False
    save =  open(csv_file,'a') 
    fileheader = ['name','password','balance']
    dict_writer = csv.DictWriter(save,fileheader)
    dict_writer.writerow({'name':new_number,'password':password,'balance':balance})
    save.close()
    return boolean

if __name__=='__main__':
    face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
    path_train = 'faces_dataset/train'
    path_test = 'faces_dataset/test'
    csv_file='list.csv'
    init_password = '123456'
    init_balance = '1000.0'
    scaling_factor = 1

    print('ready')
    if face_cascade.empty():
        raise IOError('Not find file')
    
    csvExists = os.path.exists(csv_file)
    if csvExists:
        #print_csv(csv_file)
    else:
        init_train_to_guess(csv_file,path_train,init_password,init_balance)
        csvExists = os.path.exists(csv_file)
        if csvExists:
            #print_csv(csv_file)
            
    print('\nTaining...\n')
    recognizer = cv2.face.LBPHFaceRecognizer_create()    
    images, labels, le = get_images_and_labels(path_train)
    recognizer.train(images, np.array(labels))
    
    while True:        
        n = input('\n1:新顾客\n2:付账\n0:退出\n')
        if n is '1':
            boolean = create_new(init_password,init_balance)
            if boolean:
                print("retrain...")
                images, labels, le = get_images_and_labels(path_train)
                recognizer.train(images, np.array(labels))
                print("OK")
        elif n is '2':
            dec = input('付款金额 = ')
            number = login(path_test,csv_file,dec)
        elif n is '0':
            exit()
