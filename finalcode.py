import cv2
import math
import argparse
import face_recognition
import keras
from keras.models import load_model
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten,BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint 
import tensorflow as tf

age_Array = []
gender_Array = []
image_Array = []
image_folder = './utkface-1'
image_files = os.listdir(image_folder)

for f in image_files:
    age_Value = int(f.split('_')[0])
    gender_Value = int(f.split('_')[1])
    total_Value = image_folder + '/' + f
    image  = cv2.imread(total_Value)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image= cv2.resize(image,(48,48))
    image_Array.append(image)
    age_Array.append(age_Value)
    gender_Array.append(gender_Value)

c_labels=[]
i=0

while i<len(age_Array):
  label=[]
  label.append([age_Array[i]])
  label.append([gender_Array[i]])
  c_labels.append(label)
  i+=1
  
images_v=np.array(image_Array)
labels_v=np.array(c_labels)
X_train, X_test, Y_train, Y_test= train_test_split(images_v, labels_v,test_size=0.25)
Y_train_2=[Y_train[:,1],Y_train[:,0]]
Y_test_2=[Y_test[:,1],Y_test[:,0]]

def conv(tensor_input,filtersList):
    convop = Conv2D(filters=filtersList,kernel_size=(3, 3),padding = 'same',strides=(1, 1),kernel_regularizer=l2(0.001))(tensor_input)
    convop = Dropout(0.1)(convop)
    convop= Activation('relu')(convop)
    return convop

def model(input_s):
  inputs = Input((input_s))
  convoluted_1= conv(inputs,32)
  maxpool_1 = MaxPooling2D(pool_size = (2,2)) (convoluted_1)
  convoluted_2 = conv(maxpool_1,64)
  maxpool_2 = MaxPooling2D(pool_size = (2, 2)) (convoluted_2)
  convoluted_3 = conv(maxpool_2,128)
  maxpool_3 = MaxPooling2D(pool_size = (2, 2)) (convoluted_3)
  convoluted_4 = conv(maxpool_3,256)
  maxpool_4 = MaxPooling2D(pool_size = (2, 2)) (convoluted_4)
  f_pool= Flatten() (maxpool_4)
  dense_one= Dense(64,activation='relu')(f_pool)
  dense_two= Dense(64,activation='relu')(f_pool)
  drop_one=Dropout(0.2)(dense_one)
  drop_two=Dropout(0.2)(dense_two)
  output_one= Dense(1,activation="sigmoid",name='sex_out')(drop_one)
  output_two= Dense(1,activation="relu",name='age_out')(drop_two)
  model = Model(inputs=[inputs], outputs=[output_one,output_two])
  model.compile(loss=["binary_crossentropy","mae"], optimizer="Adam",
	metrics=["accuracy"])
  return model

Model=model((48,48,3))
Model.summary()
mod='Age_sex_detection.h5'
pointcheck = ModelCheckpoint(mod, monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=False, mode='auto',save_freq='epoch')
stop=tf.keras.callbacks.EarlyStopping(patience=75, monitor='val_loss',restore_best_weights=True),
callback=[pointcheck,stop]

#History=Model.fit(X_train,Y_train_2,batch_size=64,validation_data=(X_test,Y_test_2),epochs=500,callbacks=[callback])
#Model.evaluate(X_test,Y_test_2)
#pred=Model.predict(X_test)

def hl_Face(net1, frame1, c_threshold=0.7):
    frame_Dnn1=frame1.copy()
    f_Height1=frame_Dnn1.shape[0]
    f_Width1=frame_Dnn1.shape[1]
    blob1=cv2.dnn.blobFromImage(frame_Dnn1, 1.0, (300, 300), [104, 117, 123], True, False)
    net1.setInput(blob1)
    detected_img=net1.forward()
    f_Boxes=[]
    for i in range(detected_img.shape[2]):
        conf=detected_img[0,0,i,2]
        if conf>c_threshold:
            x1=int(detected_img[0,0,i,3]*f_Width1)
            y1=int(detected_img[0,0,i,4]*f_Height1)
            x2=int(detected_img[0,0,i,5]*f_Width1)
            y2=int(detected_img[0,0,i,6]*f_Height1)
            f_Boxes.append([x1,y1,x2,y2])
            cv2.rectangle(frame_Dnn1, (x1,y1), (x2,y2), (0,255,0), int(round(f_Height1/150)), 8)
    return frame_Dnn1,f_Boxes

parser=argparse.ArgumentParser()
parser.add_argument('--image')
args=parser.parse_args()
f_Proto1="face_detector_opencv.pbtxt"
f_Model1="face_detector__opencv_uint8.pb"
age_Proto1="deploy_age.prototxt"
age_Model1="net_age.caffemodel"
gender_Proto1="deploy_gender.prototxt"
gender_Model1="net_gender.caffemodel"
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
age_List=['(0-2)', '(3-6)', '(7-12)', '(13-20)', '(21-32)', '(33-43)', '(44-53)', '(54-100)']
gender_List=['Male','Female']
emotion_dict= {'Angry': 0,'Disgust': 1, 'Fear': 2,'Happy': 3,  'Neutral': 4, 'Sad': 5,  'Surprise': 6  }
face_Net=cv2.dnn.readNet(f_Model1,f_Proto1)
age_Net=cv2.dnn.readNet(age_Model1,age_Proto1)
gender_Net=cv2.dnn.readNet(gender_Model1,gender_Proto1)
import tensorflow as tf
model = tf.keras.models.load_model('model_v6_23.hdf5')
model1 = tf.keras.models.load_model('Age_sex_detection-model.h5')
label_map = dict((v,k) for k,v in emotion_dict.items()) 
video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1)<0 :
    has_f,frame=video.read()
    if not has_f:
        cv2.waitKey()
        break  
    resultImg,face_b=hl_Face(face_Net,frame)
    if not face_b:
        print("No face detected")
    for faceBox in face_b:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        face_image = cv2.resize(face, (48,48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
        predicted_class = np.argmax(model.predict(face_image))
        predicted_label = label_map[predicted_class]
        print(f'Predicted Emotion: {predicted_label}')
        age_Net.setInput(blob)
        agePreds=age_Net.forward()
        age=age_List[agePreds[0].argmax()]
        print(f'Predicted Age: {age[1:-1]} years')
        face_image = cv2.resize(face, (48,48))
        face_image1 = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        f = cv2.resize(face_image1, (48,144))
        face_image1 = np.reshape(f,[face_image.shape[0], face_image.shape[1],3])  
        pred_1=model1.predict(np.array([face_image1]))  
        sex_f=['Male','Female']         
        print(pred_1[1])   
        sex=int(np.round(pred_1[0][0]))      
        print("Predicted Gender: "+ sex_f[sex])
        cv2.putText(resultImg, f'{sex_f[sex]}, {age} {predicted_label}', (faceBox[0]-30, faceBox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting Gender-age and Emotion", resultImg)


# In order to run the code
# ***** for webcam ****
#  python finalcode.py
# ***** for image ****
#  python finalcode.py --image=coco-1.jpg