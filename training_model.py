import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = '/home/osboxes/PycharmProjects/Face-Detection-and-Recognition/datasets/'

user_name= []

dir_name = listdir(data_path)[0]
face_path=str(data_path+str(dir_name))


onlyfiles = [f for f in listdir(face_path) if isfile(join(face_path,f))]
print(onlyfiles)

Training_data, Labels = [], [] #Two Empty List

for i, files in enumerate(onlyfiles):
    image_path = join(face_path,files)
        # data_path + files
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_data), np.asarray(Labels))

print('Model Training Complete!!!')

#Comparing datasets stored face and Realtime face

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detector(image, size = 0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)

    if face ==():
        return image,[]



    for (x,y,w,h) in face:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        roi = image[y:y+h, x:x+w] #reason of interset
        roi = cv2.resize(roi,(200,200))

        # print('image:',image,'roi',roi)

    # cv2.putText(image,str(dir_name),(x + 5, y - 5),cv2.FONT_HERSHEY_COMPLEX,1,(255, 255, 255),2)
        print(x)
       try:
         print(x)
       except:
           pass

    return image,roi


cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()


    image, face= face_detector(frame)


    try:
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            display_confidence = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_confidence,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

        if confidence > 75:
            # cv2.putText(
            #     image,
            #     str(dir_name),
            #     (x + 5, y - 5),
            #     cv2.FONT_HERSHEY_COMPLEX,
            #     1,
            #     (255, 255, 255),
            #     2
            # )
            cv2.putText(image, "Unlocked", (255, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 2)
            cv2.imshow('Face cropper',image)
        else:
            cv2.putText(image, "Locked", (255, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face cropper', image)

    except:
        cv2.putText(image, "Face Not Found", (255, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face cropper', image)
        pass

    if cv2.waitKey(10)==13:
        break

cap.release()
cv2.destroyAllWindows()

