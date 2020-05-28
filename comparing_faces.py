import cv2
import numpy as np
from os import listdir
from os.path import isfile,join


data_path = '/home/osboxes/PycharmProjects/Face-Detection-and-Recognition/datasets/'

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX  # iniciate id counter
id =  0


name,nameid = [],[]
for dir_count in range(len(listdir(data_path))):
    dir_name = listdir(data_path)[dir_count]

    face_path = str(data_path + str(dir_name))

    onlyfiles = [f for f in listdir(face_path) if isfile(join(face_path, f))]


    for files_name in onlyfiles:
        number = int(files_name.split(".")[2])
        if number == 1:
            name.append(str(files_name.split(".")[0]))
            nameid.append(int(files_name.split(".")[1]))


# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)


while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print('id',id,
              '\nconfidence',confidence)

        # user name
        for i in range(len(nameid)):
            if id == nameid[i]:
                print("user name:",name[i])
                break


        #  take name from the image and print that name with image on screen

        # If confidence is less then 100 ==> "0" : perfect match
        if (confidence < 100):
            print('id:', id)
            confidence = "  {0}%".format(round(confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(confidence))

        cv2.putText(
            img,
            str(id),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            name[i],
            (x + 80, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break  # Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()















































#Comparing datasets stored face and Realtime face
# import cv2
# import os

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trainer/trainer.yml')
# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# path = 'datasets/'
# face =[]
# id = []
# onlyfiles=[]
# # for each_dir in os.listdir(path):
# #     for user_face in os.listdir(path + each_dir):
# #         if os.path.isfile(os.path.join(path + each_dir,user_face)):
# #             onlyfiles.append(user_face)
#
#
#
# def compare_faces(model):
#
#     face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
#
#     def face_detector(image, size = 0.5):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         face = face_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
#
#         if face ==():
#             return image,[]
#
#
#
#         for (x,y,w,h) in face:
#             cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#             roi = image[y:y+h, x:x+w] #reason of interset
#             roi = cv2.resize(roi,(200,200))
#
#         return image,roi
#
#
#     cap = cv2.VideoCapture(0)
#
#     while True:
#         ret,image = cap.read()
#
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         face_class = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#
#         for (x, y, w, h) in face_class:
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             roi = image[y:y + h, x:x + w]  # reason of interset
#             roi = cv2.resize(roi, (200, 200))
#
#         # image, face= face_detector(frame)
#
#         try:
#             roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
#             result = model.predict(roi)
#             if result[1] < 500:
#                 confidence = int(100*(1-(result[1])/300))
#                 display_confidence = str(confidence)+'% Confidence it is user'
#             cv2.putText(image,display_confidence,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
#
#             if confidence > 85:
#                 # cv2.putText(
#                 #     image,
#                 #     str(dir_name),
#                 #     (x + 5, y - 5),
#                 #     cv2.FONT_HERSHEY_COMPLEX,
#                 #     1,
#                 #     (255, 255, 255),
#                 #     2
#                 # )
#                 cv2.putText(image, "Unlocked", (255, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 2)
#                 cv2.imshow('Face cropper',image)
#             else:
#                 cv2.putText(image, "Locked", (255, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#                 cv2.imshow('Face cropper', image)
#
#         except:
#             cv2.putText(image, "Face Not Found", (255, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
#             cv2.imshow('Face cropper', image)
#             pass
#
#         if cv2.waitKey(10)==13:
#             break
#
#
#     cap.release()
#     cv2.destroyAllWindows()
#
