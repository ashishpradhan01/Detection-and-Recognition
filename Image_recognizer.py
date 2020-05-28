import cv2
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join, split as ossplit

data_path = '/home/osboxes/PycharmProjects/Face-Detection-and-Recognition/datasets/'
sketch = cv2.imread("/home/osboxes/Desktop/Python Programs/Salman_khan_sketch",0)
image2 = cv2.imread("/home/osboxes/Desktop/Python Programs/salman_khan2.jpg")
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
# convert into grayscale & sketch
image2_gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
image2_gray_inv = 255 - image2_gray;
image2_blur = cv2.GaussianBlur(image2_gray_inv,(21,21),sigmaX=0,sigmaY=0)

# dodge
def dodgeV2(image, mask):
    return cv2.divide(image, 255-mask, scale=256)
#burning
def burningV2(image,mask):
    return 255-cv2.divide(255-image, 255-mask, scale=256)

image2_blend = dodgeV2(image2_gray, image2_blur)

for dir_count in range(len(listdir(data_path))):
    dir_name = listdir(data_path)[dir_count]

    print(dir_name)

    face_path = str(data_path + str(dir_name))
    print(face_path)

    onlyfiles = [f for f in listdir(face_path) if isfile(join(face_path, f))]

    print(onlyfiles)

    for files in onlyfiles:
        image_path = join(face_path, files)

        id = int(ossplit(image_path)[-1].split(".")[1])
        print(ossplit(image_path[-1]))
        print(id)

cv2.imshow("sketch", image2_blend)






cv2.waitKey(0)
cv2.destroyAllWindows()
















































# import numpy as np
# from os import listdir
# from PIL import Image
# from os.path import isfile, join, split as ossplit
# import
#
# image = cv2.imread("/home/osboxes/Desktop/Python Programs/salman_khan2.jpg")
# data_path = '/home/osboxes/PycharmProjects/Face-Detection-and-Recognition/datasets/'
#
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# # recognizer.read('trainer/trainer.yml')
# #To Resize the Image.
# scale_percent = 40
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)
# dsize = (width, height)
# image_resize= cv2.resize(image, dsize)
#
# gray_img = cv2.cvtColor(image_resize,cv2.COLOR_BGR2GRAY)
#
# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# face = face_classifier.detectMultiScale(gray_img, 1.5, 5)
#
#
# name,nameid = [],[]
# for dir_count in range(len(listdir(data_path))):
#     dir_name = listdir(data_path)[dir_count]
#
#     face_path = str(data_path + str(dir_name))
#
#     onlyfiles = [f for f in listdir(face_path) if isfile(join(face_path, f))]
#
#     for files_name in onlyfiles:
#         number = int(files_name.split(".")[2])
#         if number == 1:
#             name.append(str(files_name.split(".")[0]))
#             nameid.append(int(files_name.split(".")[1]))
#
# print("names",name,"\n id",nameid)
#
# def training_data():
#     Training_data, ids ,name,nameid= [], [],[], []  # Two Empty List
#     for dir_count in range(len(listdir(data_path))):
#         dir_name = listdir(data_path)[dir_count]
#
#         face_path=str(data_path+str(dir_name))
#
#         onlyfiles = [f for f in listdir(face_path) if isfile(join(face_path,f))]
#
#         for files in onlyfiles:
#             image_path = join(face_path,files)
#             PIL_img = Image.open(image_path).convert('L')
#             img_numpy = np.array(PIL_img,'uint8')
#             # img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#             id = int(ossplit(image_path)[-1].split(".")[1])
#
#             face = face_classifier.detectMultiScale(img_numpy)
#             for (x,y,w,h) in face:
#                 Training_data.append(img_numpy[y:y+h,x:x+w])
#                 ids.append(id)
#
#
#     return Training_data,ids
#
# print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
#
# Training_data,ids= training_data()
# recognizer.train(Training_data, np.array(ids))
#
# for x,y,w,h in face:
#     cv2.rectangle(image_resize,(x,y),(x+w,y+h),(0,255,0),1)
#     print(x,y,w,h)
#
# id, confidence = recognizer.predict(gray_img[y:y + h, x:x + w])
# # user name
# for i in range(len(nameid)):
#     if id == nameid[i]:
#         print("user name:",name[i])
#         break
#
# print('id',id,'\nconfidence',confidence)
# cv2.putText(image_resize,str(id),(x+int(w/2),y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(0,225,0),2)
# cv2.putText(image_resize,name[i],(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
# cv2.imshow("face group",image_resize)
#
#
# if cv2.waitKey(0) == 13:
#     pass
#
# cv2.destroyAllWindows()
