import cv2
import numpy as np
from PIL import Image
from os import listdir,mkdir
from os.path import isfile, join,exists, split as ossplit


data_path = '/home/osboxes/PycharmProjects/Face-Detection-and-Recognition/datasets/'
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
model = cv2.face.LBPHFaceRecognizer_create()



def training_data():
    Training_data, ids ,name,nameid= [], [],[], []  # Two Empty List
    for dir_count in range(len(listdir(data_path))):
        dir_name = listdir(data_path)[dir_count]

        face_path=str(data_path+str(dir_name))

        onlyfiles = [f for f in listdir(face_path) if isfile(join(face_path,f))]

        for files in onlyfiles:
            image_path = join(face_path,files)
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img,'uint8')
            # img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            id = int(ossplit(image_path)[-1].split(".")[1])

            face = face_detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in face:
                Training_data.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)


    return Training_data,ids

def model_train():
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

    Training_data,ids= training_data()
    model.train(Training_data, np.array(ids))


    if not exists('trainer'):
        mkdir('trainer')
    model.write('trainer/trainer.yml')  #Save the model into trainer/trainer.yml

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    cv2.destroyAllWindows()

# model_train()
import comparing_faces as cf
cf






