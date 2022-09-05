import os, os.path, sys, time
import cv2, numpy
from collections import Counter 
from PIL import Image
import shutil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DIR = 'C:\\Users\\RT-Lab\\AppData\\Roaming\\iSpy\\WebServerRoot\\Media\\video\\KXAAX\\grabs'
init_photos = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def get_images_and_labels():
    path = './yalefaces'
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = numpy.array(image_pil, 'uint8')
        subject = int(os.path.split(image_path)[1].split("_")[1])
        faces = faceCascade.detectMultiScale(image)
        if len(faces) == 0:
            print("Please replace "+image_path+" from references")
        for (x, y, w, h) in faces:
            images.append(image[y:y+h, x:x+w])
            labels.append(subject)
    recognizer.train(images, numpy.array(labels))
    print("Training complete")
    
def facerect(imagePath):
    # print(imagePath)
    try:
        image = cv2.imdecode(numpy.fromfile(imagePath, numpy.uint8), cv2.IMREAD_UNCHANGED)
        predict_image = numpy.array(Image.open(imagePath).convert('L'), 'uint8')
        faces = faceCascade.detectMultiScale(image)
        if len(faces) == 0:
            print("Face not recognized")
        for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y:y+h, x:x+h])
            print ("Is believed to be person "+str(nbr_predicted)+" with "+str(conf)+" confidence")
    except:
        print('mxnet imdecode failed to load image.')    

if __name__== "__main__":
    get_images_and_labels()
    while 1:
        folder = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
        if len(folder) > init_photos:
            init_photos = len(folder)
            facerect(newest(DIR))