from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

model_path = 'D:\matth\Documents\projects\python\models\lenet_smiles_model\lenet_smiles.hdf5'
video_path = 'D:\matth\Videos\P90x\How To Bring It.mp4'
cascade_path = r"C:\Users\matth\PycharmProjects\Python_Learning\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"

detector = cv2.CascadeClassifier(cascade_path)

model = load_model(model_path)
camera = cv2.VideoCapture(video_path)

while True:
    # grab current frame
    (grabbed, frame) = camera.read()

    # if video ends exit
    if not grabbed:
        break

    # resize the frame, convert to grascle, and clone to draw on it later
    frame = imutils.resize(frame, width=200)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()
    # rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    rects = detector.detectMultiScale(gray, 1.1, 5)
    for (fX, fY, fW, fH) in rects:
        # extract ROI of face from grascale image, resize to 28,28 and prepare for CNN classifier
        roi = gray[fY:fY+fH, fX:fX+fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float")/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # processes roi with lenet classifier
        (not_smiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > not_smiling else "Not Smiling"

        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        if label is "Smiling":
            cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (0, 255, 0), 2)
        else:
            cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (0, 0, 255), 2)

        # outpt frame
        cv2.imshow("Face", frameClone)

        fps = 30
        secpframe = 1 / fps
        milliseconds = secpframe * 1000
        milliseconds = int(milliseconds)
        # break of q is pressed
        if cv2.waitKey(milliseconds) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
