import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from keras.models import load_model
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

path = 'vid.mp4'
fourcc = cv2.VideoWriter_fourcc(*tuple([f for f in 'MP4V']))
clip = VideoFileClip(path)
writer = cv2.VideoWriter(path[:-4]+'_smile_detected.mp4', fourcc, clip.fps, tuple(clip.size), True)
video_capture = cv2.VideoCapture(path)
prog = tqdm(total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
model = load_model('128_500.h5')
while True:
    ret, img = video_capture.read()
    if not ret:
        break
    img = img[:,:,::-1]
    img_copy = img.copy()
    rects = detector(img)
    pred = predictor(img,rects[0])
    features = []
    # pt_img = img[rects[0].top():rects[0].bottom(),rects[0].left():rects[0].right()].copy()
    for x,y in [(p.x,p.y) for p in list(pred.parts())]:
        features.append((x-rects[0].left())/rects[0].width())
        features.append((y-rects[0].top())/rects[0].height())
        # cv2.circle(pt_img, (x-rects[0].left(), y-rects[0].top()), 5, (0, 0, 255), -1)
        cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)
    features = np.array(features).reshape(1,-1)
    prediction = model.predict(features)[0][0]
    if prediction > 0.5:
        text1 = 'Smiling'
        color = (100,255,100)
    else:
        text1 = 'Not Smiling'
        color = (0,0,0)
        prediction = 1 - prediction
    cv2.putText(img_copy,text1,(img.shape[1]-225,30), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = color,thickness=2)
    cv2.putText(img_copy,f'Prob.: {prediction:.4f}',(img.shape[1]-225,60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = color,thickness=2)
    writer.write(img_copy[:,:,::-1])
    prog.update(1)
prog.close()
del prog
video_capture.release()
writer.release()