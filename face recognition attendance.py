import face_recognition
import cv2
import numpy as np
import csv
import os
import glob
from datetime import *


video_capture = cv2.VideoCapture(0)

ayush_image = face_recognition.load_image_file(r"location of the pic")
ayush_encoding = face_recognition.face_encodings(ayush_image)[0]

ratan_image = face_recognition.load_image_file(r"location of the pic")
ratan_encoding = face_recognition.face_encodings(ratan_image)[0]

jeff_image = face_recognition.load_image_file(r"location of the pic")
jeff_encoding = face_recognition.face_encodings(jeff_image)[0]

known_faces_encoding = [ayush_encoding,ratan_encoding,jeff_encoding]

known_faces_names = ["ayush","ratan tata", "bald jeff"]

students = known_faces_names.copy()

face_encodings = []
face_locations = []
face_names= []


now = datetime.now()
current_Date = now.strftime("%d-%m-%y")



f = open(current_Date+'.csv','w+',newline= '')
lnwrite = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    if True:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces_encoding,face_encoding)
            name= ''

            face_distance = face_recognition.face_distance(known_faces_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name) 
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwrite.writerow([name, current_time])

                if not students:
                    video_capture.release()
                    cv2.destroyAllWindows
                    f.close()
                    break





    cv2.imshow("attendance system" , frame)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()




