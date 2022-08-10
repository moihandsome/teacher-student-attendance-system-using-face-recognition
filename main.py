import cv2
import numpy as np
import face_recognition as fr
import os
import datetime
import sqlite3
import csv
import pandas as pd

#CREATE DATABASE
con = sqlite3.connect('students.db')
c = con.cursor()
# c.execute('''
#     CREATE TABLE students (
#         student_id char(5),
#         full_name varchar(30),
#         phone_number int,
#         class char(3),
#         year_of_birth int,
#         place_of_birth varchar(20)
#     )
# ''')
# data = [
#     ('CS001', 'Bill Gates', 999999999, 'IT1', 1955, 'US'),
#     ('CS002', 'Cristiano Ronaldo ', 988888888, 'IT2', 1985, 'Portugal'),
#     ('CS003', 'Elon Musk', 911111111, 'IT3', 1971, 'South Africa'),
#     ('CS004', 'Lionel Messi', 955555555, 'IT4', 1987, 'Argentina')
# ]
# c.executemany('INSERT INTO students VALUES(?, ?, ?, ?, ?, ?)', data)

#CREATE CSV
now = datetime.datetime.now()
dayFormat = now.strftime("%m%d%Y")
timeFormat = now.strftime("%H%M%S")
subjectName = str(input('What is the subject name: '))
#CREATE CSV
def createCSV():
    with open(f'{subjectName}_{teacherId}_{dayFormat}_{timeFormat}.csv', 'w+') as f:
        filewriter = csv.writer(f, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(
            ['student_id', 'full_name', 'phone_number', 'class', 'year_of_birth', 'place_of_birth', 'checking_time'])

#WRITE CSV
def writeCSV(name):
    c.execute('select * from students')
    items = c.fetchall()
    with open(f'{subjectName}_{teacherId}_{dayFormat}_{timeFormat}.csv', 'r+') as f:
        myDataList = f.readlines()
        idList = []
        for line in myDataList:
            entry = line.split(',')
            idList.append(entry[0])
        id = name.split('_')[2]
        if id not in idList:
            for item in items:
                if item[0] == id:
                    now = datetime.datetime.now()
                    nowString = now.strftime('%H:%M:%S')
                    f.writelines(f'\n{id},{item[1]},{item[2]},{item[3]},{item[4]},{item[5]},{nowString}')
    con.commit()

#IMAGE PROCESS
pathTeachers = 'image/teachers'
pathStudents = 'image/students'
def imageProcess(path):
    images = []
    classNames = []
    myList = os.listdir(path)
    for a in myList:
        curImage = cv2.imread(f'{path}/{a}')
        images.append(curImage)
        classNames.append(os.path.splitext(a)[0])
    return images, classNames

#CONFIRM TEACHER ID
def confirmTeacherId(teacherId, path):
    images, classNames = imageProcess(path)
    idList = []
    for id in classNames:
        id = str(id.split('_')[2])
        idList.append(id)
    for teacherid in idList:
        if teacherid == teacherId:
            return True

#ENCODE IMAGE SOURCE
def findEncodings(images):
    endcodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (600, 600))
        encode = fr.face_encodings(img)[0]
        endcodeList.append(encode)
    return endcodeList

#CAMERA
camera_id = 0
cap = cv2.VideoCapture(camera_id)

#CHECK-IN TEACHER'S FACE
teacher = False
while True:
    teacherId = str(input('Teacher ID: '))
    if confirmTeacherId(teacherId, pathTeachers) == None:
        print('Wrong ID! Please try again...')
    else:
        print('ID Checked!')
        break
if confirmTeacherId(teacherId, pathTeachers) == True:
    while True:
        ret, frame = cap.read()
        frameS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
        faceCurFrame = fr.face_locations(frameS)
        encodeCurFrame = fr.face_encodings(frameS, faceCurFrame)
        images, classNames = imageProcess(pathTeachers)
        for faceLoca, faceEncode in zip(faceCurFrame, encodeCurFrame):
            matches = fr.compare_faces(findEncodings(images), faceEncode)
            faceDis = fr.face_distance(findEncodings(images), faceEncode)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoca
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f'{name}_{round((1-faceDis[matchIndex])*100)}%', (x1+6, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                createCSV()
                teacher = True
                print('Face Checked: ', name)
        cv2.imshow('Teacher Face-Check (Press ESC to exit)', frame)
        k = cv2.waitKey(0)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
df = pd.read_csv(f'{subjectName}_{teacherId}_{dayFormat}_{timeFormat}.csv')
df.dropna(how='all', inplace=True)
df.to_csv(f'{subjectName}_{teacherId}_{dayFormat}_{timeFormat}.csv', index=False)

