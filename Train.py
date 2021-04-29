import tkinter as tk
from tkinter import Message, Text
import cv2, os
import csv
import numpy as np
from PIL import Image, ImageTk
import tkinter.font as font

window = tk.Tk()
window.title("SISTEM MONITORING KEAMANAN RUMAH")
window.configure(background='black')
window.geometry('960x600')

lbl = tk.Label(window, text="Face Recognition Based Home Security System",
               bg="white" , fg="black" , width=40 , height=2,
               font=('times', 20, 'italic bold')) 
lbl.place(x=160, y=20)

lbl1 = tk.Label(window, text="Input ID", width=20 , height=2 , fg="black" ,
                bg="white", font=('times', 15, ' bold ') ) 
lbl1.place(x=80, y=150)

txt1 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
txt1.place(x=400, y=160)

lbl2 = tk.Label(window, text="Input Name", width=20 , fg="black", bg="white",
                height=2, font=('times', 15, ' bold ')) 
lbl2.place(x=80, y=250)

txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, ' bold ')  )
txt2.place(x=400, y=260)

lbl3 = tk.Label(window, text="Notifikasi →", width=25 , fg="black", bg="white", height=2, font=('times', 15, ' bold ')) 
lbl3.place(x=80, y=350)

message = tk.Label(window, text="", bg="white", fg="black", width=25, height=2, font=('times', 15, ' bold ')) 
message.place(x=400, y=350)
 
def clearId():
    txt1.delete(0, 'end')

def clearName():
    txt2.delete(0, 'end')

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

def takeImages():        
    Id=(txt1.get())
    name=(txt2.get())
    if(isNumber(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            img = cv2.flip(img, -1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                sampleNum=sampleNum+1
                cv2.imwrite("SampleImages\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('Face Detecting',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentRecord.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(isNumber(name)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(Id.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def trainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("SampleImages")
    recognizer.train(faces, np.array(Id))
    recognizer.save("DataSet\Trainner.yml")
    res = "Image Trained"
    message.configure(text= res)

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids
  
clearButton1 = tk.Button(window, text="Clear", command=clearId, fg="black", bg="white", width=15, height=1, activebackground = "Red", font=('times', 15, ' bold '))
clearButton1.place(x=700, y=150)

clearButton2 = tk.Button(window, text="Clear", command=clearName, fg="black", bg="white", width=15, height=1, activebackground = "Red", font=('times', 15, ' bold '))
clearButton2.place(x=700, y=250)  

takeImg = tk.Button(window, text="Take Images", command=takeImages, fg="black", bg="white", width=15, height=2, activebackground = "Green", font=('times', 15, ' bold '))
takeImg.place(x=90, y=450)

trainImg = tk.Button(window, text="Train Images", command=trainImages, fg="black", bg="white", width=15, height=2, activebackground = "Green" ,font=('times', 15, ' bold '))
trainImg.place(x=400, y=450)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=15, height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=680, y=450)

lbl4 = tk.Label(window, text="Created By : Dika Andra Joni | 2021", width=80, fg="white", bg="black", font=('times', 15, ' bold')) 
lbl4.place(x=160, y=550)

window.mainloop()
