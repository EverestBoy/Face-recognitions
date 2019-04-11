import requests
import cv2
import numpy as np
import os


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
nose_cascade = cv2.CascadeClassifier('cascades/third-party/Nose18x15.xml')
eye_cascade =  cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

count = 1

check = True

while check:
    user_input = input ("Enter\n1. to take input\n2. to exit")
    val = int(user_input)
    
    if val == 1:
        name = input("Enter user name: ")

        os.mkdir('images\\'+name)


        while True:

            

            img_receive = requests.get('http://192.168.10.50:8080/shot.jpg')
            imgNp = np.array(bytearray(img_receive.content), dtype=np.uint8)
            gray = cv2.imdecode(imgNp, -1)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=2.5, minNeighbors=5)
            
            
            
            for (x, y, w, h) in faces:

                roi_gray = gray[y:y+h, x:x+w]
                roi_color = gray[y:y+h, x:x+w]

                # if count < 8:
                # 	img_item = str(count)+".png"
                # 	cv2.imwrite(img_item, roi_gray)
                # 	count += 1

                

                # print(x,y,w,h) 
                color = (255,0,0,0)
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                nose = nose_cascade.detectMultiScale(roi_gray)
                eye = eye_cascade.detectMultiScale(roi_gray)
                
                for (ex,ey,ew,eh) in eye:
                    for (exe,eye,ewe,ehe) in nose:
                        if(count < 50):
                            # path = '/images/'+name+'/'+str(count)+'.png'
                            # cv2.imwrite(path, roi_gray)
                            face_file_name = "images\\" + name +"\\" + str(count) + ".jpg"
                            cv2.imwrite(face_file_name, gray)
                            print('saved')
                            count += 1
                        else:
                             exit(0)


               
                
                
            
            
            cv2.imshow('image',gray)
            
            
            

            if(ord('q') == cv2.waitKey(10)):
                exit(0)
       
    else:
        print("Thank you")
        check = False
