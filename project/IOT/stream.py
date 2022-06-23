import cv2
import PIL.Image, PIL.ImageTk
from tkinter import *
from cv2 import LINE_AA
import numpy as np
from urllib.request import urlopen
import math
import mediapipe as mp
import time
from mss import mss
from PIL import Image
import  controller as control


FRAME_WIDTH  = 1280
FRAME_HEIGHT =  750

# FRAME_WIDTH  = 1050
# FRAME_HEIGHT =  730
LED_COUNT = 0
VOLUME = 0

THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16

BLUE_LOC = 200
BLUE_VAL = 0

GREEN_LOC = 200
GREEN_VAL = 0

RED_LOC = 200
RED_VAL = 0

CON_VAL = 0




Z_THRESHOLD_PRESS = -200

VK = {
    'LED': { 'x':200,  'y':100, 'w':300, 'h':150 },
    'COUNT': { 'x':700, 'y':100, 'w':300, 'h':150 },
    'RGB': { 'x':200, 'y':400, 'w':300, 'h':150 },
    'VIDEO': { 'x':700, 'y':400, 'w':300, 'h':150 }
}

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
canvas=None
global flag 
flag = 0


def draw_keys(img, x, y, z):
    for k in VK:
        if ((VK[k]['x'] < x < VK[k]['x']+VK[k]['w']) and (VK[k]['y'] < y < VK[k]['y']+VK[k]['h'])):
            cv2.rectangle(img, (VK[k]['x'], VK[k]['y']), (VK[k]['x']+VK[k]['w'], VK[k]['y']+VK[k]['h']), (0,0,255), 1) # thickness -1 means filled rectangle
            cv2.putText(img, f"{k}", (VK[k]['x']+30,VK[k]['y']+70), cv2.FONT_HERSHEY_SIMPLEX, 2, (abs(z),255,abs(z)), abs(z//20), cv2.LINE_AA)
        else:
            cv2.rectangle(img, (VK[k]['x'], VK[k]['y']), (VK[k]['x']+VK[k]['w'], VK[k]['y']+VK[k]['h']), (0,255,0), 1) # thickness -1 means filled rectangle
            cv2.putText(img, f"{k}", (VK[k]['x']+30,VK[k]['y']+70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)


def draw_img(results,img,x,y,z):
    global flag
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # DEBUG
            try:
                index_finger_tip = handLms.landmark[INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * FRAME_WIDTH)
                y = int(index_finger_tip.y * FRAME_HEIGHT)
                z = int(index_finger_tip.z * FRAME_WIDTH)
                #print(f"x={x} , y={y} , z={z}")
                if ((z <= Z_THRESHOLD_PRESS) and (200 <=x <= 500) and (100 <= y <= 250)):
                    color = (0,0,255) # BGR
                    time.sleep(1)
                    flag =1
                elif ((z <= Z_THRESHOLD_PRESS) and (700 <=x <= 1000) and (100 <= y <= 250)):
                    color = (0,0,255) # BGR
                    time.sleep(1)
                    flag =2
                elif ((z <= Z_THRESHOLD_PRESS) and (200 <=x <= 500) and (400 <= y <= 550)):
                    color = (0,0,255) # BGR
                    time.sleep(1)
                    flag =3
                elif ((z <= Z_THRESHOLD_PRESS) and (700 <=x <= 1000) and (400 <= y <= 550)):
                    color = (0,0,255)
                    time.sleep(1)
                    flag = 4
                else:
                    color = (0,255,0)
                
                cv2.putText(img, f"{x}, {y}, {z}", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.circle(img,(x,y),30,(0,0,255),1)
                if(z<0):
                    cv2.circle(img,(x,y),-int(z/6),(0,0,255),-1)
                
                
            except IndexError:
                index_finger_tip = None
                
        draw_keys(img, x, y, z)
    cv2.imshow("OpenCV Video Capture", img)
    return img

def draw_img2(results,img,x,y,z):
    global flag
    global VOLUME
    
    op_finger_x =0 

    cv2.rectangle(img,(10,10),(120,100),(0,255,0),1)
    cv2.putText(img,"<-",(20,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5,cv2.LINE_AA)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # DEBUG
            
            
            try:
                index_finger_tip = handLms.landmark[INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * FRAME_WIDTH)
                op_finger_x = x
                
                
                y = int(index_finger_tip.y * FRAME_HEIGHT)
                z = int(index_finger_tip.z * FRAME_WIDTH)
                print(f"x={x} , y={y} , z={z}")
                if ((z <= Z_THRESHOLD_PRESS) and (10<= x <=120) and (10 <= y <= 100)):
                    color = (0,0,255) # BGR
                    time.sleep(1)
                    flag =0
                else:
                    color = (0,255,0)
                
                cv2.putText(img, f"{x}, {y}, {z}", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            except IndexError:
                index_finger_tip = None
            if (op_finger_x>500):
                finger1_x = int(handLms.landmark[4].x * FRAME_WIDTH )
                finger1_y = int(handLms.landmark[4].y * FRAME_HEIGHT )
                finger2_x = int(handLms.landmark[8].x * FRAME_WIDTH )
                finger2_y = int(handLms.landmark[8].y * FRAME_HEIGHT )
                
                dist_x = finger2_x - finger1_x
                dist_y = finger2_y - finger1_y
                dist = math.hypot(dist_x,dist_y)
                
                dist_min = 30
                old_range = 370
                scale_range = 100
                
                new_Value = (((dist-dist_min) * scale_range) // old_range)
                VOLUME = int(new_Value)
                cv2.line(img,(finger1_x,finger1_y),(finger2_x,finger2_y),(0,255,0),thickness=3,lineType=cv2.LINE_AA)
                cv2.circle(img,(finger1_x,finger1_y),15,(255,0,255),cv2.FILLED)
                cv2.circle(img,(finger2_x,finger2_y),15,(255,0,255),cv2.FILLED)
                
                
    cv2.rectangle(img,(50,150),(85,450),(0,0,255),3)
    cv2.rectangle(img,(50,450),(85,450 - (VOLUME*3)),(0,0,255),cv2.FILLED)
    cv2.putText(img,f'{VOLUME} %',(40,500), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1, cv2.LINE_AA)    
            
    cv2.line(img,(500,0),(500,700),(0,0,255),5,cv2.LINE_AA)
    cv2.putText(img,"Operating this Area",(600,50),cv2.FONT_HERSHEY_SIMPLEX,2,(123,123,0),5,cv2.LINE_AA)
            
    cv2.imshow("OpenCV",img)
    return img
    
    
    
    
    
    
def draw_img3(result,img):
    global flag
    global LED_COUNT
    mp_draw=mp.solutions.drawing_utils
    mp_hand=mp.solutions.hands
    lmList=[]
    tipIds=[4,8,12,16,20]
    op_finger_x =0 
    
    
    
    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            myHands=result.multi_hand_landmarks[0]
            for id, lm in enumerate(myHands.landmark):
                h,w,c=img.shape # 이미지의 값을 x,y,z 좌표로 출력
                cx,cy= int(lm.x*w), int(lm.y*h) # 2차원 형태로 값을 변환
                lmList.append([id,cx,cy]) # 각 포인트(랜드마크)별 좌표값을 저장
            # 즉, 각 손가락 랜드마크의 좌표값이 일정 범위보다 아래가 되었을 시 다른 값이라고 판별
            mp_draw.draw_landmarks(img, hand_landmark, mp_hand.HAND_CONNECTIONS)
            try:
                index_finger_tip = hand_landmark.landmark[INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * FRAME_WIDTH)     
                
                op_finger = hand_landmark.landmark[20]
                op_finger_x = int(op_finger.x * FRAME_WIDTH) 
                y = int(index_finger_tip.y * FRAME_HEIGHT)
                z = int(index_finger_tip.z * FRAME_WIDTH)
                print(f"x={x} , y={y} , z={z}")
                if ((z <= Z_THRESHOLD_PRESS) and (10<= x <=120) and (10 <= y <= 100)):
                    color = (0,0,255) # BGR
                    time.sleep(1)
                    flag =0
                else:
                    color = (0,255,0)
                cv2.putText(img, f"{x}, {y}, {z}", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            except IndexError:
                index_finger_tip = None
    fingers=[]
    if (len(lmList)!=0) and (op_finger_x>500): # 손이 인식되었을 경우만 작동
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]: # 손가락 마디 끝을 설정
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        total=fingers.count(1)
        if total==0:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 0 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
            LED_COUNT = 0
        elif total==1:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 1 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
            LED_COUNT = 1
        elif total==2:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 2 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
            LED_COUNT = 2
        elif total==3:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 3 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
            LED_COUNT = 3
        elif total==4:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 4 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
            LED_COUNT = 4
        elif total==5:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 5 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
            LED_COUNT = 5
            
        #control.countLED(LED_COUNT)
        
    cv2.rectangle(img,(10,10),(120,100),(0,255,0),1)
    cv2.putText(img,"<-",(20,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5,cv2.LINE_AA)
    cv2.line(img,(500,0),(500,700),(0,0,255),5,cv2.LINE_AA)
    cv2.putText(img,"Operating this Area",(600,50),cv2.FONT_HERSHEY_SIMPLEX,2,(123,123,0),5,cv2.LINE_AA)
    cv2.putText(img,f"{LED_COUNT}", (74, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
    cv2.imshow("OpenCV",img)
    return img




def draw_img4(results,img):
    global flag
    global BLUE_LOC
    global GREEN_LOC
    global RED_LOC
    
    global BLUE_VAL
    global GREEN_VAL
    global RED_VAL
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # DEBUG
            try:
                index_finger_tip = handLms.landmark[INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * FRAME_WIDTH)
                y = int(index_finger_tip.y * FRAME_HEIGHT)
                z = int(index_finger_tip.z * FRAME_WIDTH)
                
                print(f"x={x} , y={y} , z={z}")
                if ((z <= Z_THRESHOLD_PRESS) and (10<= x <=120) and (10 <= y <= 100)):
                    color = (0,0,255) # BGR
                    time.sleep(1)
                    flag =0
                else:
                    color = (0,255,0)
                cv2.putText(img, f"{x}, {y}, {z}", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                if((z <= Z_THRESHOLD_PRESS) and (200<= x <=1200) and (100 <= y <= 180)):
                    cv2.rectangle(img,(x-50,100),(x+50,180),(255,255,255),-1)
                    BLUE_LOC = x
                    BLUE_VAL = ((x-200)*255)//1000
                    
                else:
                    cv2.rectangle(img,(BLUE_LOC-50,100),(BLUE_LOC+50,180),(255,0,0),-1)
                    
                
                if((z <= Z_THRESHOLD_PRESS) and (200<= x <=1200) and (250 <= y <= 330)):
                    cv2.rectangle(img,(x-50,250),(x+50,330),(255,255,255),-1)
                    GREEN_LOC = x
                    GREEN_VAL = ((x-200)*255)//1000
                else:
                    cv2.rectangle(img,(GREEN_LOC-50,250),(GREEN_LOC+50,330),(0,255,0),-1)
                    
                    
                if((z <= Z_THRESHOLD_PRESS) and (200<= x <=1200) and (400 <= y <= 480)):
                    cv2.rectangle(img,(x-50,400),(x+50,480),(255,255,255),-1)
                    RED_LOC = x
                    RED_VAL = ((x-200)*255)//1000
                else:
                    cv2.rectangle(img,(RED_LOC-50,400),(RED_LOC+50,480),(0,0,255),-1)
                #control.control_RGB(RED_VAL/255,GREEN_VAL/255,BLUE_VAL/255)
            except IndexError:
                index_finger_tip = None
    
    
    cv2.rectangle(img,(200,100),(1200,180),(255,0,0),3,0)
    cv2.rectangle(img,(200,250),(1200,330),(0,255,0),3,0)
    cv2.rectangle(img,(200,400),(1200,480),(0,0,255),3,0)
    cv2.putText(img,f"B: {BLUE_VAL},   G: {GREEN_VAL},   R: {RED_VAL}  ",(400,550),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),5,cv2.LINE_AA)
    
    cv2.rectangle(img,(10,10),(120,100),(0,255,0),1)
    cv2.putText(img,"<-",(20,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5,cv2.LINE_AA)
    
    cv2.imshow("OpenCV",img)
    return img



def draw_img5(results,img,x,y,z):
    global CON_VAL
    global flag
    theta = 0
    op_finger_x =0 
    lmList=[]
    tipIds=[4,8,12,16,20]
    color = 0
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # DEBUG
            myHands=results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHands.landmark):
                h,w,c=img.shape # 이미지의 값을 x,y,z 좌표로 출력
                cx,cy= int(lm.x*w), int(lm.y*h) # 2차원 형태로 값을 변환
                lmList.append([id,cx,cy]) 
            
            
            
            try:
                index_finger_tip = handLms.landmark[INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * FRAME_WIDTH)
                op_finger_x = x
                
                
                y = int(index_finger_tip.y * FRAME_HEIGHT)
                z = int(index_finger_tip.z * FRAME_WIDTH)
                #print(f"x={x} , y={y} , z={z}")
                if ((z <= Z_THRESHOLD_PRESS) and (10<= x <=120) and (10 <= y <= 100)):
                    color = (0,0,255) # BGR
                    time.sleep(1)
                    flag =0
                else:
                    color = (0,255,0)
                
                cv2.putText(img, f"{x}, {y}, {z}", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            except IndexError:
                index_finger_tip = None
            
                
            finger1_x = int(handLms.landmark[4].x * FRAME_WIDTH )
            finger1_y = int(handLms.landmark[4].y * FRAME_HEIGHT )
            finger2_x = int(handLms.landmark[8].x * FRAME_WIDTH )
            finger2_y = int(handLms.landmark[8].y * FRAME_HEIGHT )
                
            dist_x = finger2_x - finger1_x
            dist_y = finger2_y - finger1_y
            dist = math.hypot(dist_x,dist_y)
                
            dist_min = 30
            old_range = 370
            scale_range = 100
                
            new_Value = (((dist-dist_min) * scale_range) // old_range)
            CON_VAL = int(new_Value)
                #print(CON_VAL)
            if (op_finger_x>650):
                if(CON_VAL >= 55):
                    cv2.line(img,(finger1_x,finger1_y),(finger2_x,finger2_y),(0,255,255),thickness=3,lineType=cv2.LINE_AA)
                    cv2.putText(img,'VOL_UP!',(660,80), cv2.FONT_HERSHEY_SIMPLEX, 3,(0,255,255), 5, cv2.LINE_AA)   
                    control.Volume_up() 
                elif(CON_VAL <= 10):
                    cv2.line(img,(finger1_x,finger1_y),(finger2_x,finger2_y),(255,255,0),thickness=3,lineType=cv2.LINE_AA)
                    cv2.putText(img,'VOL_DOWN!',(660,80), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,0), 5, cv2.LINE_AA)  
                    control.Volume_Down()
                else:
                    cv2.line(img,(finger1_x,finger1_y),(finger2_x,finger2_y),(0,0,0),thickness=3,lineType=cv2.LINE_AA)
                
                
                cv2.circle(img,(finger1_x,finger1_y),15,(255,0,255),cv2.FILLED)
                cv2.circle(img,(finger2_x,finger2_y),15,(255,0,255),cv2.FILLED)
            elif(op_finger_x<650):
                cv2.circle(img,(finger1_x,finger1_y),15,(255,255,255),cv2.FILLED)
                cv2.circle(img,(finger2_x,finger2_y),15,(255,255,255),cv2.FILLED)
                if(CON_VAL <=0):
                    cv2.circle(img,(finger1_x,finger1_y),15,(0,0,255),cv2.FILLED)
                    cv2.circle(img,(finger2_x,finger2_y),15,(0,0,255),cv2.FILLED)
                    control.Video_control()

            
            grapCircle = []
            a = 200
            b = 200
            for i in range(0,5):
                x = lmList[tipIds[i]][1] 
                y = lmList[tipIds[i]][2] 
                
                ex = lmList[tipIds[2]][1]
                ey = lmList[tipIds[2]][2]
                if(ex == 250):
                    ex = 251
                
                
                angR = math.atan((ey-500)/(ex-250))
                angD = round(math.degrees(angR))
                theta = angD
                print(theta)
                fx = ((math.pow(x-250,2)) / (math.pow(a,2))) + ((math.pow(y-500,2)) / (math.pow(b,2)))
                if fx <= 1:
                    grapCircle.append(1)
                
                
                cv2.ellipse(img,(250,500),(200,200),0,0,360,(144,144,144),2)
            if grapCircle.count(1) == 5:
                
                cv2.ellipse(img,(250,500),(200,200),theta,0,360,(144,144,144),-1)
                cv2.ellipse(img,(250,500),(200,0),theta+15,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta+30,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta+45,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta+60,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta+75,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta+90,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta-15,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta-30,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta-45,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta-60,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta-75,0,360,(0,0,0),2,2)
                cv2.ellipse(img,(250,500),(200,0),theta,0,360,(0,0,255),6,2)
                cv2.ellipse(img,(250,500),(150,150),theta,0,360,(180,180,180),-1)
                cv2.ellipse(img,(250,500),(150,150),theta,0,360,(0,0,0),2)
            
                if((theta <0 )and (theta > -40)):
                    cv2.putText(img,"FAST >>",(170,200),cv2.FONT_HERSHEY_SIMPLEX, 3, (153,255,204), 5, cv2.LINE_AA)
                    cv2.ellipse(img,(250,500),(150,150),theta,0,360,(230,230,230),-1)
                    control.Video_Jump()
                elif((theta >0) and (theta < 40)):
                    cv2.putText(img,"SLOW >>",(170,200),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,100,130), 5, cv2.LINE_AA)
                    cv2.ellipse(img,(250,500),(150,150),theta,0,360,(100,100,100),-1)
                    control.Video_Back()
                    
    
    cv2.rectangle(img,(10,10),(120,100),(0,255,0),1)
    cv2.putText(img,"<-",(20,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5,cv2.LINE_AA)
    cv2.line(img,(650,0),(650,700),(0,0,255),5,cv2.LINE_AA)
    cv2.imshow("OpenCV",img)
    return img







def get_stream_video():
    #url = "http://172.20.10.14:81/stream" #Your url
    
    cap = cv2.VideoCapture(0)
    #cap = urlopen(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1600)
    print(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1200)
    
    global flag 
    
    flag = 0


    while True:
        success, img = cap.read()
        img1 = cv2.flip(img, 1)
        img2 = cv2.flip(img,1)
        img3 = cv2.flip(img,1)
        img4 = cv2.flip(img,1)
        img5 = cv2.flip(img,1)
        imgRGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        

        x = 0
        y = 0
        z = 0
        if flag ==0:
            cv2.destroyAllWindows()
            frame = draw_img(results,img1,x,y,z)
            
        elif flag ==1:
            frame = draw_img2(results,img2,x,y,z)
            
        elif flag ==2:
            frame = draw_img3(results,img3)
        
        elif flag ==3:
            frame = draw_img4(results,img4)
        
        elif flag ==4:
            frame = draw_img5(results,img5,x,y,z)
            
        
        frame = np.array(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(frame) + b'\r\n')
        
        


# if __name__ == "__main__":
#     main()





