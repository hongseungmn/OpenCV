import math
import cv2
import mediapipe as mp
import numpy as np
import time


FRAME_WIDTH  = 1280
FRAME_HEIGHT =  720

THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16

Z_THRESHOLD_PRESS = -200

VK = {
    'LED': { 'x':200,  'y':100, 'w':300, 'h':150 },
    'TEMPER': { 'x':700, 'y':100, 'w':300, 'h':150 },
    'BRIGHT': { 'x':200, 'y':400, 'w':300, 'h':150 },
    'DOOR': { 'x':700, 'y':400, 'w':300, 'h':150 }
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
            cv2.rectangle(img, (VK[k]['x'], VK[k]['y']), (VK[k]['x']+VK[k]['w'], VK[k]['y']+VK[k]['h']), (0,0,255), -1) # thickness -1 means filled rectangle
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
                else:
                    color = (0,255,0)
                
                cv2.putText(img, f"{x}, {y}, {z}", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            except IndexError:
                index_finger_tip = None
                
        draw_keys(img, x, y, z)
    cv2.imshow("OpenCV Video Capture", img)


def draw_img2(results,img,x,y,z):
    global flag
    
    
    cv2.rectangle(img,(10,10),(120,100),(0,255,0),1)
    cv2.putText(img,"<-",(20,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5,cv2.LINE_AA)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # DEBUG
            finger1_x = int(handLms.landmark[4].x * FRAME_WIDTH )
            finger1_y = int(handLms.landmark[4].y * FRAME_HEIGHT )
            finger2_x = int(handLms.landmark[8].x * FRAME_WIDTH )
            finger2_y = int(handLms.landmark[8].y * FRAME_HEIGHT )
            dist_x = finger1_x - finger2_x
            dist_y = finger1_y - finger2_y
            
            dist = math.sqrt(abs((dist_x**2) - (dist_y**2)))
            
            cv2.rectangle(img,(1100,10),(1200,600),(0,0,255),2)
            cv2.rectangle(img,(1100,600 - int(dist)),(1200,10),(0,0,255),-1)
            cv2.line(img,(finger1_x,finger1_y),(finger2_x,finger2_y),(0,255,0),thickness=3,lineType=cv2.LINE_AA)
            try:
                index_finger_tip = handLms.landmark[INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * FRAME_WIDTH)
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

            
    cv2.imshow("OpenCV",img)
    
    
def draw_img3(result,img):
    global flag
    mp_draw=mp.solutions.drawing_utils
    mp_hand=mp.solutions.hands
    lmList=[]
    tipIds=[4,8,12,16,20]
    
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
    fingers=[]
    if len(lmList)!=0: # 손이 인식되었을 경우만 작동
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]: # 손가락 마디 끝을 설정
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        total=fingers.count(0)
        if total==0:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 0 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
        elif total==1:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 1 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
        elif total==2:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 2 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
        elif total==3:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 3 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
        elif total==4:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 4 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
        elif total==5:
            cv2.rectangle(img, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, " 5 ", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 0), 5)
        
            
    cv2.rectangle(img,(10,10),(120,100),(0,255,0),1)
    cv2.putText(img,"<-",(20,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5,cv2.LINE_AA)
    cv2.imshow("OpenCV",img)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    global flag 
    flag = 0


    while True:
        
    
        success, img = cap.read()
        img1 = cv2.flip(img, 1)
        img2 = cv2.flip(img,1)
        img3 = cv2.flip(img,1)
        imgRGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        

        x = 0
        y = 0
        z = 0
        if flag ==0:
            cv2.destroyAllWindows()
            draw_img(results,img1,x,y,z)
            
        elif flag ==1:
            draw_img2(results,img2,x,y,z)
            
        elif flag ==2:
            draw_img3(results,img3)
            
        
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # ESC
            cv2.destroyAllWindows()
            cap.release()
            break
    
    
    




if __name__ == "__main__":
    main()