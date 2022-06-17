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
    'SOUND': { 'x':700, 'y':100, 'w':300, 'h':150 },
    'BRIGHT': { 'x':200, 'y':400, 'w':300, 'h':150 },
    'SERVO': { 'x':700, 'y':400, 'w':300, 'h':150 }
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
            cv2.rectangle(img, (VK[k]['x'], VK[k]['y']), (VK[k]['x']+VK[k]['w'], VK[k]['y']+VK[k]['h']), (0,0,255),abs(z//2), 1) # thickness -1 means filled rectangle
            cv2.putText(img, f"{k}", (VK[k]['x']+30,VK[k]['y']+70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (VK[k]['x'], VK[k]['y']), (VK[k]['x']+VK[k]['w'], VK[k]['y']+VK[k]['h']), (0,255,0), 1) # thickness -1 means filled rectangle
            cv2.putText(img, f"{k}", (VK[k]['x']+30,VK[k]['y']+70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5, cv2.LINE_AA)

def draw_img2(img,dist,x_1,y_1,x_2,y_2,handLms):
    global flag
    cv2.rectangle(img,(1100,10),(1200,600),(0,0,255),2)
    cv2.rectangle(img,(1100,600 - int(dist)),(1200,10),(0,0,255),-1)
    cv2.line(img,(x_1,y_1),(x_2,y_2),(0,255,0),thickness=3,lineType=cv2.LINE_AA)
    cv2.rectangle(img,(10,10),(120,100),(0,255,0),1)
    cv2.putText(img,"<-",(20,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),5,cv2.LINE_AA)
    
    try:
        index_finger_tip = handLms.landmark[INDEX_FINGER_TIP]
        x = int(index_finger_tip.x * FRAME_WIDTH)
        y = int(index_finger_tip.y * FRAME_HEIGHT)
        z = int(index_finger_tip.z * FRAME_WIDTH)
        
        if (z <= Z_THRESHOLD_PRESS):
            color = (0,0,255) # BGR
            
        else:
            color = (0,255,0)
        
    except IndexError:
                index_finger_tip = None
    cv2.putText(img, f"{x}, {y}, {z}", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    
    
    

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    global flag 
    flag = 0


    while True:
        
    
        success, img1 = cap.read()
        img = cv2.flip(img1, 1)
        img2 = cv2.flip(img1,1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        

        x = 0
        y = 0
        z = 0
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                finger1_x = int(handLms.landmark[4].x * FRAME_WIDTH )
                finger1_y = int(handLms.landmark[4].y * FRAME_HEIGHT )
                finger2_x = int(handLms.landmark[8].x * FRAME_WIDTH )
                finger2_y = int(handLms.landmark[8].y * FRAME_HEIGHT )
                dist_x = finger1_x - finger2_x
                dist_y = finger1_y - finger2_y
                
                dist = math.sqrt(abs((dist_x**2) - (dist_y**2)))
                
                
        
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # DEBUG
                mpDraw.draw_landmarks(img2, handLms, mpHands.HAND_CONNECTIONS) # DEBUG
                try:
                    index_finger_tip = handLms.landmark[INDEX_FINGER_TIP]
                    x = int(index_finger_tip.x * FRAME_WIDTH)
                    y = int(index_finger_tip.y * FRAME_HEIGHT)
                    z = int(index_finger_tip.z * FRAME_WIDTH)
                    #print(f"x={x} , y={y} , z={z}")
                    img2(img2,dist,finger1_x,finger1_y,finger2_x,finger2_y,handLms)
                    if ((z <= Z_THRESHOLD_PRESS) and (dist < 100)):
                        time.sleep(1)
                        color = (0,0,255) # BGR
                        flag = 1
                    elif ((z <= Z_THRESHOLD_PRESS) and (10<= x <=120) and (10 <= y <= 100)):
                        flag = 0
                        color = (0,255,0)
                    else:
                        color = (0,255,0)
                    
                    cv2.putText(img, f"{x}, {y}, {z}", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    cv2.putText(img, text='f1=%d f2=%d dist=%d ' % (finger1_x,finger2_x,dist), org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=3)
                except IndexError:
                    index_finger_tip = None

            draw_keys(img, x, y, z)
            


        if flag == 1:
            cv2.imshow("OpenCV",img2)
        elif flag == 0:            
            cv2.imshow("OpenCV Video Capture", img)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # ESC
            cv2.destroyAllWindows()
            cap.release()
            break
    
    
    




if __name__ == "__main__":
    main()