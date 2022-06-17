from math import pi
import time
import pyfirmata
from math import pi, sin
import pyautogui

# comport ='/dev/cu.usbmodem1401'

# board = pyfirmata.Arduino(comport)

# pwm_r = board.get_pin('d:11:p')
# pwm_g = board.get_pin('d:10:p')
# pwm_b = board.get_pin('d:9:p')
# pin_1 = board.get_pin('d:7:o')
# pin_2 = board.get_pin('d:6:o')
# pin_3 = board.get_pin('d:5:o')
# pin_4 = board.get_pin('d:4:o')
# pin_5 = board.get_pin('d:3:o')

# def control_RGB(r,g,b):
#     pwm_r.write(r)
#     pwm_g.write(g)
#     pwm_b.write(b)
#     print("r:",r," g:",g," b:",b)
    
# def countLED(count):
#     print(count)
#     if(count == 5):
#         pin_1.write(1)
#         pin_2.write(1)
#         pin_3.write(1)
#         pin_4.write(1)
#         pin_5.write(1)
        
#     elif(count == 4):
#         pin_1.write(1)
#         pin_2.write(1)
#         pin_3.write(1)
#         pin_4.write(1)
#         pin_5.write(0)
    
#     elif(count == 3):
#         pin_1.write(1)
#         pin_2.write(1)
#         pin_3.write(1)
#         pin_4.write(0)
#         pin_5.write(0)
        
#     elif(count == 2):
#         pin_1.write(1)
#         pin_2.write(1)
#         pin_3.write(0)
#         pin_4.write(0)
#         pin_5.write(0)
        
#     elif(count == 1):
#         pin_1.write(1)
#         pin_2.write(0)
#         pin_3.write(0)
#         pin_4.write(0)
#         pin_5.write(0)
    
#     elif(count == 0):
#         pin_1.write(0)
#         pin_2.write(0)
#         pin_3.write(0)
#         pin_4.write(0)
#         pin_5.write(0)

        
def Volume_up():
    pyautogui.press('up')

def Volume_Down():
    pyautogui.press('down')
    
def Video_control():
    pyautogui.press('space')

def Video_Jump():
    pyautogui.press('right')

def Video_Back():
    pyautogui.press('left')
    
    
# pyfirmata.util.Iterator(board).start()
# board.analog[0].enable_reporting()
