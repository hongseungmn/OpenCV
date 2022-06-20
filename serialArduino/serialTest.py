import serial
import time





py_serial = serial.Serial(port='/dev/cu.usbmodem11301',baudrate=9600)

while True:
    command = input("명령어를 입력하시오")
    py_serial.write(command.encode())
    
        
    if py_serial.readable():
        response = py_serial.readline()
        print(response[:len(response)-1].decode()) 
    time.sleep(1)   

