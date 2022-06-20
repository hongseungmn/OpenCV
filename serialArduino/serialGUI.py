import tkinter
import serial
import threading
import queue
import sys

class serial_PORT(threading.Thread):

    py_serial = serial.Serial(port='/dev/cu.usbmodem1401',baudrate=9600)
    flag=True
    def __init__(self,que):
        threading.Thread.__init__(self)
        self.queue = que
        
        
    def run(self):  
        while self.flag: 
            if self.py_serial.inWaiting():
                data = self.py_serial.readline(self.py_serial.inWaiting())
                self.queue.put(data)
                print("qSize = " +str(self.queue.qsize()))
                
            
    

class SerialGUI(tkinter.Tk):
    
    def __init__(self):
        tkinter.Tk.__init__(self)
        self.title("시리얼포트")
        self.geometry() # 창 크기 설정 (가로*세로 + 초기 창위치 x좌표, y좌표)
        self.resizable(True,True) # 창 크기 조절가능 여부

        self.label = tkinter.Label(self, text='Arduino ==> User : Response')
        self.label.grid(row=0,column=0,columnspan=3)

        self.listboxArduino = tkinter.Listbox(self)
        self.listboxArduino.grid(row=1, column=0, columnspan=3, sticky='ew')


        self.label = tkinter.Label(self, text='User ==> Arduino : Request')
        self.label.grid(row=2,column=0,columnspan=3)

        self.listboxUser = tkinter.Listbox(self)
        self.listboxUser.grid(row=3, column=0, columnspan=3,sticky='ew')

        self.entry = tkinter.Entry(self)
        self.entry.grid(row=4, column=1, columnspan=2, sticky='ew')

        sendButton = tkinter.Button(self, text='전송')
        sendButton.grid(row=4, column=0, sticky='ew')
        sendButton.bind('<Button-1>', self.btn_click)

        self.queue = queue.Queue()
        self.thread = serial_PORT(self.queue)
        self.serial_thread()
        self.thread.start()
        



    def btn_click(self,event):
        i=0
        word = str(self.entry.get())
        self.listboxUser.insert([i],word)
        print("버튼이 클릭되었습니다.")
        print(word + " 명령어를 전달합니다.")
        self.entry.delete(0,tkinter.END)
        serial_PORT.py_serial.write(bytes(word,encoding='ascii'))
        i+=1
        
    def serial_thread(self):
        while self.queue.qsize():
            try:
                i=0
                data = self.queue.get()
                
                self.listboxArduino.insert([i],data)
                i+=1
            except queue.Empty:
                pass
        self.after(10,self.serial_thread)

serial_GUI = SerialGUI()
serial_GUI.mainloop()
serial_PORT.flag = False






