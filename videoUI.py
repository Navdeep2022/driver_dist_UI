from tkinter import filedialog as fd, ttk,Toplevel
from utils.nnUtils import sendNN
from utils.nnUtils import _internal
import cv2
import os

from utils.misc import load_model,activity_map as class_dict
import threading
from utils.misc import transform
from torch import nn
from PIL import Image,ImageTk
from threading import *
import numpy as np

'''
This file contains the code for tkinter widgets and UI handling for object detection from any video file.
'''

# takes the root window and fits in the other widgets local to context like progress bar, start, stop buttons

def genImageLabel(win):
    font = cv2.FONT_HERSHEY_TRIPLEX
    color_red = (255,0,0)
    color_green=(0,255,0)
    model=load_model()
    model.eval()
    # event fires when the file is selected 
    def _selectVideo():
        global vidSelect
        vidSelect = fd.askopenfilename(
            title="Select video file", filetypes=[("mp4 files", ".mp4")]
        )
    # event fires when clicked on the start button. checks for the file and initializes the VideoWriter and video capture instance from the selected file path
    def _start():
        if vidSelect:
            cap = cv2.VideoCapture(vidSelect)
            win.img_label=ttk.Label(win,image=None)
            win.img_label.grid(row=6,column=0)
            def __rec():
                def _rec():
                    _,f1=cap.read()
                    img = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
                    img=cv2.flip(img,1)
                    im_pil = Image.fromarray(img)
                    im = transform(im_pil)
                    im = im.unsqueeze(0)
                    output = model(im)
                    proba = nn.Softmax(dim=1)(output)
                    proba = [round(float(elem),4) for elem in proba[0]]
                    # print(proba)
                    one_pred=class_dict[proba.index(max(proba))]
                    print("Predicted class:",class_dict[proba.index(max(proba))])
                    proba2 = proba.copy()
                    proba2[proba2.index(max(proba2))] = 0.
                    two_pred=class_dict[proba2.index(max(proba2))]
                    # im = Image.fromarray(im_pil)
                    cv2.putText(f1,f"p1-{one_pred}",(20,22),font,1,color_green,1,cv2.LINE_AA)
                    cv2.putText(f1,f"p2-{two_pred}",(20,55),font,1,color_red,1,cv2.LINE_AA)
                    f1=cv2.cvtColor(f1,cv2.COLOR_BGR2RGB)
                    final_img=Image.fromarray(f1)
                    imgtk = ImageTk.PhotoImage(image=final_img)
                    win.img_label.imgtk=imgtk
                    win.img_label.configure(image=imgtk)
                    win.img_label.after(50,_rec)
                _rec()
            # out_vid = cv2.VideoWriter(f"annotated_video.mp4",cv2.VideoWriter_fourcc('a','v','c','1'),10,(f1.shape[1],f1.shape[0]))
            
            # win.temp=ttk.Progressbar(win,orient='horizontal',mode='indeterminate',length=200)
            # win.temp.grid(row=4,column=1)
            # win.temp.start()
            # inner function fires when stop is clicked hides the progress bar and other widgets.
            # def _stop():
            #     cap.release()
            #     win.temp.stop()
            #     _destroy()
            # win.stop=ttk.Button(win,text='Stop',command=_stop)
            # win.stop.grid(row=4,column=2)
            th=Thread(target=__rec).start()
            
            # bg=threading.Thread(target=_internal,args=(cap,net,output_layers,classes,out_vid))
            # bg.start() 
    win.button = ttk.Button(win, text="Select video file", command=_selectVideo)
    win.button.grid(row=3, column=0)
    win.start=ttk.Button(win,text='Start',command=_start)
    win.start.grid(row=4,column=0)
    # custom function to hide the controls from the video selection page. called by the ui.py file.
    def _destroy():
        try:
            win.stop.destroy()
            win.temp.destroy()
            win.start.destroy()
            cv2.destroyAllWindows()
            win.button.destroy()
        except:
            print('Already clean')
    return _destroy
