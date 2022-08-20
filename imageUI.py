from tkinter import filedialog as fd, ttk
import cv2
from utils.misc import transform
from torch import nn
from utils.misc import load_model,activity_map as class_dict
from threading import *
import PIL
from PIL import Image, ImageTk
import numpy as np

'''
This file contains the code for tkinter widgets and UI handling for object detection from any image file.
'''

# takes the root window and fits in the other widgets local to context like progress bar, start, stop buttons

def genImageLabel(win):
    font = cv2.FONT_HERSHEY_TRIPLEX
    color_red = (255,0,0)
    color_green=(0,255,0)
    model=load_model()
    model.eval()
    # event fires when the file is selected 
    def _selectImage():
        global imgSelect
        imgSelect = fd.askopenfilename(
            title="Select image file", filetypes=[("jpeg files", ".jpg")]
        )
    # event fires when clicked on the start button. checks for the file and initializes the VideoWriter and video capture instance from the selected file path
    def _start():
        if imgSelect:
            # net, classes, layer_names, output_layers = sendNN()
            cap = cv2.imread(imgSelect)
            with PIL.Image.open(imgSelect) as im:
                im = transform(im)
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
            # _,f1=cap.read()
            # out_vid = cv2.VideoWriter(f"annotated_video.mp4",cv2.VideoWriter_fourcc('a','v','c','1'),10,(f1.shape[1],f1.shape[0]))

            

            # label_pred=activity_map.get(f'c{np.argmax(y_prediction)}')
            # print(np.argmax(y_prediction))
            cv2.putText(cap,f"p1-{one_pred}",(20,22),font,1,color_green,1,cv2.LINE_AA)
            cv2.putText(cap,f"p2-{two_pred}",(20,55),font,1,color_red,1,cv2.LINE_AA)
            # cv2.putText(cap,one_pred,(20,22),font,1,color,2,cv2.LINE_AA)
            # win.temp=ttk.Progressbar(win,orient='horizontal',mode='indeterminate',length=200)
            # win.temp.grid(row=4,column=1)
            # win.temp.start()
            # inner function fires when stop is clicked hides the progress bar and other widgets.
            final_img=Image.fromarray(cap)
            imgtk = ImageTk.PhotoImage(image=final_img)
            try:
                if win.img_label:
                    win.img_label.config(photo=imgtk)
            except Exception:
                    win.img_label=ttk.Label(win,image=imgtk)
                    win.img_label.photo=imgtk
                    win.img_label.grid(row=6,column=0)
            # win.stop=ttk.Button(win,text='Stop',command=_stop)
            # win.stop.grid(row=4,column=2)
            # bg=threading.Thread(target=_internal,args=(cap,net,output_layers,classes,out_vid))
            # bg.start() 
    win.button = ttk.Button(win, text="Select Image file", command=_selectImage)
    win.button.grid(row=3, column=0)
    win.start=ttk.Button(win,text='Start',command=_start)
    win.start.grid(row=4,column=0)
    # custom function to hide the controls from the video selection page. called by the ui.py file.
    def _destroy():
        try:
            win.img_label.destroy()
            win.start.destroy()
            win.button.destroy()
            cv2.destroyAllWindows()
            
        except:
            print('Already clean')
    return _destroy
