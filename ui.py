import tkinter as tk
from tkinter import StringVar, ttk
from tkinter.constants import E, END, LEFT, W
from win32api import GetSystemMetrics
from webUI import genCamLabel
from videoUI import genImageLabel
from imageUI import genImageLabel as imageLabel

'''
this file handles the main frame of tkinter where all the widgets are linked and fires events accordingly.
'''

class Main(tk.Frame):
    # bus variable contains the list of methods that are called to clear the main window of any residual widgets.
    bus=[]
    def __init__(self, master, **kwargs):
        self.master = master
        self.master.geometry(f"{GetSystemMetrics(0)}x{GetSystemMetrics(1)}")
        ways = ["Video","Image"]
        ttk.Label(master=self.master,text='Select data stream:').grid(row=0,column=0)
        self.fd = ttk.Combobox(values=ways, state="readonly")
        ttk.Label(master=self.master,text='Select processing unit:').grid(row=0,column=1)
        self.calSel=ttk.Combobox(values=['cpu','gpu'], state="readonly")
        self.calSel.current(0)
        self.calSel.grid(row=1,column=1)
        self.fd.bind("<<ComboboxSelected>>", self.onWayChange)
        self.fd.grid(row=1,column=0)
    # clears the bus i.e removes any widgets from root frame
    def _clearBus(self):
        for i in Main.bus:
            i()
        Main.bus.clear()
    # fires when data stream is changed. clears the bus and initialises new data stream.
    def onWayChange(self, event):
        if self.fd.get() == "Live Webcam":
            self._clearBus()
            _temp = genCamLabel(self.master)
            Main.bus.append(_temp)

        elif self.fd.get() == "Video":
             self._clearBus()
             _temp=genImageLabel(self.master)
             Main.bus.append(_temp)
        else:
            self._clearBus()
            _temp=imageLabel(self.master)
            Main.bus.append(_temp)

# new instance of tkinter window is created and passed as the root frame to Main class.

root = tk.Tk()
root.title('Driver Distraction Detection')
root.iconphoto(False, tk.PhotoImage(file='C:\\Users\\nav\\Downloads\\driver-distraction\\favicon.png'))
app = Main(root)
root.mainloop()
