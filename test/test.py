import sys
import subprocess
import numpy as np
import PySimpleGUI as sg    
from PIL import Image, ImageTk
import cv2  
import pickle as pkl

def update(w, img):
    img = cv2.resize(img, (500,500))
    image = ImageTk.PhotoImage(image=Image.fromarray(img))
    w['-IMAGE-'].update(data=image)

def patchpath2img(p: str) -> np.ndarray:
   
    try:
        with open(p, "rb") as fin:
            dat = pkl.load(fin)
        tmp = np.copy(dat.RGB)
        tmp[~dat.mask] = 0
        print(tmp.shape)
        return tmp
    except:
        print("Something went wrong")
 
#sg.theme('Default')
font = ("Arial", 20)
sg.theme('Light Grey 6')


layout = [
    [sg.Image(size=(300, 600), key='-IMAGE-')],
    [sg.Button('AMPLIFIED', bind_return_key=True, key="-AMP-"), sg.Button('NON-AMPLIFIED', bind_return_key=True, key="-NAMP-")]]      

window = sg.Window('AMP VS. NON-AMP', layout, finalize=True, font=font)     

base = "/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/results/single_patches"
imgs = [os.path.join(base, x) for x in os.listdir(base)[:10]]

ev_c = 0
while True: # The Event Loop
    
    event, values = window.read() 

    if event:
        
        ev_c += 1
        update(window, patchpath2img(imgs[ev_c]))

    if event == sg.WIN_CLOSED or event == 'Exit':
        break  

window.close()