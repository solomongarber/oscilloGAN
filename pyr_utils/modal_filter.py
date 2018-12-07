import numpy as np
import cv2
import pyPyrTools3 as ppt
from pyrUtils3 import *
from scipy import signal
from vid_writer import vid_writer
import sys

#python modal_filter.py /path/to/input/vid.ext /path/to/output/folders vid t 0.027 .011 3 

in_vid=sys.argv[1]
out_dir = sys.argv[2]
stem=sys.argv[3]

diff_of_butters=sys.argv[4]=='t'
#fl=28.0/32
#fh=30.0/32
fl=np.float64(sys.argv[5])
fh=np.float64(sys.argv[6])

sigma_time=int(sys.argv[7])
k_size_x=6*sigma_time+1
k_width=k_size_x//2

style='iir'
if diff_of_butters:
    style='butter'

num_modes=10
inference_lev=1
#inference_lev=2
num_levs=inference_lev+1
y_ind=inference_lev*2+1
x_ind=inference_lev*2+2
re_sz=(1280,720)

cap=cv2.VideoCapture(in_vid)
fps=cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES,int(fps*6))
num_frames=int(fps*20)
nyquist=num_frames//2
#os.system('mkdir '+out_dir+stem)

ret,frame=cap.read()
crp=[0,frame.shape[0],0,frame.shape[1]]
bgr=cv2.resize(frame,re_sz)
im=np.uint8(np.mean(bgr,2))
im_pyr=ppt.SCFpyr(im,num_levs,1)
x_lo=im_pyr.pyr[x_ind].copy()
x_hi=x_lo.copy()
if diff_of_butters:
    x_prev=x_lo.copy()
y_lo=im_pyr.pyr[y_ind].copy()
y_hi=y_lo.copy()
if diff_of_butters:
    y_prev=y_lo.copy()
phase_x=np.zeros((bgr.shape[0]//(2**inference_lev),bgr.shape[1]//(2**inference_lev),num_frames))
phase_y=phase_x.copy()
[lo_a,lo_b,hi_a,hi_b]=[0,0,0,0]
if diff_of_butters:
    [lo_a,lo_b]=signal.butter(1,fl)
    [hi_a,hi_b]=signal.butter(1,fh)
else:
    [lo_a,lo_b]=[fl,1-fl]
    [hi_a,hi_b]=[fh,1-fh]

for i in range(num_frames):
    ret,frame=cap.read();
    im=np.uint8(np.mean(frame,2))
    bgr=cv2.resize(im[crp[0]:crp[1],crp[2]:crp[3]],re_sz)
    #im=np.uint8(np.mean(bgr,2))
    im_pyr=ppt.SCFpyr(bgr,num_levs,1)
    x_band=im_pyr.pyr[x_ind]
    y_band=im_pyr.pyr[y_ind]
    if diff_of_butters:
        x_lo[:,:]=butter(x_lo,x_prev,x_band,lo_a,lo_b)
        y_lo[:,:]=butter(y_lo,y_prev,y_band,lo_a,lo_b)
        x_hi[:,:]=butter(x_hi,x_prev,x_band,hi_a,hi_b)
        y_hi[:,:]=butter(y_hi,y_prev,y_band,hi_a,hi_b)
    else:
        x_lo[:,:]=x_band*lo_a+x_lo*lo_b
        y_lo[:,:]=y_band*lo_a+y_lo*lo_b
        x_hi[:,:]=x_band*hi_a+x_hi*hi_b
        y_hi[:,:]=y_band*hi_a+y_hi*hi_b
    #ims('xhi',np.imag(np.log(x_hi)))
    phase_x[:,:,i]=awg(np.abs(x_lo)+np.abs(x_hi),np.imag(np.log(x_lo/x_hi)),k_size_x,sigma_time)
    phase_y[:,:,i]=awg(np.abs(y_lo)+np.abs(y_hi),np.imag(np.log(y_lo/y_hi)),k_size_x,sigma_time)
    #ims('x',np.uint8(sat(phase_x[:,:,i],alpha=127/np.pi,beta=128,min=-np.pi,max=np.pi)))
    #ims('y',np.uint8(sat(phase_y[:,:,i],alpha=127/np.pi,beta=128,min=-np.pi,max=np.pi)))
    grey_show('x',phase_x[:,:,i])
    grey_show('y',phase_y[:,:,i])
    if i%300==0:
        print (i)
    if diff_of_butters:
        y_prev[:,:]=y_band[:,:]
        x_prev[:,:]=x_band[:,:]
print ("video done")



freq_x=np.fft.fftshift(np.fft.fft(phase_x,axis=2),axes=2)
print ("x analyzed")
freq_y=np.fft.fftshift(np.fft.fft(phase_y,axis=2),axes=2)
print ("y analyzed")

peaks=np.sum(np.sum(np.abs(freq_x)+np.abs(freq_y),0),0)
pks=peaks[:nyquist]
save_modes(frame,out_dir,stem,pks,num_modes,freq_x,freq_y,sigma_time=sigma_time,fl=fl,fh=fh)
