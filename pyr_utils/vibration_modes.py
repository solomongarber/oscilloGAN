import numpy as np
import cv2
import pyPyrTools as ppt
from pyrUtils import *


num_modes=4
frame_width=512
frame_height=512
num_frames=128
num_levs=5
inference_lev=2
y_ind=inference_lev*2+1
x_ind=inference_lev*2+2
k_size_x=11
k_size_t=5
k_width=k_size_x/2
sigma_time=1
sigma_freq=3
nyquist=num_frames/2
l=.8
h=.2
gamma=1
cap=cv2.VideoCapture('/Users/solomongarber/Downloads/grass14.avi')

ret,frame=cap.read()
bgr=frame[:frame_height,:frame_width,:]
im=np.uint8(np.mean(bgr,2))
im_pyr=ppt.SCFpyr(im,num_levs,1)
x_lo=im_pyr.pyr[x_ind].copy()
x_hi=x_lo.copy()
y_lo=im_pyr.pyr[y_ind].copy()
y_hi=y_lo.copy()
phase_x=np.zeros((frame_height/(2**inference_lev),frame_width/(2**inference_lev),num_frames))
phase_y=phase_x.copy()

for i in range(num_frames):
    ret,frame=cap.read();
    im=np.uint8(np.mean(frame[:frame_height,:frame_width,:],2))
    fmpyr=ppt.SCFpyr(im,5,1)
    x_lo=iir(x_lo,fmpyr.pyr[x_ind],l);y_lo=iir(y_lo,fmpyr.pyr[y_ind],l)
    x_hi=iir(x_hi,fmpyr.pyr[x_ind],h);y_hi=iir(y_hi,fmpyr.pyr[y_ind],h)
    phase_x[:,:,i]=awg(np.abs(x_lo)**gamma+np.abs(x_hi)**gamma,np.imag(np.log(x_lo/x_hi)),k_size_x,sigma_time)
    phase_y[:,:,i]=awg(np.abs(y_lo)**gamma+np.abs(y_hi)**gamma,np.imag(np.log(y_lo/y_hi)),k_size_x,sigma_time)
    ims('x',phase_x[:,:,i])
    ims('y',phase_y[:,:,i])
print "video done"
freq_x=np.fft.fftshift(np.fft.fft(phase_x,axis=2),axes=2)
print "x analyzed"
freq_y=np.fft.fftshift(np.fft.fft(phase_y,axis=2),axes=2)
print "y analyzed"
kern=cv2.getGaussianKernel(k_size_t,.5)
kern-=cv2.getGaussianKernel(k_size_t,1.41)
peaks=np.sum(np.sum(np.abs(freq_x)**gamma+np.abs(freq_y)**gamma,0),0)
pks=np.convolve(peaks,kern.squeeze(),mode='same')
pks[:k_size_t/2]=0;pks[-(k_size_t/2):]=0;
peak_inds=np.argsort(pks[:nyquist])[-num_modes:]
print pks[:nyquist]
print np.argsort(pks[:nyquist])
print peak_inds
x_modes=np.zeros((frame_height/(2**inference_lev),frame_width/(2**inference_lev),num_modes),dtype=np.complex128)
y_modes=np.zeros((frame_height/(2**inference_lev),frame_width/(2**inference_lev),num_modes),dtype=np.complex128)
for i,p in enumerate(peak_inds):
    x_modes[:,:,i]=complex_blur2d(freq_y[:,:,p],k_size_x,sigma_freq) 
    y_modes[:,:,i]=complex_blur2d(freq_x[:,:,p],k_size_x,sigma_freq) 

amps=peaks[peak_inds]
amps/=x_modes.shape[0]*x_modes.shape[1]
masses=1/amps
print peaks
print masses
print pks
dels=1/(pks[peak_inds])
dels*=x_modes.shape[0]*x_modes.shape[1]
dels/=4
print dels

for i in range(num_modes):
    ims('x'+str(i),vh(x_modes[:,:,i]))
    ims('y'+str(i),vh(y_modes[:,:,i]))
    cv2.waitKey(1000)
    x_modes[:,:,i]/=np.max(np.abs(x_modes[:,:,i]))
    y_modes[:,:,i]/=np.max(np.abs(y_modes[:,:,i]))

for i in range(num_modes):
    ys=(0,1)
    out=np.zeros((1024,))
    for inter in range(1024):
        ys=iter(.01,(num_frames/2-peak_inds[i]),dels[i],masses[i],0,ys)
        out[inter]=ys[0]
    ims(str(i),plotpts(out))






modal_coords=[]
v=np.asarray([1.,3.])
pt=(250/(2**inference_lev),166/(2**inference_lev))
for i in range(num_modes):
    x=np.abs(x_modes[pt[0],pt[1],i]*v[1])
    y=np.abs(y_modes[pt[0],pt[1],i]*v[0])
    modal_coords.append((y,x))

flow=np.zeros((frame_height,frame_width,2),dtype=np.complex128)
flow[:,:,1]=complex_resize(x_modes[:,:,-2],(frame_height,frame_width))
flow[:,:,0]=complex_resize(y_modes[:,:,-2],(frame_height,frame_width))
for i in range(100):
    ims('f',interp(bgr,np.real(flow*(np.cos(i/10.0)+1j*np.sin(i/10.0)))*10))
