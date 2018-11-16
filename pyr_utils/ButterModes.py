import numpy as np
import cv2
import pyPyrTools as ppt
from pyrUtils import *
from scipy import signal


#stem="BunnyCam"
#ext='.mp4'
#start_time=3
stem='tree_building_6450410'
ext='.avi'
start_time=0
sz=512

num_modes=5
frame_width=sz
frame_height=sz
num_frames=128
num_levs=5
inference_lev=1
y_ind=inference_lev*2+1
x_ind=inference_lev*2+2
k_size_x=11
k_size_t=5
k_width=k_size_x/2
sigma_time=3
sigma_freq=3
nyquist=num_frames/2
fl=28.0/32
fh=30.0/32
gamma=1
alpha=50
time_step=0.01
turbulence=.1
wind_speed=0

cap=cv2.VideoCapture('/Users/solomongarber/Downloads/'+stem+ext)
fps=cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES,fps*start_time)
fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
os.system('mkdir /Users/solomongarber/Documents/pyPyrTools/outvids/'+stem)
out_vid=cv2.VideoWriter('/Users/solomongarber/Documents/pyPyrTools/outvids/'+stem+'/'+stem+'_alpha_'+str(alpha)+'_synth.mp4',fourcc,fps*2,(frame_width,frame_height),True)
out_dir='/Users/solomongarber/Documents/pyPyrTools/outvids/'+stem

ret,frame=cap.read()
bgr=frame[:frame_height,:frame_width,:]
im=np.uint8(np.mean(bgr,2))
im_pyr=ppt.SCFpyr(im,num_levs,1)
x_lo=im_pyr.pyr[x_ind].copy()
x_hi=x_lo.copy()
x_prev=x_lo.copy()
y_lo=im_pyr.pyr[y_ind].copy()
y_hi=y_lo.copy()
y_prev=y_lo.copy()
phase_x=np.zeros((frame_height/(2**inference_lev),frame_width/(2**inference_lev),num_frames))
phase_y=phase_x.copy()
[lo_a,lo_b]=signal.butter(1,fl)
[hi_a,hi_b]=signal.butter(1,fh)
for i in range(num_frames):
    ret,frame=cap.read();
    im=np.uint8(np.mean(frame[:frame_height,:frame_width,:],2))
    im_pyr=ppt.SCFpyr(im,5,1)
    x_band=im_pyr.pyr[x_ind]
    y_band=im_pyr.pyr[y_ind]
    x_lo[:,:]=butter(x_lo,x_prev,x_band,lo_a,lo_b)
    y_lo[:,:]=butter(y_lo,y_prev,y_band,lo_a,lo_b)
    x_hi[:,:]=butter(x_hi,x_prev,x_band,hi_a,hi_b)
    y_hi[:,:]=butter(y_hi,y_prev,y_band,hi_a,hi_b)
    #ims('xhi',np.imag(np.log(x_hi)))
    phase_x[:,:,i]=awg(np.abs(x_lo)**gamma+np.abs(x_hi)**gamma,np.imag(np.log(x_lo/x_hi)),k_size_x,sigma_time)
    phase_y[:,:,i]=awg(np.abs(y_lo)**gamma+np.abs(y_hi)**gamma,np.imag(np.log(y_lo/y_hi)),k_size_x,sigma_time)
    ims('x',phase_x[:,:,i])
    ims('y',phase_y[:,:,i])
    #ims('im',im)
    y_prev[:,:]=y_band[:,:]
    x_prev[:,:]=x_band[:,:]
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


frequencies=num_frames/2-peak_inds
amps=peaks[peak_inds]
amps/=x_modes.shape[0]*x_modes.shape[1]
amps*=2
masses=1/amps
print peaks
print masses
print pks
dels=1/(pks[peak_inds])
dels*=x_modes.shape[0]*x_modes.shape[1]
dels/=1024
print dels


for i in range(num_modes):
    ims('x'+str(i),vh(x_modes[:,:,i]))
    ims('y'+str(i),vh(y_modes[:,:,i]))
    cv2.waitKey(1000)
    x_modes[:,:,i]/=np.max(np.abs(x_modes[:,:,i]))
    y_modes[:,:,i]/=np.max(np.abs(y_modes[:,:,i]))

forces=wind_speed*np.random.randn(num_modes)
for i in range(num_modes):
    forces[i]=forces[i]*(1-turbulence)+wind_speed*np.random.randn()*turbulence
    ys=(0,1)
    out=np.zeros((512,))
    for inter in range(512):
        ys_new=iter(.01,frequencies[i],dels[i],masses[i],forces[i],ys)
        if (np.abs(ys_new[0]+1j*ys_new[1]/frequencies[i])>np.abs(ys[0]+1j*ys[1]/frequencies[i])):
            mul=np.abs(ys[0]+1j*ys[1]/frequencies[i])/np.abs(ys_new[0]+1j*ys_new[1]/frequencies[i])
            ys_new=(ys_new[0]*mul,ys_new[1]*mul)
        ys=(ys_new[0],ys_new[1])
        out[inter]=ys[0]
    ims(str(i),plotpts(out))


modal_coords=[]
v=np.asarray([1.,1.])
v/=np.sqrt(2)
pt=(250/(2**inference_lev),166/(2**inference_lev))
motion_bases=np.zeros((frame_height/(2**inference_lev),frame_width/(2**inference_lev),num_modes,2),dtype=np.complex128)

for i in range(num_modes):
    x=np.min((3,np.abs(x_modes[pt[0],pt[1],i]*v[1])))
    y=np.min((3,np.abs(y_modes[pt[0],pt[1],i]*v[0])))
    modal_coords.append([0,amps[i]])
    motion_bases[:,:,i,0]=y_modes[:,:,i]
    motion_bases[:,:,i,1]=x_modes[:,:,i]
modal_coords=np.asarray(modal_coords)
forces=wind_speed*np.random.randn(num_modes)
big_flow=np.zeros((frame_height,frame_width,2),dtype=np.float64)
#flow[:,:,1]=complex_resize(x_modes[:,:,-2],(frame_width,frame_height))
#flow[:,:,0]=complex_resize(y_modes[:,:,-2],(frame_width,frame_height))

save_modes(bgr,motion_bases,frequencies,dels,masses,fps*2,alpha,stem,out_dir)

for i in range(512):
    div=2**inference_lev
    flow=np.zeros((frame_height/div,frame_width/div,2),dtype=np.float64)
    for j in range(num_modes):
        forces[j]=forces[j]*(1-turbulence)+wind_speed*np.random.randn()*turbulence
        mc=modal_coords[j,:]
        modal_coord=mc[0]+1j*mc[1]/frequencies[j]
        flow[:,:,:]+=np.real(modal_coord*motion_bases[:,:,j,:])
        mc=iter(time_step,frequencies[j],dels[j],masses[j],forces[j],mc)
        if (np.abs(mc[0]+1j*mc[1]/frequencies[j])>np.abs(modal_coord)):
            mul=np.abs(modal_coord)/np.abs(mc[0]+1j*mc[1]/frequencies[j])
            mc=(mc[0]*mul,mc[1]*mul)
        modal_coords[j,:]=mc
    big_flow[:,:,0]=cv2.resize(flow[:,:,0],(frame_width,frame_height))
    big_flow[:,:,1]=cv2.resize(flow[:,:,1],(frame_width,frame_height))
    warp=interp(bgr,big_flow*alpha)
    ims('x',flow[:,:,1])
    ims('y',flow[:,:,0])
    out_vid.write(np.uint8(warp))
    ims('w',np.uint8(warp))
    if i%20==0:
        print i
    #ims('f',interp(bgr,np.real(flow*(np.cos(i/10.0)+1j*np.sin(i/10.0)))*10))
#[0,0,0,0,0,5,35,25,0,0,25,15,0]
