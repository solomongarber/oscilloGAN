import numpy as np
import cv2
import pyPyrTools as ppt
import os
import json
from scipy import signal
from vid_writer import vid_writer

def ims(name,im):
    oim=np.float64(im)
    oim=255*(oim-np.min(oim))/(np.max(oim)-np.min(oim))
    if im.dtype==np.uint8:
        cv2.imshow(name,im)
    else:
        cv2.imshow(name,ate(im))
    cv2.waitKey(2)

def ate(im):
    oim=np.float64(im)
    oim=255*(oim-np.min(oim))/(np.max(oim)-np.min(oim))
    return np.uint8(oim)

def plotpts(pts):
    out=np.zeros((255,len(pts)))
    poi=(pts-np.min(pts))/(np.max(pts)-np.min(pts))
    for i,pt in enumerate(poi):
        out[np.int32(255*(1-pt)):,i]=1
    return out

def awg(aband,pband,g,s):
    b=cv2.GaussianBlur(aband,(g,g),s)
    pb=cv2.GaussianBlur(pband*aband,(g,g),s)
    b[b==0]=1
    return pb/b

def ag(wband,g,s):
    a=np.abs(wband)
    re=cv2.GaussianBlur(np.float32(a*np.real(wband)),(g,g),s)
    im=cv2.GaussianBlur(np.float32(a*np.imag(wband)),(g,g),s)
    return re+1j*im

def complex_blur2d(wband,g,s):
    re=cv2.GaussianBlur(np.float32(np.real(wband)),(g,g),s)
    im=cv2.GaussianBlur(np.float32(np.imag(wband)),(g,g),s)
    return re+1j*im

def viewHue(amps,phases):
    im=np.zeros((amps.shape[0],amps.shape[1],3),dtype=np.float32)
    im[:,:,2]=(amps-np.min(amps)/(np.max(amps)-np.min(amps)))
    im[:,:,0]=360*((phases+np.pi)/(2*np.pi))
    im[:,:,1]=.5
    return cv2.cvtColor(im,cv2.COLOR_HSV2BGR)

def vh(w):
    amps=np.abs(w)
    phases=np.imag(np.log(w))
    im=np.zeros((amps.shape[0],amps.shape[1],3),dtype=np.float32)
    im[:,:,2]=(amps-np.min(amps)/(np.max(amps)-np.min(amps)))
    im[:,:,0]=360*((phases+np.pi)/(2*np.pi))
    im[:,:,1]=.5
    return cv2.cvtColor(im,cv2.COLOR_HSV2BGR)

def phasediff(p1,p2):
    return (p1-p2+np.pi)%(2*np.pi)-np.pi

def pd(w1,w2):
    return np.imag(np.log(w1/w2))

def control(pic, flow):
    ys=np.arange(pic.shape[0]*pic.shape[1])/pic.shape[1]
    ud=(flow[:,:,0].reshape(-1)+ys)%pic.shape[0]
    xs=np.arange(pic.shape[0]*pic.shape[1])%pic.shape[1]
    lr=(flow[:,:,1].reshape(-1)+xs)%pic.shape[1]
    ans=np.zeros(pic.shape)
    ans[ys,xs,:]=pic[ud,lr,:]
    return ans

def push(pic,flow):
    p=pic.copy()
    ys=np.arange(pic.shape[0]*pic.shape[1])/pic.shape[1]
    ud=(flow[:,:,0].reshape(-1)+ys)%pic.shape[0]
    xs=np.arange(pic.shape[0]*pic.shape[1])%pic.shape[1]
    lr=(flow[:,:,1].reshape(-1)+xs)%pic.shape[1]
    u=np.int32(np.floor(ud))
    d=np.int32(np.ceil(ud))%pic.shape[0]
    l=np.int32(np.floor(lr))
    r=np.int32(np.ceil(lr))%pic.shape[1]
    p[u,l,:]=pic[ys,xs,:]
    p[u,r,:]=pic[ys,xs,:]
    p[d,l,:]=pic[ys,xs,:]
    p[d,r,:]=pic[ys,xs,:]
    return p

def push_sparse(pic,flow):
    p=pic.copy()
    weights=np.ones((pic.shape[0],pic.shape[1]),dtype=np.float32)
    ys=np.arange(pic.shape[0]*pic.shape[1])/pic.shape[1]
    ud=(flow[:,:,0].reshape(-1)+ys)%pic.shape[0]
    xs=np.arange(pic.shape[0]*pic.shape[1])%pic.shape[1]
    lr=(flow[:,:,1].reshape(-1)+xs)%pic.shape[1]
    u=np.int32(np.floor(ud))
    #d=np.int32(np.ceil(ud))%pic.shape[0]
    l=np.int32(np.floor(lr))
    #r=np.int32(np.ceil(lr))%pic.shape[1]
    p[u,l,:]=pic[ys,xs,:]
    #p[u,r,:]=pic[ys,xs,:]
    #p[d,l,:]=pic[ys,xs,:]
    #p[d,r,:]=pic[ys,xs,:]
    return p

def pull(pic,flow):
    return interp(pic,flow)

def interp(pic,flow):
    ys=np.arange(pic.shape[0]*pic.shape[1])/pic.shape[1]
    ud=(flow[:,:,0].reshape(-1)+ys)%pic.shape[0]
    xs=np.arange(pic.shape[0]*pic.shape[1])%pic.shape[1]
    lr=(flow[:,:,1].reshape(-1)+xs)%pic.shape[1]
    u=np.int32(np.floor(ud))
    d=np.int32(np.ceil(ud))%pic.shape[0]
    udiffs=ud-u
    udiffs=np.dstack((udiffs,udiffs,udiffs))
    l=np.int32(np.floor(lr))
    r=np.int32(np.ceil(lr))%pic.shape[1]
    ldiffs=lr-l
    ldiffs=np.dstack((ldiffs,ldiffs,ldiffs))
    ul=pic[u,l,:]
    ur=pic[u,r,:]
    dl=pic[d,l,:]
    dr=pic[d,r,:]
    udl=ul*(1-udiffs)+dl*udiffs
    udr=ur*(1-udiffs)+dr*udiffs
    ans=np.zeros(pic.shape)
    ans[ys,xs,:]=udl*(1-ldiffs)+udr*ldiffs
    return ans

def iter(t,w,damping,mass,force,y):
    y0=y[0]+t*y[1]
    y1=-(w*w)*t*y[0]+(1-2*damping*w*t+force*t/mass)*y[1]
    return (y0,y1)

def sat(im,alpha=1,beta=0,min='m',max='x'):
    oim=im*alpha+beta
    if min=='m':
        min=np.min(im)
    if max=='x':
        max=np.max(im)
    oim[oim>max]=max;
    oim[oim<min]=min;
    return oim

def iir(old,new,r):
    return old*r+new*(1-r)

def butter(y_prev,x_prev,x,a,b):
    return (-b[1]*y_prev+a[0]*x+a[1]*x_prev)/b[0]

def complex_resize(im,shp):
    return cv2.resize(np.real(im),shp)+1j*cv2.resize(np.imag(im),shp)

def phase_diffs(cap,fl,fh,sigma_time,sigma_freq,k_size_x,k_size_t,num_frames,inference_lev,sz,re_sz,gamma=1):
    y_ind=inference_lev*2+1
    x_ind=inference_lev*2+2
    k_width=k_size_x/2
    nyquist=num_frames/2
    num_levs=inference_lev
    
    ret,frame=cap.read()
    bgr=cv2.resize(frame[crp[0]:crp[1],crp[2]:crp[3],:],re_sz)
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
        bgr=cv2.resize(frame[crp[0]:crp[1],crp[2]:crp[3],:],re_sz)
        im=np.uint8(np.mean(bgr,2))
        ims('bgr',bgr)
        #im=np.uint8(np.mean(frame[:frame_height,:frame_width,:],2))
        im_pyr=ppt.SCFpyr(im,num_levs,1)
        x_band=im_pyr.pyr[x_ind]
        y_band=im_pyr.pyr[y_ind]
        x_lo[:,:]=butter(x_lo,x_prev,x_band,lo_a,lo_b)
        y_lo[:,:]=butter(y_lo,y_prev,y_band,lo_a,lo_b)
        x_hi[:,:]=butter(x_hi,x_prev,x_band,hi_a,hi_b)
        y_hi[:,:]=butter(y_hi,y_prev,y_band,hi_a,hi_b)
        phase_x[:,:,i]=awg(np.abs(x_lo)**gamma+np.abs(x_hi)**gamma,np.imag(np.log(x_lo/x_hi)),k_size_x,sigma_time)
        phase_y[:,:,i]=awg(np.abs(y_lo)**gamma+np.abs(y_hi)**gamma,np.imag(np.log(y_lo/y_hi)),k_size_x,sigma_time)
        ims('x',phase_x[:,:,i])
        ims('y',phase_y[:,:,i])
        y_prev[:,:]=y_band[:,:]
        x_prev[:,:]=x_band[:,:]
    print ("video done")
    return (phase_x,phase_y)

def analyze_modes_butter(cap,fl,fh,sigma_time,sigma_freq,k_size_x,k_size_t,num_frames,inference_lev,num_modes,sz,re_sz,ul,gamma=1):
    y_ind=inference_lev*2+1
    x_ind=inference_lev*2+2
    k_width=k_size_x/2
    nyquist=num_frames/2
    num_levs=inference_lev
    
    ret,frame=cap.read()
    bgr=cv2.resize(frame[crp[0]:crp[1],crp[2]:crp[3],:],re_sz)
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
        bgr=cv2.resize(frame[crp[0]:crp[1],crp[2]:crp[3],:],re_sz)
        im=np.uint8(np.mean(bgr,2))
        ims('bgr',bgr)
        #im=np.uint8(np.mean(frame[:frame_height,:frame_width,:],2))
        im_pyr=ppt.SCFpyr(im,num_levs,1)
        x_band=im_pyr.pyr[x_ind]
        y_band=im_pyr.pyr[y_ind]
        x_lo[:,:]=butter(x_lo,x_prev,x_band,lo_a,lo_b)
        y_lo[:,:]=butter(y_lo,y_prev,y_band,lo_a,lo_b)
        x_hi[:,:]=butter(x_hi,x_prev,x_band,hi_a,hi_b)
        y_hi[:,:]=butter(y_hi,y_prev,y_band,hi_a,hi_b)
        phase_x[:,:,i]=awg(np.abs(x_lo)**gamma+np.abs(x_hi)**gamma,np.imag(np.log(x_lo/x_hi)),k_size_x,sigma_time)
        phase_y[:,:,i]=awg(np.abs(y_lo)**gamma+np.abs(y_hi)**gamma,np.imag(np.log(y_lo/y_hi)),k_size_x,sigma_time)
        ims('x',phase_x[:,:,i])
        ims('y',phase_y[:,:,i])
        y_prev[:,:]=y_band[:,:]
        x_prev[:,:]=x_band[:,:]
    print ("video done")

    freq_x=np.fft.fftshift(np.fft.fft(phase_x,axis=2),axes=2)
    print ("x analyzed")
    freq_y=np.fft.fftshift(np.fft.fft(phase_y,axis=2),axes=2)
    print ("y analyzed")

    peaks=np.sum(np.sum(np.abs(freq_x)**gamma+np.abs(freq_y)**gamma,0),0)
    #(x_bases,freqs,masses,dels)=get_bases(peaks,num_modes,freq_x,k_size_x,sigma_freq)
    return (peaks,num_modes,freq_x,freq_y)


def get_bases(peaks,num_modes,spectral_vol,k_size,sigma):
    shp=spectral_vol.shape
    nyquist=shp[2]/2
    peak_inds=np.argsort(peaks[:nyquist])[-num_modes:]
    frequencies=nyquist-peak_inds
    amps=peaks[peak_inds]
    amps/=x_modes.shape[0]*x_modes.shape[1]
    amps*=2
    masses=1/amps
    dels=1/(pks[peak_inds])
    dels*=x_modes.shape[0]*x_modes.shape[1]
    dels/=1024
    modes=np.zeros((shp[0],shp[1],num_modes),dtype=np.complex128)
    for i,p in enumerate(peak_inds):
        modes[:,:,i]=complex_blur2d(spectral_vol[:,:,p],k_size,sigma)
    return (modes,frequencies,masses,dels)

def simulate(bgr,motion_bases,freqs,dels,masses,fps,alpha,name,out_dir,time):
    fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
    out_shp=(bgr.shape[1],bgr.shape[0])
    out_vid=vid_writer(name+'_alpha_'+str(alpha)+'_synth','.mp4',fourcc,fps,out_shp,out_dir+'/'+name,name,out_dir+'/'+name+'/'+name+'/'+name+'.txt',200)
    #out_vid=cv2.VideoWriter(,fourcc,fps,,True)
    inference_lev=np.int32(np.log2(bgr.shape[0]/motion_bases.shape[0]))
    frame_height=bgr.shape[0];frame_width=bgr.shape[1];
    time_step=0.01
    amps=1/masses
    modal_coords=[]
    for i in range(motion_bases.shape[2]):
        modal_coords.append([0,amps[i]])
    modal_coords=np.asarray(modal_coords)
    big_flow=np.zeros((frame_height,frame_width,2),dtype=np.float64)
    
    for i in range(time):
        div=2**inference_lev
        flow=np.zeros((frame_height/div,frame_width/div,2),dtype=np.float64)
        for j in range(motion_bases.shape[2]):
            mc=modal_coords[j,:]
            modal_coord=mc[0]+1j*mc[1]/freqs[j]
            flow[:,:,:]+=np.real(modal_coord*motion_bases[:,:,j,:])
        mc=iter(time_step,freqs[j],dels[j],masses[j],0,mc)
        if (np.abs(mc[0]+1j*mc[1]/freqs[j])>np.abs(modal_coord)):
            mul=np.abs(modal_coord)/np.abs(mc[0]+1j*mc[1]/freqs[j])
            mc=(mc[0]*mul,mc[1]*mul)
        modal_coords[j,:]=mc
        big_flow[:,:,0]=cv2.resize(flow[:,:,0],(frame_width,frame_height))
        big_flow[:,:,1]=cv2.resize(flow[:,:,1],(frame_width,frame_height))
        #warp=interp(bgr,big_flow*alpha)
        warp=push(bgr,big_flow*alpha)
        ims('x',flow[:,:,1])
        ims('y',flow[:,:,0])
        out_vid.write(np.uint8(warp))
        ims('w',np.uint8(warp))
        if i%20==0:
            print (i)
    out_vid.finish()
    

def save_modes(im,motion_bases,freqs,dels,masses,fps,amplitude,name,out_dir):
    os.system("mkdir "+out_dir+'/'+name)
    f=out_dir+'/'+name
    cv2.imwrite(out_dir+'/'+name+'.png',im)
    affines=np.zeros((len(freqs),4,2),dtype=np.float64)
    for i in range(len(freqs)):
        blank=np.zeros((motion_bases.shape[0],motion_bases.shape[1],4),dtype=np.uint8)
        flow=motion_bases[:,:,i,:]
        cv2.imwrite(out_dir+'/x_'+str(i)+'.png',ate(vh(flow[:,:,0])))
        cv2.imwrite(out_dir+'/y_'+str(i)+'.png',ate(vh(flow[:,:,1])))
    np.savez(f,im=im,motion_bases=motion_bases,freqs=freqs,dels=dels,masses=masses,fps=fps,amplitude=amplitude)


def load_modes(out_dir,name):
    f=out_dir+'/'+name+'/'+name+'.npz'
    l=np.load(f)
    return l

def sim(name,directory,alpha,time):
    modes=load_modes(directory,name)
    freqs=modes["freqs"]
    fps=modes["fps"]
    motion_bases=modes['motion_bases']
    dels=modes['dels']
    masses=modes['masses']
    im=modes['im']
    simulate(im,motion_bases,freqs,dels,masses,fps,alpha,name,directory,time)
