import numpy as np
import cv2
import pyPyrTools3 as ppt
import os
import json
from scipy import signal
from vid_writer import vid_writer
import requests
from PIL import Image
from io import BytesIO

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

def white_out(name,col):
    im=cv2.imread(name)
    oim=np.zeros((im.shape[0],im.shape[1],4),dtype=np.uint8)
    oim[:,:,:3]=im
    msk=(im[:,:,0]==col[0])*(im[:,:,1]==col[1])*(im[:,:,2]==col[2])
    oim[:,:,3]=255*(1-msk)
    cv2.imwrite(name,oim)

def clear_back(im,col):
    oim=np.zeros((im.shape[0],im.shape[1],4),dtype=np.uint8)
    oim[:,:,:3]=im
    msk=(im[:,:,0]==col[0])*(im[:,:,1]==col[1])*(im[:,:,2]==col[2])
    oim[:,:,3]=255*(1-msk)
    return oim

def gshow(name,p_im):
    top=np.max(p_im)
    bottom=np.min(p_im)
    rng=np.max((np.abs(top),np.abs(bottom)))
    ims(name,np.uint8(128+127*p_im/rng))

def complex_plt(pts):
    out=np.zeros((255,len(pts),3))
    low=np.min((np.min(np.real(pts)),np.min(np.imag(pts))))
    high=np.max((np.max(np.real(pts)),np.max(np.imag(pts))))
    x=(np.real(pts)-low)/(high-low)
    y=(np.imag(pts)-low)/(high-low)
    for i,re in enumerate(x):
        im=y[i]
        out[np.int32(255*(1-re)):,i,0]=1
        out[np.int32(255*(1-im)):,i,r]=1
    return out

def plotpts(pts):
    out=np.zeros((255,len(pts)))
    poi=(pts-np.min(pts))/(np.max(pts)-np.min(pts))
    for i,pt in enumerate(poi):
        out[np.int32(255*(1-pt)):,i]=1
    return out

def putblobs(centers,sigmas,width):
    numwide=np.int32(np.sqrt(len(centers)))
    im=np.zeros((width*numwide,width*numwide))
    x=2*(np.arange(width**2).reshape(width,width)%width)/width-1
    y=2*(np.arange(width**2).reshape(width,width)//width)/width-1
    y*=width/2
    x*=width/2
    for i in range(len(centers)):
        xind=i%numwide;yind=i//numwide;
        im[yind*width:(yind+1)*width,xind*width:(xind+1)*width]=np.exp(-(((y-centers[i][0])/sigmas[i])**2+((x-centers[i][1])/sigmas[i])**2))
    return im

def hank(mat,n):
    wid=mat.shape[1]
    full_mat=np.zeros((wid*(wid-n),n))
    for i in range(wid-n):
        full_mat[i*wid:(i+1)*wid,:]=mat[:,i:i+n]
    return full_mat

def animate_blobs(centers,sigmas,width):
    numwide=np.int32(np.sqrt(len(centers)))
    out_vol=np.zeros((width*numwide,width*numwide,len(centers[0][0])))
    for i in range(len(centers[0][0])):
        cs=[[c[0][i],c[1][i]] for c in centers]
        sigs=[sig[i] for sig in sigmas]
        out_vol[:,:,i]=putblobs(cs,sigs,width)
    return out_vol
                         

def awg(aband,pband,g,s):
    #b=cv2.GaussianBlur(aband,(g,g),s)
    pb=cv2.GaussianBlur(pband*aband,(g,g),s)
    #b[b<1]=1
    return pb

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

def grate(im):
    mx=np.max(im)
    mn=np.min(im)
    scl=np.max((np.max((mx,0)),-1*np.min((mn,0))))
    if scl==0:
        scl=1
    return np.uint8(127*(im/scl+1))

def vh(ws):
    if len(ws.shape)==3:
        w=ws[:,:,0]+1j*ws[:,:,1]
    else:
        w=ws.copy()
    amps=np.abs(w)
    ww=w.copy()
    ww[w==0]=1
    phases=np.imag(np.log(ww))
    im=np.zeros((amps.shape[0],amps.shape[1],3),dtype=np.float32)
    im[:,:,2]=(amps-np.min(amps)/(np.max(amps)-np.min(amps)))
    im[:,:,0]=360*((phases+np.pi)/(2*np.pi))
    im[:,:,1]=1
    return cv2.cvtColor(im,cv2.COLOR_HSV2BGR)
#return phases
#def vh(w):
#    amps=np.abs(w)
#    ww=w.copy()
#    ww[w==0]=1
#    phases=np.imag(np.log(ww))
#    im=np.zeros((amps.shape[0],amps.shape[1],3),dtype=np.float32)
#    im[:,:,2]=(amps-np.min(amps)/(np.max(amps)-np.min(amps)))
#    im[:,:,0]=360*((phases+np.pi)/(2*np.pi))
#    im[:,:,1]=.5
#    return cv2.cvtColor(im,cv2.COLOR_HSV2BGR)

def phasediff(p1,p2):
    return (p1-p2+np.pi)%(2*np.pi)-np.pi

def pd(w1,w2):
    return np.imag(np.log(w1/w2))

def sharpen(signal,k_size,sigma1,sigma2):           
    o=np.ones(signal.shape)
    kern1=cv2.getGaussianKernel(k_size,sigma1)
    kern2=cv2.getGaussianKernel(k_size,sigma2)
    sig1=np.convolve(signal,kern1.squeeze(),mode='same')
    sig1/=np.convolve(o,kern1.squeeze(),mode='same')
    sig2=np.convolve(signal,kern2.squeeze(),mode='same')
    sig2/=np.convolve(o,kern2.squeeze(),mode='same')
    return sig1-sig2

def control(pic, flow):
    ys=np.arange(pic.shape[0]*pic.shape[1])/pic.shape[1]
    ud=(flow[:,:,0].reshape(-1)+ys)%pic.shape[0]
    xs=np.arange(pic.shape[0]*pic.shape[1])%pic.shape[1]
    lr=(flow[:,:,1].reshape(-1)+xs)%pic.shape[1]
    ans=np.zeros(pic.shape)
    ans[ys,xs,:]=pic[ud,lr,:]
    return ans

def push_pull(frm,flow):
    p0=flow[:,:,0]
    p0=np.dstack((p0,p0,p0))
    p0[:,:,1]=flow[:,:,1]
    pp=push(p0,flow)
    pp=cv2.GaussianBlur(pp,(9,9),1.5)
    flw=flow.copy()
    flw[:,:,:2]=-pp[:,:,:2]
    return interp(frm,flw)

def push_resc_pull(frm,flow,base_flow):
    scale=base_flow.shape[0]/flow.shape[0]
    p0=flow[:,:,0]
    p0=np.dstack((p0,p0,p0))
    p0[:,:,1]=flow[:,:,1]
    pp=push(p0,flow)
    pp=cv2.GaussianBlur(pp,(5,5),1)
    pp=cv2.resize(pp*scale,(frm.shape[1],frm.shape[0]))
    flw=pp[:,:,:2].copy()
    return pull(frm,flw+base_flow)

def push(pic,flow):
    p=np.float32(pic.copy())
    ref_pic=p.copy()
    p*=0
    alpha=4
    weights=np.zeros(pic.shape,dtype=np.float32)
    ys=np.arange(pic.shape[0]*pic.shape[1])//pic.shape[1]
    ud=sat(flow[:,:,0].reshape(-1)+ys,mn=0,mx=pic.shape[0]-1)
    xs=np.arange(pic.shape[0]*pic.shape[1])%pic.shape[1]
    lr=sat(flow[:,:,1].reshape(-1)+xs,mn=0,mx=pic.shape[1]-1)
    u=np.int32(np.floor(ud))
    d=np.int32(np.ceil(ud))%pic.shape[0]
    l=np.int32(np.floor(lr))
    r=np.int32(np.ceil(lr))%pic.shape[1]
    udiffs=ud-u
    udiffs=np.squeeze(np.dstack((udiffs,udiffs,udiffs)))
    ldiffs=lr-l
    ldiffs=np.squeeze(np.dstack((ldiffs,ldiffs,ldiffs)))
    p[u,l,:]+=ref_pic[ys,xs,:]*(1-udiffs)*(1-ldiffs)*alpha
    weights[u,l,:]+=(1-udiffs)*(1-ldiffs)*alpha
    p[u,r,:]+=ref_pic[ys,xs,:]*(1-udiffs)*(ldiffs)*alpha
    weights[u,r,:]+=(1-udiffs)*(ldiffs)*alpha
    p[d,l,:]+=ref_pic[ys,xs,:]*(udiffs)*(1-ldiffs)*alpha
    weights[d,l,:]+=(udiffs)*(1-ldiffs)*alpha
    p[d,r,:]+=ref_pic[ys,xs,:]*(udiffs)*(ldiffs)*alpha
    weights[d,r,:]+=(udiffs)*(ldiffs)*alpha
    out_im=pic.copy()
    msk=weights>alpha
    out_im[msk]=p[msk]/weights[msk]
    #print (np.max(flow))
    p=cv2.GaussianBlur(p,(95,95),.5)
    weights[weights==0]=1
    weights=cv2.GaussianBlur(weights,(95,95),.5)
    out_im[msk==0]=p[msk==0]/weights[msk==0]
    return out_im

def push_sparse(pic,flow):
    p=pic.copy()
    weights=np.ones((pic.shape[0],pic.shape[1]),dtype=np.float32)
    ys=np.arange(pic.shape[0]*pic.shape[1])//pic.shape[1]
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
    flw=np.float32(flow)
    return cv2.remap(pic,flw[:,:,0],flw[:,:,1],interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    
    #return interp(pic,flow)

def interp(pic,flow):
    ys=np.arange(pic.shape[0]*pic.shape[1])//pic.shape[1]
    ud=sat((flow[:,:,0].reshape(-1)+ys),mn=0,mx=pic.shape[0]-1)
    xs=np.arange(pic.shape[0]*pic.shape[1])%pic.shape[1]
    lr=sat((flow[:,:,1].reshape(-1)+xs),mn=0,mx=pic.shape[1]-1)
    u=sat(np.int64(np.floor(ud)),mn=0,mx=pic.shape[1]-1)
    d=sat(np.int64(np.ceil(ud)),mn=0,mx=pic.shape[0]-1)
    udiffs=ud-u
    udiffs=np.dstack((udiffs,udiffs,udiffs))
    l=sat(np.int64(np.floor(lr)),mn=0,mx=pic.shape[1]-1)
    r=sat(np.int64(np.ceil(lr)),mn=0,mx=pic.shape[1]-1)
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

def push_pull(frm,flow):
     p0=flow[:,:,0]
     p0=np.dstack((p0,p0,p0))
     p0[:,:,1]=flow[:,:,1]
     pp=push(p0,flow)
     pp=cv2.GaussianBlur(pp,(9,9),1.5)
     flw=flow.copy()
     flw[:,:,:2]=-pp[:,:,:2]
     return pull(frm,flw)

def iter(t,w,damping,mass,force,y):
    y0=y[0]+t*y[1]
    y1=-(w*w)*t*y[0]+(1-2*damping*w*t)*y[1]+force*t/mass
    return (y0,y1)

def complex_iter(t,w,damping,mass,force,y):
    y0=y[0]+t*y[1]
    y1=-(w*w)*t*y[0]+(1-2*damping*w*t)*y[1]+force*t/mass
    return (y0,y1)

def sat(im,alpha=1,beta=0,mn='m',mx='x'):
    oim=im.copy()
    if mn=='m':
        mn=np.min(im)
    if max=='x':
        mx=np.max(im)
    oim[oim>mx]=mx;
    oim[oim<mn]=mn;
    return alpha*oim+beta

def iir(old,new,r):
    return old*r+new*(1-r)

def butter(y_prev,x_prev,x,a,b):
    return (-b[1]*y_prev+a[0]*x+a[1]*x_prev)/b[0]

def complex_resize(im,shp,interpolation=cv2.INTER_CUBIC):
    return cv2.resize(np.real(im),shp)+1j*cv2.resize(np.imag(im),shp)

def butter_modes(cap,fl,fh,num_frames,inference_lev,num_modes,sz,sigma_time=3,sigma_freq=3,k_size_x=11,k_size_t=5,start=0):
    return 1

def analyze_modes_butter(cap,fl,fh,num_frames,inference_lev,num_modes,sz,sigma_time=3,sigma_freq=3,k_size_x=11,k_size_t=5,start=0):
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    y_ind=inference_lev*2+1
    x_ind=inference_lev*2+2
    k_width=k_size_x/2
    nyquist=num_frames/2
    num_levs=inference_lev+1
    frame_width=sz[1]
    frame_height=sz[0]
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
    phase_x=np.zeros((np.int32(frame_height/(2**inference_lev)),np.int32(frame_width/(2**inference_lev)),num_frames))
    phase_y=phase_x.copy()
    [lo_a,lo_b]=signal.butter(1,fl)
    [hi_a,hi_b]=signal.butter(1,fh)
    for i in range(num_frames):
        ret,frame=cap.read();
        im=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #im=np.uint8(np.mean(frame[:frame_height,:frame_width,:],2))
        im_pyr=ppt.SCFpyr(im,num_levs,1)
        x_band=im_pyr.pyr[x_ind]
        y_band=im_pyr.pyr[y_ind]
        x_lo[:,:]=butter(x_lo,x_prev,x_band,lo_a,lo_b)
        y_lo[:,:]=butter(y_lo,y_prev,y_band,lo_a,lo_b)
        x_hi[:,:]=butter(x_hi,x_prev,x_band,hi_a,hi_b)
        y_hi[:,:]=butter(y_hi,y_prev,y_band,hi_a,hi_b)
        phase_x[:,:,i]=awg(np.abs(x_lo)+np.abs(x_hi),np.imag(np.log(x_lo/x_hi)),k_size_x,sigma_time)
        phase_y[:,:,i]=awg(np.abs(y_lo)+np.abs(y_hi),np.imag(np.log(y_lo/y_hi)),k_size_x,sigma_time)
        gshow('x',phase_x[:,:,i])
        gshow('y',phase_y[:,:,i])
        y_prev[:,:]=y_band[:,:]
        x_prev[:,:]=x_band[:,:]
    print ("video done")

    freq_x=np.fft.fftshift(np.fft.fft(phase_x,axis=2),axes=2)
    print ("x analyzed")
    freq_y=np.fft.fftshift(np.fft.fft(phase_y,axis=2),axes=2)
    print ("y analyzed")

    peaks=np.sum(np.sum(np.abs(freq_x)+np.abs(freq_y),0),0)
    (x_modes,y_modes,frequencies,masses,dels)= get_bases_xy(peaks,num_modes,freq_x,freq_y,k_size_x,sigma_freq)
    return (peaks, x_modes,y_modes,frequencies,masses,dels)

def get_bases_xy(peaks,num_modes,spectral_vol_x,spectral_vol_y,k_size,sigma):
    shp=spectral_vol_x.shape
    nyquist=np.int32(shp[2]/2)
    peak_inds=np.argsort(peaks[:nyquist])[-num_modes:]
    frequencies=nyquist-peak_inds
    x_modes=np.zeros((shp[0],shp[1],num_modes),dtype=np.complex128)
    y_modes=x_modes.copy()
    amps=peaks[peak_inds]
    amps/=x_modes.shape[0]*x_modes.shape[1]
    amps*=2
    masses=1/amps
    dels=1/(peaks[peak_inds])
    dels*=x_modes.shape[0]*x_modes.shape[1]
    dels/=1024
    
    for i,p in enumerate(peak_inds):
        x_modes[:,:,i]=complex_blur2d(spectral_vol_x[:,:,p],k_size,sigma)
        x_modes[:,:,i]/=np.max(np.abs(x_modes[:,:,i]))
        y_modes[:,:,i]=complex_blur2d(spectral_vol_y[:,:,p],k_size,sigma)
        y_modes[:,:,i]/=np.max(np.abs(y_modes[:,:,i]))
    return (x_modes,y_modes,frequencies,masses,dels)

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

def get_motion_bases(x_modes,y_modes):
    shp=x_modes.shape
    motion_bases=np.zeros((shp[0],shp[1],shp[2],2),dtype=np.complex128)
    for i in range(shp[2]):
        motion_bases[:,:,i,0]=y_modes[:,:,i]
        motion_bases[:,:,i,1]=x_modes[:,:,i]
    return motion_bases

def simulate(bgr,motion_bases,freqs,dels,masses,fps,alpha,name,out_dir,time):
    fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
    out_shp=(bgr.shape[1],bgr.shape[0])
    out_vid=vid_writer(name+'_alpha_'+str(alpha)+'_synth','.mp4',fourcc,fps,out_shp,out_dir+'/'+name,200)
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
        flow=np.zeros((np.int32(frame_height/div),np.int32(frame_width/div),2),dtype=np.float64)
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

def complex_simulate(bgr,motion_bases,freqs,dels,masses,fps,alpha,name,out_dir,time):
    fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
    out_shp=(bgr.shape[1],bgr.shape[0])
    out_vid=vid_writer(name+'_alpha_'+str(alpha)+'_synth','.mp4',fourcc,fps,out_shp,out_dir+'/'+name,200)
    #out_vid=cv2.VideoWriter(,fourcc,fps,,True)
    inference_lev=np.int32(np.log2(bgr.shape[0]/motion_bases.shape[0]))
    frame_height=bgr.shape[0];frame_width=bgr.shape[1];
    time_step=0.01
    amps=1/masses
    modal_coords=[]
    for i in range(motion_bases.shape[2]):
        modal_coords.append(np.mean(np.mean(motion_bases[:,:,i,:],0),0)*alpha)
    modal_coords=np.asarray(modal_coords)
    big_flow=np.zeros((frame_height,frame_width,2),dtype=np.float64)
    
    for i in range(time):
        div=2**inference_lev
        flow=np.zeros((np.int32(frame_height/div),np.int32(frame_width/div),2),dtype=np.float64)
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
    

#def save_modes(im,motion_bases,freqs,dels,masses,fps,amplitude,name,out_dir):
#    os.system("mkdir "+out_dir+'/'+name)
#    f=out_dir+'/'+name
#    cv2.imwrite(out_dir+'/'+name+'.png',im)
#    affines=np.zeros((len(freqs),4,2),dtype=np.float64)
#    for i in range(len(freqs)):
#        blank=np.zeros((motion_bases.shape[0],motion_bases.shape[1],4),dtype=np.uint8)
#        flow=motion_bases[:,:,i,:]
#        cv2.imwrite(out_dir+'/x_'+str(i)+'.png',ate(vh(flow[:,:,0])))
#        cv2.imwrite(out_dir+'/y_'+str(i)+'.png',ate(vh(flow[:,:,1])))
#    np.savez(f,im=im,motion_bases=motion_bases,freqs=freqs,dels=dels,masses=masses,fps=fps,amplitude=amplitude)

def save_modes(frame,out_dir,stem,pks,num_modes,x_freqs,y_freqs,k_size=5,sigma1=.25,sigma2=.5,sigma_time=5,fl=121/396,fh=3/4):
    name=stem
    os.system("mkdir "+out_dir+'/'+name)
    ind=np.int32(np.sqrt(num_modes+3))
    scale=np.max((255/x_freqs.shape[0],len(pks)/x_freqs.shape[1]))
    shp=[np.int32(scale*i) for i in x_freqs.shape]
    big_shp=x_freqs.shape
    x=ind
    y=ind
    if x*y<num_modes+3:
        x+=1
    if x*y<num_modes+3:
        y+=1
    xoim=np.zeros((y*shp[0],shp[1]*x,3),dtype=np.uint8)
    (oim,inds)=sharp_pks(pks,k_size,sigma1,sigma2,num_modes)
    omegas=pks.shape[0]-inds
    (masses,dels)=get_masses_dels(pks,inds,x_freqs.shape)
    xoim[:255,:2*len(pks),:]=ate(oim)
    xoim[:shp[0],2*shp[1]:3*shp[1],:]=cv2.resize(frame,(shp[1],shp[0]))
    yoim=xoim.copy()
    modes=np.zeros((big_shp[0],big_shp[1],num_modes,2),dtype=np.complex64)
    for i in range(3,num_modes+3):
        f=inds[i-3]
        omega=omegas[i-3]
        modes[:,:,i-3,0]=y_freqs[:,:,f]
        modes[:,:,i-3,1]=x_freqs[:,:,f]
        mode=np.zeros((big_shp[0],big_shp[1]*2),dtype=np.complex64)
        mode[:,:big_shp[1]]=y_freqs[:,:,f]
        mode[:,big_shp[1]:]=x_freqs[:,:,f]
        mode=ate(vh(mode))
        x_off=(i%x)*shp[1]
        y_off=(i//x)*shp[0]
        xoim[y_off:y_off+shp[0],x_off:x_off+shp[1],:]=cv2.resize(mode[:,big_shp[1]:],(shp[1],shp[0]))
        yoim[y_off:y_off+shp[0],x_off:x_off+shp[1],:]=cv2.resize(mode[:,:big_shp[1]],(shp[1],shp[0]))
        xoim=cv2.putText(xoim,str(omega),(10+x_off,15+y_off),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        yoim=cv2.putText(yoim,str(omega),(10+x_off,15+y_off),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    np.savez(out_dir+'/'+name+'/'+name,modes=modes,inds=inds,sigma1=sigma1,sigma2=sigma2,sigma_time=sigma_time,pks=pks,fl=fl,fh=fh,omegas=omegas,masses=masses,dels=dels)
    cv2.imwrite(out_dir+'/'+name+'/'+name+'x_modes_'+'.png',xoim)
    cv2.imwrite(out_dir+'/'+name+'/'+name+'y_modes_'+'.png',yoim)

def save_peaks(frame,pks,k_size,sigma1,sigma2,sigma_time,num_modes,x_freqs,y_freqs,style,fl,fh,out_dir,stem):
    name=stem+'_'+style+'_fl_'+str(fl)+'_fh_'+str(fh)+'_sigma_time_'+str(sigma_time)
    os.system("mkdir "+out_dir+'/'+name)
    ind=np.int32(np.sqrt(num_modes+3))
    scale=np.max((255/x_freqs.shape[0],len(pks)/x_freqs.shape[1]))
    shp=[np.int32(scale*i) for i in x_freqs.shape]
    big_shp=x_freqs.shape
    x=ind
    y=ind
    if x*y<num_modes+3:
        x+=1
    if x*y<num_modes+3:
        y+=1
    xoim=np.zeros((y*shp[0],shp[1]*x,3),dtype=np.uint8)
    (oim,inds)=sharp_pks(pks,k_size,sigma1,sigma2,num_modes)
    xoim[:255,:2*len(pks),:]=ate(oim)
    xoim[:shp[0],2*shp[1]:3*shp[1],:]=cv2.resize(frame,(shp[1],shp[0]))
    yoim=xoim.copy()
    modes=np.zeros((big_shp[0],big_shp[1],num_modes,2),dtype=np.complex64)
    for i in range(3,num_modes+3):
        f=inds[i-3]
        modes[:,:,i-3,0]=y_freqs[:,:,f]
        modes[:,:,i-3,1]=x_freqs[:,:,f]
        mode=np.zeros((big_shp[0],big_shp[1]*2),dtype=np.complex64)
        mode[:,:big_shp[1]]=y_freqs[:,:,f]
        mode[:,big_shp[1]:]=x_freqs[:,:,f]
        mode=ate(vh(mode))
        x_off=(i%x)*shp[1]
        y_off=(i//x)*shp[0]
        xoim[y_off:y_off+shp[0],x_off:x_off+shp[1],:]=cv2.resize(mode[:,big_shp[1]:],(shp[1],shp[0]))
        yoim[y_off:y_off+shp[0],x_off:x_off+shp[1],:]=cv2.resize(mode[:,:big_shp[1]],(shp[1],shp[0]))
        xoim=cv2.putText(xoim,str(f),(10+x_off,15+y_off),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        yoim=cv2.putText(yoim,str(f),(10+x_off,15+y_off),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    np.savez(out_dir+'/'+name+'/'+name,modes=modes,inds=inds,sigma1=sigma1,sigma2=sigma2,sigma_time=sigma_time,pks=pks,style=style,fl=fl,fh=fh)
    cv2.imwrite(out_dir+'/'+name+'/'+name+'x_modes_'+'.png',xoim)
    cv2.imwrite(out_dir+'/'+name+'/'+name+'y_modes_'+'.png',yoim)
    
def put_pts_cols(pts,pt_colors,sz,axes,hi=[],lo=[],axis_labels=[1,1],label_inds=[[]],bg_color=[255,255,255],axes_color=[0,0,0],highlight_color=[0,0,255],pt_sz=1,label_names=[[]],titles=[''],title_colors=[],axis_offsets=[-20,-5]):
    x_max=sz[1]
    y_max=sz[0]
    pts=[np.dstack(pt_set).squeeze() for pt_set in pts]
    bounds=np.max(np.asarray([np.max(np.abs(pt_set),0) for pt_set in pts]),0)
    if not len(lo):
        lo=np.asarray([-bounds[0],-bounds[1]])
    else:
        lo=np.asarray(lo)
    if not len(hi):
        hi=np.asarray([bounds[0],bounds[1]])
    else:
        hi=np.asarray(hi)
    im=np.zeros((y_max,x_max,3),dtype=np.uint8)
    im[:,:]=bg_color
    y=[y_max-np.int32((y_max)*(pt_set[:,0]-lo[0])/(hi[0]-lo[0])) for pt_set in pts]
    x=[np.int32((x_max-1)*(pt_set[:,1]-lo[1])/(hi[1]-lo[1])) for pt_set in pts]
    axis_names=axes[1]
    axes=np.int32(np.asarray(sz)*(np.asarray(axes[0])-lo)/(hi-lo))
    axes[0]=(im.shape[0]-1)-axes[0]
    im=put_label(im,axis_names[1],(axes[0]-5,10),axes_color,fontsz=.75)
    im=put_label(im,axis_names[0],(0,axes[1]-20),axes_color,fontsz=.75)
    if not title_colors:
        title_colors=[axes_color for title in titles]
    for i,title in enumerate(titles):
        im=put_label(im,title,(10+i*30,20),title_colors[i],fontsz=.75)
    im[axes[0],:]=axes_color
    im[:,axes[1]]=axes_color
    for i in np.arange(trunc(lo[0],axis_labels[0]),hi[0],axis_labels[0]):
        axis_ind=y_max-np.int32((y_max)*(i-lo[0])/(hi[0]-lo[0]))
        im=put_label(im,str(i),(axis_ind,axes[1]+axis_offsets[1]),[0,0,0])
    for i in np.arange(trunc(lo[1],axis_labels[1]),hi[1],axis_labels[1]):
        axis_ind=np.int32((x_max-1)*(i-lo[1])/(hi[1]-lo[1]))
        im=put_label(im,str(i),(axes[0]+axis_offsets[0],axis_ind),[0,0,0])
    if not label_names:
        label_names=[labify(pts[fcn][label_ind],2) for fcn,label_ind in enumerate(label_inds)]
    for fcn in range(len(pt_colors)):
        for i in range(pts[fcn].shape[0]):
            ind=(y[fcn][i],x[fcn][i])
            if ind[0]>-1 and ind[1]>-1 and ind[0]<y_max and ind[1]<x_max:
                col=pt_colors[fcn]
                if not type(col[0])==int:
                    col=col[i]
                im[ind[0],ind[1],:]=col
    for fcn,ind_set in enumerate(label_inds):
        for i,ind in enumerate(ind_set):
            highlight=(y[fcn][ind],x[fcn][ind])
            if highlight[0]>-1 and highlight[1]>-1 and highlight[0]<y_max and highlight[1]<x_max:
                im=put_label(im,label_names[fcn][i],(y[fcn][ind]-20,np.min((x[fcn][ind]+10,sz[1]-30))),highlight_color,fontsz=.5)
                im[y[fcn][ind]-pt_sz:y[fcn][ind]+pt_sz,x[fcn][ind]-pt_sz:x[fcn][ind]+pt_sz]=highlight_color
    return im

def put_pts(pts,sz,axes,hi=[],lo=[],axis_labels=[1,1],label_inds=[],bg_color=[255,255,255],axes_color=[0,0,0],pt_color=[255,0,0],highlight_color=[0,0,255],pt_sz=1,label_names=[],title='',axis_offsets=[-20,-5]):
    x_max=sz[1]
    y_max=sz[0]
    pts=np.dstack(pts).squeeze()
    if not len(lo):
        lo=np.min(pts,0)
    else:
        lo=np.asarray(lo)
    if not len(hi):
        hi=np.max(pts,0)
    else:
        hi=np.asarray(hi)
    im=np.zeros((y_max,x_max,3),dtype=np.uint8)
    im[:,:]=bg_color
    y=y_max-np.int32((y_max)*(pts[:,0]-lo[0])/(hi[0]-lo[0]))
    x=np.int32((x_max-1)*(pts[:,1]-lo[1])/(hi[1]-lo[1]))
    axis_names=axes[1]
    axes=np.int32(np.asarray(sz)*(np.asarray(axes[0])-lo)/(hi-lo))
    im=put_label(im,axis_names[1],(axes[0]+axis_offsets[0],10),axes_color,fontsz=.75)
    im=put_label(im,axis_names[0],(0,axes[1]+axis_offsets[1]),axes_color,fontsz=.75)
    im=put_label(im,title,(10,20),axes_color,fontsz=.75)
    im[axes[0],:]=axes_color
    im[:,axes[1]]=axes_color
    if not label_names:
        label_names=labify(pts[label_inds],2)
    for i in range(pts.shape[0]):
        ind=(y[i],x[i])
        if ind[0]>-1 and ind[1]>-1 and ind[0]<y_max and ind[1]<x_max:
            im[ind[0],ind[1],:]=pt_color
    for i in np.arange(trunc(lo[0],axis_labels[0]),hi[0],axis_labels[0]):
        axis_ind=y_max-np.int32((y_max)*(i-lo[0])/(hi[0]-lo[0]))
        im=put_label(im,str(i),(axis_ind,axes[1]),[0,0,0])
    for i in np.arange(trunc(lo[1],axis_labels[1]),hi[1],axis_labels[1]):
        axis_ind=np.int32((x_max-1)*(i-lo[1])/(hi[1]-lo[1]))
        im=put_label(im,str(i),(axes[0],axis_ind),[0,0,0])
    for i,ind in enumerate(label_inds):
        highlight=(y[ind],x[ind])
        if highlight[0]>-1 and highlight[1]>-1 and highlight[0]<y_max and highlight[1]<x_max:
            im=put_label(im,label_names[i],(y[ind]-20,np.min((x[ind]+10,sz[1]-30))),highlight_color,fontsz=.5)
            im[y[ind]-pt_sz:y[ind]+pt_sz,x[ind]-pt_sz:x[ind]+pt_sz]=highlight_color
    return im

def freak_show(sig,freq,prevs,num_pts,highlight_ind=[],highlight_sz=0):
    #max_abs=np.max(np.abs(np.fft.fft(sig))[1:-1])/np.sqrt(len(sig))
    bound=np.max(np.abs(sig))
    omega=freq
    mx=np.max(sig)
    mn=np.min(sig)
    phase=np.arange(0,num_pts)/num_pts
    real_phase=np.arange(0,len(sig))/len(sig)
    oim=np.zeros((600,1200,3),dtype=np.uint8)
    p_sig=point_op(sig,phase*len(sig),0,len(sig))
    titles=['','f(t)']
    #lab_inds=[[int((num_pts-1)*t/len(sig))]]
    oim[:200,600:,:]=put_pts_cols([[p_sig,phase*len(sig)]],[[0,0,255]],(200,600),[[0,0],['','']],titles=titles,hi=[mx,len(sig)],lo=[mn,0],axis_labels=[1,len(sig)/5],axis_offsets=[-20,15],)
    #for i,omega in enumerate(omegas):
    phi=phase*omega*2j*np.pi
    r_phi=real_phase*omega*2j*np.pi
    c_sig=p_sig*np.exp(phi)
    r_sig=sig*np.exp(r_phi)
    title=['omega='+str(omega)]
    expect=np.mean(r_sig)
    pts=[[np.imag(c_sig),np.real(c_sig)],[np.imag(expect)*phase,np.real(expect)*phase]]
    lab2=str(np.real(expect)-np.real(expect)%.01+1j*(np.imag(expect)-np.imag(expect)%0.01))
    if highlight_sz:
        oim[:,:600]=put_pts_cols(pts,[[255,0,0],[0,0,255]],(600,600),[[0,0],['   im','re']],hi=[bound,bound],lo=[-bound,-bound],titles=title,label_inds=[highlight_ind,[-1]],label_names=[['' for ind in highlight_ind],[lab2]],pt_sz=3,axis_offsets=[-20,-15])
    else:
        oim[:,:600]=put_pts_cols(pts,[[255,0,0],[0,0,255]],(600,600),[[0,0],['   im','re']],hi=[bound,bound],lo=[-bound,-bound],titles=title,label_inds=[[],[-1]],label_names=[[''],[lab2]],axis_offsets=[-20,-15])
    prevs.append(expect)
    if len(prevs)>4:
        amps=point_op(np.abs(prevs),phase*freq,0,freq)
        phases=point_op(np.imag(np.log(prevs)),phase*freq,0,freq)
        #amps=np.zeros((len(phase),))
        #phases=np.zeros((len(phase),))
        #amps[:len(prevs)]=np.abs(prevs)
        label_inds=[[len(prevs)-1]]
        titles=['','Amplidutde(omega)','omega='+str(omega)]
        oim[200:400,600:,:]=put_pts_cols([[amps,phase*freq]],[[255,0,0]],(200,600),[[0,0],['','']],titles=titles,hi=[np.max(amps),freq],lo=[0,0],axis_labels=[1,freq/5],axis_offsets=[-20,15])
        titles=['','Phase(omega)','omega='+str(omega)]
        oim[400:,600:,:]=put_pts_cols([[phases,phase*freq]],[[255,0,0]],(200,600),[[0,0],['','']],titles=titles,hi=[np.pi,freq],lo=[-np.pi,0],axis_labels=[1,freq/5],axis_offsets=[-20,15])
    return oim


def omega_show(sig,omegas,t,num_pts,show_means=False):
    bound=np.max(np.abs(sig))
    mx=np.max(sig)
    mn=np.min(sig)
    phase=np.arange(0,num_pts)/num_pts
    oim=np.zeros((600,900,3),dtype=np.uint8)
    p_sig=point_op(sig,phase*len(sig),0,len(sig))
    titles=['','f(t)','t='+str(t),'','','','','','g_omega(t)=exp(2*pi*i*omega*t)']
    lab_inds=[[int((num_pts-1)*t/len(sig))]]
    oim[:300,:,:]=put_pts_cols([[p_sig,phase*len(sig)]],[[255,0,0]],(300,900),[[0,0],['','']],titles=titles,hi=[mx,len(sig)],lo=[mn,0],axis_labels=[1,len(sig)/5],label_inds=lab_inds,label_names=[['f('+str(t)+')']],axis_offsets=[-20,15])
    for i,omega in enumerate(omegas):
        phi=phase*omega*2j*np.pi
        c_sig=p_sig*np.exp(phi)
        title=['omega='+str(omega),'t='+str(t)]
        time=int(num_pts*t/len(sig))
        if show_means:
            expect=np.mean(c_sig[:time])
            pts=[[np.imag(c_sig[:time]),np.real(c_sig[:time])],[np.imag(expect)*phase,np.real(expect)*phase]]
            oim[300:,i*300:(i+1)*300]=put_pts_cols(pts,[[255,0,0],[0,0,255]],(300,300),[[0,0],['   im','re']],hi=[bound,bound],lo=[-bound,-bound],titles=title,label_inds=[[-1],[-1]],label_names=[['g_'+str(omega)+'(t)'],[str(np.real(expect)-np.real(expect)%.01+1j*(np.imag(expect)-np.imag(expect)%0.01))]],axis_offsets=[-20,-15])
        else:
            oim[300:,i*300:(i+1)*300]=put_pts_cols([[np.imag(c_sig[:time]),np.real(c_sig[:time])]],[[255,0,0]],(300,300),[[0,0],['   im','re']],hi=[bound,bound],lo=[-bound,-bound],titles=title,label_inds=[[-1]],label_names=[['g_'+str(omega)+'(t)']],axis_offsets=[-20,-15])
    return oim

def point_op(y,x,domin,domax):
     phase=x.copy()
     phase[phase>domax]=domax
     phase[phase<domin]=domin
     phase=(len(y)-1)*(phase-domin)/(domax-domin)
     #print(phase)
     #print(np.int64(np.floor(phase)))
     lo=y[np.int64(np.floor(phase))]
     hi=y[np.int64(np.ceil(phase))]
     mix=phase-np.floor(phase)
     return hi*mix+lo*(1-mix)

def gmm_f_sig(modes,sigmas,amps,sz):
    phase=np.pi*(2*np.arange(sz)/sz-1)
    freq=np.zeros((sz,),dtype=np.complex128)
    for i,mode in enumerate(modes):
        bumps=np.exp((-(sigmas[i]*(phase-mode))**2))+np.exp((-(sigmas[i]*(phase+mode))**2))
        bumps/=np.max(bumps)
        bumps*=amps[i]
        freq+=bumps
    noise=np.random.randn(len(freq))+1j*(np.random.randn(len(freq)))
    noise/=np.abs(noise)
    freq*=noise
    return np.real(np.fft.ifft(np.fft.ifftshift(freq)))

def put_label(im,label,loc,color,thickness=1,fontsz=.5):
    return cv2.putText(im,label,(loc[1]-10,15+loc[0]),cv2.FONT_HERSHEY_SIMPLEX,fontsz,color,thickness,cv2.LINE_AA)

def trunc(a,b):
    return np.sign(a)*(np.abs(a)//b)*b


def labify(pts,decimals):
    return [str((pt-pt%(10**-decimals))[::-1]) for pt in pts]

def view_sharp_pks(pks,k_size,sigma1,sigma2,num_modes):
    points=sharpen(pks,k_size,sigma1,sigma2)
    inds=np.argsort(points)[-num_modes:]
    out_im=np.zeros((255,len(pks)*2,3))
    out_im[:,:len(pks),:]=view_amps_pks(points,inds)
    out_im[:,len(pks):,:]=view_amps_pks(pks,inds)
    return out_im

def sharp_pks(pks,k_size,sigma1,sigma2,num_modes):
    points=sharpen(pks,k_size,sigma1,sigma2)
    inds=np.argsort(points)[-num_modes:]
    out_im=np.zeros((255,len(pks)*2,3))
    out_im[:,:len(pks),:]=view_amps_pks(points,inds)
    out_im[:,len(pks):,:]=view_amps_pks(pks,inds)
    return (out_im,inds)
    
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

def daw():
    cv2.destroyAllWindows()
    cv2.waitKey(2)

def view_amps_pks(amps,peaks):
    points=np.zeros((255,len(amps),3))
    poi=(amps-np.min(amps))/(np.max(amps)-np.min(amps))
    for i,pt in enumerate(poi):
        if i in peaks:
            points[np.int32(255*(1-pt)):,i,:2]=1
            points[:np.int32(255*(1-pt)),i,2]=1
        else:
            points[np.int32(255*(1-pt)):,i,:]=1
    points=cv2.putText(points,str(len(amps)),(10,250),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
    points=cv2.putText(points,str(0),(len(amps)-30,250),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
    return points

def grey_show(name,im):
    mx=np.max(im)
    mn=np.min(im)
    scl=np.max((np.max((mx,0)),-1*np.min((mn,0))))
    if scl==0:
        scl=1
    cv2.imshow(name,np.uint8(127*(im/scl+1)))
    cv2.waitKey(2)

def get_masses_dels(pks,inds,shp,consts=[.5,1024]):
    amps=pks[inds]
    amps/=shp[0]*shp[1]
    amps/=consts[0]
    dels=1/(pks[inds])
    dels*=shp[0]*shp[1]
    dels/=consts[1]
    return (1/amps,dels)



def encode(cap, out, xmos, ymos, cpf, damp, out_sz, gain=1,fl=.6,fh=.5, sigma_time=2, sigma_freq=2, k_size_x=9, k_size_t=9, inference_lev=1, re_sz=.5, gamma=1, num_frames=1):
    #cps = cycles per frame = hz/fps
    #re_sz should be approximate downsampling factor
    ret,frame=cap.read()
    frame0=cv2.resize(frame,(out_sz[1],out_sz[0]))
    base_x=np.arange(frame0.shape[0]*frame0.shape[1]).reshape(frame0.shape[0],-1)
    base_y=base_x//frame0.shape[1]
    base_x=base_x%frame0.shape[1]
    base_flow=np.float32(np.dstack((base_x,base_y)))
    feedback_tap=(1-damp)*np.exp(-2j*np.pi*cpf)
    print (feedback_tap)
    input_tap=damp
    out_x=np.zeros((xmos.shape[0],xmos.shape[1],1),dtype=np.complex128)
    out_y=out_x.copy()
    if num_frames==1:
        num_frames=np.int32(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_sig=np.zeros((xmos.shape[2],num_frames,2),dtype=np.complex128)
    ret,frame=cap.read()
    y_ind=inference_lev*2+1
    x_ind=inference_lev*2+2
    k_width=k_size_x/2
    num_levs=inference_lev+1
    new_sz=np.int32(np.array(frame.shape)*re_sz)
    sz=(2**num_levs)*(new_sz//(2**num_levs))
    sz=np.int32(sz/re_sz)
    re_sz=(new_sz[1],new_sz[0])
    bgr=cv2.resize(frame,re_sz,interpolation=cv2.INTER_AREA)
    im=np.uint8(np.mean(bgr,2))
    im_pyr=ppt.SCFpyr(im,num_levs,1)
    x_lo=im_pyr.pyr[x_ind].copy()
    x_hi=x_lo.copy()
    x_prev=x_lo.copy()
    y_lo=im_pyr.pyr[y_ind].copy()
    y_hi=y_lo.copy()
    y_prev=y_lo.copy()
    #phase_x=np.zeros((np.int(crp[0]/(2**inference_lev)),np.int(crp[1]/(2**inference_lev)),num_frames))
    phase_x=np.zeros(np.int32((re_sz[1]/(2**inference_lev),re_sz[0]/(2**inference_lev),1)))
    phase_y=phase_x.copy()
    [lo_a,lo_b]=signal.butter(1,fl)
    [hi_a,hi_b]=signal.butter(1,fh)
    pos=cap.get(cv2.CAP_PROP_POS_FRAMES)
    for i in range(num_frames-1):
        if i%80==0:
            print('frame = '+str(i))
        ret,frame=cap.read();
        if not ret:
            while not ret:
                numb=np.int32(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print('error'+str(numb))
                cap.set(cv2.CAP_PROP_POS_FRAMES,numb)
                ret,frame=cap.read();
        bgr=cv2.resize(frame,re_sz,interpolation=cv2.INTER_AREA)
        im=np.uint8(np.mean(bgr,2))
        im_pyr=ppt.SCFpyr(im,num_levs,1)
        x_band=im_pyr.pyr[x_ind]
        y_band=im_pyr.pyr[y_ind]
        x_lo[:,:]=butter(x_lo,x_prev,x_band,lo_a,lo_b)
        y_lo[:,:]=butter(y_lo,y_prev,y_band,lo_a,lo_b)
        x_hi[:,:]=butter(x_hi,x_prev,x_band,hi_a,hi_b)
        y_hi[:,:]=butter(y_hi,y_prev,y_band,hi_a,hi_b)
        phase_x[:,:,0]=awg(np.abs(x_lo)**gamma+np.abs(x_hi)**gamma,np.imag(np.log(x_lo/x_hi)),k_size_x,sigma_time)
        phase_y[:,:,0]=awg(np.abs(y_lo)**gamma+np.abs(y_hi)**gamma,np.imag(np.log(y_lo/y_hi)),k_size_x,sigma_time)
        gshow('x',phase_x[:,:,0])
        gshow('y',phase_y[:,:,0])
        out_x[:,:,:]=out_x[:,:,:]*feedback_tap+phase_x[:,:,:]*input_tap
        out_y[:,:,:]=out_y[:,:,:]*feedback_tap+phase_y[:,:,:]*input_tap
        state_x=np.mean(np.mean(xmos*out_x,0),0).reshape(1,1,xmos.shape[2])
        state_y=np.mean(np.mean(ymos*out_y,0),0).reshape(1,1,xmos.shape[2])
        out_sig[:,i,0]=state_y[0,0,:]
        out_sig[:,i,1]=state_x[0,0,:]
        ims('out_x',vh(out_x[:,:,0]))
        ims('out_y',vh(out_y[:,:,0]))
        y_prev[:,:]=y_band[:,:]
        x_prev[:,:]=x_band[:,:]
            
    print('input done')
    for i in range(out_sig.shape[1]):
        state_y[0,0,:]=out_sig[:,i,0]
        state_x[0,0,:]=out_sig[:,i,1]
        xmo=np.sum(xmos*state_x,2)
        ymo=np.sum(ymos*state_y,2)
        ims('xmo',vh(xmo))
        ims('ymo',vh(ymo))
        disp_x=cv2.resize(np.real(xmo),(frame0.shape[1],frame0.shape[0]))
        disp_y=cv2.resize(np.real(ymo),(frame0.shape[1],frame0.shape[0]))
        ims('statey',state_y)
        if i>1:
            ims('ptsy',plotpts(out_sig[0,:i,0]))
        pp=push_resc_pull(frame0,gain*np.dstack((disp_y,disp_x)),base_flow)
        ims('pp',pp)
        pp=ate(pp)
        out.write(pp)

    print ("video done")
    out.release()
    return (out_sig)

def decode(frame0, out_sig, xmos, ymos, gain, out, interp='push_resc_pull'):
    state_x=np.zeros((1,1,xmos.shape[2]),dtype=np.complex128)
    state_y=np.zeros((1,1,ymos.shape[2]),dtype=np.complex128)
    base_x=np.arange(frame0.shape[0]*frame0.shape[1]).reshape(frame0.shape[0],-1)
    base_y=base_x//frame0.shape[1]
    base_x=base_x%frame0.shape[1]
    base_flow=np.float32(np.dstack((base_x,base_y)))
    for i in range(out_sig.shape[1]):
        if i % 100 == 0:
            print(i)
        state_y[0,0,:]=out_sig[:,i,0]
        state_x[0,0,:]=out_sig[:,i,1]
        xmo=np.sum(xmos*state_x,2)
        ymo=np.sum(ymos*state_y,2)
        ims('xmo',vh(xmo))
        ims('ymo',vh(ymo))
        disp_x=cv2.resize(np.real(xmo),(frame0.shape[1],frame0.shape[0]))
        disp_y=cv2.resize(np.real(ymo),(frame0.shape[1],frame0.shape[0]))
        ims('statey',state_y)
        if i>1:
            ims('ptsy',plotpts(out_sig[0,:i,0]))
        if interp == 'push_resc_pull':
            pp=push_resc_pull(frame0,gain*np.dstack((disp_y,disp_x)),base_flow)
        elif interp == 'push':
            pp=push(frame0,gain*np.dstack((disp_y,disp_x)))
        else:
            pp=pull(frame0,base_flow+gain*np.dstack((disp_y,disp_x)))
        ims('pp',pp)
        pp=ate(pp)
        out.write(pp)
    print ("video done")
    out.release()

def phase_vol(vol,fl,fh,sigma_time,sigma_freq,k_size_x,k_size_t,num_frames,inference_lev,sz,re_sz,gamma=1):
    #re_sz should be approximate downsampling factor
    
    y_ind=inference_lev*2+1
    x_ind=inference_lev*2+2
    k_width=k_size_x/2
    nyquist=num_frames/2
    num_levs=inference_lev+1
    sz=2**np.int32(np.log2(np.array(sz)))
    re_sz=np.int32(sz[:-1]*re_sz)
    re_sz=(re_sz[1],re_sz[0])
    crp=[0,sz[0],0,sz[1]]



def phase_diffs(cap,fl=.6,fh=.5,sigma_time=2,sigma_freq=2,k_size_x=9,k_size_t=9,num_frames=800,inference_lev=1,re_sz=.5,gamma=1):
    #re_sz should be approximate downsampling factor
    ret,frame=cap.read()
    y_ind=inference_lev*2+1
    x_ind=inference_lev*2+2
    k_width=k_size_x/2
    nyquist=num_frames/2
    num_levs=inference_lev+1
    new_sz=np.int32(np.array(frame.shape)*re_sz)
    sz=(2**num_levs)*(new_sz//(2**num_levs))
    sz=np.int32(sz/re_sz)
    #sz=2**np.int32(np.log2(np.array(sz)*re_sz))
    #re_sz=np.int32(sz[:-1])
    re_sz=(new_sz[1],new_sz[0])
    crp=[0,sz[0],0,sz[1]]
    #ret,frame=cap.read()
    bgr=cv2.resize(frame[crp[0]:crp[1],crp[2]:crp[3],:],re_sz,interpolation=cv2.INTER_AREA)
    im=np.uint8(np.mean(bgr,2))
    im_pyr=ppt.SCFpyr(im,num_levs,1)
    x_lo=im_pyr.pyr[x_ind].copy()
    x_hi=x_lo.copy()
    x_prev=x_lo.copy()
    y_lo=im_pyr.pyr[y_ind].copy()
    y_hi=y_lo.copy()
    y_prev=y_lo.copy()
    #phase_x=np.zeros((np.int(crp[0]/(2**inference_lev)),np.int(crp[1]/(2**inference_lev)),num_frames))
    phase_x=np.zeros(np.int32((re_sz[1]/(2**inference_lev),re_sz[0]/(2**inference_lev),num_frames)))
    phase_y=phase_x.copy()
    [lo_a,lo_b]=signal.butter(1,fl)
    [hi_a,hi_b]=signal.butter(1,fh)
    pos=cap.get(cv2.CAP_PROP_POS_FRAMES)
    #for i in range(num_frames):
    #    cap.set(cv2.CAP_PROP_POS_FRAMES,np.int32(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    #    ret,frame=cap.read()
    #    #cap.set(cv2.CAP_PROP_POS_FRAMES,cap.get(cv2.CAP_PROP_POS_FRAMES)-1)
    #    if i%100==0:
    #        print(i)
    #    if not ret:
    #        print (cap.get(cv2.CAP_PROP_POS_FRAMES))
    for i in range(num_frames):
        if i%80==0:
            print('frame = '+str(i))
        #cap.set(cv2.CAP_PROP_POS_FRAMES,np.int32(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        ret,frame=cap.read();
        if not ret:
            numb=np.int32(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print('error'+str(numb))
            cap.set(cv2.CAP_PROP_POS_FRAMES,numb)
            ret,frame=cap.read();
        #if(cap.get(cv2.CAP_PROP_POS_FRAMES)>2100):
        #    print (ret)
        #    print (cap.get(cv2.CAP_PROP_POS_FRAMES))
        #    print (cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #    print (frame.shape)
        #    print (crp)
        #print (frame.shape)
        bgr=cv2.resize(frame[crp[0]:crp[1],crp[2]:crp[3],:],re_sz,interpolation=cv2.INTER_AREA)
        im=np.uint8(np.mean(bgr,2))
        #ims('bgr',bgr)
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
        gshow('x',phase_x[:,:,i])
        gshow('y',phase_y[:,:,i])
        y_prev[:,:]=y_band[:,:]
        x_prev[:,:]=x_band[:,:]
    print ("video done")
    return (phase_x,phase_y)


def phase_vol(vol,fl,fh,sigma_time,sigma_freq,k_size_x,k_size_t,num_frames,inference_lev,sz,re_sz,gamma=1):
    #re_sz should be approximate downsampling factor
    
    y_ind=inference_lev*2+1
    x_ind=inference_lev*2+2
    k_width=k_size_x/2
    nyquist=num_frames/2
    num_levs=inference_lev+1
    sz=2**np.int32(np.log2(np.array(sz)))
    re_sz=np.int32(sz[:-1]*re_sz)
    re_sz=(re_sz[1],re_sz[0])
    crp=[0,sz[0],0,sz[1]]
    
    frame=vol[:,:,0]
    im=cv2.resize(frame[crp[0]:crp[1],crp[2]:crp[3]],re_sz)
    #im=np.uint8(np.mean(bgr,2))
    im_pyr=ppt.SCFpyr(im,num_levs,1)
    x_lo=im_pyr.pyr[x_ind].copy()
    x_hi=x_lo.copy()
    x_prev=x_lo.copy()
    y_lo=im_pyr.pyr[y_ind].copy()
    y_hi=y_lo.copy()
    y_prev=y_lo.copy()
    #phase_x=np.zeros((np.int(crp[0]/(2**inference_lev)),np.int(crp[1]/(2**inference_lev)),num_frames))
    phase_x=np.zeros(np.int32((re_sz[1]/(2**inference_lev),re_sz[0]/(2**inference_lev),num_frames)))
    phase_y=phase_x.copy()
    [lo_a,lo_b]=signal.butter(1,fl)
    [hi_a,hi_b]=signal.butter(1,fh)
    for i in range(1,num_frames):
        if i%20==0:
            print('frame = '+str(i))
        frame=vol[:,:,i]
        im=cv2.resize(frame[crp[0]:crp[1],crp[2]:crp[3]],re_sz)
        #im=np.uint8(np.mean(bgr,2))
        ims('bgr',im)
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


def complex_plt(pts):
    out=np.zeros((255,len(pts),3))
    low=np.min((np.min(np.real(pts)),np.min(np.imag(pts))))
    high=np.max((np.max(np.real(pts)),np.max(np.imag(pts))))
    x=(np.real(pts)-low)/(high-low)
    y=(np.imag(pts)-low)/(high-low)
    for i,re in enumerate(x):
        im=y[i]
        out[np.int32(255*(1-re)):,i,1]=1
        out[np.int32(255*(1-im)):,i,2]=1
    return out


def lorenz(ends,center,n_out,s):
    phi=np.arange(ends[0],ends[1],(ends[1]-ends[0])/n_out)-center
    #lo=(1/(np.pi*s))*
    lo=((s**2)/(phi**2+s**2))
    #lo/=np.max(lo)
    return lo

def lorenz_cost(a,w,s):
    l=lorenz((0,len(a)),w,len(a),s)
    return np.sum(np.square(a-l))


def quick_params(a,w,eps=.001,mag=800,mx=800):
    s=1
    phi=np.arange(0,len(a))
    phi_hat=phi-w
    alpher=s/(phi_hat**2+s**2)
    dw=-(alpher**2)*(2*w-2*phi)
    #dw/=np.sqrt(np.sum(dw**2))
    ds=2*alpher*(1-s*alpher)
    #ds/=np.sqrt(np.sum(ds**2))
    step=(np.mean(dw*a),np.mean(ds*a))
    tot=1
    #steps=[np.sqrt(np.sum(np.square(step)))]
    while np.sqrt(step[0]**2+step[1]**2)>eps and tot<mx:
        tot+=1
        w+=step[0]*mag
        s+=step[1]*mag
        phi_hat=phi-w
        alpher=s/(phi_hat**2+s**2)
        dw=-(alpher**2)*(2*w-2*phi)
        ds=2*alpher*(1-s*alpher)
        step=(np.mean(dw*a),np.mean(ds*a))
        #steps.append(np.sqrt(np.sum(np.square(step))))
        #ims('p',plotpts(steps))
        #print (step)
    #print (tot)
    return (w,s)   

def lorenz_params(a,w,step=[.1,.1]):
    s=1
    done=0
    ct=0
    while done<2:
        done=0
        c=lorenz_cost(a,w,s)
        c2=lorenz_cost(a,w+step[0],s)
        ct+=2
        if (c2<c):
            while(c2<c):
                c=c2
                w=w+step[0]
                c2=lorenz_cost(a,w+step[0],s)
                ct+=1
        else:
            c2=lorenz_cost(a,np.max((w-step[0],0)),s)
            ct+=1
            if (c2<c):
                while(c2<c):
                    c=c2
                    w=np.max((w-step[0],0))
                    c2=lorenz_cost(a,np.max((w-step[0],0)),s)
                    ct+=1
            else:
                done+=1
        c2=lorenz_cost(a,w,s+step[1])
        ct+=1
        if (c2<c):      
            while(c2<c):
                c=c2
                s=s+step[1]                  
                c2=lorenz_cost(a,w,s+step[1])
                ct+=1
        else:                       
            c2=lorenz_cost(a,w,s-step[1])
            ct+=1
            if (c2<c):
                while(c2<c):
                    c=c2  
                    s=s-step[1]                  
                    c2=lorenz_cost(a,w,s-step[1])
                    ct+=1
            else:      
                done+=1
    return (w,s)

def tensor_params(a,ww):
    #np.arange(len(a))
    ap=a.reshape(-1,a.shape[2])
    arg=np.arange(np.product(ww.shape))
    s=np.ones(arg.shape)
    w=ww.reshape(-1,1).copy()
    

def get_sig(fp,loc):
    pt_fr=fp[loc[0],loc[1],:]
    a=np.abs(pt_fr[:len(pt_fr)//2])
    ma=np.max(a)
    a/=ma
    w=np.argmax(a)
    w,s=quick_params(a,w)
    phase=np.exp(1j*np.imag(np.log(pt_fr)))
    out_sig=ma*np.fft.ifft(phase*lorenz((0,fp.shape[2]),w,fp.shape[2],s))
    return (s,w,out_sig)

def get_sig_amps(amps):
    a=amps[:len(amps)//2]
    ma=np.max(a)
    a/=ma
    w=np.argmax(a)
    w,s=quick_params(a,w)
    

def sim_gt(im,u,hz,ori,s,ends,j,t,power,fps,full_size=False,out=None):
    bgr=cv2.resize(im,(ori.shape[1],ori.shape[0]))
    mode_clp=u[ends[0]:ends[1],ends[2]:ends[3],j]
    w_clp=hz[ends[0]:ends[1],ends[2]:ends[3]]
    s_clp=s[ends[0]:ends[1],ends[2]:ends[3]]
    bgr_clp=bgr[ends[0]:ends[1],ends[2]:ends[3],:]
    ori_clp=ori[ends[0]:ends[1],ends[2]:ends[3],:]
    for i in range(t):
        state=np.exp(2j*np.pi*i*(w_clp/fps)-((s_clp-.13)/27.4)*(i/800))
        #print(state.shape)
        arg=power*mode_clp*state
        ims('x',vh(arg))
        disp=ori_clp*np.real(arg).reshape(arg.shape[0],arg.shape[1],1)
        if full_size:
            p=push_pull(im,cv2.resize(disp,(im.shape[1],im.shape[0])))
        else:
            p=push_pull(bgr_clp,disp)
        ims('p',p)
        if out is not None:
            out.write(np.uint8(p))
def sim_freqs(im,x_du,y_du,freqs,s,ends,js,t,powers,fps,full_size=False,out=None):
    bgr=cv2.resize(im,(x_du.shape[1],x_du.shape[0]))
    mode_clp=np.zeros((x_du.shape[0],x_du.shape[1],2,len(powers)),dtype=np.complex128)
    for i,j in enumerate(js):
        mode_clp[:,:,:,i]=np.dstack((y_du[ends[0]:ends[1],ends[2]:ends[3],j],
                        x_du[ends[0]:ends[1],ends[2]:ends[3],j]))
        mode_clp[:,:,:,i]/=np.max(np.abs(mode_clp[:,:,:,i]))
    power=np.array(powers).reshape(1,1,1,len(powers))
    hz=np.array(freqs).reshape(1,1,1,len(freqs))
    mode_clp/=np.max(np.abs(mode_clp))
    s=np.array(s).reshape(1,1,1,len(s))
    #shp=mode_clp.shape
    bgr_clp=bgr[ends[0]:ends[1],ends[2]:ends[3],:]
    for i in range(t):
        state=np.exp(2j*np.pi*i*(hz/fps)-((s-.13)/27.4)*(i/fps))
        #print(state)
        #print(state.shape)
        arg=np.sum(power*mode_clp*state,3)
        ims('x',vh(arg[:,:,0]))
        ims('y',vh(arg[:,:,1]))
        #disp=np.real(arg).reshape(arg.shape[0],arg.shape[1],1)
        disp=np.real(arg)
        if full_size:
            p=push_pull(im,cv2.resize(disp,(im.shape[1],im.shape[0])))
        else:
            p=push_pull(bgr_clp,disp)
        ims('p',p)
        if out is not None:
            #print (p.shape)
            out.write(np.uint8(p))

def sim_sigs(im,x_du,y_du,ends,js,powers,sigs,full_size=False,out=None):
    bgr=cv2.resize(im,(x_du.shape[1],x_du.shape[0]))
    mode_clp=np.zeros((x_du.shape[0],x_du.shape[1],2,len(powers)),dtype=np.complex128)
    for i,j in enumerate(js):
        mode_clp[:,:,:,i]=np.dstack((y_du[ends[0]:ends[1],ends[2]:ends[3],j],
                                     x_du[ends[0]:ends[1],ends[2]:ends[3],j]))
        mode_clp[:,:,:,i]/=np.max(np.abs(mode_clp[:,:,:,i]))
    power=np.array(powers).reshape(1,1,1,len(powers))
    mode_clp/=np.max(np.abs(mode_clp))
    bgr_clp=bgr[ends[0]:ends[1],ends[2]:ends[3],:]
    if full_size:
        shp=im.shape
    else:
        shp=x_du.shape
    base_x=np.arange(shp[0]*shp[1]).reshape(shp[0],-1)
    base_y=base_x//shp[1]
    base_x=base_x%shp[1]
    base_flow=np.float32(np.dstack((base_x,base_y)))
    for i in range(sigs.shape[0]):
        state=sigs[i,::-1]
        arg=np.sum(power*mode_clp*state,3)
        ims('x',vh(arg[:,:,0]))
        ims('y',vh(arg[:,:,1]))
        #disp=np.real(arg).reshape(arg.shape[0],arg.shape[1],1)
        disp=np.real(arg)
        if full_size:
            p=push_resc_pull(im,cv2.resize(disp,(im.shape[1],im.shape[0])),base_flow)
        else:
            p=push_resc_pull(bgr_clp,disp,base_flow)
        if i%100==0:
            print(i)
        ims('p',p)
        if out is not None:
            #print (p.shape)
            out.write(np.uint8(p))

def sim_iter_freqs(im,x_du,y_du,freqs,s,ends,js,t,powers,fps,full_size=False,out=None,r1=.95):
    bgr=cv2.resize(im,(x_du.shape[1],x_du.shape[0]))
    mode_clp=np.zeros((x_du.shape[0],x_du.shape[1],2,len(powers)),dtype=np.complex128)
    for i,j in enumerate(js):
        mode_clp[:,:,:,i]=np.dstack((y_du[ends[0]:ends[1],ends[2]:ends[3],j],
                                     x_du[ends[0]:ends[1],ends[2]:ends[3],j]))
        mode_clp[:,:,:,i]/=np.max(np.abs(mode_clp[:,:,:,i]))
    power=np.array(powers).reshape(1,1,1,len(powers))
    hz=np.array(freqs).reshape(1,1,1,len(freqs))
    mode_clp/=np.max(np.abs(mode_clp))
    #s=np.array(s).reshape(1,1,1,len(s))
    #shp=mode_clp.shape
    bgr_clp=bgr[ends[0]:ends[1],ends[2]:ends[3],:]
    state=np.zeros((1,1,len(powers)),dtype=np.complex128)
    fs=np.random.randn(len(powers))
    max_phase_diff=.05/np.max(freqs)
    tot_time=0
    ps=[]
    qs=[]
    rs=[]
    for i in range(t):
        if i%100==0:
            print(i)
        while tot_time<i*2*np.pi/fps:
            fs=r1*fs+(1-r1)*np.random.randn(len(powers))
            for j,f in enumerate(fs):
                sj=state[0,0,j]
                next=iter(max_phase_diff,freqs[j],1/s,1/powers[j],f,(np.real(sj),np.imag(sj)*freqs[j]))
                state[0,0,j]=n=next[0]+1j*next[1]/freqs[j]
                #ps.append(next[0])
                #qs.append(next[1]/freqs[j])
                #if i>0:
                #rs.append(ps[i-1]-ps[i])
                #ims('me',plotpts(ps+qs))
            tot_time+=max_phase_diff
        #state=np.exp(2j*np.pi*i*(hz/fps)-((s-.13)/27.4)*(i/fps))
        #print(state)
        #print(state.shape)
        arg=np.sum(mode_clp*state,3)
        
        ims('x',vh(arg[:,:,0]))
        ims('y',vh(arg[:,:,1]))
        #disp=np.real(arg).reshape(arg.shape[0],arg.shape[1],1)
        disp=np.real(arg)
        if full_size:
            p=push_pull(im,cv2.resize(disp,(im.shape[1],im.shape[0])))
        else:
            p=push_pull(bgr_clp,disp)
        ims('p',p)
        if out is not None:
            #print (p.shape)
            out.write(np.uint8(p))

def sim_xy(im,x_du,y_du,hz,s,ends,j,t,power,fps,interpolation='push_pull',full_size=False,out=None):
    bgr=cv2.resize(im,(x_du.shape[1],x_du.shape[0]))
    mode_clp=np.dstack((y_du[ends[0]:ends[1],ends[2]:ends[3],j],
                        x_du[ends[0]:ends[1],ends[2]:ends[3],j]))
    mode_clp/=np.max(np.abs(mode_clp))
    shp=mode_clp.shape
    bgr_clp=bgr[ends[0]:ends[1],ends[2]:ends[3],:]
    for i in range(t):
        state=np.exp(2j*np.pi*i*(hz/fps)-((s-.13)/27.4)*(i/fps))
        #print(state)
        #print(state.shape)
        arg=power*mode_clp*state
        ims('x',vh(arg[:,:,0]))
        ims('y',vh(arg[:,:,1]))
        #disp=np.real(arg).reshape(arg.shape[0],arg.shape[1],1)
        disp=np.real(arg)
        if full_size:
            if interpolation=='push_pull':
                p=push_pull(im,cv2.resize(disp,(im.shape[1],im.shape[0])))
            elif interpolation=='push_resc_pull':
                p=push_resc_pull(im,disp)
            elif interpolation=='pull':
                p=pull(im,cv2.resize(disp,(im.shape[1],im.shape[0])))
            elif interpolation=='push':
                p=push(im,cv2.resize(disp,(im.shape[1],im.shape[0])))
            elif interpolation=='push_sparse':
                p=push_sparse(im,cv2.resize(disp,(im.shape[1],im.shape[0])))
            if i%10==0:
                print (i)
        else:
            p=push_pull(bgr_clp,disp)
        ims('p',p)
        if out is not None:
            #print (p.shape)
            out.write(np.uint8(p))




def load_gt(name,rd,im):
     gt=np.load('/volumes/LaCie/gt/'+name+'_gt/round_'+str(rd)+'/'+name+'_'+str(rd)+'answers.npz')
     im=cv2.imread('/Volumes/LaCie/gt/'+name+'_gt/round_'+str(rd)+'/in_'+str(im)+'.png')
     return (im,gt['u'],gt['hz'],gt['ori'],gt['s'])


def mode_from_html(checkpts_dir,chkpt,fixins,epoch):
    pref=checkpts_dir+'/'+chkpt+'/web/images/epoch'+epoch+'_'+fixins[0]+'_'
    prefA=checkpts_dir+'/'+chkpt+'/web/images/epoch'+epoch+'_real_'
    suf='_'+fixins[1]+'.png'
    sufA='_encoded.png'
    response=requests.get(prefA+'A'+sufA)
    im=np.asarray(Image.open(BytesIO(response.content)))
    response=requests.get(pref+'w'+suf)
    w=np.asarray(Image.open(BytesIO(response.content))) 
    w=(w[:,:,0]-127)/32
    response=requests.get(pref+'s'+suf)
    s=np.asarray(Image.open(BytesIO(response.content))) 
    s=(s[:,:,0]-127)/4
    response=requests.get(pref+'orix'+suf)
    orix=np.asarray(Image.open(BytesIO(response.content))) 
    orix=(np.float32(orix[:,:,0])-127)/128
    response=requests.get(pref+'oriy'+suf)
    oriy=np.asarray(Image.open(BytesIO(response.content)))        
    oriy=(np.float32(oriy[:,:,0])-127)/128
    ori=np.dstack((orix,oriy))
    ori/=np.sqrt(np.sum(np.square(ori),2)).reshape(ori.shape[0],-1,1)
    response=requests.get(pref+'real'+suf)
    real=np.asarray(Image.open(BytesIO(response.content)))        
    real=(np.float32(real[:,:,0])-127)/128    
    response=requests.get(pref+'imag'+suf)
    imag=np.asarray(Image.open(BytesIO(response.content)))        
    imag=(np.float32(imag[:,:,0])-127)/128
    return (im,w,s,ori,real+1j*imag)

def nofreq_html(checkpts_dir,chkpt,fixins,epoch,freq,t,power):
    pref=checkpts_dir+'/'+chkpt+'/web/images/epoch'+epoch+'_'+fixins[0]+'_'
    prefA=checkpts_dir+'/'+chkpt+'/web/images/epoch'+epoch+'_real_'
    suf='_'+fixins[1]+'.png'
    sufA='_encoded.png'
    response=requests.get(prefA+'A'+sufA)
    im=np.asarray(Image.open(BytesIO(response.content)))
    response=requests.get(pref+'xre'+suf)
    xre=np.float32(Image.open(BytesIO(response.content))) 
    xre=(xre[:,:,0]-127)/128
    response=requests.get(pref+'xre'+suf)
    xim=np.float32(Image.open(BytesIO(response.content))) 
    xim=(xim[:,:,0]-127)/128
    response=requests.get(pref+'yre'+suf)
    yre=np.float32(Image.open(BytesIO(response.content))) 
    yre=(yre[:,:,0]-127)/128
    response=requests.get(pref+'yim'+suf)
    yim=np.asarray(Image.open(BytesIO(response.content))) 
    yim=(np.float32(yim[:,:,0])-127)/128
    ends=[0,im.shape[0]-1,0,im.shape[1]-1]
    xmo=(xre+1j*xim).reshape(im.shape[0],-1,1)
    ymo=(yre+1j*yim).reshape(im.shape[0],-1,1)
    fps=1
    sim_xy(im,ymo,xmo,freq,.13,ends,0,t,power,fps)
    return (im,ymo,xmo)

def nofreq_test(test_dir,input_ind,test_ind,freq,t,power):
    #pref=checkpts_dir+'/'+chkpt+'/web/images/epoch'+epoch+'_'+fixins[0]+'_'
    #prefA=checkpts_dir+'/'+chkpt+'/web/images/epoch'+epoch+'_real_'
    #suf='_'+fixins[1]+'.png'
    #sufA='_encoded.png'
    #response=requests.get(prefA+'A'+sufA)
    pref=test_dir+'/input_%3.3d'%input_ind
    #address=pref+'_input.png'
    response=requests.get(pref+'_input.png')
    im=np.asarray(Image.open(BytesIO(response.content)))
    if test_ind>0:
        pref+='_random_sample'
        suf='_%2.2d.png'%test_ind
    else:
        pref+='_fake_B'
        suf='.png'
    #print(pref)
    #print(suf)
    response=requests.get(pref+'_xr'+suf)
    xre=np.float32(Image.open(BytesIO(response.content))) 
    xre=(xre[:,:,0]-127)/128
    response=requests.get(pref+'_xi'+suf)
    xim=np.float32(Image.open(BytesIO(response.content))) 
    xim=(xim[:,:,0]-127)/128
    response=requests.get(pref+'_yr'+suf)
    yre=np.float32(Image.open(BytesIO(response.content))) 
    yre=(yre[:,:,0]-127)/128
    response=requests.get(pref+'_yi'+suf)
    yim=np.asarray(Image.open(BytesIO(response.content))) 
    yim=(np.float32(yim[:,:,0])-127)/128
    ends=[0,im.shape[0]-1,0,im.shape[1]-1]
    xmo=(xre+1j*xim).reshape(im.shape[0],-1,1)
    ymo=(yre+1j*yim).reshape(im.shape[0],-1,1)
    fps=1
    sim_xy(im,ymo,xmo,freq,.13,ends,0,t,power,fps)
    return (im,ymo,xmo)

def sim_html(checkpoints_dir,chkpt,fixins,epoch,power,fps,t):
    im,w,s,ori,u=mode_from_html(checkpoints_dir,chkpt,fixins,epoch)
    ends=[0,im.shape[0]-1,0,im.shape[1]-1]
    sim_gt(im,u.reshape(u.shape[0],-1,1),w,ori,s,ends,0,t,power,fps)

def distort(frame,mx,my,phase,amp):
    arg=np.exp(1j*phase)
    mode=amp*np.dstack((np.real(my*arg),np.real(mx*arg)))
    return push_pull(frame,mode)

#def sim_single_html_w(checkpoints_dir,chkptfixins,epoch,power,fps,t):


#checkpoints_dir='http://gpgpu.cs-i.brandeis.edu/solomon/im2im/checkpoints/npz_dataset'

#In [525]: chkpt='npz_dataset_osc_25_08_2018_14'

#In [526]: fixins=['real','encoded']

#In [527]: sim_html(checkpoints_dir,chkpt,fixins,24,10,20,60)
