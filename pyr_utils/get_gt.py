import numpy as np
import cv2
from pyrUtils3 import *


def get_udf(cap,ori,s_gt,hz_gt,fl=.6,fh=.5,sigma=2, k_sz=9,num_frames=800,re_sz=.5,lev=0):
    ret,frame=cap.read()
    print (ret)
    
    x_dp,y_dp=phase_diffs(cap,fl,fh,sigma,sigma,k_sz,k_sz,num_frames,lev,frame.shape,re_sz)
    u_dp=np.zeros(x_dp.shape)
    for y in range(x_dp.shape[0]):
        for x in range(y_dp.shape[1]):
            sig=np.stack((x_dp[y,x,:],y_dp[y,x,:]))
            u_dp[y,x,:]=np.dot(ori[y,x,:].reshape(1,2),sig)
    del x_dp
    del y_dp      
    freq_u=np.fft.fft(u_dp,axis=2)
    del u_dp
    u_df=np.zeros(freq_u.shape,dtype=np.complex64)
    factor=cap.get(cv2.CAP_PROP_FPS)/num_frames
    for y in range(freq_u.shape[0]):
        for x in range(freq_u.shape[1]):
            
            w=hz_gt[y,x]/factor
            s=s_gt[y,x]
            pt_fr=freq_u[y,x,:]
            a=np.abs(pt_fr[:len(pt_fr)//2])
            ma=np.max(a)
            a/=ma
            phase=np.exp(1j*np.imag(np.log(pt_fr)))
            out_sig=ma*np.fft.ifft(phase*lorenz((0,freq_u.shape[2]),w,freq_u.shape[2],s))
            u_df[y,x,:]=out_sig
    #skip_factor=np.int32(cap.get(cv2.CAP_PROP_FPS)/sample_hz)
    return (freq_u,u_df)


def get_single_gt(cap,fl=.6,fh=.5,sigma=2,k_sz=9,num_frames=800,re_sz=.5,lev=1):

    ret,frame=cap.read()
    print (ret)
    nyquist=num_frames//2
    x_dp,y_dp=phase_diffs(cap,
                          fl=fl,fh=fh,
                          sigma_time=sigma,sigma_freq=sigma,
                          k_size_x=k_sz,k_size_t=k_sz,
                          num_frames=num_frames,
                          inference_lev=lev,re_sz=re_sz)

    freq_x=np.fft.fft(x_dp,axis=2)
    freq_y=np.fft.fft(y_dp,axis=2)
    print('fft done')

    factor=cap.get(cv2.CAP_PROP_FPS)/num_frames
    aft=np.sqrt(np.sum(np.sum(np.abs(freq_x)**2+np.abs(freq_y)**2,axis=0),axis=0))
    args=np.argsort(aft[:nyquist])
    amask=np.sum(np.abs(freq_x[:,:,args[-5:]])+np.abs(freq_y[:,:,args[-5:]]),axis=2).reshape(freq_x.shape[0],-1,1)
    aft=np.sqrt(np.sum(np.sum(np.abs(freq_x*amask)**2+np.abs(freq_y*amask)**2,axis=0),axis=0))
    print('mask done')
    w=np.argmax(aft[:nyquist])
    w,s=quick_params(aft[:nyquist]/np.max(aft),w)
    response=lorenz((0,num_frames),w,num_frames,s)
    x_du=np.fft.ifft(freq_x*response,axis=2)
    y_du=np.fft.ifft(freq_y*response,axis=2)
    print('ifft done')
    #for y in range(freq_u.shape[0]):
    #    #ims('s',s_gt)
    #    #ims('w',w_gt)
    #    if y%40==0:
    #        print('running w,s for row '+str(y)+' of '+str(s_gt.shape[0]))
    ##        #print (y)
    #    for x in range(freq_u.shape[1]):
    #        (s,w,o)=get_sig(freq_u,(y,x))
    #        s_gt[y,x]=s
    #        hz_gt[y,x]=w*factor
    #        u_df[y,x,:]=o
    #skip_factor=np.int32(cap.get(cv2.CAP_PROP_FPS)/sample_hz)
    return (w*factor,s*factor,x_du,y_du)

    
def get_gt(cap,fl=.6,fh=.5,sigma=2,k_sz=9,num_frames=800,re_sz=.5,lev=1):


    #in_vid='/Volumes/G-DRIVE_mobile_SSD_R-Series/data/video_data/Ann_Arbor/00310.mp4'

    #cap=cv2.VideoCapture(in_vid)
    ret,frame=cap.read()
    print (ret)
    
    x_dp,y_dp=phase_diffs(cap,fl,fh,sigma,sigma,k_sz,k_sz,num_frames,lev,frame.shape,re_sz)
    #s_gt=np.zeros((x_dp.shape[0],x_dp.shape[1]))
    
    #x_df=np.complex128(x_dp*0)
    
    #w_gt=s_gt.copy()
    
    
    aco_gt=np.zeros((x_dp.shape[0],x_dp.shape[1],2,2),dtype=np.float32)
    ori_gt=np.zeros((x_dp.shape[0],x_dp.shape[1],2,2),dtype=np.float32)
    u_dp=np.zeros(x_dp.shape)
    for y in range(x_dp.shape[0]):
        for x in range(y_dp.shape[1]):
            sig=np.stack((x_dp[y,x,:],y_dp[y,x,:]))
            sig-=np.mean(sig,1).reshape(2,1)
            aco_gt[y,x,:,:]=np.dot(sig,sig.T)
            u,_,_=np.linalg.svd(aco_gt[y,x,:,:])
            ori_gt[y,x,:,:]=u
            u_dp[y,x,:]=np.dot(u[:,0].reshape(1,2),sig)
        if y%40==0:
            print('running ori for row '+str(y)+' of '+str(x_dp.shape[0]))
            #ims('i',vh(ori_gt[:,:,0,0]+1j*ori_gt[:,:,1,0]))

    del x_dp
    del y_dp
            
    #ori=ori_gt[:,:,:,0].copy()
    #abri=np.dstack((np.sum(np.abs(x_dp),2),np.sum(np.abs(y_dp),2)))
    #sigab=np.sign(np.sum(abri*ori,2))

    freq_u=np.fft.fft(u_dp,axis=2)
    print('fft done')
    del u_dp
    s_gt=np.zeros((freq_u.shape[0],freq_u.shape[1]))
    hz_gt=s_gt.copy()
    u_df=np.zeros(freq_u.shape,dtype=np.complex64)
    factor=cap.get(cv2.CAP_PROP_FPS)/num_frames
    for y in range(freq_u.shape[0]):
        #ims('s',s_gt)
        #ims('w',w_gt)
        if y%40==0:
            print('running w,s for row '+str(y)+' of '+str(s_gt.shape[0]))
            #print (y)
        for x in range(freq_u.shape[1]):
            (s,w,o)=get_sig(freq_u,(y,x))
            s_gt[y,x]=s
            hz_gt[y,x]=w*factor
            u_df[y,x,:]=o
    #skip_factor=np.int32(cap.get(cv2.CAP_PROP_FPS)/sample_hz)
    return (hz_gt,ori_gt[:,:,:,0],s_gt,u_df)
#amap=np.abs(u_df[:,:,0])**2/s_gt
#hist, xbins, ybins = np.histogram2d(w_gt.ravel(),phase.ravel(),[400,256],[[0,100],[-np.pi,np.pi]],weights=amap.ravel())

#phase=np.imag(np.log(u_df[:,:,i]))
#      ...:     state1=np.abs(complex_scatter((ww+1j*phase).reshape(-1),1,acc=True,amps=amap.reshape(-1),align_axes=True))
#      ...:     f1=np.fft.fft(state1,axis=1)
#      ...:     f1[f1==0]=1
#      ...:     fd=f0*np.abs(f1)/(f1*np.abs(f0))
#      ...:     fdd=np.fft.fft(fd,axis=1)
#      ...:     fdd*=np.abs(f0[:,0]).reshape(256,1)
#      ...:     a=np.argmax(np.abs(fdd),axis=1)
#      ...:     maps2.append(a)
#      ...:     ims('i',vh(u_df[:,:,i]*np.exp((-2j*np.pi*a/255)[np.uint8(w_inds)])))
#      ...:     ims('j',vh(u_df[:,:,0]))
#     ...:     ims('p',plotpts(np.argmax(np.abs(fdd),axis=1)))

