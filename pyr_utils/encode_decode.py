import numpy as np
import cv2
import pyrUtils3 as pu
import sys
import os

cap=cv2.VideoCapture(sys.argv[1])
out_dir=sys.argv[2]
out_name=sys.argv[3]
out_gain=np.float32(sys.argv[4])

out_dir=out_dir+'/'+out_name
bg_dir=out_dir+'/bg'
os.system('mkdir '+out_dir)
os.system('mkdir '+bg_dir)


fps=cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES,fps*3)
ret,frame0=cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES,0)
num_frames=np.int32(fps*60)
fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
print (fps)









x_dp,y_dp=pu.phase_diffs(cap)
freq_y=np.fft.fft(y_dp,axis=2);freq_x=np.fft.fft(x_dp,axis=2)
print('fft done')
aft=np.sqrt(np.sum(np.sum(np.abs(freq_x)**2+np.abs(freq_y)**2,axis=0),axis=0))
args=np.argsort(aft[:400])
amask=np.sum(np.abs(freq_x[:,:,args[-5:]])+np.abs(freq_y[:,:,args[-5:]]),axis=2).reshape(freq_x.shape[0],-1,1)
aft2=np.sqrt(np.sum(np.sum(np.abs(freq_x*amask)**2+np.abs(freq_y*amask)**2,axis=0),axis=0))
print('amps done')
pu.ims('a',pu.plotpts(aft2))


whz=np.argmax(aft2[:400]);
w,s=pu.quick_params(aft2[:400]/np.max(aft2),whz);
hz=w*fps/800;
sz=s*fps/800

xmos=freq_x[:,:,whz].reshape(-1,freq_x.shape[1],1).copy()
ymos=freq_y[:,:,whz].reshape(-1,freq_x.shape[1],1).copy()

print('encoding')
cap.set(cv2.CAP_PROP_POS_FRAMES,0)


tap=(s-.13)/(27.4*800)

out=cv2.VideoWriter(out_dir+'/small_synth_'+out_name+'.mp4',fourcc,fps,(x_dp.shape[1],x_dp.shape[0]),True)
out_sig=pu.encode(cap,out,xmos,ymos,hz/fps,tap,x_dp.shape,gain=0.3,num_frames=num_frames)
pu.daw()
out.release()
print('saving')
ymo=ymos[:,:,0].copy()
xmo=xmos[:,:,0].copy()
lymo=np.log(ymo)
lxmo=np.log(xmo)
cv2.imwrite(bg_dir+'/bg.jpg',frame0)
cv2.imwrite(bg_dir+'/ylamp.jpg',pu.ate(np.real(lymo)))
cv2.imwrite(bg_dir+'/yphase.jpg',pu.ate(np.imag(lymo)))
cv2.imwrite(bg_dir+'/xlamp.jpg',pu.ate(np.real(lxmo)))
cv2.imwrite(bg_dir+'/xphase.jpg',pu.ate(np.imag(lxmo)))
params=np.zeros((2,2))
params[:,0]=(np.min(np.real(lymo)),np.max(np.real(lymo)))
params[:,1]=(np.min(np.real(lxmo)),np.max(np.real(lxmo)))
np.savez(bg_dir+'/outsig.npz',out_sig=np.complex64(out_sig),params=params,fps=fps)





#decoding

print('decoding')
fourcc=cv2.VideoWriter_fourcc('m','p','4','v')
frame0=cv2.imread(bg_dir+'/bg.jpg')
sig=np.load(bg_dir+'/outsig.npz')
params=sig['params']
in_sig=sig['out_sig']
fps=sig['fps']
newx=cv2.imread(bg_dir+'/xlamp.jpg')[:,:,0]
newx=newx*(params[1,1]-params[0,1])/255+params[0,1]
newxp=cv2.imread(bg_dir+'/xphase.jpg')[:,:,0]
newxp=2*(np.float64(newxp)-127.5)*np.pi/255 
newy=cv2.imread(bg_dir+'/ylamp.jpg')[:,:,0]
newy=newy*(params[1,0]-params[0,0])/255+params[0,0]
newyp=cv2.imread(bg_dir+'/yphase.jpg')[:,:,0]
newyp=2*(np.float64(newyp)-127.5)*np.pi/255
newmoy=np.exp(newy-1j*newyp)   
newmox=np.exp(newx-1j*newxp)




out=cv2.VideoWriter('/encoded_clip_gain_'+str(out_gain)+'.mp4',fourcc,fps,(frame0.shape[1],frame0.shape[0]),True) 

pu.decode(frame0,in_sig,newmox.reshape(270,480,1),newmoy.reshape(270,480,1),out_gain,out)
out.release()





