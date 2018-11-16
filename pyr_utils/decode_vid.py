import numpy as np
import cv2
import pyrUtils3 as pu
import sys
import os


out_dir=sys.argv[1]
out_name=sys.argv[2]
out_gain=np.float32(sys.argv[3])
interp=sys.argv[4]


out_dir=out_dir+'/'+out_name
bg_dir=out_dir+'/bg'



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

gstr=str

out=cv2.VideoWriter(out_dir+'/encoded_clip_gain_'+interp+'_'+str(out_gain).replace('.','_') +'.mp4',fourcc,fps,(frame0.shape[1],frame0.shape[0]),True) 

pu.decode(frame0,in_sig[:,:,:],newmox.reshape(270,480,1),newmoy.reshape(270,480,1),out_gain,out, interp=interp)
out.release()

