import numpy as np
import cv2
from get_gt import get_single_gt as get_gt
import os
import sys

in_file=open(sys.argv[1]).read()
in_vids=in_file.split('\n')
num_frames=800
buff=1.5
sample_hz=1
out_dir='/Volumes/LaCie/gt_sets/gt_single'
out_dir2='/Volumes/LaCie/gt_sets/gt_single_redo'
for in_vid in in_vids:
    cap=cv2.VideoCapture(in_vid)
    ret,frame=cap.read()
    if ret:
        fps=cap.get(cv2.CAP_PROP_FPS)
        skip_frames=np.int32(fps*buff)
        print (in_vid)
        #print(-2*skip_frames)
        #print (np.int32(cap.get(cv2.CAP_PROP_FRAME_COUNT)-2*skip_frames))
        #print(num_frames)
        #print ((np.int32(cap.get(cv2.CAP_PROP_FRAME_COUNT)-2*skip_frames))//num_frames)
        num_rounds=np.int32(cap.get(cv2.CAP_PROP_FRAME_COUNT)-2*skip_frames)//num_frames
        print (num_rounds)
        leftovers=np.int32(cap.get(cv2.CAP_PROP_FRAME_COUNT)-num_frames*num_rounds)//2
        print (leftovers)
        print (cap.get(cv2.CAP_PROP_FRAME_COUNT))
        in_stem=in_vid.split('/')[-1][:-4]
        od=out_dir+'/'+in_stem+'_gt'
        od2=out_dir2+'/'+in_stem+'_gt'
        #os.system('mkdir '+od)
        for rd in range(np.min((4,num_rounds))):
            print ('round '+str(rd))
            start_time=leftovers+rd*num_frames
            end_time=leftovers+(rd+1)*(num_frames+1)
            cap.set(cv2.CAP_PROP_POS_FRAMES,end_time)
            ret,frame=cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES,start_time)
                infile=od+'/round_'+str(rd)+'/'+in_stem+'_'+str(rd)+'answers.npz'
                gt=np.load(infile)
                hz=gt['hz']
                factor=num_frames/cap.get(cv2.CAP_PROP_FPS)
                nq=num_frames//2
                print ('hz '+str(hz))
                if hz*factor>nq:
                    os.system('mkdir '+od2)
                    print ('redoing '+infile)
                    hz,s,x_du,y_du=get_gt(cap)
                    print ('new hz '+str(hz))
                    sample_factor=np.int32(cap.get(cv2.CAP_PROP_FPS)/sample_hz)
                    x_du=x_du[:,:,::sample_factor]
                    y_du=y_du[:,:,::sample_factor]
                    odd=od+'/round_'+str(rd)
                    odd2=od2+'/round_'+str(rd)
                    os.system('mkdir '+odd2)
                    np.savez(odd+'/'+in_stem+'_'+str(rd)+'answers.npz',
                         hz=hz,s=s,x_du=x_du,y_du=y_du)
                    np.savez(odd2+'/'+in_stem+'_'+str(rd)+'answers.npz',
                         hz=hz,s=s,x_du=x_du,y_du=y_du)
                    for i in range(num_frames//sample_factor):
                        cap.set(cv2.CAP_PROP_POS_FRAMES,np.int32(start_time+i*sample_factor))
                        ret,frame=cap.read()
                        cv2.imwrite(odd+'/in_'+str(i)+'.png',frame)
                        cv2.imwrite(odd2+'/in_'+str(i)+'.png',frame)
                
