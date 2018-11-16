import numpy as np
import cv2
import pyrUtils3 as pu
import os

per_clump=2

for root, _, fnames in sorted(os.walk('/Volumes/LaCie/gt_sets/gt_single/')):
    roopat=root.split('/')
    if roopat[-1][:5]=='round':
        print (root)
        gt=np.load(root+'/'+roopat[-2][:-2]+root[-1]+'answers.npz')
        num_ims=gt['x_du'].shape[2]-1
        #ind1=np.random.randint(num_ims)
        #ind2=np.random.randint(num_ims-1)
        #if ind2>=ind1:
        #    ind2+=1
        inds=np.random.permutation(num_ims)
        done=0
        for ind in inds:
            if done<2:
                im=cv2.imread(root+'/in_'+str(ind)+'.png')
                im=cv2.resize(im,(gt['x_du'].shape[1],gt['x_du'].shape[0]),interpolation=cv2.INTER_AREA)
                x_du=gt['x_du'][:,:,ind]
                y_du=gt['y_du'][:,:,ind]
                pu.ims('im',im)
                pu.ims('y',pu.vh(x_du))
                pu.ims('x',pu.vh(y_du))
                pu.ims('a',(np.abs(x_du)+np.abs(y_du)).reshape(x_du.shape[0],x_du.shape[1],1)*im)
                print(ind)
                print ('y/n?')
                key=cv2.waitKey(1)&0xFF
                while key==0xFF:
                    key=cv2.waitKey(1)&0xFF
                    #print (key)
                if key==ord("y"):
                    done+=1
                    print ("yes")
                    out=np.zeros((x_du.shape[0],x_du.shape[1],4))
                    out[:,:,0]=np.real(x_du)
                    out[:,:,1]=np.imag(x_du)
                    out[:,:,2]=np.real(y_du)
                    out[:,:,3]=np.imag(y_du)
                    out/=np.max(np.sqrt(np.sum(np.square(out),2)))
                    hz=gt['hz']
                    s=gt['s']
                    print (np.max(out))
                    print (np.min(out))
                    outfile='/Volumes/LaCie/supervised_dataset/'+roopat[-2][:-2]+root[-1]+'_'+str(ind)+'.npz'
                    np.savez(outfile,frame=im,mode=out,hz=hz,s=s)
                else:
                    print (key)
