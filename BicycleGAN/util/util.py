from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import pickle


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8, ends=[-1,1]):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    #print (image_numpy.shape)
    if len(image_numpy.shape)==1:
        print('one')
        print(image_numpy)
        image_numpy=plotpts(image_numpy)
    elif len(image_numpy.shape)==3:
        if image_numpy.shape[2]==image_numpy.shape[1] and image_numpy.shape[1]==1:
            print('two')
            #print(image_numpy[:,0,0])
            print("two_two")
            #iprint(image_numpy)
            image_numpy=plotpts(image_numpy[:,0,0])
    if image_numpy.shape[0] == 1:
        #print('three')
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    elif len(image_numpy.shape)==2:
        image_numpy = np.dstack((image_numpy,image_numpy,image_numpy))
        image_numpy = np.transpose(image_numpy, (2, 0, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) - ends[0]) / (ends[1]-ends[0]) * 255.0
    #print (image_numpy.shape)
    return image_numpy.astype(imtype)

def plotpts(arr):
    #print('arr')
    #print(arr)
    z=np.zeros((3,256,256),dtype=np.uint8)
    for i in range(arr.shape[0]):
        z[:,np.int32((1-arr[i])*255):,i]=255
    return z

def tensor2vec(vector_tensor):
    numpy_vec = vector_tensor.data.cpu().numpy()
    if numpy_vec.ndim == 4:
        return numpy_vec[:, :, 0, 0]
    else:
        return numpy_vec


def pickle_load(file_name):
    data = None
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def pickle_save(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def interp_z(z0, z1, num_frames, interp_mode='linear'):
    zs = []
    if interp_mode == 'linear':
        for n in range(num_frames):
            ratio = n / float(num_frames - 1)
            z_t = (1 - ratio) * z0 + ratio * z1
            zs.append(z_t[np.newaxis, :])
        zs = np.concatenate(zs, axis=0).astype(np.float32)

    if interp_mode == 'slerp':
        z0_n = z0 / (np.linalg.norm(z0) + 1e-10)
        z1_n = z1 / (np.linalg.norm(z1) + 1e-10)
        omega = np.arccos(np.dot(z0_n, z1_n))
        sin_omega = np.sin(omega)
        if sin_omega < 1e-10 and sin_omega > -1e-10:
            zs = interp_z(z0, z1, num_frames, interp_mode='linear')
        else:
            for n in range(num_frames):
                ratio = n / float(num_frames - 1)
                z_t = np.sin((1 - ratio) * omega) / sin_omega * z0 + np.sin(ratio * omega) / sin_omega * z1
                zs.append(z_t[np.newaxis, :])
        zs = np.concatenate(zs, axis=0).astype(np.float32)

    return zs


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path, 'JPEG', quality=100)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    #print (paths)
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1)
                             ).repeat(1, in_feat.size()[1], 1, 1)
    return in_feat / (norm_factor + eps)


def cos_sim(in0, in1):
    in0_norm = normalize_tensor(in0)
    in1_norm = normalize_tensor(in1)
    return torch.mean(torch.sum(in0_norm * in1_norm, dim=1))
