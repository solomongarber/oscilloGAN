import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html


# options
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1   # test code only supports batchSize=1
opt.serial_batches = True  # no shuffle

# create dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

# test stage
for i, data in enumerate(islice(dataset, opt.how_many)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.how_many))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        #print (fake_B.shape)
        real_B_xr=real_B[:,0,:,:]
        real_B_xi=real_B[:,1,:,:]
        real_B_yr=real_B[:,2,:,:]
        real_B_yi=real_B[:,3,:,:]
        fake_B_xr=fake_B[:,0,:,:]
        fake_B_xi=fake_B[:,1,:,:]
        fake_B_yr=fake_B[:,2,:,:]
        fake_B_yi=fake_B[:,3,:,:]
        if nn == 0:
            images = [real_A, real_B_xr, real_B_xi, real_B_yr, real_B_yi, fake_B_xr, fake_B_xi, fake_B_yr, fake_B_yi]
            names = ['input', 'real_B_xr', 'real_B_xi', 'real_B_yr', 'real_B_yi', 'fake_B_xr', 'fake_B_xi', 'fake_B_yr', 'fake_B_yi']
        else:
            images+=[fake_B_xr, fake_B_xi, fake_B_yr, fake_B_yi]
            titles=['random_sample_%s_%2.2d' %(x,nn) for x in ['xr','xi','yr','yi']]
            names+=titles

    img_path = 'input_%3.3d' % i
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.fineSize)

webpage.save()
