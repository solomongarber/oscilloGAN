import torch
from .base_model import BaseModel
from . import networks


class OsCycleGANModel(BaseModel):
    def name(self):
        return 'OsCycleGANModel'

    def initialize(self, opt):
        if opt.isTrain:
            assert opt.batchSize % 2 == 0  # load two images at one time.

        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'z_L1', 'kl','G2_L1']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A_encoded', 'real_xre_encoded','real_xim_encoded','real_yre_encoded','real_yim_encoded','fake_xre_random','fake_xim_random','fake_yre_random','fake_yim_random' ,'fake_xre_encoded','fake_xim_encoded','fake_yre_encoded','fake_yim_encoded','fake_C_random','fake_C_encoded','real_C_encoded']
        #self.visual_names = ['real_w_encoded']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_E = opt.isTrain or not opt.no_encode
        use_mse = opt.isTrain and opt.use_mse
        self.use_G2=opt.input_nz>0
        self.conditional_E=self.use_G2
        use_vae = True
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, which_model_netG=opt.which_model_netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, where_add=self.opt.where_add, upsample=opt.upsample, split_outer=False)
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        use_sigmoid = opt.gan_mode == 'dcgan'
        if self.use_G2:
            self.model_names += ['G2']
            self.netG2=networks.define_E(opt.input_nc+opt.nz,opt.input_nz,opt.nef,which_model_netE=opt.which_model_netE,norm=opt.norm,nl=opt.nl,init_type=opt.init_type,gpu_ids=self.gpu_ids,vaeLike=False)
            #self.netG2=networks.define_G2(opt.nz, opt.input_nz, 3, opt.ndf)
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, which_model_netD=opt.which_model_netD, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, which_model_netD=opt.which_model_netD2, norm=opt.norm, nl=opt.nl,
                                           use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_E:
            self.model_names += ['E']
            self.netE = networks.define_E(opt.output_nc+opt.input_nz, opt.nz, opt.nef, which_model_netE=opt.which_model_netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, gpu_ids=self.gpu_ids, vaeLike=use_vae)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(mse_loss=not use_sigmoid).to(self.device)
            if use_mse:
                self.criterionL1 = torch.nn.MSELoss()
            else:
                self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)
            if self.use_G2:
                self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G2)
                
    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batchSize

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['Z'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batchSize, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batchSize, nz)
        return z.to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:  # use encoded z
                if self.conditional_E:
                    self.z_img=self.real_C_encoded.view(
                        self.real_C_encoded.size(0),self.real_C_encoded.size(1),1,1).expand(
                            self.real_C_encoded.size(0),
                            self.real_C_encoded.size(1),
                            self.real_B_encoded.size(2),
                            self.real_B_encoded.size(3))
                    self.x_and_z=torch.cat([self.real_B_encoded,self.z_img],1)
                    z0,_=self.netE(self.x_and_z)
                else:
                    z0, _ = self.netE(self.real_B)
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            self.fake_B = self.netG(self.real_A, z0)
            return self.real_A, self.fake_B, self.real_B

    def forward(self):
        # get real images
        half_size = self.opt.batchSize // 2
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded     = self.real_A[0:half_size]
        self.real_B_encoded     = self.real_B[0:half_size]
        self.real_C_encoded      = self.real_C[0:half_size]
        self.real_xre_encoded     = self.real_B_encoded[:,0,:,:]
        self.real_xim_encoded     = self.real_B_encoded[:,1,:,:]
        self.real_yre_encoded  = self.real_B_encoded[:,2,:,:]
        self.real_yim_encoded  = self.real_B_encoded[:,3,:,:]
        #self.real_real_encoded  = self.real_B_encoded[:,4,:,:]
        #self.real_imag_encoded  = self.real_B_encoded[:,5,:,:]
        self.real_B_random      = self.real_B[half_size:]
        
        # get encoded z
        if self.conditional_E:
            self.c_img=self.real_C_encoded.view(
                self.real_C_encoded.size(0),self.real_C_encoded.size(1),1,1).expand(
                    self.real_C_encoded.size(0),self.real_C_encoded.size(1),self.real_B_encoded.size(2),self.real_B_encoded.size(3))
            self.b_and_c=torch.cat([self.real_B_encoded,self.c_img],1)
            self.z_encoded, self.mu, self.logvar = self.encode(self.b_and_c)
        else:
            self.z_encoded,self.mu,self.logvar=self.encode(self.real_B_encoded)
        # get random z
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
        # generate fake_B_encoded
        self.fake_B_encoded     = self.netG(self.real_A_encoded, self.z_encoded)
        self.fake_xre_encoded     = self.fake_B_encoded[:,0,:,:]
        self.fake_xim_encoded     = self.fake_B_encoded[:,1,:,:]
        self.fake_yre_encoded  = self.fake_B_encoded[:,2,:,:]
        self.fake_yim_encoded  = self.fake_B_encoded[:,3,:,:]
        #self.fake_real_encoded  = self.fake_B_encoded[:,4,:,:]
        #self.fake_imag_encoded  = self.fake_B_encoded[:,5,:,:]
        # generate fake_B_random
        self.fake_B_random     = self.netG(self.real_A_encoded, self.z_random)
        if self.use_G2:
            self.z_encoded_slice=self.z_encoded.view(self.z_encoded.size(0),
                                                     self.z_encoded.size(1),1,1).expand(
                self.z_encoded.size(0),self.z_encoded.size(1),self.real_A_encoded.size(2),self.real_A_encoded.size(3))
            self.a_z_encoded=torch.cat([self.real_A_encoded,self.z_encoded_slice],1)
            self.fake_C_encoded=self.netG2(self.a_z_encoded)
            self.z_random_slice=self.z_random.view(self.z_random.size(0),
                                              self.z_random.size(1),1,1).expand(
                self.z_random.size(0),self.z_random.size(1),self.real_A_encoded.size(2),self.real_A_encoded.size(3))
            self.a_z_random=torch.cat([self.real_A_encoded,self.z_random_slice],1)
            self.fake_C_random    = self.netG2(self.a_z_random)
        self.fake_xre_random     = self.fake_B_random[:,0,:,:]
        self.fake_xim_random     = self.fake_B_random[:,1,:,:]
        self.fake_yre_random  = self.fake_B_random[:,2,:,:]
        self.fake_yim_random  = self.fake_B_random[:,3,:,:]
        #self.fake_real_random  = self.fake_B_random[:,4,:,:]
        #self.fake_imag_random  = self.fake_B_random[:,5,:,:]
        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A[half_size:], self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            self.real_data_encoded = self.real_B_encoded
            self.real_data_random = self.real_B_random

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            if self.use_G2:
                self.c_slice=self.fake_C_random.view(self.fake_C_random.size(0),
                                                self.fake_C_random.size(1),
                                                1,1).expand(self.fake_C_random.size(0),
                                                            self.fake_C_random.size(1),
                                                            self.fake_B_random.size(2),
                                                            self.fake_B_random.size(3))
                self.b_and_c=torch.cat([self.fake_B_random,self.c_slice],1)
                self.mu2,logvar2=self.netE(self.b_and_c)
            else:
                self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate

    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
            self.loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        #4, reconstruction |fake_z-real_z|
        if self.opt.lambda_G2>0.0:
            self.loss_G2_L1 = self.criterionL1(self.fake_C_encoded.view(self.fake_C_encoded.size(0),self.fake_C_encoded.size(1),1,1),
                                               self.real_C_encoded)*self.opt.lambda_G2
        else:
            self.loss_G2_L1 = 0.0
        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl + self.loss_G2_L1
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad(self.netD, True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(torch.abs(self.mu2- self.z_random)) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        if self.use_G2:
            self.optimizer_G2.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        if self.use_G2:
            self.optimizer_G2.step()
        self.optimizer_E.step()
        # update G only
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            if self.use_G2:
                self.optimizer_G2.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()
