def create_model(opt):
    model = None
    print('Loading model %s...' % opt.model)

    if opt.model == 'bicycle_gan':
        from .bicycle_gan_model import BiCycleGANModel
        model = BiCycleGANModel()
    elif opt.model == 'pix2pix':
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'osc':
        from .oscycle_gan_model import OsCycleGANModel
        model = OsCycleGANModel()
    elif opt.model=='bicondle_gan':
        from .bicondle_gan_model import BiCondleGANModel
        model = BiCondleGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
