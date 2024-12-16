import argparse
import os
from util import util
import torch
import models
import data

class BaseOptions():
    def __init__(self):
        self.initialized = False
    
    def initialize(self, parser):
        """Initialize options used during both training and test time."""
        # Basic options
        parser.add_argument('--dataroot', required=False, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')  # Modified default
        parser.add_argument('--crop_size', type=int, default=1024, help='then crop to this size')    # Modified default
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')     # Modified default
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')   # Modified default
        parser.add_argument('--nz', type=int, default=64, help='#latent vector')                     # Modified default
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        parser.add_argument('--name', type=str, default='color2manga_cycle_ganstft', help='name of the experiment')  # Modified default
        parser.add_argument('--preprocess', type=str, default='none', help='not implemented')         # Modified default
        parser.add_argument('--dataset_mode', type=str, default='aligned', help='aligned,single')
        parser.add_argument('--model', type=str, default='cycle_ganstft', help='chooses which model to use')
        parser.add_argument('--direction', type=str, default='BtoA', help='AtoB or BtoA')            # Modified default
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--local_rank', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default=self.model_global_path+'/ScreenStyle/color2manga/', help='models are saved here')  # Modified default
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset.')
        parser.add_argument('--no_flip', action='store_false', help='if specified, do not flip the images for data argumentation')  # Modified default

        # Model parameters
        parser.add_argument('--level', type=int, default=0, help='level to train')
        parser.add_argument('--num_Ds', type=int, default=2, help='number of Discriminators')
        parser.add_argument('--netD', type=str, default='basic_256_multi', help='selects model to use for netD')
        parser.add_argument('--netD2', type=str, default='basic_256_multi', help='selects model to use for netD2')
        parser.add_argument('--netG', type=str, default='unet_256', help='selects model to use for netG')
        parser.add_argument('--netC', type=str, default='unet_128', help='selects model to use for netC')
        parser.add_argument('--netE', type=str, default='conv_256', help='selects model to use for netE')
        parser.add_argument('--nef', type=int, default=48, help='# of encoder filters in the first conv layer')  # Modified default
        parser.add_argument('--ngf', type=int, default=48, help='# of gen filters in the last conv layer')       # Modified default
        parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in the first conv layer')  # Modified default
        parser.add_argument('--norm', type=str, default='layer', help='instance normalization or batch normalization')
        parser.add_argument('--upsample', type=str, default='bilinear', help='basic | bilinear')                  # Modified default
        parser.add_argument('--nl', type=str, default='prelu', help='non-linearity activation: relu | lrelu | elu')
        parser.add_argument('--no_encode', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--color2screen', action='store_true', help='continue training: load the latest model including RGB model')  # Modified default

        # Extra parameters
        parser.add_argument('--where_add', type=str, default='all', help='input|all|middle; where to add z in the network G')
        parser.add_argument('--conditional_D', action='store_true', help='if use conditional GAN for D')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--center_crop', action='store_true', help='if apply for center cropping for the test')  # Modified default
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')

        # Special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options (only once)."""
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        # Get the basic options
        opt, _ = parser.parse_known_args()

        # Modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # Parse again with new defaults

        # Modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # Save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options."""
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # Save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            try:
                util.mkdirs(expr_dir)
            except:
                pass
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, model_global_path):
        """Parse options, create checkpoints directory suffix, and set up gpu device."""
        self.model_global_path = model_global_path
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test
        

        # Process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # Set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt