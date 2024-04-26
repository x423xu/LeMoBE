'''
Here we want to train nvae on:
1. CelebA dataset
2. shapenet
3. LSUN
'''
import sys,os 
sys.path.append('/home/xxy/Documents/code/LeMoBE/NVAE')
sys.path.append('/home/xxy/Documents/code/LeMoBE/SnowflakeNet')
import torch
import argparse
import pprint
import wandb

from NVAE import utils  
from configs import args as siren_args
from models.shapenet_model import ShapeNetAutoEncoder

def get_timestr():
    import pytz
    from datetime import datetime
    toronto_tz = pytz.timezone('America/Toronto')
    utc_now = datetime.utcnow()
    toronto_now = utc_now.astimezone(toronto_tz)
    timestr = toronto_now.strftime("%Y%m%d-%H%M%S")
    return timestr

def load_checkpoint(siren_args):
    checkpoint = torch.load(siren_args.pretrained_nvae, map_location='cpu')
    print(checkpoint.keys())
    return checkpoint

def get_args(siren_args, checkpoint):
    timestr = get_timestr()
    nvae_args = checkpoint['args']
    args1 = vars(siren_args)
    args2 = vars(nvae_args)
    merged_args = {**args2, **args1}
    args = argparse.Namespace(**merged_args)
    # get logging
    if not os.path.exists(args.save):
        os.makedirs(args.save,exist_ok=True)
    logging = utils.Logger(0, args.save)
    if not hasattr(args, 'ada_groups'):
        logging.info('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        logging.info('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        logging.info('old model, no num_mixture_dec was found.')
        args.num_mixture_dec = 10

    if args.batch_size > 0:
        args.batch_size = args.batch_size
    pprint.pprint(merged_args)
    # init path
    # clean empty path
    for dir in os.listdir(os.path.join(args.save, args.checkpoint)):
        if not os.listdir(os.path.join(args.save, args.checkpoint, dir)):
            os.rmdir(os.path.join(args.save, args.checkpoint, dir))
            logging.info(f'remove empty dir {dir}')
    ckpt_path = os.path.join(args.save, args.checkpoint, timestr)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path,exist_ok=True)
    args.checkpoint_path = ckpt_path
    if args.wandb: 
        wandb.init(entity='xxy', project='siren_vae', dir=os.path.join(args.save, args.checkpoint))
    logging.info(f'checkpoint path {ckpt_path}')
    
    return args, logging

def init_model(args, checkpoint, logging):
    logging.info('loaded the model at epoch %d', checkpoint['epoch'])
    arch_instance = utils.get_arch_cells(args.arch_instance)
    nvae = ShapeNetAutoEncoder(args, None, arch_instance)

if __name__ == '__main__':
    checkpoint = load_checkpoint(siren_args)
    args, logging = get_args(siren_args, checkpoint)
    init_model(args, checkpoint, logging)