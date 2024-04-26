'''
We imitate the implementation of NVAE:
half precision
distributed training
'''
import os,sys
import torch
import argparse
import pprint
import numpy as np
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler
import wandb
import torch.distributed as dist

from configs import args as siren_args
from models.siren_nvae import MSiren
from NVAE import utils
from NVAE.model import AutoEncoder
from NVAE import datasets
from tqdm import tqdm


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
    
    return args,


def init_model(args, checkpoint, logging):
    # init nvae
    logging.info('loaded the model at epoch %d', checkpoint['epoch'])
    arch_instance = utils.get_arch_instance(args.arch_instance)
    nvae = AutoEncoder(args, None, arch_instance)
    nvae.load_state_dict(checkpoint['state_dict'], strict=False)
    nvae_optimizer = torch.optim.Adamax(nvae.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)
    nvae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        nvae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    grad_scalar = GradScaler(2**10)
    siren = MSiren(routing_mode=args.routing_mode)
    return nvae, siren, [nvae_optimizer, nvae_scheduler, grad_scalar]


def cleanup():
    dist.destroy_process_group()
def init_processes(rank, size, fn, args):
    nvae, siren, train_queue, valid_queue, args, device, train_stuff = args
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6061'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(nvae, siren, train_queue, valid_queue, args, device, train_stuff)
    cleanup()

def get_2d_mgrid(shape):
    pixel_coords = np.stack(np.mgrid[:shape[0], :shape[1]], axis=-1).astype(np.float32)

    # normalize pixel coords onto [-1, 1]
    pixel_coords[..., 0] = pixel_coords[..., 0] / max(shape[0] - 1, 1)
    pixel_coords[..., 1] = pixel_coords[..., 1] / max(shape[1] - 1, 1)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    # flatten 
    pixel_coords = torch.tensor(pixel_coords).view(-1, 2)

    return pixel_coords

def train(args, checkpoint):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logging = utils.Logger(args.global_rank, args.save)
    nvae, siren, [nvae_optimizer, nvae_scheduler, grad_scalar] = init_model(args, checkpoint, logging)
    siren_optimizer = torch.optim.Adam(siren.parameters(), lr=1e-4)
    siren_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(siren_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)

    train_queue, valid_queue, num_classes = datasets.get_loaders(args)

    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs

    if args.global_rank == 0:
        logging.info('param size nvae = %fM | siren = %fM ', utils.count_parameters_in_M(nvae), utils.count_parameters_in_M(siren))

    nvae.train()
    siren.train()
    nvae.cuda()
    siren.cuda()

    global_step=0

    alpha_i = utils.kl_balancer_coeff(num_scales=nvae.num_latent_scales,
                                      groups_per_scale=nvae.groups_per_scale, fun='square')
    coords = get_2d_mgrid((64, 64)).cuda()
    
    for epoch in range(args.epochs):
        for step, x in enumerate(train_queue):
            x = x[0] if len(x) > 1 else x
            x = x.cuda()
            with autocast(enabled=args.half):
                if not args.train_nvae:
                    nvae.eval()
                    with torch.no_grad():
                        logits, log_q, log_p, kl_all,kl_diag, latents = nvae(x, return_latent=True)
                        pred = nvae.decoder_output(logits)
                        y_save = pred.sample()
                    rgb, _ = siren(coords, logits, latents, return_routings=False)
                    mse_loss = torch.nn.functional.mse_loss(rgb, x)
                    loss = mse_loss
                else:
                    logits, log_q, log_p, kl_all,kl_diag, latents = nvae(x, return_latent=True)
                    with torch.no_grad():
                        pred = nvae.decoder_output(logits)
                        y_save = pred.sample()
                    rgb, _ = siren(coords, logits, latents, return_routings=False, return_latent=False)
                    kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
                                            args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)
                    mse_loss = torch.nn.functional.mse_loss(rgb, x)
                    balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)
                    nelbo_batch = mse_loss + balanced_kl
                    loss = torch.mean(nelbo_batch)
                    norm_loss = nvae.spectral_norm_parallel()
                    bn_loss = nvae.batchnorm_loss()
                    # get spectral regularization coefficient (lambda)
                    if args.weight_decay_norm_anneal:
                        assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                        wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
                        wdn_coeff = np.exp(wdn_coeff)
                    else:
                        wdn_coeff = args.weight_decay_norm

                    loss += 0.01*(norm_loss * wdn_coeff + bn_loss * wdn_coeff)
            
            grad_scalar.scale(loss).backward()
            if args.train_nvae:
                utils.average_gradients(nvae.parameters(), args.distributed)
            utils.average_gradients(siren.parameters(), args.distributed)
            if args.train_nvae:
                grad_scalar.step(nvae_optimizer)
            grad_scalar.step(siren_optimizer)
            grad_scalar.update()

def main():
    # some presetup in the main process
    checkpoint = load_checkpoint(siren_args)
    args, logging = get_args(siren_args, checkpoint)

    size = args.num_process_per_node
    if args.distributed:
        assert size > 1, 'distributed training requires at least 2 processes per node'
        processes = []
        try:
            for rank in range(size):
                args.local_rank = rank
                global_rank = rank + args.node_rank * args.num_process_per_node
                global_size = args.num_proc_node * args.num_process_per_node
                args.global_rank = global_rank
                print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
                p = Process(target=init_processes, args=(global_rank, global_size, train, (args, checkpoint)))
                p.start()
                processes.append(p)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            for p in processes:
                p.terminate()
        except Exception as e:
            print(e)
            for p in processes:
                p.terminate()
        finally:
            for p in processes:
                p.join()
        
main()