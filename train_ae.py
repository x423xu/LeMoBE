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
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from SnowflakeNet.loss_functions import chamfer_3DDist 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Process

from NVAE import utils  
from configs import args as siren_args
from model_zoo.shapenet_model import ShapeNetAutoEncoder



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

def cleanup():
    dist.destroy_process_group()
def init_processes(rank, size, fn, args):
    args = args
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6061'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()

def get_args(siren_args, checkpoint):
    timestr = get_timestr()
    nvae_args = checkpoint['args']
    args1 = vars(siren_args)
    args2 = vars(nvae_args)
    merged_args = {**args2, **args1}
    args = argparse.Namespace(**merged_args)
    if not os.path.exists(args.save):
        os.makedirs(args.save,exist_ok=True)
    if not hasattr(args, 'ada_groups'):
        print('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        print('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        print('old model, no num_mixture_dec was found.')
        args.num_mixture_dec = 10

    if args.batch_size > 0:
        args.batch_size = args.batch_size
    # init path
    # clean empty path
    for dir in os.listdir(os.path.join(args.save, args.checkpoint)):
        if not os.listdir(os.path.join(args.save, args.checkpoint, dir)):
            os.rmdir(os.path.join(args.save, args.checkpoint, dir))
            print(f'remove empty dir {dir}')
    ckpt_path = os.path.join(args.save, args.checkpoint, timestr)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path,exist_ok=True)
    args.checkpoint_path = ckpt_path
    if args.wandb: 
        wandb.init(entity='xxy', project='siren_vae', dir=os.path.join(args.save, args.checkpoint))
    print(f'checkpoint path {ckpt_path}')
    print(args)
    return args

def init_model(args):
    arch_instance = utils.get_arch_cells(args.arch_instance)
    shapenet_nvae = ShapeNetAutoEncoder(args, None, arch_instance)
    return shapenet_nvae

def get_data(args):
    from SnowflakeNet.generation.utils.data import DataLoader, get_data_iterator
    from SnowflakeNet.generation.utils.dataset import ShapeNetCore
    transform = None
    train_dset = ShapeNetCore(
    path=args.data,
    cates=args.categories,
    split='train',
    scale_mode=args.scale_mode,
    transform=transform,
    )
    val_dset = ShapeNetCore(
        path=args.data,
        cates=args.categories,
        split='val',
        scale_mode=args.scale_mode,
        transform=transform,
    )
    train_iter = get_data_iterator(DataLoader(
        train_dset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    ))
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=args.num_workers)
    return train_iter, val_loader, len(train_dset)//args.batch_size, len(val_dset)
def train(args):
    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    shapenet_nvae = init_model(args)
    shapenet_nvae.train()
    shapenet_nvae.cuda()

    '''get optimizer and scheduler for vae'''
    vae_optimizer = torch.optim.Adamax(shapenet_nvae.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)
    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    grad_scalar = GradScaler(2**10) # use grad scalar for mixed precision training

    # get dataloader
    '''
    Here we have to be careful since for image data nvae uses value range from [-1,1].
    In point cloud data, the shape_bbox scale mode should be used.
    '''
    train_iter, val_loader, train_len, val_len = get_data(args)
    args.num_total_iter = train_len * args.epochs
    

    '''
    training loop
    '''
    alpha_i = utils.kl_balancer_coeff(num_scales=shapenet_nvae.num_latent_scales,
                                      groups_per_scale=shapenet_nvae.groups_per_scale, fun='square')
    chamfer_loss = chamfer_3DDist()
    global_step = 0
    nelbo = utils.AvgrageMeter()
    for epoch in range(args.epochs):
        for step, train_batch in enumerate(train_iter):
            x = train_batch['pointcloud'].cuda()
            with autocast(args.half):
                if epoch == 0 and step <500:
                    warmup = True
                else:
                    warmup = False
                logits, log_q, log_p, kl_all,kl_diag = shapenet_nvae(x)
                output = shapenet_nvae.decoder_output(logits.unsqueeze(-1))
                mu = output.dist.mu.squeeze(-1).transpose(1,2)
                sigma = output.dist.sigma.squeeze(-1)
                chamfer_d1, chamfer_d2, idx1, idx2 = chamfer_loss(x, mu)
                std_loss = 0.5* torch.log(sigma).mean()
                # recon_loss1 = torch.sum(chamfer_d1 ,dim = (1))
                # recon_loss2 = torch.sum(chamfer_d2 , dim = (1))
                # recon_loss = recon_loss1 + recon_loss2
                recon_loss = chamfer_d1.mean()+chamfer_d2.mean() + std_loss
                if warmup:
                    loss = recon_loss
                else:
                    kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
                                                args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)
                    balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)
                    nelbo_batch = recon_loss + balanced_kl.mean()
                    loss = nelbo_batch
                    norm_loss = shapenet_nvae.spectral_norm_parallel()
                    bn_loss = shapenet_nvae.batchnorm_loss()
                    # get spectral regularization coefficient (lambda)
                    if args.weight_decay_norm_anneal:
                        assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                        wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
                        wdn_coeff = np.exp(wdn_coeff)
                    else:
                        wdn_coeff = args.weight_decay_norm

                    loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff
                

            grad_scalar.scale(loss).backward()
            orig_grad_norm = clip_grad_norm_(shapenet_nvae.parameters(), args.max_grad_norm)
            utils.average_gradients(shapenet_nvae.parameters(), args.distributed)
            grad_scalar.step(vae_optimizer)
            grad_scalar.update()
            nelbo.update(loss.data, 1)
            global_step += 1
            if args.global_rank == 0:
                if warmup:
                    log = '[Train] Epoch {} | Iter {}/{} | Loss {:.6f} | d1 {:.6f} | d2 {:.6f}, | std_loss {:.6f}'.format(
                            epoch,step, args.num_total_iter,nelbo.avg, chamfer_d1.mean().cpu().item(), chamfer_d2.mean().cpu().item(), std_loss.mean().cpu().item())
                else:
                    log = '[Train] Epoch {} | Iter {}/{} | Loss {:.6f} | kld {:.6f} | d1 {:.6f} | d2 {:.6f} | std_loss {:.6f} | norm_loss {:.6f} | bn_loss {:.6f}'.format(
                            epoch,step, args.num_total_iter,nelbo.avg, balanced_kl.mean().detach().cpu().numpy(), chamfer_d1.mean().cpu().item(), chamfer_d2.mean().cpu().item(), std_loss.mean().cpu().item(), norm_loss.cpu().item(), bn_loss.cpu().item())
                print(log)
                if args.wandb:
                    wandb.log({'train_loss': nelbo.avg, 'd1': chamfer_d1.mean().cpu().item(), 'd2': chamfer_d2.mean().cpu().item()})
                    # wandb.log({'train_loss': nelbo.avg, 'kld': balanced_kl.mean().detach().cpu().numpy(), 'd1': chamfer_d1.mean().cpu().item(), 'd2': chamfer_d2.mean().cpu().item(), 'std_loss': std_loss.mean().cpu().item(), 'norm_loss': norm_loss.cpu().item(), 'bn_loss': bn_loss.cpu().item()})
                if step%100 == 0:
                    y_pred =mu.detach().cpu().numpy()
                    x_gt = x.cpu().numpy()
                    plot_3d(x_gt[0],y_pred[0], epoch, step, idx1[0].detach().cpu().numpy(), idx2[0].detach().cpu().numpy())

def plot_3d(x,y, e, step, idx1, idx2):
    y_ = y[idx1]
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    # ax.view_init(elev=20, azim=120)
    ax.scatter(x[:,0], x[:,1], x[:,2], label='gt', color='darkorange', alpha=0.2)
    ax.scatter(y_[:,0], y_[:,1], y_[:,2], label='pred', color='dodgerblue', alpha=0.2)
    plt.legend()
    for i in range(x.shape[0]):
        p1 = x[i]
        p2 = y[idx1[i]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray')
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(y_[:,0], y_[:,1], y_[:,2], label='pred', color='dodgerblue')
    plt.legend()
    plt.savefig('experiments/imgs/3de{}s{}.png'.format(e,step))
    plt.close()

if __name__ == '__main__':
    checkpoint = load_checkpoint(siren_args)
    args = get_args(siren_args, checkpoint)

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
                p = Process(target=init_processes, args=(global_rank, global_size, train, args))
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
    else:
        init_processes(0, 1, train, args)