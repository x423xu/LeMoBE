import argparse

parser = argparse.ArgumentParser('encoder decoder examiner')
# experimental results
parser.add_argument('--pretrained_nvae', type=str, default='/home/xxy/Documents/code/NVAE/pretrained_weights/celeba_64-20240403T002440Z-001/celeba_64/checkpoint.pt',
                    help='location of the checkpoint')
parser.add_argument('--save', type=str, default='experiments',
                    help='location of the checkpoint')

parser.add_argument('--checkpoint', type=str, default='checkpoints',help='location of the checkpoint')
parser.add_argument('--wandb', action='store_true', default=False, help='enabling this will enable wandb logging')

# data
parser.add_argument('--dataset', type=str, default='celeba', help='dataset to use')
parser.add_argument('--data', type=str, default='/home/xxy/Documents/data/celebA/celeba64_lmdb',
                    help='location of the data corpus')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers to use for data loading')


# DDP.
parser.add_argument('--distributed', action='store_true', default=False, help='enabling this will enable distributed training')
parser.add_argument('--local_rank', type=int, default=0,
                    help='rank of process')
parser.add_argument('--seed', type=int, default=1,
                    help='seed used for initialization')
parser.add_argument('--master_address', type=str, default='127.0.0.1',
                    help='address for master')
parser.add_argument('--num_process_per_node', type=int, default=4, help='number of processes per node')
parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')

# training
parser.add_argument('--half', action='store_true', default=False,help='enabling this will enable half precision training')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size used during evaluation. If set to zero, training batch size is used.')
parser.add_argument('--epochs', type=int, default=100,help='number of epochs')
parser.add_argument('--warmup_epochs', type=int, default=0, help='number of warmup epochs')
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')

# model vae
parser.add_argument('--model', type=str, default='msiren', choices=['msiren', 'rsiren'],
                    help='model to use')
parser.add_argument('--routing_mode', type=int, default=0, choices=[0, 1], help='routing mode: 0 to disable, 1 to enable')
parser.add_argument('--train_nvae', action='store_true', default=False, help='train nvae weights')
parser.add_argument('--num_channels_enc', type=int, default=6, help='number of channels in encoder')
parser.add_argument('--num_mixture_dec', type=int, default=1, help='number of mixtures in decoder')

# for model shapenet
parser.add_argument('--pc_encoder_dim', type=int, default=128, help='point cloud encoder dimension')
parser.add_argument('--num_points', type=int, default=2048, help='number of points in point cloud')
parser.add_argument('--scale_mode', type=str, default='shape_bbox', help='scale mode')
parser.add_argument('--categories', type=str, default=['chair'], help='categories to use')

args = parser.parse_args()