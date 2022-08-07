import argparse

# my files
from MISU.constants import LOG_PATH

parser = argparse.ArgumentParser()

dataset_choices = ['PCBA', 'MUV', 'HIV', 'BACE', 'BBBP', 'Tox21', 'ToxCast', 'SIDER', 'ClinTox', 'PCQ']

# general / saving params #
##############################
parser.add_argument('--dataset', type=str, default='PCQ', choices=dataset_choices, help='the dataset name')
parser.add_argument('--gpus', type=str, default='0', help='gpus parameter used for pytorch_lightning')
parser.add_argument('--log_path', type=str, default=LOG_PATH, help='tensorboard log path')
parser.add_argument('--save_path', type=str, default='', help='checkpoints saving path')
parser.add_argument('--disable_logging', default=False, action='store_true', help='disables logging')
parser.add_argument('--random_split', default=False, action='store_true', help='changes the dataset to pyg from ogb')
parser.add_argument('--seed', type=int, default=42, help='seed for data splitting randomness')
parser.add_argument('--verbose', default=False, action='store_true', help='whether to print more information')
# training params #
##############################
parser.add_argument('--adamw', action='store_true', help='whether to use AdamW optimizer or Adam')
parser.add_argument('--batch_size', type=int, default=5120, help='the desired batch size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout for the classifier')
parser.add_argument('--hidden_channels', type=int, default=600, help='the dimension of embeddings of nodes and edges')
parser.add_argument('--fp_loss_factor', type=float, default=100.0, dest='fp_factor', help='the weight of the fp loss')
parser.add_argument('--kl_factor', type=float, default=0.1, help='the weight of the kl loss')
parser.add_argument('--jt_kl_factor', type=float, default=1e-5, dest='fg_kl_factor',
                    help='the weight of the kl loss for jtvae')
parser.add_argument('--load_ckpt', action='store_true', help='whether to use checkpoints for previous training')
parser.add_argument('--ckpt_path', type=str, default='')
parser.add_argument('--lr', type=float, default=3e-4, dest='learning_rate', help='learning rate')
parser.add_argument('--lr_scheduler', action='store_true', help='whether to use learning rate scheduler')
parser.add_argument('--max_epochs', type=int, default=80, help='number of maximum epochs to run')
parser.add_argument('--maccs_dim', type=int, default=167, help='size of maccs fingerprint')
parser.add_argument('--mgf_dim', type=int, default=2048, help='size of morgan fingerprint')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers used in dataloader')
parser.add_argument('--optimize_fp', action='store_true', help='whether to optimize with fps during pretraining')
parser.add_argument('--patience', type=int, default=20, help='number of epochs to wait where val loss is not the min')
parser.add_argument('--vae', action='store_true', help='whether to train with VAE')
parser.add_argument('--jt_vae', action='store_true', help='whether to train with JTVAE')
parser.add_argument('--jt_factor', type=float, default=1e3, dest='fg_factor', help='the weight of the jt_vae loss')
# initially I wanted to name this part fgvae however since it does not relate to functional groups but rather clusters
# as it makes more sense to change it to jtvae and change all related variable however to ensure nothing changes I
# added a "dest" parameter to this argument with the original naming scheme.
parser.add_argument('--wd', type=float, default=1e-7, dest='weight_decay', help='weight_decay')
# model params #
##############################
parser.add_argument('--activations', type=str, default='relu')
parser.add_argument('--add_virtual_node', action='store_true', help='whether to add virtual node')
# note that pyg version 2.0.4 added support for virtual node transform, this means that you can in the future just
# add it as a transformation and not implement yourself.
parser.add_argument('--backbone', type=str, default='gin', choices=['DeeperGCN', 'gin', 'gcn'],
                    help='backbone for method')
parser.add_argument('--block', default='res+', type=str, help='graph backbone block type {res+, res, dense, plain} for'
                                                              'DeeperGCN')
parser.add_argument('--conv', type=str, default='gen', help='the type of GCNs for DeeperGCN')
parser.add_argument('--gcn_aggr', type=str, default='max',
                    choices=['mean', 'max', 'add', 'softmax', 'softmax_sg', 'power'], help='the aggregator of GENConv')
parser.add_argument('--mlp_layers', type=int, default=3, help='the number of layers of mlp in conv')
parser.add_argument('--norm', type=str, default='batch', choices=['batch', 'layer', 'instance'],
                    help='the type of normalization layer')
parser.add_argument('--num_layers', type=int, default=5, help='the number of layers of the networks')
# learnable parameters
parser.add_argument('--learn_p', action='store_true')
parser.add_argument('--learn_t', action='store_true')
parser.add_argument('--learn_y', action='store_true')
parser.add_argument('--p', type=float, default=1.0, help='the power of PowerMean')
parser.add_argument('--t', type=float, default=1.0, help='the temperature of SoftMax')
parser.add_argument('--y', type=float, default=0.0, help='the power of softmax_sum and powermean_sum')
# message norm
parser.add_argument('--learn_msg_scale', action='store_true')
parser.add_argument('--msg_norm', action='store_true')
# encode edge in conv
parser.add_argument('--conv_encode_edge', action='store_true')
# graph pooling type
parser.add_argument('--graph_pooling', type=str, default='sum', choices=['max', 'mean', 'sum'],
                    help='graph pooling method')

