import argparse

# my files
from MISU.constants import LOG_PATH

parser = argparse.ArgumentParser()

dataset_choices = ['PCBA', 'MUV', 'HIV', 'BACE', 'BBBP', 'Tox21', 'ToxCast', 'SIDER', 'ClinTox']

# general / saving params #
##############################
parser.add_argument('--dataset', type=str, default='HIV', choices=dataset_choices, help='the dataset name')
parser.add_argument('--gpus', type=str, default='0', help='gpus parameter used for pytorch_lightning')
parser.add_argument('--log_path', type=str, default=LOG_PATH, help='tensorboard log path')
parser.add_argument('--disable_logging', default=False, action='store_true', help='disables logging')
parser.add_argument('--random_split', default=False, action='store_true', help='changes the dataset to pyg from ogb')
parser.add_argument('--seed', type=int, default=42, help='seed for data splitting randomness')
parser.add_argument('--verbose', default=False, action='store_true', help='whether to print more information')
# training params #
##############################
parser.add_argument('--adamw', action='store_true', help='whether to use AdamW optimizer or Adam')
parser.add_argument('--batch_size', type=int, default=256, help='the desired batch size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout for the classifier')
parser.add_argument('--hidden_channels', type=int, default=256, help='the dimension of embeddings of nodes and edges')
parser.add_argument('--lr', type=float, default=3e-4, dest='learning_rate', help='learning rate')
parser.add_argument('--lr_scheduler', action='store_true', help='whether to use learning rate scheduler')
parser.add_argument('--max_epochs', type=int, default=150, help='number of maximum epochs to run')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers used in dataloader')
parser.add_argument('--no_pretrain', action='store_true', help='whether to use the pretrained model or not')
parser.add_argument('--patience', type=int, default=30, help='number of epochs to wait where val loss is not the min')
parser.add_argument('--run_times', type=int, default=10, help='number of times to train the model')
parser.add_argument('--wd', type=float, default=1e-7, dest='weight_decay', help='weight_decay')
# fine tuning params #
##############################
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/PCQ/pretrain/latest_run.ckpt')
# model params #
##############################
parser.add_argument('--mlp_layers', type=int, default=3, help='the number of layers of mlp in conv')
parser.add_argument('--last_layer', type=int, default=5, help='the number of layers of the networks')

