import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from MISU.models.conv import GNN_node, GNN_node_Virtualnode


# from ogb/examples/lsc/pcqm4m-v2/gnn.py
class GNN(torch.nn.Module):

    def __init__(self, args):
        """
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        """
        super(GNN, self).__init__()
        residual = False
        self.add_virtual_node = args.add_virtual_node
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.task_type = args.task_type
        self.mean_layer = nn.Linear(args.hidden_channels, args.hidden_channels)
        self.std_layer = nn.Linear(args.hidden_channels, args.hidden_channels)

        self.JK = "last"
        self.hidden_channels = args.hidden_channels
        self.num_tasks = args.num_tasks
        self.graph_pooling = args.graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        if args.add_virtual_node:
            self.gnn_node = GNN_node_Virtualnode(args.num_layers, args.hidden_channels, JK=self.JK,
                                                 drop_ratio=args.dropout, residual=residual, gnn_type=args.backbone,
                                                 graph_pooling=args.graph_pooling)
        else:
            self.gnn_node = GNN_node(args.num_layers, args.hidden_channels, JK=self.JK, drop_ratio=args.dropout,
                                     residual=residual, gnn_type=args.backbone, graph_pooling=args.graph_pooling)

        # Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data, fp_emb: bool = False, last_layer: int = None):
        assert last_layer is None or last_layer <= self.num_layers, \
            "last layer argument in GNN.forward is bigger than number of layers"
        if last_layer is None:
            last_layer = self.num_layers

        h_node, fps = self.gnn_node(batched_data, last_layer)
        if self.task_type == 'pretrain':
            return self.mean_layer(h_node), fps if fp_emb else self.std_layer(h_node)
        # we need this because we use VAE and the pyg implementation expects the module to return both mu and log_std
        # with a single call
        else:
            h_graph = self.pool(h_node, batched_data.batch)  # N, hidden_channels
            return h_graph

    def change_task(self, new_args):
        self.dropout = new_args.dropout
        self.task_type = new_args.task_type
        self.gnn_node.drop_ratio = self.dropout
        if self.task_type != 'pretrain':
            del self.mean_layer
            del self.std_layer
