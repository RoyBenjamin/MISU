from torch import nn
from torch_geometric.graphgym.models.layer import LayerConfig, MLP
from torch_geometric.nn import GIN, global_add_pool


class GIN_model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.task_type = hparams.task_type
        self.gnn = GIN(hparams.num_node_features, hparams.hidden_channels, hparams.num_layers, dropout=hparams.dropout,
                       jk='cat')
        classifier_config = LayerConfig(num_layers=hparams.mlp_layers, dim_in=hparams.hidden_channels,
                                        dim_out=hparams.num_tasks, final_act=False, dropout=hparams.dropout)
        self.classifier = MLP(classifier_config)
        self.mean = nn.Linear(hparams.hidden_channels, hparams.hidden_channels)
        self.std = nn.Linear(hparams.hidden_channels, hparams.hidden_channels)

    def forward(self, batch, fp_emb: bool = False, last_layer: int = None):
        x, edge_index, batch_ind = batch.x.float(), batch.edge_index, batch.batch
        h = self.gnn(x, batch.edge_index)
        if self.task_type == 'pretrain':
            return self.mean(h), h.reshape(-1, self.batch_size) if fp_emb else self.std(h)
        else:
            return global_add_pool(h, batch_ind)  # N, hidden_channels

    def change_task(self, new_args):
        self.task_type = new_args.task_type
