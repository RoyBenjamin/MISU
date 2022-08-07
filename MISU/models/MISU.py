from argparse import Namespace
# for AUC margin loss
from ogb.graphproppred import Evaluator
from ogb.lsc import PCQM4Mv2Evaluator
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.graphgym.models.layer import LayerConfig, MLP
from torch_geometric.utils import batched_negative_sampling
from torchmetrics import MeanMetric

# my files
from MISU.models.DeeperGCN import DeeperGCN
from MISU.models.gnn import GNN
from MISU.models.VGAE import VGAENoNeg
from MISU.utils import dataset_contain_nans, get_ogb_name, get_junction_tree


class MISU(nn.Module):
    def __init__(self, args, encoder: nn.Module):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        classifier_config = LayerConfig(num_layers=args.mlp_layers, dim_in=args.hidden_channels, dim_out=args.num_tasks,
                                        final_act=False, dropout=args.dropout)
        self.classifier = MLP(classifier_config)  # to be assigned with the classifier for fine-tuning later
        self.encoder = encoder
        self.hparams = args
        if args.task_type == 'pretrain' and args.optimize_fp:
            aligner_config = LayerConfig(num_layers=1, dim_in=args.hidden_channels,
                                         dim_out=args.mgf_dim + args.maccs_dim,
                                         final_act=False, dropout=args.dropout)
            self.fp_aligners = nn.ModuleList([MLP(aligner_config) for _ in range(args.num_layers - 1)])
        else:
            classifier_config = LayerConfig(num_layers=args.mlp_layers, dim_in=self.hparams.hidden_channels,
                                            dim_out=args.num_tasks, final_act=False, dropout=args.dropout)
            self.classifier = MLP(classifier_config)

    def forward(self, batch, fp_emb: bool = False, last_layer: int = None):
        if self.hparams.task_type == 'pretrain':
            embeddings, fp_embeddings = self.encoder(batch, fp_emb, last_layer)
            if self.hparams.optimize_fp:
                fp_preds = [module(fp_embeddings[:, i]) for i, module in enumerate(self.fp_aligners)]
                fp_embeddings = torch.stack(fp_preds, dim=1)
            return embeddings, fp_embeddings
        elif self.classifier is None:
            raise AttributeError("can not use MISU module without MLP classifier defined, you should change the task "
                                 "using the change_task_type method only!")
        else:
            embeddings = self.encoder(batch, fp_emb, last_layer)
        return self.classifier(embeddings)

    def change_task(self, new_args):
        if self.hparams.task_type == new_args.task_type:
            return
        self.hparams.dropout = new_args.dropout
        self.hparams.mlp_layers = new_args.mlp_layers
        self.hparams.num_tasks = new_args.num_tasks
        self.hparams.task_type = new_args.task_type
        if new_args.task_type == 'finetune':
            if hasattr(self, 'fp_aligners'):
                del self.fp_aligners  # this reduces the size of the model effectively on the GPU
            classifier_config = LayerConfig(num_layers=new_args.mlp_layers, dim_in=self.hparams.hidden_channels,
                                            dim_out=new_args.num_tasks, final_act=False,
                                            dropout=new_args.dropout)
            self.classifier = MLP(classifier_config)


class MISUModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
            if not hasattr(hparams, 'optimize_fp'):
                hparams.optimize_fp = False
            if not hasattr(hparams, 'vae'):
                hparams.vae = False
            if not hasattr(hparams, 'jt_vae'):
                hparams.jt_vae = False
            if not hasattr(hparams, 'fg_factor'):
                hparams.fg_factor = 1.0
        self.save_hyperparameters(hparams)

        if self.hparams.backbone == 'DeeperGCN':
            self.encoder = DeeperGCN(hparams)
        elif self.hparams.backbone == 'gin' or self.hparams.backbone == 'gcn':
            self.encoder = GNN(hparams)

        self.metric_calculation_results = {'y_pred': {}, 'y_true': {}}  # to be reassigned later for fine-tuning
        for split in ['train', 'val', 'test']:
            self.metric_calculation_results['y_pred'][split] = None
            self.metric_calculation_results['y_true'][split] = None

        self.metrics = {}
        for split in ['train', 'val', 'test']:
            for metric_type in ['total_loss', 'fps_loss', 'loss']:
                self.metrics[f'{split}_{metric_type}'] = MeanMetric(nan_strategy='ignore', compute_on_step=False)

        self.gnn = MISU(hparams, self.encoder)
        self.vae = VGAENoNeg(self.encoder) if hparams.task_type == 'pretrain' and hparams.vae else None
        self.jt_vae = VGAENoNeg(self.encoder) if hparams.task_type == 'pretrain' and hparams.jt_vae else None
        # set the pos_weight to None for Tox21/ToxCast to be able to ignore nan values present as labels,
        # if pos_weight is not None then an exception is raised as the module tries to multiply the weight of size
        # num_tasks with an input of different size
        if hparams.task_type == 'pretrain':
            self.evaluator = PCQM4Mv2Evaluator()  # to be assigned with the molnet evaluator for fine-tuning later
            self.evaluator.eval_metric = 'mae'  # add eval metric for compatability with the properties of graph
            self.criterion_labels = nn.L1Loss()
            self.criterion_fps = nn.BCEWithLogitsLoss()
            self.last_layer = None  # to be assigned later for finetuning
        else:
            self.evaluator = Evaluator(get_ogb_name(self.hparams.dataset))
            # self.last_layer = hparams.last_layer if hasattr(hparams, 'last_layer') else 5 #change for qualitative
            self.last_layer = hparams.last_layer
            self.criterion_labels = nn.BCEWithLogitsLoss(pos_weight=self.hparams.pos_weight if not
            dataset_contain_nans(self.hparams.dataset) else None)

    def forward(self, batch, fp_emb: bool = False):
        return self.gnn(batch, fp_emb, self.last_layer)

    def step(self, batch, name):
        loss_type = 'vae' if self.hparams.task_type == 'pretrain' else 'labels'
        pbar = {}
        labels = batch.y.float()
        loss_graph, loss_fp, pred = self._loss(batch)
        pbar[f'{name}_{loss_type}_loss'] = loss_graph.detach()
        if self.hparams.task_type == 'pretrain':
            pbar[f'{name}_fps_loss'] = self.hparams.fp_factor * loss_fp.detach()
            loss = loss_graph + self.hparams.fp_factor * loss_fp
            for metric_type, value in zip(['total_loss', 'fps_loss'],
                                           [loss_graph.detach(), loss_fp.detach()]):
                if self.metrics[f'{name}_{metric_type}'].device != loss_graph.device:
                    self.metrics[f'{name}_{metric_type}'] = \
                        self.metrics[f'{name}_{metric_type}'].to(loss_graph.device)
                self.metrics[f'{name}_{metric_type}'].update(value)
        else:
            loss = loss_graph

        if self.hparams.task_type != 'pretrain':
            previous_preds = self.metric_calculation_results['y_pred'][name]
            previous_labels = self.metric_calculation_results['y_true'][name]
            self.metric_calculation_results['y_pred'][name] = torch.cat([previous_preds, pred], dim=0) \
                if previous_preds is not None else pred
            self.metric_calculation_results['y_true'][name] = torch.cat([previous_labels, labels], dim=0) \
                if previous_labels is not None else labels

        if self.metrics[f'{name}_loss'].device != loss.device:
            self.metrics[f'{name}_loss'] = self.metrics[f'{name}_loss'].to(loss.device)
        self.metrics[f'{name}_loss'].update(loss.detach())

        return {'loss': loss, 'progress_bar': pbar}

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, name='train')

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, name='val')

    def test_step(self, batch: dict, batch_idx: int):
        return self.step(batch, name='test')

    def epoch_end(self, step_outputs: EPOCH_OUTPUT, name: str) -> None:
        results = {f'avg_{name}_loss': self.metrics[f'{name}_loss'].compute()}
        self.metrics[f'{name}_loss'].reset()
        if self.hparams.task_type == 'pretrain':
            results[f'avg_{name}_total_loss'] = self.metrics[f'{name}_total_loss'].compute()
            self.metrics[f'{name}_total_loss'].reset()

            if self.hparams.optimize_fp:
                results[f'avg_{name}_fps_loss'] = self.metrics[f'{name}_fps_loss'].compute()
                self.metrics[f'{name}_fps_loss'].reset()

        else:
            input_dict = {"y_true": self.metric_calculation_results['y_true'][name],
                          "y_pred": self.metric_calculation_results['y_pred'][name]}
            metric = self.evaluator.eval(input_dict)
            results[f'avg_{name}_auc'] = metric[self.evaluator.eval_metric]
            self.metric_calculation_results['y_pred'][name] = None
            self.metric_calculation_results['y_true'][name] = None
        for k, v in results.items():
            p_bar = False if any(substring in k for substring in ['total', 'fps']) else True
            self.log(k, v, prog_bar=p_bar, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)

    def training_epoch_end(self, train_step_outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end(train_step_outputs, name='train')

    def validation_epoch_end(self, val_step_outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end(val_step_outputs, name='val')

    def test_epoch_end(self, test_step_outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end(test_step_outputs, name='test')

    def configure_optimizers(self):
        if self.hparams.adamw:
            opt = torch.optim.AdamW(params=self.parameters(), lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay)
        else:
            opt = torch.optim.Adam(params=self.parameters(), lr=self.hparams.learning_rate,
                                   weight_decay=self.hparams.weight_decay)
        if not self.hparams.lr_scheduler:
            return opt

        scheduler = CosineAnnealingLR(opt, T_max=self.hparams.max_epochs, eta_min=1e-5)
        scheduler.verbose = self.hparams.verbose
        configs = {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "epoch"}
        }
        return configs

    def transfer_model_to_device(self):
        if hasattr(self, 'encoder'):
            self.encoder.to(self.device)
        if hasattr(self.gnn, 'vae'):
            self.vae.to(self.device)
        if hasattr(self.gnn, 'jt_vae'):
            self.jt_vae.to(self.device)
        self.gnn.to(self.device)
        if hasattr(self.gnn, 'fp_aligners'):
            for module in self.gnn.fp_aligners:
                module.to(self.device)
        if self.gnn.classifier is not None:
            self.gnn.classifier.to(self.device)

    def on_fit_start(self) -> None:
        self.transfer_model_to_device()

    def on_train_start(self) -> None:
        self.transfer_model_to_device()

    def _loss(self, batch):
        loss_fp = torch.zeros(1, device=self.device)
        if self.hparams.task_type != 'pretrain':
            y_true = batch.y.float()
            pred = self(batch)
            out = torch.sigmoid(pred).detach()
            if dataset_contain_nans(self.hparams.dataset):
                y_pred_reshaped = pred.view(-1)
                y_gt = y_true[:, :].reshape(-1, 1).view(-1)
                mask = ~torch.isnan(y_gt)
                loss_y = self.criterion_labels(y_pred_reshaped[mask], y_gt[mask])
            else:
                loss_y = self.criterion_labels(pred, y_true)
        else:
            if self.hparams.optimize_fp or self.hparams.jt_vae:
                embeddings, fp_embeddings = self(batch, fp_emb=True)
            # use PCQM4M label for optimization
            out = torch.zeros(1, device=self.device)

            # use vae for adjacency matrix creation for the whole graph
            if self.hparams.vae:
                # calculate the vae related losses
                try:
                    neg_edge_index = batched_negative_sampling(batch.edge_index, batch.batch).long()
                except AssertionError:
                    # The most common reason for an assertion error in _loss is when batched_negative_sampling can not
                    # sample for example in the dataset there is a molecule ClCl this is a graph with two nodes, and it
                    # is fully connected (with one edge), hence we can not use batched_negative_sampling.
                    neg_edge_index = None
                recon_loss = self.vae.recon_loss(self.vae.encode(batch, self.device), batch.edge_index.long(),
                                                 neg_edge_index)
                kl_loss = self.vae.kl_loss()
            else:
                recon_loss = torch.zeros(1, device=self.device)
                kl_loss = torch.zeros(1, device=self.device)

            # use fg vae for adjacency matrix creation for the junction tree
            if self.hparams.jt_vae:
                _, fg_edge_index, fg_batch = get_junction_tree(embeddings, batch, self.device)
                # calculate the vae related losses
                try:
                    fg_neg_edge_index = batched_negative_sampling(fg_edge_index, fg_batch).long()
                except AssertionError:
                    fg_neg_edge_index = None
                fg_recon_loss = self.jt_vae.recon_loss(self.jt_vae.encode(batch, device=self.device, fgvae=True),
                                                       fg_edge_index.long(), fg_neg_edge_index)
                fg_kl_loss = self.jt_vae.kl_loss()
            else:
                fg_recon_loss = torch.zeros(1, device=self.device)
                fg_kl_loss = torch.zeros(1, device=self.device)

            loss_y = recon_loss + self.hparams.kl_factor*kl_loss + self.hparams.fg_factor*fg_recon_loss + \
                     self.hparams.fg_kl_factor*fg_kl_loss

            # optimize using fingerprints with BCE as listed in the paper
            if self.hparams.optimize_fp:
                fp_true = torch.cat([batch.morgan, batch.maccs], dim=1)
                for i in range(self.hparams.num_layers - 1):
                    loss_fp += self.criterion_fps(fp_embeddings[:, i], fp_true) / (self.hparams.num_layers - 1)
        return loss_y, loss_fp, out

    def change_task(self, new_args):
        if self.hparams.task_type == new_args.task_type:
            return
        self.hparams.adamw = new_args.adamw
        self.hparams.batch_size = new_args.batch_size
        self.hparams.dataset = new_args.dataset
        self.hparams.dropout = new_args.dropout
        self.hparams.imratio = new_args.imratio
        self.hparams.learning_rate = new_args.learning_rate
        self.hparams.lr_scheduler = new_args.lr_scheduler
        self.hparams.num_tasks = new_args.num_tasks
        self.hparams.pos_weight = new_args.pos_weight
        self.hparams.task_type = new_args.task_type
        self.hparams.verbose = new_args.verbose
        self.hparams.weight_decay = new_args.weight_decay
        self.encoder.change_task(new_args)
        self.last_layer = new_args.last_layer
        self.gnn.change_task(new_args)
        if self.hparams.task_type != 'pretrain':
            if hasattr(self, 'criterion_fps'):
                del self.criterion_fps  # remove the loss for fingerprints
            if hasattr(self, 'vae'):
                del self.vae  # remove the VAE as it is not needed for finetuning
            if hasattr(self, 'jt_vae'):
                del self.jt_vae  # remove the jt_vae as it is not needed for finetuning
            self.evaluator = Evaluator(get_ogb_name(self.hparams.dataset))
            self.metric_calculation_results = {'y_pred': {}, 'y_true': {}}
            for split in ['train', 'val', 'test']:
                self.metric_calculation_results['y_pred'][split] = None
                self.metric_calculation_results['y_true'][split] = None

            self.criterion_labels = nn.BCEWithLogitsLoss(pos_weight=self.hparams.pos_weight if not
            dataset_contain_nans(self.hparams.dataset) else None)
