from datetime import datetime
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import sys
import torch
import wandb
sys.path.append(".")

# my files
from constants import CHECKPOINTS_PATH, LOG_PATH, SINGLE_MLP
from MISU.data.dataset import MolecularDataModule
from MISU.models.MISU import MISUModule
from MISU.finetune_configs import parser


def add_fields_to_hparams(args, dm: MolecularDataModule):
    """
    Add fields to hparams
    :param args: args from parser
    :param dm: data module
    :return: hparams
    """
    args.num_node_features = dm.num_node_features  # dm is short for data module
    args.num_tasks = dm.num_tasks
    args.pos_weight = dm.pos_weight
    args.single_mlp = SINGLE_MLP  # indicates whether to use a single mlp for all tasks or separate.
    args.imratio = []
    for i in range(args.num_tasks):
        labels = dm.dataset.data.y[:, i]
        labels = labels[~torch.isnan(labels)]
        num_samples = labels.shape[0]  # after filtering nans
        args.imratio.append(float((labels.sum() / num_samples).numpy()))
    return args


if __name__ == '__main__':
    # setup initial variables
    hparams = parser.parse_args()
    hparams.task_type = 'finetune'
    training_checkpoint = os.path.join(CHECKPOINTS_PATH, hparams.dataset)
    results = {}
    for split in ['train', 'val', 'test']:
        results[f'{split}_loss'] = np.array([])
        results[f'{split}_auc'] = np.array([])
    for seed in range(hparams.seed, hparams.seed+hparams.run_times):

        pl.seed_everything(seed=seed, workers=True)

        # define lightning related stuff
        project_name = os.path.join(hparams.dataset, f'seed_{seed}')
        logger = WandbLogger(name=project_name, version=datetime.now().strftime('%y%m%d_%H%M%S.%f'), project='MISU',
                             config=hparams, save_dir=LOG_PATH) if not hparams.disable_logging else False
        datamodule = MolecularDataModule(hparams, seed=seed)

        datamodule.prepare_data()  # called only because num_node_features, num_classes are initialized in prepare_data
        hparams = add_fields_to_hparams(hparams, datamodule)
        model = MISUModule.load_from_checkpoint(hparams.checkpoint_path)
        model.change_task(hparams)
        gnn_name = model.encoder.__class__.__name__ if model.hparams.backbone == 'DeeperGCN' else \
            model.encoder.gnn_node.__class__.__name__

        # deal with checkpoints for seed
        if hparams.no_pretrain:  # for baseline testing, we still need the checkpoints file to reuse model arch. params
            model.hparams.last_layer = model.last_layer
            model = MISUModule(model.hparams)  # only use the hparams from the pretraining for consistency
            model.change_task(hparams)
            seed_checkpoint = os.path.join(training_checkpoint, gnn_name, 'train')
        else:
            seed_checkpoint = os.path.join(training_checkpoint, gnn_name, 'finetune')
        seed_checkpoint = os.path.join(seed_checkpoint, 'adam')
        ablation_dir = os.path.basename(os.path.dirname(hparams.checkpoint_path))
        seed_checkpoint = os.path.join(seed_checkpoint, ablation_dir)
        seed_checkpoint = os.path.join(seed_checkpoint, f'seed_{seed}')
        monitor = 'avg_val_loss'
        early_stooping = EarlyStopping(monitor=monitor, patience=hparams.patience, mode='min')
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=seed_checkpoint, monitor='avg_val_auc', save_top_k=1,
                                                  mode='max')
        trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, logger=logger, num_sanity_val_steps=0,
                             log_every_n_steps=1, callbacks=[early_stooping, checkpoint], gradient_clip_val=50)
        
        # train
        seed_results = {}
        trainer.fit(model, datamodule=datamodule)
        seed_results['test'] = trainer.test(ckpt_path='best', datamodule=datamodule)[0]
        seed_results['val'] = trainer.validate(ckpt_path='best', datamodule=datamodule)[0]

        for split in ['val', 'test']:
            latest_loss = seed_results[split][f'avg_{split}_loss']
            latest_metric = seed_results[split][f'avg_{split}_auc']
            results[f'{split}_loss'] = np.append(results[f'{split}_loss'], latest_loss)
            results[f'{split}_auc'] = np.append(results[f'{split}_auc'], latest_metric)

        wandb.finish()

    for split in ['val', 'test']:
        for metric in ['loss', 'auc']:
            mean = np.mean(results[f'{split}_{metric}'])
            std = np.std(results[f'{split}_{metric}'])
            print(f'{split} {metric} - {mean:.4f} ± {std:.4f}')
