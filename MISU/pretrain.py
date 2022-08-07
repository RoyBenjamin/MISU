from datetime import datetime
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import sys
sys.path.append(".")

# my files
from constants import CHECKPOINTS_PATH, LOG_PATH
from MISU.pretrain_configs import parser
from MISU.constants import SINGLE_MLP
from MISU.data.dataset import MolecularDataModule
from MISU.models.MISU import MISUModule


def add_fields_to_hparams(args, data: MolecularDataModule):
    # args.deepauc = False  # old flag used for some tests, not used and is not deleted for backward compatibility
    args.num_node_features = data.num_node_features
    args.num_tasks = data.num_tasks
    args.pos_weight = data.pos_weight
    hparams.single_mlp = SINGLE_MLP  # indicates whether to use a single mlp for all tasks or separate
    # hparams.use_fp = False  # old flag used for some tests, not used and is not deleted for backward compatibility
    return args


if __name__ == '__main__':
    # setup initial variables
    hparams = parser.parse_args()
    if not (hparams.vae or hparams.optimize_fp or hparams.jt_vae):
        parser.error('Can not pretrain at least one of {\'--vae\', \'--jt_vae\', \'--optimize_fp\'}')
    if hparams.load_ckpt and hparams.ckpt_path == '':
        parser.error('In order to load checkpoints you must also provide a path for loading')
    hparams.task_type = 'pretrain'
    training_checkpoint = os.path.join(CHECKPOINTS_PATH, hparams.dataset, hparams.task_type)
    pl.seed_everything(seed=hparams.seed, workers=True)

    time = datetime.now().strftime('%y%m%d_%H%M%S')
    save_path = time if hparams.save_path == '' else hparams.save_path
    # define lightning related stuff
    if hparams.save_path != '':
        project_name = os.path.join(hparams.dataset, hparams.task_type, hparams.backbone, hparams.save_path)
    else:
        project_name = os.path.join(hparams.dataset, hparams.task_type, hparams.backbone)
    logger = WandbLogger(name=project_name, version=time, project='MISU', config=hparams,
                         save_dir=LOG_PATH) if not hparams.disable_logging else False
    datamodule = MolecularDataModule(hparams, seed=hparams.seed)

    datamodule.prepare_data()  # called only because num_node_features, num_classes are initialized in prepare_data
    hparams = add_fields_to_hparams(hparams, datamodule)
    model = MISUModule.load_from_checkpoint(hparams.ckpt_path) if hparams.load_ckpt else MISUModule(hparams)

    early_stooping = EarlyStopping(monitor='avg_val_loss', patience=hparams.patience, mode='min')
    checkpoint_monitor = 'avg_val_loss' if hparams.task_type == 'pretrain' else 'avg_val_auc'
    checkpoint_mode = 'min' if hparams.task_type == 'pretrain' else 'max'
    gnn_name = model.encoder.__class__.__name__ if hparams.backbone == 'DeeperGCN' else \
        model.encoder.gnn_node.__class__.__name__
    checkpoint = ModelCheckpoint(dirpath=os.path.join(training_checkpoint, gnn_name, save_path),
                                 monitor=checkpoint_monitor, save_top_k=1, mode=checkpoint_mode)
    callbacks = [early_stooping, checkpoint]
    if hparams.lr_scheduler:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    if int(hparams.gpus) > 1:
        torch.multiprocessing.set_sharing_strategy('file_system')
        trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, logger=logger,
                             strategy='ddp', log_every_n_steps=1, callbacks=callbacks, precision=16)
    else:
        trainer = pl.Trainer(gpus=hparams.gpus, max_epochs=hparams.max_epochs, logger=logger,
                             log_every_n_steps=1, callbacks=callbacks)

    # train
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)
    trainer.validate(ckpt_path='best', datamodule=datamodule)
