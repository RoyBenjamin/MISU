import os.path as osp
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import random_split
from torch_geometric.loader.dataloader import DataLoader

# my files
from MISU.constants import DATA_PATH_MOLECULENET, DATA_PATH_PCQ
from MISU.pretrain_configs import dataset_choices
from MISU.utils import extract_fingerprints, extract_junction_trees, get_smiles_df, get_ogb_name, calc_pos_weight


class MolecularDataModule(LightningDataModule):
    """
    DataModule handling data for the MISU model.
    """

    def __init__(self, args, seed: int = 0):
        super().__init__()
        if args.dataset not in dataset_choices:
            raise Exception('name is not supported')
        self.batch_size = args.batch_size
        self.dataset = None
        self.lengths = None
        self.name = args.dataset
        self.num_tasks = None
        self.num_node_features = None
        self.num_workers = args.num_workers
        self.use_fp = True if hasattr(args, 'optimize_fp') and args.optimize_fp else False
        self.jt_vae = True if hasattr(args, 'jt_vae') and args.jt_vae else False
        self.path = DATA_PATH_PCQ if self.name == 'PCQ' else osp.join(DATA_PATH_MOLECULENET, 'ogb')
        self.pos_weight = None
        self.random_split = args.random_split
        self.seed = seed
        self.train = None
        self.test = None
        self.val = None
        self.verbose = args.verbose

    def prepare_data(self):
        """
        Prepares the dataset (download or create)
        """
        if self.name in dataset_choices:
            self.dataset = PygPCQM4Mv2Dataset(self.path, smiles2graph=smiles2graph) if self.name == 'PCQ' else \
                PygGraphPropPredDataset(root=self.path, name=get_ogb_name(self.name))
            self.add_representations()
            self.num_node_features = self.dataset.num_node_features
            self.num_tasks = self.dataset.num_tasks if self.name != 'PCQ' else 1
            self.lengths = [int(0.8 * len(self.dataset)), int(0.1 * len(self.dataset))]
            self.lengths.append(len(self.dataset) - sum(self.lengths))
            self.pos_weight = calc_pos_weight(self.dataset.data.y.float()) if self.name != 'PCQ' else None
        else:
            raise Exception(f'dataset {self.dataset} not supported')

    def setup(self, stage=None):
        """
        Splits the data into train/val/test
        """
        if self.train is None:
            if self.random_split:
                self.train, self.val, self.test = random_split(self.dataset, self.lengths,
                                                               generator=torch.Generator().manual_seed(self.seed))
            else:
                test_str = "test-dev" if self.name == 'PCQ' else "test"
                split_idx = self.dataset.get_idx_split()
                self.train = self.dataset[split_idx["train"]]
                self.val = self.dataset[split_idx["valid"]]
                self.test = self.dataset[split_idx[test_str]]
            if self.verbose and self.name != 'PCQ':
                print(f'{sum(self.train[i].y for i in range(len(self.train)))} positive samples in train')
                print(f'{sum(self.val[i].y for i in range(len(self.val)))} positive samples in val')
                print(f'{sum(self.test[i].y for i in range(len(self.test)))} positive samples in test')

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dataloader()

    def add_representations(self):
        """
        Adds smiles and fingerprint tensors to dataset
        """
        ogb_name = get_ogb_name(self.name)
        if self.name == 'PCQ':
            file_directory = osp.join(DATA_PATH_PCQ, ogb_name)
        else:
            file_directory = osp.join(DATA_PATH_MOLECULENET, 'ogb', ogb_name).replace("-", "_")
        csv_path = osp.join(DATA_PATH_PCQ, ogb_name, 'raw', 'data.csv.gz') if self.name == 'PCQ' \
            else osp.join(file_directory, 'mapping', 'mol.csv.gz')
        smiles_df = get_smiles_df(csv_path)
        self.dataset.data['smiles'] = list(smiles_df.values)
        self.dataset.slices['smiles'] = self.dataset.slices['y']
        if self.use_fp:
            if not osp.isfile(osp.join(file_directory, "mgf_feat.pt")):
                extract_fingerprints(file_directory, smiles_df)
            self.dataset.data['morgan'] = torch.load(osp.join(file_directory, "mgf_feat.pt"))
            self.dataset.data['maccs'] = torch.load(osp.join(file_directory, "maccs_feat.pt"))
            self.dataset.slices['morgan'] = self.dataset.slices['y']
            self.dataset.slices['maccs'] = self.dataset.slices['y']
        if self.jt_vae:
            if not osp.isfile(osp.join(file_directory, "junction_trees.pt")):
                extract_junction_trees(file_directory, smiles_df)
            trees_list = torch.load(osp.join(file_directory, "junction_trees.pt"))
            self.dataset.data['functional_group'] = trees_list
            self.dataset.slices['functional_group'] = self.dataset.slices['y']
