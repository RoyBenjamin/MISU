import os.path as osp
import pandas as pd
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import tree_decomposition
from tqdm import tqdm
# my files
from MISU.pretrain_configs import dataset_choices


def get_ogb_name(name: str = 'HIV'):
    _ogb_dataset_choices = ['ogbg-molpcba', 'ogbg-molmuv', 'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp',
                            'ogbg-moltox21', 'ogbg-moltoxcast', 'ogbg-molsider', 'ogbg-molclintox', 'pcqm4m-v2']
    ogb_dict = {dataset_choices[i]: _ogb_dataset_choices[i] for i in range(len(dataset_choices))}
    return ogb_dict[name]


def calc_pos_weight(y_true: torch.Tensor):
    assert y_true.dtype == torch.float
    num_true = y_true.sum(dim=0)
    num_samples = y_true.shape[0]  # shape: (B, L) when B is batch size and L is the number of tasks
    pos_weight = [float(num_samples - num_pos) / num_pos if num_pos != 0 else torch.tensor(1.0) for num_pos in num_true]
    return torch.tensor(pos_weight).type_as(y_true)


def get_morgan_fingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))  # default size for vector is 2048


def get_maccs_fingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]


def get_smiles_df(csv_path: str):
    if not osp.isfile(csv_path):
        raise FileNotFoundError(f'{csv_path} does not exists')
    return pd.read_csv(csv_path)["smiles"]


def extract_fingerprints(file_directory: str, smiles):
    # adapted from PierreHao/YouGraph
    mgf_feat_list = []
    maccs_feat_list = []
    print("Extracting fingerprints...")
    for ii in tqdm(range(len(smiles))):
        rdkit_mol = AllChem.MolFromSmiles(smiles.iloc[ii])

        mgf = get_morgan_fingerprint(rdkit_mol)
        mgf_feat_list.append(mgf)

        maccs = get_maccs_fingerprint(rdkit_mol)
        maccs_feat_list.append(maccs)

    mgf_feat = torch.tensor(mgf_feat_list, dtype=torch.float)
    maccs_feat = torch.tensor(maccs_feat_list, dtype=torch.float)
    print("morgan feature shape: ", mgf_feat.shape)
    print("maccs feature shape: ", maccs_feat.shape)

    torch.save(mgf_feat, osp.join(file_directory, "mgf_feat.pt"))
    torch.save(maccs_feat, osp.join(file_directory, "maccs_feat.pt"))
    print(f'saved features in {file_directory}')


def extract_junction_trees(file_directory: str, smiles):
    trees_list = []
    print("Extracting junction trees...")
    for ii in tqdm(range(len(smiles))):
        rdkit_mol = AllChem.MolFromSmiles(smiles.iloc[ii])
        tree = tree_decomposition(rdkit_mol)
        data = HeteroData()  # while not the most elegant solution this solves two problem, one is if during training we
        # try to generate the junction tree the runtime explodes. Second is that using HeteroData allow us to add
        # different fields to the dataset relatively easy. A more elegant solution would be to rewrite some
        # ogb/PygPCQM4Mv2Dataset functions, for example the 'collate' fn could be rewritten to handle the JTs
        data['atom_mapping'].edge_index = tree[0]
        data['junction_tree'].edge_index = tree[1]
        data['junction_tree'].num_nodes = tree[2]
        trees_list.append(data)
    torch.save(trees_list, osp.join(file_directory, "junction_trees.pt"))
    print(f'saved junction trees in {file_directory}')


# allowable multiple choice node and edge features
# code from https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
    ]))


def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))


def get_junction_tree(z, batch, device, pool='sum'):
    assert pool == 'sum', "get_junction_tree::only sum operations are supported"
    assert hasattr(batch, "smiles"), "batch does not have smiles attribute"
    pool = global_add_pool if pool == 'sum' else global_add_pool
    # numOfNodesGraph holds the cumulative sum for the number of nodes in the molecular graph
    numOfNodesGraph = torch.tensor([torch.sum(sample == batch.batch) for sample in range(len(batch.smiles))],
                                   dtype=torch.long, device=device)
    numOfNodesGraph = torch.cumsum(numOfNodesGraph, dim=0)
    numOfNodesGraph = torch.cat([torch.zeros(1, device=device), numOfNodesGraph]).long()
    # numOfNodesJT holds the cumulative sum for the number of nodes in the junction tree graph
    numOfNodesJT = torch.tensor([fg['junction_tree'].num_nodes for fg in batch.functional_group], dtype=torch.long,
                                device=device)
    numOfNodesJT = torch.cumsum(numOfNodesJT, dim=0)
    numOfNodesJT = torch.cat([torch.zeros(1, device=device), numOfNodesJT]).long()
    assert z.shape[0] == numOfNodesGraph[-1], f'get_junction_tree::size of embedding {z.shape[0]} and number of nodes' \
                                              f' {numOfNodesGraph[-1]} in batch are not the same'
    group_embs = torch.empty(0, dtype=torch.float, requires_grad=True, device=device)
    jt_index = torch.zeros((numOfNodesJT[-1].item()), dtype=torch.int64, device=device)
    group_edge_index = torch.empty(0, dtype=torch.long, device=device)
    for tree_index, tree in enumerate(batch.functional_group):
        # tree[0] contains the graph connectivity for the junction tree
        # tree[1] holds the mapping from atom embedding to functional group embedding
        # tree[2] holds the number of clusters

        tree = tree.to(device)
        atom_mapping = tree['atom_mapping'].edge_index
        jt_edges = tree['junction_tree'].edge_index
        num_clusters = tree['junction_tree'].num_nodes
        current_graph = numOfNodesGraph[tree_index]
        current_jt = numOfNodesJT[tree_index]
        next_jt = numOfNodesJT[tree_index + 1]
        jt_index[current_jt:next_jt] = tree_index
        group_edge_index = torch.cat([group_edge_index, current_jt + atom_mapping], dim=1)
        group_embs = torch.cat([group_embs, pool(z[current_graph + jt_edges[0]], jt_edges[1], size=num_clusters)])

    return group_embs, group_edge_index.long(), jt_index


def move_tree_to_device(tree, device):
    moved = []
    for i, element in enumerate(tree):
        if torch.is_tensor(element):
            moved.append(element.to(device))
        else:
            moved.append(element)
    return tuple(moved)


def dataset_contain_nans(dataset_name: str) -> bool:
    nan_datasets = ['PCBA', 'MUV', 'Tox21', 'ToxCast']
    return dataset_name in nan_datasets
