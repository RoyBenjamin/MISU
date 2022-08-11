# MISU: Graph Neural Networks Pretraining Through Inherent Supervision for Molecular Property Prediction

### Roy Benjamin, Uriel Singer and Kira Radinsky. 

> **Abstract:**
> Recent global events have emphasized the importance of accelerating the drug discovery process. 
> A way to deal with the issue is to use machine learning to increase the rate at which drugs are made available to the public. However, chemical labeled data for real-world applications is extremely scarce making traditional approaches less effective. 
> A fruitful course of action for this challenge is to pretrain a model using related tasks with large enough datasets, with the next step being finetuning it for the desired task. This is challenging as creating these datasets requires labeled data or expert knowledge.
> To aid in solving this pressing issue, we introduce MISU - Molecular Inherent SUpervision, a unique method for pretraining graph neural networks for molecular property prediction. 
> Our method leapfrogs past the need for labeled data or any expert knowledge by introducing three innovative components that utilize inherent properties of molecular graphs to induce information extraction at different scales, from the local neighborhood of an atom to substructures in the entire molecule. 
> Our empirical results for six chemical-property-prediction tasks show that our method reaches state-of-the-art results compared to numerous baselines.

This repository provides a reference implementation of MISU as described in the paper.

## Requirements
 - ogb==1.3.2
 - pandas==1.4.1
 - python==3.9.10
 - pytorch-lightning==1.5.10
 - rdkit==2021.09.4
 - scikit-learn=1.0.2
 - torch==1.10.2
 - torch-geometric==2.0.3
 - tqdm==4.64.0
 - wandb==0.12.11

### Creating an Environment
First clone repository, then follow these steps.
```bash
cd MISU/
conda env create --file environment.yml
conda activate MISU
```

## Data
During the first run, the dataset is automatically downloaded and cached. This may take some time (on our tests about 3 hours for 4 millions molecules).

## Usage
### Pretraining
```bash
python MISU/pretrain.py\
    --gpus <number of gpus you want to use> \ 
    --num_workers <number of workers> \
    --batch_size <batch size> \
    --optimize_fp  \
    --jt_vae \
    --vae \
    --add_virtual_node 
```
There are other parameters that can be passed for more info look at pretrain_configs.py.

### Finetuning
```bash
python MISU/finetune.py \
    --dataset <dataset you want to use> \
    --gpu <number of gpus you want to use> \
    --disable_logging \
    --checkpoint_path <path to saved checkpoint> \
```
There are other parameters that can be passed for more info look at finetune_configs.py.

## Cite
Please cite our paper if you use this code in your own work:
```
@inproceedings{misu,
  title     = {Graph Neural Networks Pretraining Through Inherent Supervision for Molecular Property Prediction},
  author    = {Benjamin, Roy and Singer, Uriel and Radinsky, Kira},
  year      = {2022},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  booktitle = {Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
  numpages  = {10},
  keywords  = {Drug Discovery, ML for Healthcare, Molecular Property Prediction},
  location  = {Atlanta, Georgia, USA},
  series    = {CIKM '22}
}
```
