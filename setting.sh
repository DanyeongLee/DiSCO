#!/bin/sh


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install rdkit wandb hydra-core hydra-colorlog torch_ema e3nn
pip install torch_geometric==1.7.2
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install multiprocess easydict lmdb 
pip install xtb
