# TGNet
Code for “TGNet: A Graph Neural and Transformer Network for Multi-Task Machining Feature Recognition in B-rep CAD Models”
![B-rep data processing](image/network.png)
![Network Architecture](image/archi.png)
TGNet, a neural network combining graph neural networks for local feature extraction with a Transformer module for global context modeling. It efficiently processes B-rep data by unifying geometric sampling, attribute embedding, and hierarchical feature fusion. TGNet is designed as a multi-task network that enables precise recognition and localization of machining features and related surfaces, providing essential support for downstream operations such as part dimension measurement. Experimental results demonstrate that TGNet delivers strong performance on various tasks, showing great potential for real-world industrial applications.

## Environment Setup
```bash
conda env create -f environment.yaml -n XXX  # Replace XXX with your environment name
conda activate XXX
```
## Dataset
Download the piping and sheet metal dataset from the following link:
https://drive.google.com/drive/folders/1TevFvOuHBV50hqkWHwlLMgEiZvSA9XwX?usp=drive_link
make sure each dataset contains these folders: fag, labels, MFInstseg_partition, steps.

## Train
Open [dataprocess.py](Code/dataprocess.py) and choose according to the dataset type:
```bash
Line 21 # return 17  # steel metal
Line 22 # return 13  # piping
```
Then Open [train.py](Code/train.py) and change the dataset path to your local dataset path:
```bash
Line 62 "dataset": "D:\\dataset\\sheet metal dataset"  # Path containing 'fag', 'labels', 'MFInstseg_partition', and 'steps'
```
Finally, run: 
```bash
python train.py
```
After training starts, all logs and models will be saved automatically (the `output` folder will be created in the same directory where you run `train.py`):
- Logs: `output/<timestamp>/log.txt`
- Model checkpoints: `output/<timestamp>/weight_xx-epoch.pth`
- Best model: `output/<timestamp>/best_model.pth`
(Additionally, offline wandb logs will be stored in the `wandb/` folder.)
