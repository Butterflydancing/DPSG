# DPSG-model
This repository introduces the use of DPSG model.


## 0. Requirements
This repository is coded in `python==3.8.5`.
Please run the following command to install the other requirements from `requirements.txt`.
```
pip install -r requirements.txt
```

## 1. Dataset Collection
Three datasets, PolitiFact, GossipCop and PHEME are used. 

### Collect PolitiFact and GossipCop raw data
Please follow the steps in [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet).
### Collect PHEME raw data
Run the following on command line to collect PHEME under `data/`, unzip it, and rename it.
```
cd data
wget -O PHEME.tar.bz2 "https://ndownloader.figshare.com/files/6453753"
tar -vxf PHEME.tar.bz2
mv pheme-rnr-dataset PHEME
cd ..
```
The zipped file is only 25M and can be downloaded in around 3 minutes.

## 2. Data Pre-processing
Data pre-processing includes image fetching, image encoding, text encoding, graph construction, and the extraction of other features.

We also provide a processed version of the three datasets via this [Google Drive link](https://drive.google.com/drive/folders/118YqbFodriKwQ5LUCiQT5mObIxsOrZZD?usp=drive_link).

You can download them and place them under `data/processed_data/FakeNewsNet/GossipCop/`, `data/processed_data/FakeNewsNet/PolitiFact/` and `data/processed_data/PHEME/` respectively in align with the data paths described in training_and_evaluation scripts.

## 3. Model Training and evaluation
You can find pre-trained models in models `models/pre-trained/` .

Run scripts in `models/train_and_evaluation/` to train the model and get the evaluation results. 

The algorithm has been patented.
