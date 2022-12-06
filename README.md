## Overview
HiDRA (Hierarchical Network for Drug Response Prediction with Attention) is a drug response prediction network published published by [Iljung Jim and Hojung Nam] (https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00706). Here we have brought it into the IMPROVE framework.

HiDRA consists of four smaller networks: a drug feature encoding network, which takes SMILES strings converted to Morgan fingerprints; a set of gene-level networks which encode expression data for genes in each pathway; a pathway-level network that takes in output of the individual gene-level networks; and a prediction network that uses drug encodings and pathway output to predict ln(IC50). Each sub-network consists of two dense layers and an attention module (tanh + softmax activation).

## Dependencies
- CANDLE-ECP (develop branch)
- tensorflow-gpu (2.4.2)
- scikit-learn (0.24.2)
- pandas (1.1.5)
- openpyxl (3.0.9)

## Data Required

Four data files are required:
- A KEGG pathway file, where each line is a pathway name followed by a list of gene symbols (included in the repository as geneset.gmt)
- A gene expression file with gene symbols as rows and cell lines as columns. Ideally this file will contain all or most genes present in geneset.gmt. The original publication uses GDSC1000 data which can be downloaded [here](https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip)
- A 512-bit Morgan fingerprint drug file. The original publication converts SMILES strings for 235 drugs to fingerprints using RDKit.
- A response table of cancer/drug pairs. The original publication uses GDSC1000 IC50 values which can be downloaded [here](https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/suppData/TableS4A.xlsx)

## Data Preprocessing

Data preprocessing includes the following steps:
- Download gene expression, drug fingerprint, and response pair data from FTP site
- Transform gene expressions to z-scores
- Load KEGG pathway data and remove all genes from the expression file not present in a pathway
- Save KEGG pathways as a JSON file
- Load response data and remove unnecessary columns
	
If using the original GDSC1000 data, a few extra reformatting steps are needed:
- Load expression dataset, remove genes without valid symbols, and change COSMIC identifiers to sample names to be consistent with response data
- Load response data, reformat pivot table into a list of pairs, and remove any entries with no fingerprint or expression data
	
Preprocessing steps are performed in HiDRA_FeatureGeneration.py
	
## Model Training

Training hyperparameters are specified in hidra_default_model.txt. HiDRA_training.py splits data into training and validation, reformats expression data by pathway, and builds the model. It produces an hdf5 model file and validation results.

HiDRA_predict.py also requires the hidra_default_model.txt parameters file, reformats test data into pathways, and saves predictions.

train.sh sets which GPU to use, sets the data directory, and runs preprocessing and training scripts. 
