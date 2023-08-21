import numpy as np
import pandas as pd
import os
import json
import argparse
from scipy.stats import zscore
import candle
import sys
from data_utils import Downloader, DataProcessor, add_smiles
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf


def main(args):

    dw = Downloader(args)
    data_path= args.data_path
    metric=args.metric
    data_type=args.data_type
    split_id = args.data_split_id
    source_data_name= data_type
 
    dw.download_candle_data(data_type=data_type, split_id=split_id, data_dest=data_path)
    proc = DataProcessor(args.data_version)


#---------------------
    # train_tmp = proc.load_drug_response_data(data_path=data_path, 
    #                                     data_type=data_type, split_id=split_id, split_type='train', response_type=metric)

    # val_tmp = proc.load_drug_response_data(data_path=data_path, 
    #                                     data_type=data_type, split_id=split_id, split_type='val', response_type=metric)

    # test_tmp = proc.load_drug_response_data(data_path=data_path, 
    #                                     data_type=data_type, split_id=split_id, split_type='test', response_type=metric)


    # smiles_df = proc.load_smiles_data(data_dir=data_path)

    # train_tmp = add_smiles(smiles_df, train_tmp, metric)
    # val_tmp = add_smiles(smiles_df, val_tmp, metric)
    # test_tmp = add_smiles(smiles_df, test_tmp, metric)

    # df_all = pd.concat([train_tmp, val_tmp, test_tmp], axis=0)
    # df_all.reset_index(drop=True, inplace=True)
    # train, val, test = split_df(df_all, args.data_split_seed)
#---------------------


    # sys.path.append('../benchmark-dataset-generator')
    # import improve_utils

    # data_dir = os.environ['CANDLE_DATA_DIR'].rstrip('/')
    # split = os.environ['SPLIT'].rstrip('/')
    # train_source = os.environ['TRAIN_SOURCE'].rstrip('/')

    # y_col_name = "auc1"
    # source_data_name = "CCLE"
    # y_col_name = "auc"

    # if isinstance(split, int):
    #     split += 1
    #     rs_tr = improve_utils.load_single_drug_response_data_v2(
    #     source=source_data_name,
    #     split_file_name=f"{train_source}_split_{split}_train.txt",
    #     y_col_name=y_col_name)

    #     rs_vl = improve_utils.load_single_drug_response_data_v2(
    #     source=source_data_name,
    #     split_file_name=f"{train_source}_split_{split}_val.txt",
    #     y_col_name=y_col_name)

    #     rs_te = improve_utils.load_single_drug_response_data_v2(
    #     source=source_data_name,
    #     split_file_name=f"{train_source}_split_{split}_test.txt",
    #     y_col_name=y_col_name)

    #     rs_tr.to_csv(data_dir + '/rsp_' + train_source + '_split' + str(split) + '_train.csv')
    #     rs_vl.to_csv(data_dir + '/rsp_' + train_source + '_split' + str(split) + '_val.csv')
    #     rs_te.to_csv(data_dir + '/rsp_' + train_source + '_split' + str(split) + '_test.csv')

    # else:
    #     rs_te = improve_utils.load_single_drug_response_data_v2(
    #     source=source_data_name,
    #     split_file_name=f"{train_source}_all.txt",
    #     y_col_name=y_col_name)

    #     rs_te.to_csv(data_dir + '/rsp_' + source_data_name + '_all.csv')


    # Loading cell line expression data
    expression_df = proc.load_gene_expression_data(data_dir=data_path, gene_system_identifier="Gene_Symbol")
    expression_df = expression_df.transpose()
    expression_df.index = [x.strip('ge_') for x in expression_df.index]
    expression_df.index = expression_df.index.rename('Gene_Symbol')

    # Transform expression values into z-score
    expression_df = expression_df.apply(zscore)

    # Loading Gene Sets
    GeneSetFile = 'raw_data/geneset.gmt'
    GeneSet = []
    GeneSet_Dic = {}

    with open(GeneSetFile) as f:
        for line in f:
            line = line.rstrip().split('\t')
            pathway = line[0]

            # Use only genes present in expression data
            example_cell_line = expression_df.columns[5]
            genes = [x for x in line[2:] if x in expression_df[example_cell_line]]
            GeneSet.extend(genes)
            GeneSet_Dic[pathway] = genes

    GeneSet = set(GeneSet)
    print(str(len(GeneSet)) + ' KEGG genes found in expression file.')

    # Remove genes not present in pathways from expression data
    drop_rows = [x for x in expression_df.index if x not in GeneSet]
    expression_df = expression_df.drop(drop_rows)

    # Save trimmed and normalized expression file and pathway/gene dictionary
    expression_df.to_csv(data_path + '/ge_' + source_data_name + '.csv')
    json.dump(GeneSet_Dic, open(data_path + '/geneset.json', 'w'))

    # Load drug fingerprints file
    drug_fng = proc.load_morgan_fingerprint_data(data_dir=data_path)
    drug_fng.to_csv(data_path + '/ecfp2_' + source_data_name + '.csv')


import argparse
if __name__=="__main__":


    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config.yaml')
    # args = parser.parse_args()

    # if args.config:  # args priority is higher than yaml
    #     args_ = OmegaConf.load(args.config)
    #     OmegaConf.resolve(args_)
    #     args=args_

    parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does')
    parser.add_argument('--metric',  default='auc', help='')
    parser.add_argument('--run_id',  default=0, help='')
    # parser.add_argument('--epochs',  default=1, help='')
    parser.add_argument('--data_split_seed',  default=1, help='')
    parser.add_argument('--out_dir',  default='./Output', help='')
    parser.add_argument('--data_path',  default='./Data', help='')
    parser.add_argument('--data_version',  default='benchmark-data-pilot1', help='')
    parser.add_argument('--data_type',  default='CCLE', help='')
    parser.add_argument('--data_split_id',  default=0, help='')
    # parser.add_argument('--encoder_type',  default='gnn', help='')
    args = parser.parse_args()

    main(args)

