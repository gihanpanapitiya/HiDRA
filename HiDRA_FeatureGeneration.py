def main():
    import numpy as np
    import pandas as pd
    import os
    import json
    import argparse
    from scipy.stats import zscore
    import candle
#    os.environ['CANDLE_DATA_DIR'] = 'raw_data/'
    data_dir = os.environ['CANDLE_DATA_DIR'].rstrip('/')

    dir_url = 'ftp://ftp.mcs.anl.gov/pub/candle/public/improve/reproducability/July2020/data.gdsc1/'
    candle.file_utils.get_file('ge_gdsc1.csv', dir_url + 'ge_gdsc1.csv')
    candle.file_utils.get_file('ecfp2_gdsc1.csv', dir_url + 'ecfp2_gdsc1.csv')
    candle.file_utils.get_file('rsp_gdsc1.csv', dir_url + 'rsp_gdsc1.csv')

#    if not os.path.isdir('preprocessed_data/'):
#        os.mkdir('preprocessed_data/')

    # Loading cell line expression data
    expression_df = pd.read_csv(data_dir + '/common/ge_gdsc1.csv', index_col=0)

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
    expression_df.to_csv(data_dir + '/ge_gdsc1.csv')
    json.dump(GeneSet_Dic, open(data_dir + '/geneset.json', 'w'))

    # Load drug fingerprints file
    drug_fng = pd.read_csv(data_dir + '/common/ecfp2_gdsc1.csv', index_col=0)
    drug_fng.to_csv(data_dir + '/ecfp2_gdsc1.csv')

    # Read IC50 table and remove drugs/cell lines not present in metadata
    ic50_tbl = pd.read_csv(data_dir + '/common/rsp_gdsc1.csv')
    drop_columns = ['SOURCE', 'IC50', 'EC50', 'EC50se', 'R2fit', 'Einf', 'HS',
                    'AAC1', 'AUC1', 'DSS1', 'AUC_bin']
    ic50_tbl = ic50_tbl.drop(columns=drop_columns)
    ic50_tbl.to_csv(data_dir + '/rsp_gdsc1.csv')


if __name__=="__main__":
    main()
