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

    dir_url = 'ftp://ftp.mcs.anl.gov/pub/candle/public/improve/hidra/raw_data/'
    files = ['Cell_line_RMA_proc_basalExp.txt', 'TableS1E.csv', 'TableS4A.csv',
             'drug.csv', 'drug_alias.txt']

    for fname in files:
        candle.file_utils.get_file(fname, dir_url + fname)

    # Loading cell line expression data
    expression_df = pd.read_csv(data_dir + '/common/' + files[0], sep='\t', index_col=0)
    cl_metadata = pd.read_csv(data_dir + '/common/' + files[1], header=2, dtype=str)

    # Change COSMIC IDs for cell line names and remove extra columns
    cl_dict = dict(zip(cl_metadata['COSMIC identifier'], cl_metadata['Sample Name']))

    cell_lines = [x for x in expression_df.columns]
    del_col = ['GENE_title']
    new_col = []

    for i in range(1,len(cell_lines)):
        c = cell_lines[i].split('.')[1]

        if c in cl_dict:
            new_col.append(cl_dict[c])

        else:
            del_col.append(cell_lines[i])

    expression_df = expression_df.drop(del_col, axis=1)
    expression_df.columns = new_col

    # Remove genes listed as 'NaN'
    expression_df = expression_df[expression_df.index.notnull()]

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
    expression_df.to_csv(data_dir + '/ge_GDSC1000.csv')
    json.dump(GeneSet_Dic, open(data_dir + '/geneset.json', 'w'))

    # Load drug fingerprints file
    drug_fng = pd.read_csv(data_dir + '/common/' + files[3], index_col=0)
    drug_alias = pd.read_csv(data_dir + '/common/' + files[4], sep='\t', header=None, names=['Old File', 'New File'])

    # Update drug names
    dr_dict = dict(zip(drug_alias['Old File'], drug_alias['New File']))

    for x in drug_fng.index:
        if x in dr_dict:
            drug_fng = drug_fng.rename(index={x: dr_dict[x]})

    drug_fng.to_csv(data_dir + '/ecfp2_GDSC1000.csv')

    # Read IC50 table and remove drugs/cell lines not present in metadata
    response = pd.read_csv(data_dir + '/common/' + files[2])

    # Reformat response data
    response = response.melt(id_vars=['Sample Names'])
    response = response.dropna()
    response.columns = ['CancID', 'DrugID', 'IC50']

    # Remove response data without CL or drug data
    idx_drop = []

    for idx in response.index:
        if response['DrugID'][idx] not in drug_fng.index:
            idx_drop.append(idx)

        elif response['CancID'][idx] not in expression_df.columns:
            idx_drop.append(idx)

    response = response.drop(index=idx_drop)

    response.to_csv(data_dir + '/rsp_GDSC1000.csv')


if __name__=="__main__":
    main()
