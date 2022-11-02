# import packages
import numpy as np
import pandas as pd
import json
from scipy.stats import zscore

# Loading cell line expression data
expression_df = pd.read_csv('raw_data/Cell_line_RMA_proc_basalExp.txt',
                            sep='\t', index_col=0)

expression_df = expression_df.drop(columns='GENE_title')
expression_df.columns = [x.strip('DATA.') for x in expression_df.columns]
expression_df.index = expression_df.index.rename('Gene_Symbol')

# Loading cell line metadata
cl_info = pd.read_csv('raw_data/TableS1E.csv', sep=',', index_col=0,
                      skiprows=[0,1], usecols=[0,1,2], dtype=str)

cl_info.columns = ['Cell line name','COSMIC identifier']

# Excluding cell lines whose expression values are not valid
cosmic_list = [x for x in cl_info['COSMIC identifier']]
drop_list = [x for x in expression_df.columns if x not in cosmic_list]
expression_df = expression_df.drop(columns=drop_list)

# Converting COSMIC identifier into Cell line name
cl_name = [x for x in cl_info['Cell line name']]
name_dict = dict(zip(cosmic_list, cl_name))
expression_df.columns = [name_dict[x] for x in expression_df.columns]

# Excluding expressions that are not gene
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
        # 'ES3' is an example cell line name
        genes = [x for x in line[2:] if x in expression_df['ES3']]
        GeneSet.extend(genes)
        GeneSet_Dic[pathway] = genes

        if pathway=='KEGG_GLYCOLYSIS_GLUCONEOGENESIS':
            print(line)
            print(genes)

GeneSet = set(GeneSet)
print(str(len(GeneSet)) + ' KEGG genes found in expression file.')

# Remove genes not present in pathways from expression data
drop_rows = [x for x in expression_df.index if x not in GeneSet]
expression_df = expression_df.drop(drop_rows)

# Save trimmed and normalized expression file and pathway/gene dictionary
expression_df.to_csv('new_processed_data/expression.csv')
json.dump(GeneSet_Dic, open("new_processed_data/geneset.json", 'w'))

# Get alternate drug names for conversion in fingerprint file
drug_alias = {}

with open('raw_data/drug_alias.txt', 'r') as alias_file:
    for line in alias_file:
        old_name, new_name = line.rstrip().split('\t')
        drug_alias[old_name] = new_name

drug_fng = pd.read_csv('raw_data/drug.csv', sep=',', dtype=str, index_col=0)
new_rows = [drug_alias[x] if x in drug_alias else x for x in drug_fng.index]
drug_fng.index = new_rows
drug_fng.to_csv('new_processed_data/drug_fp.csv')

# Read IC50 table and remove drugs/cell lines not present in metadata
ic50_tbl = pd.read_csv('raw_data/TableS4A.csv', sep=',', dtype=str)
drop_columns = [x for x in ic50_tbl.columns[1:] if x not in drug_fng.index]
ic50_tbl = ic50_tbl.drop(columns=drop_columns)
drop_rows = [x for x in ic50_tbl.index if ic50_tbl['Sample Names'][x] not in expression_df.columns]
ic50_tbl = ic50_tbl.drop(index=drop_rows)

# Convert IC50 table to drug/cell line IC50 pairs
ic50_tbl = ic50_tbl.melt(id_vars=['Sample Names'])
ic50_tbl = ic50_tbl.dropna()
ic50_tbl.columns = ['Cell line name', 'Drug name', 'log(IC50)']
ic50_tbl.reset_index(drop=True, inplace=True)
print(ic50_tbl)
ic50_tbl.to_csv('new_processed_data/ic50.csv')
