"""
Predict with the HiDRA model

Requirements:
    model.hdf5: Pre-trained model file
"""
# Import basic packages
import numpy as np
import pandas as pd
import os
import sys
import argparse
import json

# Import keras modules
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, multiply, dot, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold, train_test_split

file_path = os.path.dirname(os.path.realpath(__file__))
import candle
data_dir = os.environ['CANDLE_DATA_DIR'].rstrip('/')
#config_file = os.environ['CANDLE_CONFIG'].rstrip('/')

additional_definitions = []

required = [
    'epochs',
    'batch_size',
    'optimizer',
    'loss',
    'output_dir'
]


class HIDRA(candle.Benchmark):
    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


if K.backend() == 'tensorflow' and 'NUM_INTRA_THREADS' in os.environ:
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
                                            intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS'])))
    K.set_session(sess)


def initialize_parameters():
    hidra_common = HIDRA(file_path,
        'hidra_default_model.txt',
        'keras',
        prog='HiDRA_candle',
        desc='HiDRA run'
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(hidra_common)

    return gParameters


def parse_data(ic50, expr, GeneSet_Dic, drugs):
    # Divide expression data based on pathway
    X = []
    expr = expr.transpose()

    for pathway in GeneSet_Dic.keys():
        df = expr[GeneSet_Dic[pathway]]
        X.append(df.loc[ic50['improve_sample_id']])

    X.append(drugs.loc[ic50['improve_chem_id']])

    return X


def run(gParameters):
    batch_size = gParameters['batch_size']
    epochs = gParameters['epochs']
    optimizer = gParameters['optimizer']
    loss = gParameters['loss']
    output_dir = gParameters['output_dir']

    # y_col_name = "auc1"
    # source_data_name = "CCLE"
    y_col_name =  "ic50"
    source_data_name = "GDSCv1"
    split = 0

    expr = pd.read_csv(data_dir + '/ge_' + source_data_name + '.csv', index_col=0)
    GeneSet_Dic = json.load(open(data_dir + '/geneset.json', 'r'))
    drugs = pd.read_csv(data_dir + '/ecfp2_' + source_data_name + '.csv', index_col=0)

    # Training
    file_start = data_dir + '/rsp_' + source_data_name + '_split' + str(split)
    auc_test = pd.read_csv(file_start + '_test.csv', index_col=0)
    test_label = auc_test[y_col_name]
    test_input = parse_data(auc_test, expr, GeneSet_Dic, drugs)

    model = load_model(output_dir + '/model.hdf5')

    result = model.predict(test_input)
    result = [y[0] for y in result]
    auc_test['result'] = result
    auc_test.to_csv(output_dir + '/test_results.csv')


def main():
    gParameters = initialize_parameters()
    run(gParameters)


if __name__ == '__main__':
    main()
    try:
        K.clear_session()

    except AttributeError:
        pass
