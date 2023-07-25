"""
Training the HiDRA model
Requirements:
    Run the shell script train.sh to preprocess data and set the CANDLE data dir and config files.
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


def Making_Model(GeneSet_Dic):
    # HiDRA model with keras
    # Drug-level network
    Drug_feature_length = 512
    Drug_Input = Input((Drug_feature_length,), dtype='float32', name='Drug_Input')

    Drug_Dense1 = Dense(256, name='Drug_Dense_1')(Drug_Input)
    Drug_Dense1 = BatchNormalization(name='Drug_Batch_1')(Drug_Dense1)
    Drug_Dense1 = Activation('relu', name='Drug_RELU_1')(Drug_Dense1)

    Drug_Dense2 = Dense(128, name='Drug_Dense_2')(Drug_Dense1)
    Drug_Dense2 = BatchNormalization(name='Drug_Batch_2')(Drug_Dense2)
    Drug_Dense2 = Activation('relu', name='Drug_RELU_2')(Drug_Dense2)

    # Drug network that will be used to attention network in the Gene-level network and Pathway-level network
    Drug_Dense_New1 = Dense(128, name='Drug_Dense_New1')(Drug_Input)
    Drug_Dense_New1 = BatchNormalization(name='Drug_Batch_New1')(Drug_Dense_New1)
    Drug_Dense_New1 = Activation('relu', name='Drug_RELU_New1')(Drug_Dense_New1)

    Drug_Dense_New2 = Dense(32, name='Drug_Dense_New2')(Drug_Dense_New1)
    Drug_Dense_New2 = BatchNormalization(name='Drug_Batch_New2')(Drug_Dense_New2)
    Drug_Dense_New2 = Activation('relu', name='Drug_RELU_New2')(Drug_Dense_New2)

    #Gene-level network
    GeneSet_Model=[]
    GeneSet_Input=[]

    #Making networks whose number of node is same with the number of member gene in each pathway    
    for GeneSet in GeneSet_Dic.keys():
        Gene_Input=Input(shape=(len(GeneSet_Dic[GeneSet]),),dtype='float32', name=GeneSet+'_Input')
        Drug_effected_Model_for_Attention=[Gene_Input]
        #Drug also affects to the Gene-level network attention mechanism
        Drug_Dense_Geneset=Dense(int(len(GeneSet_Dic[GeneSet])/4)+1,dtype='float32',name=GeneSet+'_Drug')(Drug_Dense_New2)
        Drug_Dense_Geneset=BatchNormalization(name=GeneSet+'_Drug_Batch')(Drug_Dense_Geneset)
        Drug_Dense_Geneset=Activation('relu', name=GeneSet+'Drug_RELU')(Drug_Dense_Geneset)
        Drug_effected_Model_for_Attention.append(Drug_Dense_Geneset) #Drug feature to attention layer

        Gene_Concat=concatenate(Drug_effected_Model_for_Attention,axis=1,name=GeneSet+'_Concat')
        #Gene-level attention network
        Gene_Attention = Dense(len(GeneSet_Dic[GeneSet]), activation='tanh', name=GeneSet+'_Attention_Dense')(Gene_Concat)
        Gene_Attention=Activation(activation='softmax', name=GeneSet+'_Attention_Softmax')(Gene_Attention)
        Attention_Dot=dot([Gene_Input,Gene_Attention],axes=1,name=GeneSet+'_Dot')
        Attention_Dot=BatchNormalization(name=GeneSet+'_BatchNormalized')(Attention_Dot)
        Attention_Dot=Activation('relu',name=GeneSet+'_RELU')(Attention_Dot)

	#Append the list of Gene-level network (attach new pathway)
        GeneSet_Model.append(Attention_Dot)
        GeneSet_Input.append(Gene_Input)

    Drug_effected_Model_for_Attention=GeneSet_Model.copy()

    #Pathway-level network
    Drug_Dense_Sample=Dense(int(len(GeneSet_Dic)/16)+1,dtype='float32',name='Sample_Drug_Dense')(Drug_Dense_New2)
    Drug_Dense_Sample=BatchNormalization(name=GeneSet+'Sample_Drug_Batch')(Drug_Dense_Sample)
    Drug_Dense_Sample=Activation('relu', name='Sample_Drug_ReLU')(Drug_Dense_Sample)    #Drug feature to attention layer
    Drug_effected_Model_for_Attention.append(Drug_Dense_Sample)
    GeneSet_Concat=concatenate(GeneSet_Model,axis=1, name='GeneSet_Concatenate')
    Drug_effected_Concat=concatenate(Drug_effected_Model_for_Attention,axis=1, name='Drug_effected_Concatenate')
    #Pathway-level attention
    Sample_Attention=Dense(len(GeneSet_Dic.keys()),activation='tanh', name='Sample_Attention_Dense')(Drug_effected_Concat)
    Sample_Attention=Activation(activation='softmax', name='Sample_Attention_Softmax')(Sample_Attention)
    Sample_Multiplied=multiply([GeneSet_Concat,Sample_Attention], name='Sample_Attention_Multiplied')
    Sample_Multiplied=BatchNormalization(name='Sample_Attention_BatchNormalized')(Sample_Multiplied)
    Sample_Multiplied=Activation('relu',name='Sample_Attention_Relu')(Sample_Multiplied)
    
    #Making input list
    Input_for_model=[]
    for GeneSet_f in GeneSet_Input:
        Input_for_model.append(GeneSet_f)
    Input_for_model.append(Drug_Input)

    #Concatenate two networks: Pathway-level network, Drug-level network
    Total_model=[Sample_Multiplied,Drug_Dense2]
    Model_Concat=concatenate(Total_model,axis=1, name='Total_Concatenate')

    #Response prediction network
    Concated=Dense(128, name='Total_Dense')(Model_Concat)
    Concated=BatchNormalization(name='Total_BatchNormalized')(Concated)
    Concated=Activation(activation='relu', name='Total_RELU')(Concated)

    Final=Dense(1, name='Output')(Concated)
    Activation(activation='sigmoid', name='Sigmoid')(Final)
    model=Model(inputs=Input_for_model,outputs=Final)

    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def run(gParameters):
    batch_size = gParameters['batch_size']
    epochs = gParameters['epochs']
    optimizer = gParameters['optimizer']
    loss = gParameters['loss']
    output_dir = gParameters['output_dir']

#    y_col_name = "auc1"
#    source_data_name = "CCLE"
    y_col_name =  "auc"
    source_data_name = "CTRPv2"
    split = 0

    expr = pd.read_csv(data_dir + '/ge_' + source_data_name + '.csv', index_col=0)
    GeneSet_Dic = json.load(open(data_dir + '/geneset.json', 'r'))
    drugs = pd.read_csv(data_dir + '/ecfp2_' + source_data_name + '.csv', index_col=0)

    # Training
    file_start = data_dir + '/rsp_' + source_data_name + '_split' + str(split)
    auc_tr = pd.read_csv(file_start + '_train.csv', index_col=0)
    auc_val = pd.read_csv(file_start + '_val.csv', index_col=0)
    train_label = auc_tr[y_col_name]
    val_label = auc_val[y_col_name]
    train_input = parse_data(auc_tr, expr, GeneSet_Dic, drugs)
    val_input = parse_data(auc_val, expr, GeneSet_Dic, drugs)

    model_saver = ModelCheckpoint(output_dir + '/model.h5', monitor='val_loss',
                                  save_best_only=True, save_weights_only=False)

    callbacks = [model_saver]

    model = Making_Model(GeneSet_Dic)
    model.compile(loss=loss, optimizer=optimizer)
    history = model.fit(train_input, train_label, shuffle=True,
                     epochs=epochs, batch_size=batch_size, verbose=2,
                     validation_data=(val_input,val_label),
                     callbacks=callbacks)

    result = model.predict(val_input)
    result = [y[0] for y in result]
    auc_val['result'] = result
    auc_val.to_csv(output_dir + '/val_results.csv')
    model.save(output_dir + '/model.hdf5')

    pcc = auc_val.corr(method='pearson').loc[y_col_name, 'result']
    scc = auc_val.corr(method='spearman').loc[y_col_name, 'result']
    rmse = ((auc_val[y_col_name] - auc_val['result']) ** 2).mean() ** .5
    val_loss = np.min(history.history['val_loss'])

    val_scores = {'val_loss':val_loss, 'pcc':pcc, 'scc':scc, 'rmse':rmse}

    with open(output_dir + "/scores.json", "w", encoding="utf-8") as f:
        json.dump(val_scores, f, ensure_ascii=False, indent=4)

    print('IMPROVE_RESULT val_loss:\t' + str(val_loss))

    return history


def main():
    gParameters = initialize_parameters()
    history = run(gParameters)


if __name__ == '__main__':
    main()
    try:
        K.clear_session()

    except AttributeError:
        pass
