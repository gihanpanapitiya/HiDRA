"""
Training new HiDRA model
Requirement:
    expression.csv: Expression of all genes for all cell lines
    geneset.gmt: Gene set description file. Consists of Gene set name, Source, Gene set member 1, Gene set member 2, ...
    Training.csv: Training pair list. Consists of idx, Drug name, Cell line name, IC50 value for that pair.
    Validation.csv: Validation pair list. Consists of idx, Drug name, Cell line name, IC50 value for that pair.
    input_dir: The directory that includes input files.
"""

# Import basic packages
import numpy as np
import pandas as pd
import os
import argparse
import json

# Import keras modules
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, multiply, dot, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold

# Import rdkit
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols

# Set number of GPUs to use
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[6:], 'GPU')

# Fix the random seed
np.random.seed(5)


def parse_data(ic50_tr, ic50_val, expr, GeneSet_Dic, drugs):
    # Split data into a fold and divide expression data based on pathway
    X_train = []
    X_val = []
    expr = expr.transpose()

    for pathway in GeneSet_Dic.keys():
        df = expr[GeneSet_Dic[pathway]]
        X_train.append(df.loc[ic50_tr['Cell line name']])
        X_val.append(df.loc[ic50_val['Cell line name']])

    X_train.append(drugs.loc[ic50_tr['Drug name']])
    X_val.append(drugs.loc[ic50_val['Drug name']])

    return X_train, X_val


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
    model=Model(inputs=Input_for_model,outputs=Final)

    return model


def root_mean_squared_error(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def main():
    # Reading argument
    parser = argparse.ArgumentParser(description='HiDRA:Hierarchical Network for Drug Response Prediction with Attention-Training')

    # Options
    parser.add_argument('d', type=str, help='Directory of input data')
    parser.add_argument('f', type=int, help='The number of folds')
    parser.add_argument('e', type=int, help='The epoch in the training process')
    parser.add_argument('o', type=str, help='The output path that model file be stored')
    args = parser.parse_args()

    # Read Training and Validation files
    dir = args.d.rstrip('/')
    expr = pd.read_csv(dir + '/expression.csv', index_col=0)
    GeneSet_Dic = json.load(open(dir + '/geneset.json', 'r'))
    ic50 = pd.read_csv(dir + '/ic50.csv', index_col=0)
    drugs = pd.read_csv(dir + '/drug_fp.csv', index_col=0)

    # Training
    kf = KFold(n_splits=args.f, shuffle=True, random_state=1234)
    fn = 0

    for train_index, test_index in kf.split(ic50):
        ic50_tr = ic50.iloc[train_index]
        ic50_val = ic50.iloc[test_index]
        train_label, val_label = ic50_tr['log(IC50)'], ic50_val['log(IC50)']
        train_input, val_input = parse_data(ic50_tr, ic50_val, expr,
                                            GeneSet_Dic, drugs)

        model = Making_Model(GeneSet_Dic)
        #model.compile(loss='mean_squared_error',optimizer='adam')
        model.compile(loss=root_mean_squared_error,optimizer='adam')
        #print(model.summary())
        hist = model.fit(train_input, train_label, shuffle=True,
                         epochs=args.e, batch_size=1024, verbose=2,
                         validation_data=(val_input,val_label))

        result = model.predict(val_input)
        result = [y[0] for y in result]
        ic50_val['result'] = result
        ic50_val.to_csv(args.o + '_fold' + str(fn) + '.csv')
        fn += 1

    model.save(args.o + '.hdf5') #Save the model to the output directory


if __name__=="__main__":
    main()


