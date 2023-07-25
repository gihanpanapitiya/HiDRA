#!/bin/bash
  
#########################################################################
### THIS IS A TEMPLATE FILE. SUBSTITUTE #PATH# WITH THE MODEL EXECUTABLE.
#########################################################################


# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 TRAIN_DATA_SOURCE
# arg 4 SPLIT
# arg 5 TEST_DATA_SOURCE
# arg 6 CANDLE_CONFIG

### Path to your CANDLEized model's main Python script###
CANDLE_MODEL=csa_training.py
DATA_PREPROCESSOR=csa_feature_gen.py
CANDLE_TEST=csa_predict.py

if [ $# -lt 5 ] ; then
        echo "Illegal number of parameters"
        echo "CUDA_VISIBLE_DEVICES, CANDLE_DATA_DIR, CANDLE_CONFIG, TRAIN_DATA_SOURCE, SPLIT, AND TEST_DATA_SOURCE are required"
        exit -1
fi

if [ $# -eq 5 ] ; then
        CUDA_VISIBLE_DEVICES=$1 ; shift
        CANDLE_DATA_DIR=$1 ; shift
        TRAIN_DATA_SOURCE=$1 ; shift
        SPLIT=$1 ; shift
        TEST_DATA_SOURCE=$1 ; shift

        CMD="python ${CANDLE_MODEL}"
        echo "CMD = $CMD"

elif [ $# -ge 6 ] ; then
        CUDA_VISIBLE_DEVICES=$1 ; shift
        CANDLE_DATA_DIR=$1 ; shift
        TRAIN_DATA_SOURCE=$1 ; shift
        SPLIT=$1 ; shift
        TEST_DATA_SOURCE=$1 ; shift

        # if original $3 is a file, set candle_config and passthrough $@
        if [ -f $CANDLE_DATA_DIR/$1 ] ; then
		echo "$1 is a file"
                CANDLE_CONFIG=$1 ; shift
                CMD="python ${CANDLE_MODEL} --config_file $CANDLE_CONFIG $@"
                echo "CMD = $CMD $@"

        # else passthrough $@
        else
		echo "$1 is not a file"
                CMD="python ${CANDLE_MODEL} $@"
                echo "CMD = $CMD"

        fi
fi

# Set data preprocessing command
DPP_CMD="python ${DATA_PREPROCESSOR}"
TEST_CMD="python ${CANDLE_TEST}"

# Display runtime arguments
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"

# Set up environmental variables and execute model
EXE_DIR=$(dirname ${CANDLE_MODEL})
cd $EXE_DIR

echo "running command ${DPP_CMD}"
time CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $DPP_CMD
echo "running command ${CMD}"
time CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD

time CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $TEST_CMD

# Change into directory and execute tests
# cd ${MODEL_DIR}


# model data processing and loading
# python HiDRA_FeatureGeneration.py

# model training
# python HiDRA_training.py

# Check if successful
exit 0


