#!/usr/bin/env bash
# Uncomment the following two lines to step through each command and to print
# the command being executed.
#set -x
#trap read debug

# Start the training job

# read common variables (between installation, training, and serving)
source variables.bash

cd ${APP_NAME}
pwd

# Set training job specific environment variables in `envs` variable(comma
# separated key-value pair). These key-value pairs are passed on to the
# training job when created.
ENV="TF_DATA_DIR=$TF_DATA_DIR,TF_CHECKPOINT_DIR=$TF_CHECKPOINT_DIR,TRAIN_DATA_FILE_PATH=$TRAIN_DATA_FILE_PATH,TF_MODEL_EXPORT_PATH=$TF_MODEL_EXPORT_PATH,TF_MODEL_VERSION=$TF_MODEL_VERSION,BOWORDS_PATH=$BOWORDS_PATH,TEST_DATA_PATH=$TEST_DATA_PATH,TEST_CLASSES_PATH=$TEST_CLASSES_PATH"

JOB=tf-${APP_NAME}job
ks generate ${JOB} ${JOB}

# Set tf training job specific environment params
ks param set ${JOB} image ${IMAGE}
ks param set ${JOB} envs ${ENV}


# Deploy and start training
ks apply ${KF_ENV} -c ${JOB}

# Check that the container is up and running
kubectl get pods -n ${NAMESPACE} | grep ${JOB}
