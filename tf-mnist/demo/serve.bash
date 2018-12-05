#!/usr/bin/env bash

# Start TF serving on the trained results

# Read variables
source variables.bash

cd ${APP_NAME}
pwd

#2. Create namespace if not present

ks generate tf-serving tfserving --name=${APP_NAME}

# Set tf serving job specific environment params
ks param set tfserving modelPath ${NFS_MODEL_PATH}
ks param set tfserving modelStorageType nfs
ks param set tfserving nfsPVC ${NFS_PVC_NAME}

# Deploy and start serving
ks apply ${KF_ENV} -c tfserving
