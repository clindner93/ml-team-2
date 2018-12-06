#!/usr/bin/env bash

#1. Read variables
source variables.bash

#2. Find model IP
cd ${APP_NAME}
MNIST_SERVING_IP=`kubectl -n ${NAMESPACE} get svc/mnist --output=jsonpath={.spec.clusterIP}`
   echo "MNIST_SERVING_IP is ${MNIST_SERVING_IP}"

#3. Run
ks generate tf-mnist-client-local tf-mnist-client --mnist_serving_ip=${MNIST_SERVING_IP} --image=${CLIENT_IMAGE}
ks apply ${KF_ENV} -c tf-mnist-client
