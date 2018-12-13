# read common variables (between installation, training, and serving)
source variables.bash

# define new variables
TF_MODEL_SERVER_HOST=`kubectl describe pod helix -n ${NAMESPACE} | grep IP | sed -E 's/IP:[[:space:]]+//'`
MNIST_SERVING_IP=`kubectl -n ${NAMESPACE} get svc/helix --output=jsonpath={.spec.clusterIP}`

# move to ksonnet project
cd ${APP_NAME}
ks env set ${KF_ENV} --namespace ${NAMESPACE}

# generate from local template to use NodePort
ks generate tf-mnist-client tf-helix-client --mnist_serving_ip=${TF_MODEL_SERVER_HOST} --image=${CLIENT_IMAGE}

ks apply ${KF_ENV} -c tf-helix-client

# ensure that all pods are running in the namespace set in variables.bash.
kubectl get pods -n ${NAMESPACE}

# get nodePort
NODE_PORT=`kubectl get svc/tf-helix-client -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].nodePort}'`
CLUSTER_IP=`hostname -I | awk '{print $1}'`
echo "Visit your webapp at ${CLUSTER_IP}:${NODE_PORT}"
