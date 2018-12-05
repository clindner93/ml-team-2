# Demo

This directory contains instructions and scripts needed for a demo on local (on-prem) Kubernetes clusters.

## Prerequisites

- Start from a fresh Kubernetes cluster
- Install Ksonnet >= 0.11.0

## Preparation

- Have an environment variable of your github token (NOTE: replace ${YOUR_GITHUB_TOKEN} with your own token value)

```
export GITHUB_TOKEN=${YOUR_GITHUB_TOKEN}
```

- Clone demo repository

```
git clone https://github.com/CiscoAI/kubeflow-workflows.git
cd kubeflow-workflows
git checkout -b local-demo origin/local-demo
cd tf-mnist/demo
```

## Installation

```
source ./variables.bash
./install.bash
```

## Bring up Jupyter Notebook

- Check the port of Jupyter Hub

```
kubectl get svc -n ${NAMESPACE} -o wide | grep tf-hub-lb
```

- Access the server and port where your Jupyter Hub service is located.

- Create a Jupyter Notebook server from the UI
  - For GPU use image `gcr.io/kubeflow-images-public/tensorflow-1.8.0-notebook-gpu:v20180523-2a68f293`

## Run training

- Find your Jupyter Notebook pod

```
kubectl get pods -n ${NAMESPACE} | grep jupyter-
```

- Create training script (NOTE: replace ${YOUR_NOTEBOOK_POD} with your pod name)

```
kubectl cp ./train.ipynb kubeflow/${YOUR_NOTEBOOK_POD}:/home/jovyan/work/train.ipynb 
```

- Go to the Notebook and run the script

- Copy out the trained model

```
mkdir ./model
kubectl cp kubeflow/jupyter-cisco:/home/jovyan/work/model/export/mnist ./model/mnist
```

## Run Serving

- Find your NFS server pod

```
kubectl get pods -n ${NAMESPACE} | grep nfs-server-
```

- Copy model to NFS server (NOTE: replace ${YOUR_NFS_SERVER_POD} with your NFS server pod name)

```
kubectl cp ./model/mnist kubeflow/${YOUR_NFS_SERVER_POD}:/exports/mnist
```

- Run serving

```
./serve.bash
```

## Run App

- Deploy

```
./app.bash
```

- Find your app's server and port

```
kubectl get svc -n ${NAMESPACE} -o wide | grep tf-mnist-client
```

- Access the app from browser using the port above
