#!/usr/local/env bash

## Namespace to be used in k8s cluster for your application
NAMESPACE=sandbox

MNIST=mnist

## Ksonnet app name
APP_NAME=helix

## GITHUB version for official kubeflow components
KUBEFLOW_GITHUB_VERSION=v0.3.0-rc.3

## GITHUB version for ciscoai components
CISCOAI_GITHUB_VERSION=master

## Ksonnet environment name
KF_ENV=nativek8s

## Name of the NFS Persistent Volume
NFS_PVC_NAME=nfs

## Used in training.bash
# Enviroment variables for mnist training job (See mnist_model.py)
TRAIN_DATA_DIR_LOCAL=/traindata
TRAIN_DATA_PATH=/mnt${TRAIN_DATA_DIR_LOCAL}

TF_DATA_DIR=/mnt/data
TF_CHECKPOINT_DIR=/mnt/checkpoint
NFS_MODEL_PATH=/mnt/export
TF_EXPORT_DIR=${NFS_MODEL_PATH}

# If you want to use your own image,
# make sure you have a dockerhub account and change
# DOCKER_BASE_URL and IMAGE below.
DOCKER_BASE_URL=chrlindn/sjc-team-2
IMAGE=${DOCKER_BASE_URL}:${APP_NAME}


#docker build . --no-cache  -f Dockerfile -t ${IMAGE}
#docker push ${IMAGE}

# Used in portf.bash and webapp.bash
# If using without an application, source this file before using
PORT=9000
export TF_MODEL_SERVER_PORT=${PORT}

# Used in webapp.bash
DOCKER_HUB=gcr.io
DOCKER_USERNAME=cpsg-ai-demo
DOCKER_IMAGE=mnist-client
WEBAPP_FOLDER=webapp

