#!/usr/local/env bash

## Namespace to be used in k8s cluster for your application
NAMESPACE=team-sjc-2

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
NFS_PVC_NAME=team-sjc-2-nfs-pvc

## Used in training.bash
# Enviroment variables for mnist training job (See mnist_model.py)
TRAIN_DATA_FILE_LOCAL=/traindata/TrainingDataset.json
TRAIN_DATA_FILE_PATH=/mnt${TRAIN_DATA_FILE_LOCAL}

TF_DATA_DIR=/mnt/data
TF_CHECKPOINT_DIR=/mnt/checkpoint
NFS_MODEL_PATH=/mnt/modelpath
TF_MODEL_EXPORT_PATH=${NFS_MODEL_PATH}
TF_MODEL_VERSION=1

BOWORDS_PATH=/mnt/traindata/global_words.txt

TEST_CLASSES_PATH=/mnt/traindata/TestDataClasses.txt
TEST_DATA_PATH=/mnt/traindata/TestDataset.txt

# If you want to use your own image,
# make sure you have a dockerhub account and change
# DOCKER_BASE_URL and IMAGE below.
DOCKER_BASE_URL=chrlindn/sjc-team-2
IMAGE=${DOCKER_BASE_URL}:${APP_NAME}6


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

