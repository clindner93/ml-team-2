#!/usr/local/env bash

## Namespace to be used in k8s cluster for your application
NAMESPACE=kubeflow

## Ksonnet app name
APP_NAME=mnist

## GITHUB version for official kubeflow components
KUBEFLOW_GITHUB_VERSION=v0.2.0-rc.0

## GITHUB version for ciscoai components
CISCOAI_GITHUB_VERSION=dd81f2e8ad4db20e091576a724c7d58eb738f26e

## Ksonnet environment name
KF_ENV=nativek8s

## Name of the NFS Persistent Volume
NFS_PVC_NAME=nfs

## Used in training.bash
# Enviroment variables for mnist training job (See mnist_model.py)
TF_DATA_DIR=/mnt/data
TF_MODEL_DIR=/mnt/model
NFS_MODEL_PATH=/mnt/mnist
TF_EXPORT_DIR=${NFS_MODEL_PATH}

# If you want to use your own image,
# make sure you have a dockerhub account and change
# DOCKER_BASE_URL and IMAGE below.
DOCKER_BASE_URL=gcr.io/cpsg-ai-demo
IMAGE=${DOCKER_BASE_URL}/tf-mnist-demo:v1
CLIENT_IMAGE=${DOCKER_BASE_URL}/tf-mnist-demo-webapp:20180719-1
