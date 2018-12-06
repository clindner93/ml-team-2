local env = std.extVar("__ksonnet/environments");
local params = std.extVar("__ksonnet/params").components["tf-mnistjob"];

local k = import "k.libsonnet";
local deployment = k.extensions.v1beta1.deployment;
local container = deployment.mixin.spec.template.spec.containersType;
local podTemplate = k.extensions.v1beta1.podTemplate;
local util = import "ciscoai/tf-mnistjob/util.libsonnet";

// updatedParams uses the environment namespace if
// the namespace parameter is not explicitly set
local updatedParams = params {
  namespace: if params.namespace == "null" then env.namespace else params.namespace,
};

local tfJob = import "ciscoai/tf-mnistjob/tf-job.libsonnet";

local name = params.name;
local namespace = updatedParams.namespace;

local argsParam = params.args;
local args =
  if argsParam == "null" then
    []
  else
    std.split(argsParam, ",");

local envsParam = params.envs;
local envs =
  if envsParam == "null" then
    []
  else
    util.nameValuePair(std.split(envsParam,","));



local image = params.image;
local imageGpu = params.image_gpu;
local imagePullSecrets = params.image_pull_secrets;
local numPs = params.num_ps;
local numMasters = params.num_masters;
local numWorkers = params.num_workers;
local numGpus = params.num_gpus;
local path = params.volume_mount_path;

local mountPath = if path == "null" then "/mnt" else path;

local workerSpec = if numGpus > 0 then
  tfJob.parts.tfJobReplica("WORKER", numWorkers, args, imageGpu, imagePullSecrets, numGpus)
else
  tfJob.parts.tfJobReplica("WORKER", numWorkers, args, image, imagePullSecrets);

local masterSpec = tfJob.parts.tfJobReplica("MASTER", numMasters, args, image, imagePullSecrets);

local replicas = std.map(function(s)
                           s {
                             template+: {
                               spec+: {
                                  volumes: [ { name: "nfsvolume", persistentVolumeClaim: { claimName: "nfs" } } ] ,
                                 
                                  containers: [
                                   s.template.spec.containers[0] {
                                           volumeMounts: [ { mountPath: mountPath , name: "nfsvolume" } ],
                                           env: envs 
                                
                                   },
                                 ],
                               },
                             },
                           },
                         std.prune([masterSpec, workerSpec, tfJob.parts.tfJobReplica("PS", numPs, args, image, imagePullSecrets)]));


local job =
  if numWorkers < 1 then
    error "num_workers must be >= 1"
  else
    if numPs < 1 then
      error "num_ps must be >= 1"
    else {
      apiVersion: "kubeflow.org/v1alpha2",
      kind: "TFJob",
      metadata: {
        name: name,
        namespace: namespace,
      },
      spec: {
        tfReplicaSpecs: {
          Master: replicas[0],
          Worker: replicas[1],
          Ps: replicas[2],
        },
      },
    };

std.prune(k.core.v1.list.new([job]))
