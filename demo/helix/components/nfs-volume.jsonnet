local env = std.extVar("__ksonnet/environments");
local params = std.extVar("__ksonnet/params").components["nfs-volume"];


local k = import "k.libsonnet";
local nfs = import "ciscoai/nfs-volume/nfs-volume.libsonnet";

// updatedParams uses the environment namespace if
// the namespace parameter is not explicitly set
local updatedParams = params {
  namespace: if params.namespace == "null" then env.namespace else params.namespace,
};


local name = params.name;
local namespace = updatedParams.namespace;


local nfs_server_ip = params.nfs_server_ip;
local capacity = params.capacity;
local path = params.mountpath;
local storage_request = params.storage_request;

std.prune(k.core.v1.list.new([
  nfs.parts.nfsPV(name, namespace, nfs_server_ip, capacity, path),
  nfs.parts.nfsPVC(name, namespace, storage_request)
]))

