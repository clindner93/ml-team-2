{
  global: {},
  components: {
    // Component-level parameters, defined initially from 'ks prototype use ...'
    // Each object below should correspond to a component in the components/ directory
    "nfs-server": {
      name: 'nfs-server',
      namespace: 'null',
    },
    "tf-helixjob": {
      args: 'null',
      envs: 'TF_DATA_DIR=/mnt/data,TF_EXPORT_DIR=/mnt/model_path,TF_CHECKPOINT_DIR=/mnt/checkpoint,TRAIN_DATA_PATH=/mnt/traindata',
      image: 'chrlindn/sjc-team-2:helix',
      image_gpu: 'null',
      image_pull_secrets: 'null',
      name: 'tf-helixjob',
      namespace: 'null',
      num_gpus: 0,
      num_masters: 1,
      num_ps: 1,
      num_workers: 1,
      volume_mount_path: 'null',
    },
    tfserving: {
      modelPath: '/mnt/model_path/model_trial',
      modelStorageType: 'nfs',
      name: 'helix',
      nfsPVC: 'team-sjc-2-nfs-pvc',
    },
  },
}
