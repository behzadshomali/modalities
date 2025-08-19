CUDA_VISIBLE_DEVICES="0,1,2" torchrun \
  --rdzv-endpoint localhost:29516  \
  --nnodes 1   \
  --nproc_per_node 3   \
  $(which modalities) run \
  --config_file_path "/home/behzad_shomali/modalities/config_files/warmstart/warmstart_config_2b_2*15.yaml"