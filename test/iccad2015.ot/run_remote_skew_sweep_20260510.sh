#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all \
  -v /home/shengyuanjia/SkewPlace:/DREAMPlace \
  -v /home/shengyuanjia/heterosta_license:/tmp/heterosta_license:ro \
  -w /DREAMPlace \
  shengyuanjia/dreamplace:cuda \
  bash -lc 'ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so; export HeteroSTA_Lic=$(cat /tmp/heterosta_license); export PYTHONPATH=/DREAMPlace/install:/DREAMPlace/install/dreamplace; export DREAMPLACE_SUMMARY_MAX_PATHS=20000; python test/iccad2015.ot/run_skew_experiment_suite.py /DREAMPlace/results/skew_sweep_20260510 test/iccad2015.ot/superblue1.json,test/iccad2015.ot/superblue3.json,test/iccad2015.ot/superblue4.json,test/iccad2015.ot/superblue5.json 1,100,1000,9000 1000 1000 100'
