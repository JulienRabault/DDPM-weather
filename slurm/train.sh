#!/bin/bash
set -x
cd $SOURCE_DIR
HYDRA_FULL_ERROR=1 python -m torch.distributed.run --standalone --nproc\_per\_node gpu main.py --yaml\_path $@