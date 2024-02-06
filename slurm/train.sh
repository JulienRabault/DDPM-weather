#!/bin/bash
set -x
cd /ddpm
HYDRA_FULL_ERROR=1 python3 -m torch.distributed.run --standalone --nproc\_per\_node gpu main.py --yaml\_path $@
