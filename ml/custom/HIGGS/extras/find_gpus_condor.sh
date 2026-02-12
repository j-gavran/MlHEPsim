#!/bin/bash

condorsub -J FST_find_gpus -q express -g 1 -m 500 \
"source ~/.bashrc && \
source /etc/profile && \
shopt -s expand_aliases && \
source /project/atlas/users/mveldijk/MLHEPsim/venv311/bin/activate && \
cd /project/atlas/users/mveldijk/MLHEPsim && \
TORCH_C_FILE=\$(python3 -c 'import torch; print(torch._C.__file__)') && \
ldd \$TORCH_C_FILE | grep cuda && \
echo '===== /dev/nvidia* =====' && \
ls -l /dev/nvidia* && \
python find_gpus.py"

