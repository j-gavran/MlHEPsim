#!/bin/bash

cd /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim

# Remove old condor log files (suppress errors if files don't exist)
rm -f /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim/ml/custom/DrellYan/run/mafmademog*
rm -f /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim/ml/custom/DrellYan/run/condor/mveldijk/condorsub/*
# Remove old condor submission files
rm -f /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim/condor/mveldijk/condorsub/mafmademog*
rm -f /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim/condor/mveldijk/condorsub/enviromentvariables*

condorsub -J mafmademog -q long -n 1 -g 1 -m 64000 \
"source /etc/profile && \
source /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim/venv311/bin/activate && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH && \
export PYTHONPATH=$PYTHONPATH:/project/atlas/users/mvedlijk/MLHEPsimtest/MLHEPsim && \
cd /project/atlas/users/mveldijk/MLHEPsimtest/MLHEPsim && \
python -m ml.custom.DrellYan.main_flows_reduced"
