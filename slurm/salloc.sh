#!/bin/bash

# This script allocates resources using the salloc command.
#
# Argument: ${1} - Number of nodes that you want to be allocated.
#
# Available partition list:
# - cas_v100nv_8: Does not have NVMe storage.
# - amd_a100nv_8: NVMe is available and mounted at /etmp, but it is the busiest queue.
# - eme_h200nv_8: Idlest queue but does not have NVMe storage.
#
# Notes:
# - --ntasks-per-node: Should be at least 2 (HVAC server + training agent).
# - --cpus-per-task: Determines the memory capacity allocated for each task.
# 
# For more details, refer to the official documentation:
# https://docs-ksc.gitbook.io/neuron-user-guide/undefined/running-jobs-through-scheduler-slurm

salloc -n ${1} --time=1:00:00 --partition=cas_v100nv_8 --nodes=${1} --ntasks-per-node=3 --cpus-per-task=8 --gres=gpu:1 --comment=etc

