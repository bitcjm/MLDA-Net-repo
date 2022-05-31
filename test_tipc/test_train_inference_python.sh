#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer', 'whole_infer']
MODE=$2

dataline=$(awk 'NR==1, NR==32{print}' $FILENAME)

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")
gpu_list=$(func_parser_value "${lines[3]}")
use_gpu=$(func_parser_value "${lines[4]}")

# train py
#trainer_list=$(func_parser_value "${lines[5]}")
trainer_py=$(func_parser_value "${lines[6]}")
echo trainer_py
# infer params
export_py=$(func_parser_value "${lines[12]}")
inference_py=$(func_parser_value "${lines[13]}")

# log
LOG_PATH="./log/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}
status_log="${LOG_PATH}/results_python.log"

run_train=${trainer_py}
cmd="${python} ${run_train}"
eval $cmd
status_check $? "${cmd}" "${status_log}"
cmd="${python} ${export_py}"
eval $cmd
status_check $? "${cmd}" "${status_log}"
cmd="${python} ${inference_py}"
eval $cmd
status_check $? "${cmd}" "${status_log}"