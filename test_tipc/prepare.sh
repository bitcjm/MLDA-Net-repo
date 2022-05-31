#!/bin/bash
FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',
#                 'whole_infer']

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})
function func_parser_key(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[0]}
    echo ${tmp}
}
function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}
IFS=$'\n'
# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

if [ ${MODE} = "benchmark_train" ];then
    pip install -v -e .
    MODE="lite_train_lite_infer"
fi

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',
#                 'whole_infer

if [ ${MODE} = "lite_train_lite_infer" ];then
    case ${model_name} in
    GPEN)
        rm -rf ./data/ffhq*
        wget -nc -P ./data/ https://paddlegan.bj.bcebos.com/datasets/ffhq.tar --no-check-certificate
        cd ./data/ && tar xf ffhq.tar && cd ../ ;;
    esac
fi
