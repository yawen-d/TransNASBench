#!/bin/bash

task_name=""
gpu_id="0"
encoder_str=""
seed="666"
ddp="0"

function usage
{
    echo "train_a_net.sh -g or --gpu for gpu_id; -t or --task for task_name; --seed for seed;
    -e or --encoder_str for encoder_str; --ddp for distributed data parallel"
    echo "task_name can only be from [autoencoder, class_object, class_scene, jigsaw, normal,
    room_layout, segmentsemantic]"
}

while [ "$1" != "" ]
do
   case "$1" in
        -g | --gpu )           shift
                               gpu_id=$1
                               ;;
        -t | --task )          shift
                               task_name=$1
                               ;;
        -e | --encoder_str )   shift
                               encoder_str=$1
                               ;;
        --ddp )                shift
                               ddp=$1
                               ;;
        --seed )               shift
                               seed=$1
                               ;;
        -h | --help )          usage
                               exit
                               ;;
        * )                    usage
                               exit 1
    esac
    shift
done

echo "Running Experiment for Task: $task_name"

if [ "$task_name" = "" ]; then
    echo "Task Name is empty..."
    exit 1
fi

if [ "$gpu_id" = "" ]; then
    echo "GPU id is empty..."
    exit 1
fi

scripts_dir="$( cd "$( dirname "$0" )" && pwd )"
code_dir=$(dirname "$scripts_dir")
project_dir=$(dirname "$code_dir")

# Change the paths here to relative paths
config_dir="$project_dir/configs/task_cfg/train_from_scratch"

if [ "$ddp" = "0" ]; then
    CUDA_VISIBLE_DEVICES=$gpu_id python3 $code_dir/tools/main.py $config_dir/$task_name/ \
      --encoder_str=$encoder_str --seed $seed
else
    CUDA_VISIBLE_DEVICES=$gpu_id python3 $code_dir/tools/main.py $config_dir/$task_name/ \
      --encoder_str=$encoder_str --seed $seed --ddp
fi
