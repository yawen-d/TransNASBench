# TransNAS-Bench-101: Improving Transferrability and Generalizability of Cross-Task Neural Architecture Search

We propose TransNAS-Bench-101, a benchmark containing network performance across seven tasks, covering classification, regression, pixel-level prediction, and self-supervised tasks. This diversity provides opportunities to transfer NAS methods among the tasks and allows for more complex transfer schemes to evolve. We explore two fundamentally different types of search spaces: cell-level search space and macro-level search space. With 7,352 backbones evaluated on seven tasks, 51,464 trained models with detailed training information are provided. With TransNASBench-101, we hope to encourage the advent of exceptional NAS algorithms that raise cross-task search efficiency and generalizability to the next level.

In this Markdown file, we show an example how to use TransNAS-Bench-101. The complete network training information file can be found through [VEGA](https://www.noahlab.com.hk/opensource/vega/page/doc.html?path=datasets/transnasbench101).

## How to use TransNAS-Bench-101

1. Import the API object in `./code/api/api.py` and create an API instance from the `.pth` file in `./api_home/`:
​
```python
from api import TransNASBenchAPI as API
path2nas_bench_file = "./api_home/transnas-bench_v10141024.pth"
api = API(path2nas_bench_file)
```

2. Check the task information, number of architectures evaluated, and search spaces:

```python
# show number of architectures and number of tasks
length = len(api)
task_list = api.task_list # list of tasks
print(f"This API contains {length} architectures in total across {len(task_list)} tasks.")
# This API contains 7352 architectures in total across 7 tasks.

# Check all model encoding
search_spaces = api.search_spaces # list of search space names
all_arch_dict = api.all_arch_dict # {search_space : list_of_architecture_names}
for ss in search_spaces:
   print(f"Search space '{ss}' contains {len(all_arch_dict[ss])} architectures.")
print(f"Names of 7 tasks: {task_list}")
# Search space 'macro' contains 3256 architectures.
# Search space 'micro' contains 4096 architectures.
# Names of 7 tasks: ['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder']
```

3. Since different tasks may require different evaluation metrics, hence `metric_dict` showing the used metrics can be retrieved from `api.metrics_dict`. TransNAS-Bench API also recorded the model inference time, backbone/model parameters, backbone/model FLOPs in `api.infor_names`.

```python
metrics_dict = api.metrics_dict # {task_name : list_of_metrics}
info_names = api.info_names # list of model info names

# check the training information of the example task
task = "class_object"
print(f"Task {task} recorded the following metrics: {metrics_dict[task]}")
print(f"The following model information are also recorded: {info_names}")
# Task class_object recorded the following metrics: ['train_top1', 'train_top5', 'train_loss', 'valid_top1', 'valid_top5', 'valid_loss', 'test_top1', 'test_top5', 'test_loss', 'time_elapsed']
# The following model information are also recorded: ['inference_time', 'encoder_params', 'model_params', 'model_FLOPs', 'encoder_FLOPs']
```

4. Query the results of an architecture by arch string
​
```python
# Given arch string
xarch = api.index2arch(1) # '64-2311-basic'
for xtask in api.task_list:
    print(f'----- {xtask} -----')
    print(f'--- info ---')
    for xinfo in api.info_names:
        print(f"{xinfo} : {api.get_model_info(xarch, xtask, xinfo)}")
    print(f'--- metrics ---')
    for xmetric in api.metrics_dict[xtask]:
        print(f"{xmetric} : {api.get_single_metric(xarch, xtask, xmetric, mode='best')}")
        print(f"best epoch : {api.get_best_epoch_status(xarch, xtask, metric=xmetric)}")
        print(f"final epoch : {api.get_epoch_status(xarch, xtask, epoch=-1)}")
        if ('valid' in xmetric and 'loss' not in xmetric) or ('valid' in xmetric and 'neg_loss' in xmetric):
            print(f"\nbest_arch -- {xmetric}: {api.get_best_archs(xtask, xmetric, 'micro')[0]}")
```

A complete example is given in `code/api/example.py`
- `cd code/api`
- `python example.py`

## Example network encoding in both search spaces

```
Macro example network: 64-1234-basic
- Base channel: 64
- Macro skeleton: 1234 (4 stacked modules)
  - [m1(normal)-m2(channelx2)-m3(resolution/2)-m4(channelx2 & resolution/2)]
- Cell structure: basic (ResNet Basic Block)

Micro example network: 64-41414-1_02_333
- Base channel: 64
- Macro skeleton: 41414 (5 stacked modules)
  - [m1(channelx2 & resolution/2)-m2(normal)-m3(channelx2 & resolution/2)-m4(normal)-m5(channelx2 & resolution/2)]
- Cell structure: 1_02_333 (4 nodes, 6 edges)
  - node0: input tensor
  - node1: Skip-Connect( node0 ) # 1
  - node2: None( node0 ) + Conv1x1( node1 ) # 02
  - node3: Conv3x3( node0 ) + Conv3x3( node1 ) + Conv3x3( node2 ) # 333
```

## How to regenerate TransNAS-Bench-101

- Taskonomy_mini data used to train the networks
    - Raw images and labels should be downloaded from this [link]().
    - train/val/test split is located in `configs/dataset_split/final5k/`
- Train a new neural network
    - `code/scripts/train_a_net.sh -g <gpu_id> -t <task_name> -e <network_encoding> --seed <seed>`
        - Example usage: `code/scripts/train_a_net.sh -g 0 -t class_scene -e 64-41414-2_03_323 --seed 666`
- Regenerate a new benchmark dataset
    - `code/scripts/get_bm_sp.sh -t <task_name> -g <gpu_num> --start <start_net_idx> --end <end_net_idx> --trial <log_name> --mode <fill or force>`
        - `<start_net_idx>`: start training from `<start_net_idx>`-th network in `benchmark_status/net_strings_all.json` 
        - `<end_net_idx>`: stop training until `<end_net_idx>`-th network in `benchmark_status/net_strings_all.json` 
        - `<mode>`: `force` == train selected networks by force; `fill` == only evaluate untrained ones in selected networks 
        - Example usage: `code/scripts/gen_bm_sp.sh -t class_object -g 4 --start 4 --end 8 --trial test --mode fill`

# Citation

If you find that TransNAS-Bench-101 helps your research, please consider citing it:

​
​








