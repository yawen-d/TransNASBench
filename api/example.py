from api import TransNASBenchAPI


if __name__ == '__main__':
    api = TransNASBenchAPI("../api_home/transnas-bench_v10141024.pth")
    xarch = '64-4111-basic'
    for xtask in api.task_list:
        print(f'----- {xtask} -----')
        print(f'--- info ---')
        for xinfo in api.info_names:
            print(f"{xinfo} : {api.get_model_info(xarch, xtask, xinfo)}")
        print(f'--- metrics ---')
        for xmetric in api.metrics_dict[xtask]:
            # print(f"{xmetric} : {api.get_single_metric(xarch, xtask, xmetric, mode='best')}")
            # print(f"best epoch : {api.get_best_epoch_status(xarch, xtask, metric=xmetric)}")
        # print(f"final epoch : {api.get_epoch_status(xarch, xtask, epoch=-1)}")
            if ('valid' in xmetric and 'loss' not in xmetric) or ('valid' in xmetric and 'neg_loss' in xmetric):
                print(f"\nbest_arch -- {xmetric}: {api.get_best_archs(xtask, xmetric, 'micro')[0]}")

