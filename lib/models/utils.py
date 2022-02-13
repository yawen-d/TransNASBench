import os
import sys
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

lib_dir = (Path(__file__).parent / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from procedures import task_demo


############################
# operations for all tasks #
############################


class Timer:
    def __init__(self, start_step, total_step):
        self.start = time.time()
        self.start_step = start_step
        self.total_step = total_step

    def elapse_string(self):
        now = time.time()
        hours, rem = divmod(now - self.start, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))

    def total_time_string(self, current_step):
        now = time.time()
        avg_time_per_step = (now - self.start) / (current_step - self.start_step)
        total_time = (self.total_step - self.start_step) * avg_time_per_step
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))

    def time_status(self, current_step):
        return f'[{self.elapse_string()}/{self.total_time_string(current_step)}]'


def logging(cfg, epoch, metrics, logger, timer, training=True, step=0, total_step=0, extra_message=''):
    if 'time_elapsed' in metrics.keys():
        metrics.pop('time_elapsed', None)
    msg = [met_name + ' [%.4f]' % met_value for met_name, met_value in metrics.items()]
    msg.insert(0, f"{timer.elapse_string()}" if training else f"[{timer.time_status(epoch + 1)}]")
    msg.insert(1, cfg['encoder_str'])
    msg.insert(2, 'epoch [%d/%d]' % (epoch, cfg['num_epochs']))
    if training:
        msg.insert(3, 'step [%d/%d]' % (step, total_step))
    logger.write(', '.join(msg) + extra_message)


def end_epoch_log(cfg, epoch, metrics, model_dic, model_db, logger, recorder, timer):
    metrics_cp = metrics.copy()
    logging(cfg, epoch, metrics_cp, logger, timer, training=False)
    recorder.update(metrics)

    # metrics plotting
    model_db.save(model_dic, epoch, metrics)
    if cfg['task_name'] in ['autoencoder', 'inpainting', 'normal']:
        recorder.plot_curve(
            f"{cfg['log_dir']}/{cfg['encoder_str']}_glr{cfg['initial_lr']}_dlr{cfg['d_lr']}.png")
    else:
        recorder.plot_curve(
            f"{cfg['log_dir']}/{cfg['encoder_str']}_lr{cfg['initial_lr']}.png")

    logger.write('Time elapsed: %s' % timer.elapse_string())
    if cfg['task_name'] in ['class_object', 'class_scene']:
        logger.write('Highest validation acc: %.4f' % recorder.max_metric('valid_top1'))
    elif cfg['task_name'] in ['autoencoder', 'normal', 'inpainting']:
        logger.write('Highest validation ssim: %.4f' % recorder.max_metric('val_ssim'))
    elif cfg['task_name'] == 'segmentsemantic':
        logger.write('Highest validation mIoU: %.4f' % recorder.max_metric('valid_mIoU'))


def demo(cfg, epoch, step, imgs, tars, preds, extra_msg=''):
    if not os.path.exists(f"{cfg['log_dir']}/img_output"):
        os.makedirs(f"{cfg['log_dir']}/img_output")
    store_name = os.path.join(cfg['log_dir'], 'img_output',
                              f"{cfg['task_name']}_epoch{epoch}_step{step}_{cfg['plot_msg']}{extra_msg}.png")
    task_demo(cfg['task_name'], store_name, imgs=imgs[:3], tars=tars[:3], preds=preds[:3],
              **cfg['demo_kwargs'])


def merge_list(lists):
    merged = []
    for li in lists:
        merged += li
    return merged


########################
# classification tasks #
########################

def get_topk_acc(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    Support both classification and segmentation
    """
    maxk = max(topk)
    batch_size = target.size(0)
    data_point_size = torch.tensor(target.size()[1:]).prod()

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.transpose(0, 1)
    correct = pred.eq(target.view_as(pred[0]))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k /= data_point_size
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_confusion_matrix(gt_labels, pred_labels, class_num):
    """
    Calcute the confusion matrix by given label and pred
    Args:
        gt_labels: the ground truth label, torch.tensor [B, C, H, W]
        pred_labels: the pred label, torch.tensor [B, C, H, W]
        class_num: the number of class
    Returns: the confusion matrix
    """
    cm = np.zeros((class_num, class_num))
    for i, (gt_label, pred_label) in enumerate(zip(gt_labels, pred_labels)):
        gt_label, pred_label = gt_label.cpu().numpy(), pred_label.cpu().numpy()
        index = (gt_label.flatten() * class_num + pred_label.flatten()).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]
        cm += confusion_matrix
    return cm


def get_iou(cm):
    pos = cm.sum(1)
    res = cm.sum(0)
    tp = np.diag(cm)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array[(pos + res) == 0.] = np.nan
    mean_IU = np.nanmean(IoU_array)
    return IoU_array, mean_IU * 100.


#########
# algos #
#########




# if __name__ == '__main__':
#     sfm_output = torch.rand([1, 4, 3, 3]).softmax(1)
#     label = torch.randint(0, 4, [1, 3, 3])
#     print(f"sfm_output, {sfm_output.shape},\n{sfm_output}")
#     print(f"label, {label.shape},\n{label}")
#     # print(get_topk_acc(sfm_output, label))
#     _, pred = sfm_output.topk(1, 1, True, True)
#     print(f"pred, {pred.shape},\n{pred}")
#     cm = get_confusion_matrix(label, pred, 4)
#     print(f'confusion matrix: {cm};\niou: {get_iou(cm)}')
