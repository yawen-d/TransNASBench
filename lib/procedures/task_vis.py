import os
import sys
import math
import torch
import skimage
import itertools
import matplotlib
import numpy as np
import transforms3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from torchvision import utils as tv_utils
from torchvision.transforms import functional as F

lib_dir = (Path(__file__).parent / '..').resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from data import load_ops

##############
# base tools #
##############


def tensor2np_img(tensor):
    if len(tensor.shape) == 3:
        return tensor.clone().cpu().permute(1, 2, 0).detach().numpy()
    elif len(tensor.shape) == 4:
        return tensor.clone().cpu().permute(0, 2, 3, 1).detach().numpy()
    elif len(tensor.shape) == 2:
        return tensor.clone().cpu().detach().numpy()


##################
# classification #
##################

def classification(preds, synset, **kwargs):
    assert len(preds.shape) == 2
    batch_size = preds.shape[0]
    values, indices = preds.topk(5, 1, True, True)
    img_list = []
    for b in range(batch_size):
        top_5_pred = [synset[indices[b][i]] for i in range(5)]
        to_print_pred = "Top 5 prediction: \n {}\n {}\n {}\n {} \n {}".format(*top_5_pred)
        img_size = (256, 256) if 'img_size' not in kwargs.keys() else tuple(kwargs['img_size'])
        img = Image.new('RGB', img_size, (255, 255, 255))
        d = ImageDraw.Draw(img)
        fnt = ImageFont.truetype(str((Path(__file__).parent / 'DejaVuSerifCondensed.ttf').resolve()), 22)
        d.text((5, 5), to_print_pred, fill=(255, 0, 0), font=fnt)
        img_list.append(F.to_tensor(img).float())
    return torch.stack(img_list)


#####################
# segmentation demo #
#####################

def set_img_color(colors, background, img, gt, show255=False):
    for i in range(0, len(colors)):
        if i != background:
            img[np.where(gt == i)] = tuple([value * 255. for value in colors[i]])
        else:
            img[np.where(gt == i)] = (0., 0., 0.)
    if show255:
        img[np.where(gt == 255)] = 255
    return img


def seg_img(background, img, gt):
    """

    Args:
        background: choosing the color of background
        img: [C, W, H] torch.float [0, 1]
        gt: [W, H] torch.int

    Returns: torch_imgs [B, C, W, H]

    """
    labels = ('bottle', 'chair', 'couch', 'plant',
              'bed', 'd.table', 'toilet', 'tv', 'microw',
              'oven', 'toaster', 'sink', 'fridge', 'book',
              'clock', 'vase')
    colors = ('white', 'red', 'blue', 'yellow', 'magenta',
              'green', 'indigo', 'darkorange', 'cyan', 'pink',
              'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
              'purple', 'darkviolet')
    colors = [matplotlib.colors.to_rgb(c) for c in colors]
    im = tensor2np_img(img)
    im = np.array(im * 255., np.uint8)
    set_img_color(colors, background, im, gt, True)
    return F.to_tensor(im)


def semseg_imgs(labels, imgs):
    """
    Args:
        labels: [B, H, W]
        imgs: [B, C, H, W]
    """
    seg_imgs = []
    for i in range(labels.shape[0]):
        tmp_img = seg_img(0, imgs[i], labels[i])
        seg_imgs.append(tmp_img.unsqueeze(0))
    return torch.cat(seg_imgs, 0)


####################
# room layout demo #
####################

def get_cam_corners_and_edge_ordered(input_array):
    center = input_array[:3]
    edge_lengths = input_array[-3:]
    axis = transforms3d.euler.euler2mat(*input_array[3:6], axes='sxyz')
    y = axis[0] * edge_lengths[0] / 2
    x = axis[1] * edge_lengths[1] / 2
    z = axis[2] * edge_lengths[2] / 2
    corners_for_cam = np.empty((8, 3))
    corners_for_cam[0] = center - x + y - z
    corners_for_cam[1] = center + x + y - z
    corners_for_cam[2] = center + x - y - z
    corners_for_cam[3] = center - x - y - z
    corners_for_cam[4] = center - x + y + z
    corners_for_cam[5] = center + x + y + z
    corners_for_cam[6] = center + x - y + z
    corners_for_cam[7] = center - x - y + z
    return corners_for_cam, edge_lengths


def permute_orig_cols_display(array):
    return np.stack([array[:, 0], array[:, 2], array[:, 1]], axis=1)


def check_if_point_in_fustrum(point, fov):
    return all([np.abs(math.atan(coord / point[2])) < fov / 2. for coord in point[:2]])


def get_corner_idxs_in_view(corners, fov):
    in_view = []
    for idx, point in enumerate(corners):
        if check_if_point_in_fustrum(point, fov):
            in_view.append(idx)
    return in_view


def plot_bb_c(pred_corners, pred_edge, ax=None):
    if ax is None:
        ax = plt
    # dark_edge = [(0, 1),(1, 2),(2, 3),(0, 3)]
    # mid_edge = [(0, 4),(1, 5),(2, 6),(3, 7)]
    # light_edge = [(4, 5),(5, 6),(6, 7),(0, 7)]
    for (s_idx, s), (e_idx, e) in itertools.combinations(enumerate(pred_corners), 2):
        if any([np.isclose(np.linalg.norm(s - e), el, atol=1e-04) for el in pred_edge]):
            if min(s_idx, e_idx) < 4 and max(s_idx, e_idx) < 4:
                c = (0.54, 0, 0)
            elif min(s_idx, e_idx) < 4 and max(s_idx, e_idx) > 3:
                c = (0.77, 0, 0)
            else:
                c = 'r'

            ax.plot3D(*zip(s, e), color=c, linewidth=5)
    return ax


def plot_points_with_bb(pred_corners, pred_edge, cube_only=False, fov=None, space='camera',
                        fig=None, subplot=(1, 1, 1)):
    # is_camera_space = space.lower() == 'camera'
    # in_view_pred = get_corner_idxs_in_view(pred_corners, fov)
    pred_corners = permute_orig_cols_display(pred_corners)
    # total_corners = pred_corners
    # mins = np.min(total_corners, axis=0)
    # maxes = np.max(total_corners, axis=0)
    # largest_range = (maxes - mins).max()
    # axis_ranges = [[m, m + largest_range] for m in mins ]
    if cube_only:
        axis_ranges = [[-6, 6], [-6, 6], [-6, 6]]
    else:
        axis_ranges = [[-6, 6], [-8, 1.5], [-1.2, 7]]
    axes = ['x', 'z', 'y'] if space.lower() == 'camera' else ['x', 'y', 'z']
    axis_idx = {v: k for k, v in enumerate(axes)}
    from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
    ax = fig.add_subplot(*subplot, projection='3d')
    ax._axis3don = False
    ax.set_xlim(axis_ranges[axis_idx['x']])
    ax.set_zlim(axis_ranges[axis_idx['y']])
    ax.set_ylim(axis_ranges[axis_idx['z']])
    ax.set_xlabel(axes[0], fontsize=12)
    ax.set_ylabel(axes[1], fontsize=12)
    ax.set_zlabel(axes[2], fontsize=12)
    plot_bb_c(pred_corners, pred_edge, ax=ax)
    if not cube_only:
        ax.scatter(0, 0, 0, zdir='r', c='m', s=50)
    theta = np.arctan2(1, 0) * 180 / np.pi
    ax.view_init(30, theta)
    ax.invert_xaxis()
    return ax


def plot_room_layout(predicted, cube_only=False, overlay=False, keep_ratio=True):
    from PIL import Image, ImageDraw, ImageFont
    fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(1, 1, 1)
    predicted[-3:] = np.absolute(predicted[-3:])
    if cube_only:
        predicted[:3] = [0, 0, -1]
        if keep_ratio:
            predicted[-3:] = 7 * predicted[-3:] / np.prod(predicted[-3:]) ** (1 / 3)
        else:
            predicted[-3:] = [8, 8, 8]
    corners_for_cam_prediction, edge_lengths_pred = get_cam_corners_and_edge_ordered(predicted)
    camera_space_plot = plot_points_with_bb(pred_corners=corners_for_cam_prediction[:, :3],
                                            pred_edge=edge_lengths_pred, cube_only=cube_only,
                                            fov=1, space='camera', subplot=(1, 1, 1), fig=fig)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    x = skimage.transform.resize(data, [256, 256])
    if not overlay:
        x = skimage.img_as_ubyte(x)
        out = Image.fromarray(x)
        return out
    else:
        raise NotImplementedError


def room_layout(preds):
    preds = preds.detach().numpy()
    mean = np.array([-1.64378987e-02, 7.03680296e-02, -2.72496318e+00,
                     1.56155458e+00, -2.83141191e-04, -1.57136446e+00,
                     4.81219593e+00, 3.69077521e+00, 2.61998101e+00])
    std = np.array([0.73644884, 0.53726124, 1.45194796,
                    0.19338562, 0.01549811, 0.42258508,
                    2.80763433, 1.92678054, 0.89655357])
    assert len(preds.shape) == 2
    batch_size = preds.shape[0]
    img_tensors = []
    for b in range(batch_size):
        predicted = preds[b] * std + mean
        out = plot_room_layout(np.squeeze(predicted))
        img_tensors.append(F.to_tensor(out))
    return torch.stack(img_tensors)


###############
# Jigsaw demo #
###############
def get_permutations(indices, permutation_set):
    """
    :param permutation_set:
    :param indices: Tensor (N,)
    :return: list of np.array(9)
    """
    return [permutation_set[idx] for idx in indices]


def reverse_permutations(permutations):
    results = []
    for p in permutations:
        tmp = sorted([(y, x) for x, y in enumerate(p)])
        results.append([t[-1] for t in tmp])
    return results


def tiles2image(tiles, permutations=None):
    """
    :param tiles: Tensor (N, 9, 3, 64, 64)
    :param permutations: Tensor/Array of size (N, 9)
    :return: Tensor
    """
    N, T, C, W, H = tiles.shape
    if permutations is None:
        permutations = torch.stack([torch.arange(T) for _ in range(N)])
    assert N == len(permutations)
    all_result = []
    for n in range(N):
        width = W + 5
        height = H + 5
        result = Image.new('RGB', (width * 3, height * 3))
        for idx in range(9):
            H_idx = idx // 3
            W_idx = idx % 3
            tile_idx = permutations[n][idx]
            image = F.to_pil_image(tiles[n, tile_idx, :, :, :])
            image = F.pad(image, (5, 5, 0, 0), 0, 'constant')
            result.paste(image, box=(W_idx * width, H_idx * height))
        result = F.pad(result, (0, 0, 5, 5), 0, 'constant')
        all_result.append(F.to_tensor(result))
    return torch.stack(all_result)


##############
# store imgs #
##############

def task_demo(task_name, store_path, imgs, tars, preds, **kwargs):
    imgs, tars, preds = imgs.cpu(), tars.cpu(), preds.cpu()
    if task_name in ['autoencoder', 'normal', 'inpainting', 'denoise']:
        imglist = torch.cat([imgs, tars, preds], 0)
    elif task_name in ['room_layout']:
        imglist = torch.cat([imgs, room_layout(tars), room_layout(preds)], 0)
    elif task_name in ['segmentsemantic']:
        imglist = torch.cat([imgs, semseg_imgs(tars, imgs), semseg_imgs(preds.squeeze(1), imgs)], 0)
    elif task_name in ['class_object', 'class_scene']:
        synset = load_ops.get_synset(task_name, **kwargs)
        img_size = imgs.shape[-2:]
        imglist = torch.cat([imgs, classification(tars, synset, img_size=img_size),
                             classification(preds, synset, img_size=img_size)], 0)
    elif task_name in ['jigsaw']:
        permutation_set = load_ops.get_permutation_set(**kwargs)
        raw_imgs = tiles2image(imgs)
        tar_imgs = tiles2image(imgs, reverse_permutations(get_permutations(tars, permutation_set)))
        pred_imgs = tiles2image(imgs, reverse_permutations(get_permutations(preds.argmax(dim=-1), permutation_set)))
        imglist = torch.cat([raw_imgs, tar_imgs, pred_imgs], 0)
    else:
        raise ValueError(f'Invalid task name {task_name}!')
    # grid_draws(store_path, imgs, tars, preds)
    assert len(imglist.shape) == 4
    tv_utils.save_image(imglist, store_path, padding=25, nrow=3, scale_each=True, normalize=True)
