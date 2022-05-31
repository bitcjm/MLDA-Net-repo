import argparse
import cv2
import numpy as np
import os

import paddle
import skimage
from paddle.nn import functional as F
from paddle.vision import transforms as tfs

import networks
from networks.layers import disp_to_depth
from datasets.mono_dataset import pil_loader
from utils.kitti_utils import generate_depth_map

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu())
    mi = float(x.min().cpu())
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def get_depth(calib_path, velo_filename):

    depth_gt = generate_depth_map(calib_path, velo_filename)
    depth_gt = skimage.transform.resize(
        depth_gt, (375, 1242), order=0, preserve_range=True, mode='constant')

    return depth_gt

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    gt = gt.numpy()
    pred = pred.numpy()
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def parse_args():
    parser = argparse.ArgumentParser(description='Model predict.')
    parser.add_argument('--scales', nargs="+", type=int, help="scales used in the loss", default=[0, 1, 2, 3])
    parser.add_argument('--height', dest='height', default=192, type=int)
    parser.add_argument('--width', dest='width', default=640, type=int)
    parser.add_argument('--min_depth', dest='min_depth', default=0.1, type=float)
    parser.add_argument('--max_depth', dest='max_depth', default=100, type=float)
    parser.add_argument('--num_layers', dest='num_layers', default=18, type=int)

    parser.add_argument("--load_weights_folder", type=str, required=True)

    # RGB image
    parser.add_argument("--color_path", type=str, default="./data/lite_data/10_03/1/0000000001.jpg")
    # If you want to test an arbitrary color map and there is no corresponding point cloud data
    # to provide depth information, set no_rmse to True
    parser.add_argument("--no_rmse", type=bool, default=False)
    parser.add_argument("--calib_path", type=str, default="./data/lite_data/10_03")
    # The point cloud data corresponding to the color image is used to calculate the actual depth information
    parser.add_argument("--velo_filename", type=str, default="./data/lite_data/10_03/1/0000000001.bin")

    return parser.parse_args()

if __name__ == '__main__':

    opt = parse_args()

    color_img = pil_loader(opt.color_path)
    resize = tfs.Resize((opt.height, opt.width), interpolation="lanczos")
    color_aug = paddle.to_tensor(np.array(resize(color_img))).astype('float32')
    color_aug = color_aug.transpose((2,0,1)).unsqueeze(0) / 255

    models = {}
    models['encoder'] = networks.ResnetEncoder_multi_sa_add_reduce_640(opt.num_layers, False)
    models['depth'] = networks.DepthDecoderAttention_edge(models['encoder'].num_ch_enc)

    models_to_load = ["encoder", "depth"]
    load_weights_folder = opt.load_weights_folder
    for n in models_to_load:
        path = os.path.join(load_weights_folder, "{}.pdparams".format(n))
        model_dict = models[n].state_dict()
        pretrained_dict = paddle.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[n].set_state_dict(model_dict)
    for m in models.values():
        m.eval()

    num = 0
    with paddle.no_grad():
        features = models["encoder"](color_aug)
        outputs = models["depth"](features)
        
        disp = outputs[("disp", 0)]

        color_aug = color_aug[0]
        input_color = color_aug.transpose([1, 2, 0]).cpu().numpy() * 255
        input_color = cv2.cvtColor(input_color, cv2.COLOR_RGB2BGR)
        pred_depth = normalize_image(outputs[("disp", 0)][0, 0]).detach().cpu().numpy() * 255
        pred_depth = cv2.cvtColor(pred_depth, cv2.COLOR_GRAY2BGR)
        img2save = np.concatenate((input_color, pred_depth), 0)

        save_dir = './predict_figs/'
        os.makedirs(save_dir, exist_ok=True)
        path_to_save = save_dir + 'depth_predict.jpg'
        cv2.imwrite(path_to_save, img2save)
        print("predict_img saved to {}".format(path_to_save))

        if opt.no_rmse or opt.calib_path==None or opt.velo_filename==None:
            print("can't find depth files, can't count rmse")
        else:
            if not os.path.exists(opt.velo_filename):
                print("can't find depth files, can't count rmse")
                assert opt.calib_path==None

            depth_gt = get_depth(opt.calib_path, opt.velo_filename)
            depth_gt = paddle.to_tensor(depth_gt).unsqueeze(0).unsqueeze(0)
            _, depth_pred = disp_to_depth(disp, opt.min_depth, opt.max_depth)

            depth_pred = paddle.clip(F.interpolate(
                depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)

            depth_pred = depth_pred.detach()

            mask = depth_gt > 0

            # garg/eigen crop
            crop_mask = paddle.zeros_like(mask)
            crop_mask[:, :, 153:371, 44:1197] = 1
            mask = mask * crop_mask

            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]
            depth_pred *= np.median(depth_gt.numpy()) / np.median(depth_pred.numpy())
            depth_pred = paddle.clip(depth_pred, min=1e-3, max=80)

            depth_errors = compute_errors(depth_gt, depth_pred)

            mean_errors = np.array(depth_errors)

            print("eval")
            print("  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
            print("-> Done!")


