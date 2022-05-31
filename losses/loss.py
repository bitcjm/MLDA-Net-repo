import numpy as np
import paddle
import paddle.nn.functional as F
from networks.layers import SSIM, compute_depth_errors, get_smooth_loss

def compute_reprojection_loss(opt, pred, target):
    """Computes reprojection losses between a batch of predicted and target images
    """
    abs_diff = paddle.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim = SSIM()
    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


def compute_proxy_supervised_loss(pred, target, valid_pixels, loss_mask):
    """ Compute proxy supervised losses (depth hint losses) for prediction.

        - valid_pixels is a mask of valid depth hint pixels (i.e. non-zero depth values).
        - loss_mask is a mask of where to apply the proxy supervision (i.e. the depth hint gave
        the smallest reprojection error)"""

    # first compute proxy supervised losses for all valid pixels
    depth_hint_loss = paddle.log(paddle.abs(target - pred) + 1) * valid_pixels

    # only keep pixels where depth hints reprojection losses is smallest
    depth_hint_loss = depth_hint_loss * loss_mask

    return depth_hint_loss



def compute_loss_masks(reprojection_loss, identity_reprojection_loss,
                       depth_hint_reprojection_loss):
    """ Compute losses masks for each of standard reprojection and depth hint
    reprojection.

    identity_reprojections_loss and/or depth_hint_reprojection_loss can be None"""

    if identity_reprojection_loss is None:
        # we are not using automasking - standard reprojection losses applied to all pixels
        reprojection_loss_mask = paddle.ones_like(reprojection_loss)

        if depth_hint_reprojection_loss:
            all_losses = paddle.concat([reprojection_loss, depth_hint_reprojection_loss], axis=1)
            idxs = paddle.argmin(all_losses, axis=1, keepdim=True)
            depth_hint_loss_mask = (idxs == 1).astype('float32')

    else:
        # we are using automasking
        if depth_hint_reprojection_loss is not None:
            all_losses = paddle.concat([reprojection_loss, identity_reprojection_loss,
                                        depth_hint_reprojection_loss], 1)
        else:
            all_losses = paddle.concat([reprojection_loss, identity_reprojection_loss], 1)

        idxs = paddle.argmin(all_losses, axis=1, keepdim=True)
        reprojection_loss_mask = (idxs != 1).astype('float32')  # automask has index '1'
        depth_hint_loss_mask = (idxs == 2).astype('float32')  # will be zeros if not using depth hints

    # just set depth hint mask to None if not using depth hints
    depth_hint_loss_mask = None if depth_hint_reprojection_loss is None else depth_hint_loss_mask

    return reprojection_loss_mask, depth_hint_loss_mask


def compute_losses(opt, inputs, outputs):
    """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
    """
    losses = {}
    total_loss = 0

    # compute depth hint reprojection losses
    if opt.use_depth_hints:
        pred = outputs[("color_depth_hint", 's', 0)]
        depth_hint_reproj_loss = compute_reprojection_loss(opt, pred, inputs[("color", 0, 0)])
        # set losses for missing pixels to be high so they are never chosen as minimum
        depth_hint_reproj_loss += 1000 * (1 - inputs['depth_hint_mask'])
    else:
        depth_hint_reproj_loss = None

    for scale in opt.scales:
        loss = 0
        reprojection_losses = []
        #source_scale = scale
        source_scale = 0

        disp = outputs[("disp", scale)]
        color = inputs[("color", 0, scale)]
        target = inputs[("color", 0, source_scale)]

        for frame_id in opt.frame_ids[1:]:
            pred = outputs[("color", frame_id, scale)]
            reprojection_losses.append(compute_reprojection_loss(opt, pred, target))

        reprojection_losses = paddle.concat(reprojection_losses, 1)

        identity_reprojection_losses = []
        for frame_id in opt.frame_ids[1:]:
            pred = inputs[("color", frame_id, source_scale)]
            identity_reprojection_losses.append(
                compute_reprojection_loss(opt, pred, target))

        identity_reprojection_losses = paddle.concat(identity_reprojection_losses, 1)
        identity_reprojection_loss = paddle.min(identity_reprojection_losses, axis=1, keepdim=True)
        reprojection_loss = paddle.min(reprojection_losses, axis=1, keepdim=True)

        reprojection_loss_mask, depth_hint_loss_mask = \
                compute_loss_masks(reprojection_loss,
                                    identity_reprojection_loss,
                                    depth_hint_reproj_loss)

        # standard reprojection losses
        reprojection_loss = reprojection_loss * reprojection_loss_mask
        reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

        outputs["identity_selection/{}".format(scale)] = (1 - reprojection_loss_mask).astype('float32')
        losses['reproj_loss/{}'.format(scale)] = reprojection_loss

        # proxy supervision losses
        depth_hint_loss = 0
        if opt.use_depth_hints:
            target = inputs['depth_hint']
            pred = outputs[('depth', 0, scale)]
            valid_pixels = inputs['depth_hint_mask']

            depth_hint_loss = compute_proxy_supervised_loss(pred, target, valid_pixels,
                                                                 depth_hint_loss_mask)
            depth_hint_loss = depth_hint_loss.sum() / (depth_hint_loss_mask.sum() + 1e-7)
            # save for logging
            outputs["depth_hint_pixels/{}".format(scale)] = depth_hint_loss_mask
            losses['depth_hint_loss/{}'.format(scale)] = depth_hint_loss

        loss += reprojection_loss + depth_hint_loss
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)
        loss += opt.disparity_smoothness * smooth_loss / (2 ** scale)
        total_loss += loss / (2 ** scale)

        losses["losses/{}".format(scale)] = loss / (2 ** scale)

    losses["losses"] = total_loss

    return losses


def compute_depth_losses(inputs, outputs, print_info=True):
    """Compute depth metrics, to allow monitoring during training

    This isn't particularly accurate as it averages over the entire batch,
    so is only used to give an indication of validation performance
    """
    depth_pred = outputs[("depth", 0, 0)]
    depth_pred = paddle.clip(F.interpolate(
        depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_pred = depth_pred.detach()

    depth_gt = inputs["depth_gt"]
    mask = depth_gt > 0

    # garg/eigen crop
    crop_mask = paddle.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= np.median(depth_gt) / np.median(depth_pred)

    depth_pred = paddle.clip(depth_pred, min=1e-3, max=80)

    depth_errors = compute_depth_errors(depth_gt, depth_pred)

    mean_errors = np.array(depth_errors)
    if print_info:
        print("eval")
        print("  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("-> Done!")

    return mean_errors[2]