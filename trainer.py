# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import time
import random

from utils import *
from networks.layers import *
from losses.loss import compute_depth_losses, compute_losses
import datasets
import networks

import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.io import DataLoader

import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.models = {}
        self.parameters_to_train = []

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2
        self.use_pose_net = True
        self.opt.frame_ids.append("s")
        self.min_rmse = 100

        self.models["encoder"] = networks.ResnetEncoder_multi_sa_add_reduce_640(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoderAttention_edge(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["pose_encoder"] = networks.ResnetEncoder_multi_sa_add_reduce_640(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.load_weights_folder is not None:
            self.load_model()

        self.model_lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=self.opt.learning_rate,
                                                                step_size=self.opt.scheduler_step_size, gamma=0.1,
                                                                verbose=True)
        self.model_optimizer = optim.Adam(learning_rate=self.model_lr_scheduler, parameters=self.parameters_to_train)

        print("Models and images files are saved to:\n  ", self.opt.log_dir)

        # data
        self.dataset = datasets.KITTIRAWDataset
        self.split = self.opt.split
        fpath = os.path.join(os.path.dirname(__file__), "splits/{}/{}_files.txt")
        img_ext = '.jpg' 
        train_filenames = readlines(fpath.format(self.split, "train"))
        mid_filenames = []
        for name in train_filenames:
            f_str = "{:010d}{}".format(int(name.split(' ')[1]) - 1, img_ext)
            image_path1 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)

            f_str = "{:010d}{}".format(int(name.split(' ')[1]) + 1, img_ext)
            image_path2 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)

            f_str = "{:010d}{}".format(int(name.split(' ')[1]), img_ext)
            image_path3 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)
            if os.path.exists(image_path1) and os.path.exists(image_path2) and os.path.exists(image_path3):
                mid_filenames.append(name)
            # else:
            #     print(name)
        train_filenames = mid_filenames

        val_filenames = readlines(fpath.format(self.split, "val"))
        mid_filenames = []
        for name in val_filenames:
            #if (name.split('/')[0] == '2011_10_03'):
            f_str = "{:010d}{}".format(int(name.split(' ')[1]) - 1, img_ext)
            image_path1 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)

            f_str = "{:010d}{}".format(int(name.split(' ')[1]) + 1, img_ext)
            image_path2 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)

            f_str = "{:010d}{}".format(int(name.split(' ')[1]), img_ext)
            image_path3 = os.path.join(self.opt.data_path, name.split(' ')[0], "image_00/data", f_str)
            if os.path.exists(image_path1) and os.path.exists(image_path2) and os.path.exists(image_path3):
                mid_filenames.append(name)
            # else:
            #     print(name)
        val_filenames = mid_filenames

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * (self.opt.num_epochs-self.opt.start_epoch)

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.use_depth_hints, self.opt.depth_hint_path,
            is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, drop_last=True)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.use_depth_hints, self.opt.depth_hint_path,
            is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, drop_last=True)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)

        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()
            for param in m.parameters():
                param.trainable = True

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        os.makedirs(self.opt.log_dir, exist_ok=True)
        self.logger = get_logger(self.opt.log_dir + '/train.log')
        self.logger.info('start training!')

        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
        self.logger.info('finish training!')

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()
        print(F'EPOCH {self.epoch}:')
        for batch_idx, inputs in enumerate(self.train_loader):
            if batch_idx % 20 == 0:
                print('.', end='')
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.clear_grad()
            losses["losses"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:
                self.log_time(duration, losses["losses"].cpu())
                if "depth_gt" in inputs:
                    rmse = compute_depth_losses(inputs, outputs)
                    self.logger.info("train_dateset_rmse={}".format(rmse))
                self.save_img("train", inputs, outputs)
                self.val()

            self.step += 1
        self.val()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)

        losses = compute_losses(self.opt, inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.models["pose_encoder"](paddle.concat(pose_inputs, 1))]

                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def val(self, only_val=False):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        if only_val:
            os.makedirs(self.opt.log_dir, exist_ok=True)
            self.logger = get_logger(self.opt.log_dir + '/val.log')
            self.epoch = 0
            self.step = 0

        with paddle.no_grad():
            rmses = []
            print("eval: ", end='')
            rmse_min = 100
            for batch_idx, inputs in enumerate(self.val_loader):
                print(".", end='')
                outputs, losses = self.process_batch(inputs)

                if "depth_gt" in inputs:
                    rmse = compute_depth_losses(inputs, outputs, print_info=False)
                    rmses.append(rmse)
                    if rmse < rmse_min:
                        rmse_min = rmse
                        inputs_to_save, outputs_to_save = inputs, outputs
            rmse_avg = np.mean(rmses)
            rmse_max = np.max(rmses)
            self.logger.info("val_dateset: rmse_avg={},rmse_min={} ,rmse_max={} ".format(rmse_avg, rmse_min, rmse_max))
            if rmse_avg > 10 and self.min_rmse != 100:
                print("load_ckpt")
                self.load_best_model()
            if rmse_avg < self.min_rmse:
                self.min_rmse = rmse_avg
                if not only_val:
                    self.save_best_model()
                    

            self.save_img("val", inputs_to_save, outputs_to_save)
            del inputs, outputs, losses, inputs_to_save, outputs_to_save

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                inputs[("color", frame_id, source_scale)].stop_gradient = False
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]

                if self.opt.use_depth_hints:
                    if frame_id == 's' and scale == 0:
                        # generate depth hint warped image (only max scale and for stereo image)
                        depth = inputs['depth_hint']
                        cam_points = self.backproject_depth[source_scale](
                            depth, inputs[("inv_K", source_scale)])
                        pix_coords = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T)

                        inputs[("color", frame_id, source_scale)].stop_gradient = False
                        outputs[("color_depth_hint", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords, padding_mode="border")

    def log_time(self, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | step {:>6} | examples/s: {:5.1f}" + \
                       " | losses: {} | time elapsed: {} | time left: {}"
        loss = loss.numpy()
        # print(loss)
        self.logger.info(print_string.format(self.epoch, self.step, samples_per_sec, loss, time_sofar // 60,
                                             training_time_left // 3600))

    def save_img(self, mode, inputs, outputs):
        """Write imgs file
        """
        save_dir = os.path.join(self.opt.log_dir, "imgs_{}".format(mode))
        os.makedirs(save_dir, exist_ok=True)
        path2save = os.path.join(save_dir, "epoch_{}_step_{}.jpg".format(self.epoch, self.step))
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            input_color = inputs[("color", 0, 0)][j].transpose([1, 2, 0]).cpu().numpy() * 255
            input_color = cv2.cvtColor(input_color, cv2.COLOR_RGB2BGR)
            pred_depth = normalize_image(outputs[("disp", 0)][j, 0]).detach().cpu().numpy() * 255
            pred_depth = cv2.cvtColor(pred_depth, cv2.COLOR_GRAY2BGR)
            mid_img2save = np.concatenate((input_color, pred_depth), 0)
            disp = 1 / (inputs['depth_hint'] + 1e-7) * inputs['depth_hint_mask']
            disp = normalize_image(disp[j, 0]).cpu().numpy() * 255
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            mid_img2save = np.concatenate((mid_img2save, disp), 0)
            if j == 0:
                img2save = mid_img2save
            else:
                img2save = np.concatenate((img2save, mid_img2save), 1)

        cv2.imwrite(path2save, img2save)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.opt.log_dir, "models", "weights_{}".format(self.epoch))
        print("save_model to {}".format(save_folder))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pdparams".format(model_name))
            to_save = model.state_dict()
            paddle.save(to_save, save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pdparams".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = paddle.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].set_state_dict(model_dict)

    def save_best_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.opt.log_dir, "models", "best_weights")
        print("save best model to {}".format(save_folder))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pdparams".format(model_name))
            to_save = model.state_dict()
            paddle.save(to_save, save_path)

    def load_best_model(self):
        """Load model(s) from disk
        """
        save_folder = os.path.join(self.opt.log_dir, "models", "best_weights")

        print("loading best model from folder {}".format(save_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(save_folder, "{}.pdparams".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = paddle.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].set_state_dict(model_dict)
