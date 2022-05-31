import argparse
import os

import paddle
from paddle.vision import transforms as tfs

import networks
from datasets.mono_dataset import pil_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument('--scales', nargs="+", type=int, help="scales used in the loss", default=[0, 1, 2, 3])
    parser.add_argument('--height', dest='height', default=192, type=int)
    parser.add_argument('--width', dest='width', default=640, type=int)
    parser.add_argument('--min_depth', dest='min_depth', default=0.1, type=float)
    parser.add_argument('--max_depth', dest='max_depth', default=100, type=float)
    parser.add_argument('--num_layers', dest='num_layers', default=18, type=int)

    parser.add_argument("--load_weights_folder", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./inference")

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()

    models = {}
    models['encoder'] = networks.ResnetEncoder_multi_sa_add_reduce_640(opt.num_layers, False)
    models['depth'] = networks.DepthDecoderAttention_edge(models['encoder'].num_ch_enc)

    models_to_load = ["encoder", "depth"]
    load_weights_folder = opt.load_weights_folder
    if os.path.exists(load_weights_folder):
        for n in models_to_load:
            path = os.path.join(load_weights_folder, "{}.pdparams".format(n))
            model_dict = models[n].state_dict()
            pretrained_dict = paddle.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            models[n].set_state_dict(model_dict)
        print('Loaded trained params of model successfully.')
    else:
        print('Weight file dose not exist.')

    for m in models.values():
        m.eval()

    input_spec1 = paddle.static.InputSpec(shape=[1, 3, opt.height, opt.width], dtype='float32', name='image')
    model1 = paddle.jit.to_static(models['encoder'], input_spec=[input_spec1])
    # [1, 64, 96, 320]
    # [1, 64, 48, 160]
    # [1, 128, 24, 80]
    # [1, 256, 12, 40]
    # [1, 512, 6, 20]
    
    input_spec2 = [paddle.static.InputSpec(shape=[1, 64, 96, 320], dtype='float32', name='feature1'),
                    paddle.static.InputSpec(shape=[1, 64, 48, 160], dtype='float32', name='feature2'),
                    paddle.static.InputSpec(shape=[1, 128, 24, 80], dtype='float32', name='feature3'),
                    paddle.static.InputSpec(shape=[1, 256, 12, 40], dtype='float32', name='feature4'),
                    paddle.static.InputSpec(shape=[1, 512, 6, 20], dtype='float32', name='feature5')]
    model2 = paddle.jit.to_static(models['depth'], input_spec=[input_spec2])
    save_path1 = os.path.join(opt.save_dir, 'model_encoder')
    paddle.jit.save(model1, save_path1)
    save_path2 = os.path.join(opt.save_dir, 'model_depth')
    paddle.jit.save(model2, save_path2)
    print(f'Model is saved in {opt.save_dir}.')
