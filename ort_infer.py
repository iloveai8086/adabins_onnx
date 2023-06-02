import os
import glob
from tqdm import tqdm
import numpy as np
import onnxruntime
import matplotlib
from PIL import Image


def colorize(value, vmin=None, vmax=None, cmap='magma_r'):
    value = value[0, :, :]
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    # img = value[:, :, :3]
    # return img.transpose((2, 0, 1))
    return value


def preprocess(image, mean, std):
    image = (image - mean) / std  # normalize
    image = image.astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = image[None, ...]
    return image


def predict_dir(test_dir, out_dir, session):
    os.makedirs(out_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(test_dir, "*"))
    min_depth = 1e-3
    max_depth = 10
    saving_factor = 256
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    for f in tqdm(all_files):
        image = np.asarray(Image.open(f), dtype='float32') / 255.
        image = preprocess(image, imagenet_mean, imagenet_std)
        pred = session.run(["centers", "final"], {"image": image})
        centers, final = pred
        # print(final.shape)  # (1, 1, 540, 960)
        # final = final.squeeze().cpu().numpy()
        final[final < min_depth] = min_depth
        final[final > max_depth] = max_depth
        final[np.isinf(final)] = max_depth
        final[np.isnan(final)] = min_depth
        centers = centers[centers > min_depth]
        centers = centers[centers < max_depth]

        # final centers
        final = (final * saving_factor).astype('uint16')
        print(final.shape)
        # basename = os.path.basename(f).split('.')[0]
        # save_path = os.path.join(out_dir, basename + ".png")
        img = colorize(final)
        img = img.squeeze()[:, :, :3]
        basename = os.path.basename(f).split('.')[0]
        save_path = os.path.join(out_dir, basename + ".png")
        Image.fromarray(img).save(save_path)


if __name__ == '__main__':
    session = onnxruntime.InferenceSession("sim.onnx", providers=["CUDAExecutionProvider"])  # CPU 后端cpu cuda trt的,可选“CPUExecutionProvider”
    predict_dir("video/img/",
                "video/out4/", session)
