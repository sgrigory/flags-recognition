import numpy as np
import os
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import cv2
import glob

from tensorflow.keras.layers import Dense
from tensorflow.keras import applications, models


def load_backgrounds(path, module, width, height, lim=1000):
    all_backgrounds = glob.glob(path)
    backgrounds_raw = [mpimage.imread(img_file) for img_file in all_backgrounds[:lim]]
    backgrounds = [module.preprocess_input(x) for x in backgrounds_raw if (len(x.shape) == 3) & (x.shape[0] >= width) & (x.shape[1] >= height)]
    return backgrounds


def load_images(df, images_path, module):

    images = []
    images_raw = []
    masks = []
    images_orig = []
    for fname in df["flag_128"]:
        #print(fname)
        file_path = os.path.join(images_path, fname)
        img = mpimage.imread(file_path)
        images_orig.append(img)
        if img.shape[2] == 4:
            mask = img[:, :, 3]
            img = img[:, :, :3] * 256
        else:
            mask = np.ones(img.shape[:2])
            img *= 256
        #img = cv2.resize(img, (width, height))
        images_raw.append(img.copy())
        #print(img.shape)
        img = module.preprocess_input(img)
        images.append(img)
        masks.append(mask)
    images = np.stack(images)
    masks = np.stack(masks)
    masks = np.transpose(np.array([masks] * 3), axes=(1, 2, 3, 0))

    return images, masks



def create_model(n_classes, train_batch_norm=False, num_frozen=250):
    # head_layers = [n_classes * 4, n_classes * 2]

    trained_model = applications.InceptionV3(weights="imagenet")

    head_layers = []
    layer_inner = trained_model.layers[-2].output

    for n_nodes in head_layers:
        layer_inner = Dense(n_nodes, activation="relu")(layer_inner)

    last_layer = Dense(n_classes, activation="softmax")(layer_inner)

    seq2 = models.Model(inputs=trained_model.input, outputs=last_layer)
    for i, layer in enumerate(seq2.layers):
        if layer.name.find("batch_normalization") != -1:
            layer.trainable = train_batch_norm
        if i < num_frozen:
            layer.trainable = False
    return seq2


def train_gen_overlay2(batch_size, backgrounds, images, masks, y, min_scale=1):
    assert len(images) == len(masks)
    width, height = 299, 299
    while True:

        # create a random batch of backgrounds
        backgrounds_sample = get_backgrounds_sample(backgrounds, width, height, batch_size)

        # create a random batch of images
        sample = random.choices(range(len(images)), k=batch_size)
        imgs_sample = images[sample]
        masks_sample = masks[sample]

        imgs_trans, masks_trans = [], []

        max_scale_x = backgrounds_sample.shape[1] / masks.shape[1]
        max_scale_y = backgrounds_sample.shape[2] / masks.shape[2]
        max_scale = min(max_scale_x, max_scale_y)
        scale = np.random.rand() * (max_scale - min_scale) + min_scale
        for img, mask in zip(imgs_sample, masks_sample):
            img_trans, mask_trans = transform_image_mask(img, mask, scale)
            imgs_trans.append(img_trans)
            masks_trans.append(mask_trans)

        imgs_trans = np.array(imgs_trans)
        masks_trans = np.array(masks_trans)

        imgs_width, imgs_height = masks_trans.shape[1:-1]
        # select a random position in the background window
        x_img, y_img = (random.randint(0, backgrounds_sample.shape[1] - imgs_width),
                        random.randint(0, backgrounds_sample.shape[2] - imgs_height))

        overlayed = lay_over_background(backgrounds_sample, imgs_trans, masks_trans, x_img, y_img)
        yield overlayed, y[sample]


def get_backgrounds_sample(backgrounds, width, height, batch_size):
    backgrounds_samples = []
    idx_backgrounds = np.random.choice(range(len(backgrounds)), size=batch_size)
    for i in idx_backgrounds:
        x0, y0 = random.randint(0, backgrounds[i].shape[0] - width), random.randint(0, backgrounds[i].shape[1] - height)
        backgrounds_samples.append(backgrounds[i][x0: x0 + width, y0: y0 + height])
    return np.array(backgrounds_samples)


def lay_over_background(overlayed, imgs_trans, masks_trans, x_img, y_img):

    imgs_width, imgs_height = imgs_trans.shape[1], imgs_trans.shape[2]
    assert (imgs_width > 0) & (imgs_height > 0)
    from_images = imgs_trans * masks_trans

    if len(overlayed.shape) != 4:
        print(overlayed.shape)
        print(imgs_trans.shape)
        print(x_img, y_img)
    assert len(overlayed.shape) == 4

    from_background = overlayed[:, x_img: x_img + imgs_width, y_img: y_img + imgs_height] * (1 - masks_trans)
    overlayed[:, x_img: x_img + imgs_width, y_img: y_img + imgs_height] = from_images + from_background
    return overlayed


def transform_image_mask(img, mask, scale=2, brightness=(0.7, 1.5)):
    assert img.shape == mask.shape
    img_w = img.shape[0]
    img_h = img.shape[1]
    transf, img_h_new, img_w_new = get_random_transform(img_h, img_w, scale)
    out_mask = cv2.warpAffine(mask, transf, (img_h_new, img_w_new))
    out_img = cv2.warpAffine(img * 0.5 + 0.5, transf, (img_h_new, img_w_new))

    out_img *= np.random.rand() * (brightness[1] - brightness[0]) + brightness[0]

    out_img = out_img.clip(0, 1)

    out_img = (out_img - 0.5) * 2

    return out_img, out_mask


def get_random_transform(img_w, img_h, scale, alpha=0.3):
    src = np.float32([(0, 0), (0, img_h), (img_w, 0)])
    max_shift_h = int(img_h * alpha)
    max_shift_w = int(img_w * alpha)
    dst = np.float32([(random.randint(0, max_shift_w), random.randint(0, max_shift_h)),
                      (random.randint(0, max_shift_w), img_h - random.randint(0, max_shift_h)),
                      (img_w - random.randint(0, max_shift_w), random.randint(0, max_shift_h))])
    dst *= scale
    transf = cv2.getAffineTransform(src, dst)
    # sc_factor = max(dst[:, 1].max() / img_h, dst[:, 0].max() / img_w)
    # print(f"sc_factor {sc_factor}")
    return transf, int(img_w * scale), int(img_h * scale)


def rand_point(w, h):
    return random.randint(0, int(w)), random.randint(0, int(h))


def show_next(train_gen, bs1=16):
    batch, y_batch = next(train_gen)
    show_batch(batch, bs1=bs1)


def show_batch(batch, bs1=16):
    fig, axs = plt.subplots(bs1 // 4, 4, figsize=(10, 10))
    for i in range(bs1):
        axs[i // 4, i % 4].imshow((batch[i] + 1) * 0.5)
        # axs[i // 4, i % 4].set_title(f"{y_one}")