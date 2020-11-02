import numpy as np
import os
import random
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import cv2
import glob

from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import applications, models

N_COORDS = 8

class PredictionEngine:

    def __init__(self, db_path, model_path):
        self.model = models.load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})

        self.df = load_df(db_path)
        self.n_classes = self.df.shape[0]

        self.module = applications.inception_v3
        self.input_width = 299
        self.input_height = 299


    def preprocess_pred(self, example_raw, d_contr=1, d_light=0, scale=1, shift=0):
        example = self.module.preprocess_input(example_raw)
        # example = cluster_spatial(example, n_clusters=20)
        #example = cv2.GaussianBlur(example, ksize=(11, 11), sigmaX=1)
        example_tuned = tune_contrast_light(example, d_contr=d_contr, d_light=d_light)

        preds, coords, _ = self.pred_resize_one_image(example_tuned, scale=scale, shift=shift)
        return preds[:5], coords

    def pred_resize_one_image(self, example, scale=1, shift=0):
        ex_w, ex_h = example.shape[:2]
        min_factor = min(self.input_width / ex_w, self.input_height / ex_h) * scale
        resized_example1 = cv2.resize(example, (int(ex_h * min_factor), int(ex_w * min_factor)))
        blank = np.random.rand(self.input_width, self.input_height, 3)  # np.zeros((width, height, 3))
        x0, y0 = int(self.input_width * shift), int(self.input_height * shift)
        x_min = min(resized_example1.shape[0] + x0, blank.shape[0])
        y_min = min(resized_example1.shape[1] + y0, blank.shape[1])
        blank[x0: x_min, y0: y_min, :] = resized_example1[: x_min - x0, : y_min - y0, :]
        preds, coords = self.pred_one_image(blank)
        return preds, coords, blank

    def pred_one_image(self, blank):
        resized_example = np.expand_dims(blank, axis=0)

        preds, coords = self.model(resized_example, training=False)
        preds = preds[0]
        preds = pd.Series(preds, index=self.df["name"][:self.n_classes]).sort_values(ascending=False)
        return preds, coords.numpy()[0]


def load_df(path):
    df = pd.read_sql_table("Countries", path)
    df = df[(df["name"] != "United States Minor Outlying Islands") &
            (df["name"] != "Bouvet Island") &
            (df["name"] != "Svalbard and Jan Mayen") &
            (df["name"] != "Bonaire, Saint Eustatius And Saba") &
            (df["name"] != "Heard Island And McDonald Islands")
            ]

    df = df[(df["name"] != "Indonesia") &
            (df["name"] != "Monaco")
            ]
    # df.loc[df["name"] == "Poland", "name"] == "Poland, Monaco, or Indonesia"

    df = df[df["name"] != "Chad"]

    # df.loc[df["name"] == "Romania", "name"] == "Chad or Romania"

    df = df[~df.name.isin(['Martinique', 'Saint Pierre and Miquelon', 'Réunion', 'Guadeloupe',
                           'Wallis and Futuna', 'Saint Martin', 'Saint Barthélemy',
                           'Mayotte', 'French Guiana'])]

    return df



def load_backgrounds(path, module, width, height, lim=1000):
    all_backgrounds = glob.glob(path)
    backgrounds_raw = [mpimage.imread(img_file) for img_file in all_backgrounds[:lim]]
    backgrounds = [module.preprocess_input(x) for x in backgrounds_raw if (len(x.shape) == 3) & (x.shape[0] >= width) & (x.shape[1] >= height)]
    return backgrounds


def load_images(df, images_path, module, lim=None):

    images = []
    images_raw = []
    masks = []
    images_orig = []
    for fname in df["flag_128"][:lim]:
        #print(fname)
        file_path = os.path.join(images_path, fname)
        img = mpimage.imread(file_path)
        images_orig.append(img)
        if img.shape[2] == 4:
            mask = img[:, :, 3]
            img = img[:, :, :3] * 255
        else:
            mask = np.ones(img.shape[:2])
            img *= 255
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


def create_model_box(n_classes, train_batch_norm=False, num_frozen=250):
    # head_layers = [n_classes * 4, n_classes * 2]

    trained_model = applications.InceptionV3(weights="imagenet")

    head_layers = []
    layer_inner = trained_model.layers[-2].output

    for n_nodes in head_layers:
        layer_inner = Dense(n_nodes, activation="relu")(layer_inner)

    last_layer = Dense(n_classes, activation="softmax", name="class_probs")(layer_inner)

    box_coords = Dense(N_COORDS, activation=LeakyReLU(alpha=0.1), name="box_coords")(layer_inner)

    seq2 = models.Model(inputs=trained_model.input, outputs=[last_layer, box_coords])
    for i, layer in enumerate(seq2.layers):
        if layer.name.find("batch_normalization") != -1:
            layer.trainable = train_batch_norm
        if i < num_frozen:
            layer.trainable = False
    return seq2


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


def train_gen_overlay_box(batch_size, backgrounds, images, masks, y, min_scale=1.5,
                       contrast_bounds=(0.3, 1.5), light_bounds=(-0.2, 0.2),
                        alpha_min=0.7, affine_shift=0.3):
    assert len(images) == len(masks)
    width, height = 299, 299
    while True:

        # create a random batch of backgrounds
        backgrounds_sample = get_backgrounds_sample(backgrounds, width, height, batch_size)

        # create a random batch of images
        sample = random.choices(range(len(images)), k=batch_size)
        imgs_sample = images[sample]
        masks_sample = masks[sample]

        imgs_trans, masks_trans, coords = [], [], []

        max_scale_x = backgrounds_sample.shape[1] / masks.shape[1]
        max_scale_y = backgrounds_sample.shape[2] / masks.shape[2]
        max_scale = min(max_scale_x, max_scale_y)
        scale = np.random.rand() * (max_scale - min_scale) + min_scale
        for img, mask in zip(imgs_sample, masks_sample):
            img_trans, mask_trans, dst = transform_image_mask_box(img, mask, scale,
                                                         contrast_bounds=contrast_bounds,
                                                         light_bounds=light_bounds,
                                                         alpha=affine_shift
                                                         )
            x0y0 = dst[0][::-1]
            x1y1 = dst[1][::-1] + dst[2][::-1] - x0y0
            x0y0x1y1 = np.concatenate([x0y0, dst[1][::-1], x1y1, dst[2][::-1]])
            coords.append(x0y0x1y1)
            imgs_trans.append(img_trans)
            masks_trans.append(mask_trans)

        imgs_trans = np.array(imgs_trans)
        masks_trans = np.array(masks_trans)
        coords = np.array(coords)

        imgs_width, imgs_height = masks_trans.shape[1:-1]
        # select a random position in the background window
        x_img, y_img = (random.randint(0, backgrounds_sample.shape[1] - imgs_width),
                        random.randint(0, backgrounds_sample.shape[2] - imgs_height))

        alpha = np.random.rand() * (1 - alpha_min) + alpha_min
        overlayed = lay_over_background(backgrounds_sample, imgs_trans, masks_trans,
                                        x_img, y_img, alpha)

        coords[:, 0] = (coords[:, 0] + x_img) / width
        coords[:, 2] = (coords[:, 2] + x_img) / width
        coords[:, 4] = (coords[:, 4] + x_img) / width
        coords[:, 6] = (coords[:, 6] + x_img) / width

        coords[:, 1] = (coords[:, 1] + y_img) / height
        coords[:, 3] = (coords[:, 3] + y_img) / height
        coords[:, 5] = (coords[:, 5] + y_img) / height
        coords[:, 7] = (coords[:, 7] + y_img) / height

        eps = 1e-4
        coords = coords.clip(eps, 1 - eps)

        yield overlayed, {"class_probs": y[sample], "box_coords": coords}


def train_gen_overlay2(batch_size, backgrounds, images, masks, y, min_scale=1.5,
                       contrast_bounds=(0.3, 1.5), light_bounds=(-0.2, 0.2), alpha_min=0.7):
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
            img_trans, mask_trans = transform_image_mask(img, mask, scale,
                                                         contrast_bounds=contrast_bounds,
                                                         light_bounds=light_bounds,
                                                         )
            imgs_trans.append(img_trans)
            masks_trans.append(mask_trans)

        imgs_trans = np.array(imgs_trans)
        masks_trans = np.array(masks_trans)

        imgs_width, imgs_height = masks_trans.shape[1:-1]
        # select a random position in the background window
        x_img, y_img = (random.randint(0, backgrounds_sample.shape[1] - imgs_width),
                        random.randint(0, backgrounds_sample.shape[2] - imgs_height))

        alpha = np.random.rand() * (1 - alpha_min) + alpha_min
        overlayed = lay_over_background(backgrounds_sample, imgs_trans, masks_trans,
                                        x_img, y_img, alpha)
        yield overlayed, y[sample]


def get_backgrounds_sample(backgrounds, width, height, batch_size, color_shift=0.8):
    backgrounds_samples = []
    idx_backgrounds = np.random.choice(range(len(backgrounds)), size=batch_size)
    rgb = np.array([(np.random.rand() - 0.5) * color_shift for _ in range(3)])
    rgb -= sum(rgb) / 3
    rgb += 1
    for i in idx_backgrounds:
        x0, y0 = random.randint(0, backgrounds[i].shape[0] - width), random.randint(0, backgrounds[i].shape[1] - height)
        cutout = backgrounds[i][x0: x0 + width, y0: y0 + height]
        cutout = tune_contrast_light(cutout,
                                     d_light=0.8 * (np.random.rand() - 0.5) * 2,
                                     d_contr=np.random.rand() * 2,
                                     rgb=rgb
                                     )
        backgrounds_samples.append(cutout)
    return np.array(backgrounds_samples)


def lay_over_background(overlayed, imgs_trans, masks_trans, x_img, y_img, alpha):

    imgs_width, imgs_height = imgs_trans.shape[1], imgs_trans.shape[2]
    assert (imgs_width > 0) & (imgs_height > 0)

    masks_trans *= alpha
    from_images = imgs_trans * masks_trans

    if len(overlayed.shape) != 4:
        print(overlayed.shape)
        print(imgs_trans.shape)
        print(x_img, y_img)
    assert len(overlayed.shape) == 4

    from_background = overlayed[:, x_img: x_img + imgs_width, y_img: y_img + imgs_height] * (1 - masks_trans)
    overlayed[:, x_img: x_img + imgs_width,
                 y_img: y_img + imgs_height] = from_images + from_background
    return overlayed


def transform_image_mask_box(img, mask, scale=4, contrast_bounds=(0.2, 1.5),
                             light_bounds=(-0.3, 0.3), color_shift=0.8, alpha=0.3):
    assert img.shape == mask.shape
    img_w = img.shape[0]
    img_h = img.shape[1]
    transf, img_h_new, img_w_new, dst = get_random_transform_box(img_h, img_w, scale, alpha=alpha)
    out_mask = cv2.warpAffine(mask, transf, (img_h_new, img_w_new))
    out_img = cv2.warpAffine(img * 0.5 + 0.5, transf, (img_h_new, img_w_new))
    out_img = (out_img - 0.5) * 2

    contrast = np.random.rand() * (contrast_bounds[1] - contrast_bounds[0]) + contrast_bounds[0]

    light = np.random.rand() * (light_bounds[1] - light_bounds[0]) + light_bounds[0]

    rgb = np.array([(np.random.rand() - 0.5) * color_shift for _ in range(3)])
    rgb -= sum(rgb) / 3
    rgb += 1
    out_img = tune_contrast_light(out_img, light, contrast, rgb=rgb)

    return out_img, out_mask, dst

def transform_image_mask(img, mask, scale=4, contrast_bounds=(0.2, 1.5), light_bounds=(-0.3, 0.3), color_shift=0.8):
    assert img.shape == mask.shape
    img_w = img.shape[0]
    img_h = img.shape[1]
    transf, img_h_new, img_w_new = get_random_transform(img_h, img_w, scale)
    out_mask = cv2.warpAffine(mask, transf, (img_h_new, img_w_new))
    out_img = cv2.warpAffine(img * 0.5 + 0.5, transf, (img_h_new, img_w_new))
    out_img = (out_img - 0.5) * 2

    contrast = np.random.rand() * (contrast_bounds[1] - contrast_bounds[0]) + contrast_bounds[0]

    light = np.random.rand() * (light_bounds[1] - light_bounds[0]) + light_bounds[0]

    rgb = np.array([(np.random.rand() - 0.5) * color_shift for _ in range(3)])
    rgb -= sum(rgb) / 3
    rgb += 1
    out_img = tune_contrast_light(out_img, light, contrast, rgb=rgb)

    return out_img, out_mask


def get_random_transform_box(img_w, img_h, scale, alpha=0.3):
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
    return transf, int(img_w * scale), int(img_h * scale), dst


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


def tune_contrast_light(im, d_light, d_contr, rgb=None):
    im = ((im + d_light) * d_contr)
    if rgb is not None:
        im = (im + 1) * 0.5
        im *= np.array(rgb)
        im = im * 2 - 1
    return im.clip(-1, 1)