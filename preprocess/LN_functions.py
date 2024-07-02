'''
step 0
functions for LN patch processing
date: 2021-02-01
author: xiaw@sibet.ac.cn
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pylab

def diameter_calculate(LN_patch):
    # edge extraction
    LN_patch = LN_patch.astype(np.uint8)
    contour, hierarchy = cv2.findContours(LN_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = contour[0]
    rect = cv2.minAreaRect(edge)
    x, y = rect[0]
    w, h = rect[1]
    angle = rect[2]
    long_diameter = max(w, h)
    short_diameter = min(w, h)

    # # cv2.fitEllipse(points)
    # # points类型要求是numpy.array（[[x,y],[x1,y1]...]）:拟合出一个椭圆尽量使得点都在圆上
    # try:
    #     # edge extraction
    #     contours, hierarchy = cv2.findContours(LN_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     # fit the eclipse
    #     edge = contours[0]
    #     ellipse = cv2.fitEllipse(edge)
    #     # extract the long and short diameter
    #     a = ellipse[1][0]
    #     b = ellipse[1][1]
    #     long_diameter = max(a, b)
    #     short_diameter = min(a, b)
    # except:
    #     long_diameter = 0
    #     short_diameter = 0

    return long_diameter, short_diameter


def change_dataset_size(x, y, batch_size):
    length = len(x)
    if (length % batch_size != 0):
        remainder = length % batch_size
        x = x[:(length - remainder)]
        y = y[:(length - remainder)]
        print(len(x))
        print(len(y))
    return x, y

def img_norl(img):
    # 0-1
    img_min = np.min(img)
    img = img - img_min
    img_max = np.max(img)

    if img_max == 0:
        return img
    else:
        return img/img_max


# zero-padding and resize 2D LN patches for each patient
def get_pLevel_LN_patches_Padding(P_LN_patches_img, LN_z=56, LN_y=32, LN_x=32):
    P_LN_patches_img_padding = np.zeros([LN_z, LN_y, LN_x])
    patch_num = len(P_LN_patches_img)
    for ind_patch in range(0, patch_num):
        LN_patch = P_LN_patches_img[ind_patch]
        # LN_patch = img_norl(LN_patch)
        LN_size = LN_patch.shape

        resize_0 = LN_size[0]
        resize_1 = LN_size[1]

        y_pad_diff = 0
        x_pad_diff = 0

        if resize_0 > resize_1:
            y_pad_diff = resize_0 - resize_1
        else:
            x_pad_diff = resize_1 - resize_0

        y_pad_size_0 = y_pad_diff // 2
        y_pad_size_1 = y_pad_diff - y_pad_size_0

        x_pad_size_0 = x_pad_diff // 2
        x_pad_size_1 = x_pad_diff - x_pad_size_0

        LN_patch_zero_padding = np.pad(LN_patch,
                                       ((y_pad_size_0, y_pad_size_1),
                                        (x_pad_size_0, x_pad_size_1)),
                                       'constant', constant_values=(0, 0))

        LN_patch_resize_slice = cv2.resize(LN_patch_zero_padding, (LN_y, LN_x),
                                                          interpolation=cv2.INTER_LINEAR)
        # # show LN patch
        # plt.imshow(LN_patch_resize_slice, cmap='gray')
        # pylab.show()

        P_LN_patches_img_padding[ind_patch] = LN_patch_resize_slice

    return P_LN_patches_img_padding


# resize 2D LN patches for each patient  -- correct the zero padding directions
def get_pLevel_LN_patches_Resize(P_LN_patches_img, LN_y=32, LN_x=32):
    patch_num = len(P_LN_patches_img)
    P_LN_patches_img_padding = np.zeros([patch_num, LN_y, LN_x])
    for ind_patch in range(0, patch_num):
        LN_patch = P_LN_patches_img[ind_patch]
        # LN_patch = img_norl(LN_patch)
        LN_size = LN_patch.shape

        resize_0 = LN_size[0]
        resize_1 = LN_size[1]

        y_pad_diff = 0
        x_pad_diff = 0

        if resize_0 > resize_1:
            y_pad_diff = resize_0 - resize_1
        else:
            x_pad_diff = resize_1 - resize_0

        y_pad_size_0 = y_pad_diff // 2
        y_pad_size_1 = y_pad_diff - y_pad_size_0

        x_pad_size_0 = x_pad_diff // 2
        x_pad_size_1 = x_pad_diff - x_pad_size_0

        LN_patch_zero_padding = np.pad(LN_patch,
                                       ((x_pad_size_0, x_pad_size_1),
                                        (y_pad_size_0, y_pad_size_1)),
                                       'constant', constant_values=(0, 0))

        LN_patch_resize_slice = cv2.resize(LN_patch_zero_padding, (LN_y, LN_x),
                                                          interpolation=cv2.INTER_LINEAR)
        # # show LN patch
        # plt.imshow(LN_patch_resize_slice, cmap='gray')
        # pylab.show()

        P_LN_patches_img_padding[ind_patch] = LN_patch_resize_slice

    return P_LN_patches_img_padding

# zero-padding and resize 2D LN patches for each patient
def get_pLevel_LN_patches_Padding_MultiModal(x_train, LN_z=56, LN_y=56, LN_x=56, channel = 2):
    x_train_padding = np.zeros([len(x_train), LN_z, LN_y, LN_x, channel])
    for ind_p in range(0, len(x_train)):
        p_sample = x_train[ind_p]
        patch_num = len(p_sample)
        p_sample_resize = np.zeros([LN_z, LN_y, LN_x, channel])
        for ind_channel in range(0, channel):
            for ind_patch in range(0, patch_num):
                LN_patch = p_sample[ind_patch]
                LN_patch = LN_patch[:, :, ind_channel]
                # LN_patch = img_norl(LN_patch)
                LN_size = LN_patch.shape

                resize_0 = LN_size[0]
                resize_1 = LN_size[1]

                y_pad_diff = 0
                x_pad_diff = 0

                if resize_0 > resize_1:
                    y_pad_diff = resize_0 - resize_1
                else:
                    x_pad_diff = resize_1 - resize_0

                y_pad_size_0 = y_pad_diff // 2
                y_pad_size_1 = y_pad_diff - y_pad_size_0

                x_pad_size_0 = x_pad_diff // 2
                x_pad_size_1 = x_pad_diff - x_pad_size_0

                LN_patch_zero_padding = np.pad(LN_patch,
                                               ((y_pad_size_0, y_pad_size_1),
                                                (x_pad_size_0, x_pad_size_1)),
                                               'constant', constant_values=(0, 0))

                LN_patch_resize_slice = cv2.resize(LN_patch_zero_padding, (LN_y, LN_x),
                                                                  interpolation=cv2.INTER_LINEAR)
                # plt.imshow(LN_patch_resize_slice, cmap='gray')
                # pylab.show()
                p_sample_resize[ind_patch, :, :, ind_channel] = LN_patch_resize_slice
        x_train_padding[ind_p] = p_sample_resize

    return x_train_padding

def get_pLevel_LN_patches_Padding_Size(x_train, LN_num=56):

    x_train_temp = x_train[0]
    LN_feas_temp = x_train_temp[0]
    LN_feas_num = len(LN_feas_temp)

    x_train_padding = np.zeros([len(x_train), LN_num, LN_feas_num])
    for ind_p in range(0, len(x_train)):
        p_sample = x_train[ind_p]
        patch_num = len(p_sample)
        p_sample_padding = np.zeros([LN_num, LN_feas_num])
        for ind_patch in range(0, patch_num):
            LN_feas = p_sample[ind_patch]
            LN_feas_norl = LN_feas
            for Ln_fea_ind in range(0, LN_feas_num):
                LN_feas_norl[Ln_fea_ind] = LN_feas[Ln_fea_ind]/56
            p_sample_padding[ind_patch] = LN_feas_norl
        x_train_padding[ind_p] = p_sample_padding
    return x_train_padding


# for variable size input and resize the patch according to the pooling number
def get_x_patches_data_paddingPooling(x_train):
    x_train_patches = []
    for ind_p in range(0, len(x_train)):
        p_sample = x_train[ind_p]
        patch_num = len(p_sample)
        for ind_patch in range(0, patch_num):

            LN_patch = p_sample[ind_patch]
            LN_patch = img_norl(LN_patch)
            LN_size = LN_patch.shape

            resize_slice_num = LN_size[0]
            resize_1 = LN_size[1]
            resize_2 = LN_size[2]

            # adjust the patch size according to the pooling nums
            # if resize_slice_num < 4:
            #     resize_slice_num = 4
            #
            # if (resize_slice_num % 2) != 0:
            #     resize_slice_num = resize_slice_num + 1

            if (resize_1 % 2) != 0:
                resize_1 = resize_1 + 1
                if (resize_1 % 4) != 0:
                    resize_1 = resize_1 + 2

            if (resize_1 % 4) != 0:
                resize_1 = resize_1 + 2

            if (resize_2 % 2) != 0:
                resize_2 = resize_2 + 1
                if (resize_2 % 4) != 0:
                    resize_2 = resize_2 + 2

            if (resize_2 % 4) != 0:
                resize_2 = resize_2 + 2

            # # resize image
            # for ind_slice in range(0, LN_size[0]):
            #     LN_patch_slice = LN_patch[ind_slice]
            #     LN_patch_resize_slice = cv2.resize(LN_patch_slice, (resize_2, resize_1),
            #                                                   interpolation=cv2.INTER_LINEAR)
            #     LN_patch_resize[ind_slice] = LN_patch_resize_slice

            # zero padding
            y_pad_diff = resize_1 - LN_size[1]
            x_pad_diff = resize_2 - LN_size[2]
            y_pad_size_0 = y_pad_diff // 2
            y_pad_size_1 = y_pad_diff - y_pad_size_0

            x_pad_size_0 = x_pad_diff // 2
            x_pad_size_1 = x_pad_diff - x_pad_size_0

            LN_patch_zero_padding = np.pad(LN_patch,
                                           ((0, 0), (y_pad_size_0, y_pad_size_1),
                                            (x_pad_size_0, x_pad_size_1)),
                                           'constant', constant_values=(0, 0))

            LN_patch_zero_padding = np.expand_dims(LN_patch_zero_padding, axis=3)

            x_train_patches.append(LN_patch_zero_padding)
    return x_train_patches

# do not change the LN patch size
def get_x_patches_data(x_train):
    x_train_patches = []
    z_max = 0
    y_max = 0
    x_max = 0
    for ind_p in range(0, len(x_train)):
        p_sample = x_train[ind_p]
        patch_num = len(p_sample)
        for ind_patch in range(0, patch_num):
            LN_patch = p_sample[ind_patch]
            LN_size = LN_patch.shape

            z = LN_size[0]
            y = LN_size[1]
            x = LN_size[2]

            if z > z_max:
                z_max = z
            if y > y_max:
                y_max = y
            if x > x_max:
                x_max = x

            LN_patch = img_norl(LN_patch)
            # LN_patch_resize = np.expand_dims(LN_patch_resize, axis=3)
            # LN_patch_resize = np.expand_dims(LN_patch_resize, axis=0)
            x_train_patches.append(LN_patch)
        shape_max = [z_max, y_max, x_max]
    return x_train_patches, shape_max

def cal_shape_max(shape_max_train, shape_max_val):
    z_max = shape_max_train[0]
    if shape_max_val[0] > z_max:
        z_max = shape_max_val[0]

    y_max = shape_max_train[1]
    if shape_max_val[1] > y_max:
        y_max = shape_max_val[1]

    x_max = shape_max_train[2]
    if shape_max_val[2] > x_max:
        x_max = shape_max_val[2]

    # # adjust the patch size according to the pooling nums
    # if z_max < 4:
    #     z_max = 4
    # if (z_max % 2) != 0:
    #     z_max = z_max + 1
    #
    # if (y_max % 2) != 0:
    #     y_max = y_max + 1
    #     if (y_max % 4) != 0:
    #         y_max = y_max + 2
    # if (y_max % 4) != 0:
    #     y_max = y_max + 2
    #
    # if (x_max % 2) != 0:
    #     x_max = x_max + 1
    #     if (x_max % 4) != 0:
    #         x_max = x_max + 2
    # if (x_max % 4) != 0:
    #     x_max = x_max + 2

    return [z_max, y_max, x_max]

def get_x_patches_zero_padding(x_data, padding_shape):
    x_data_zero_padding = []
    z_pad = padding_shape[0]
    y_pad = padding_shape[1]
    x_pad = padding_shape[2]
    for ind_patch in range(0, len(x_data)):
        LN_patch = x_data[ind_patch]
        LN_patch_shape = LN_patch.shape
        z_patch = LN_patch_shape[0]
        y_patch = LN_patch_shape[1]
        x_patch = LN_patch_shape[2]

        z_pad_diff = z_pad - z_patch

        y_pad_diff = y_pad - y_patch
        if y_pad_diff < 0:
            # y_crop = y_pad_diff * (-1)
            # LN_patch = np.delete(LN_patch, np.s_[0:y_crop], axis=1)
            LN_patch = np.resize(LN_patch, (z_patch, y_pad, x_patch))
            y_pad_diff = 0

        x_pad_diff = x_pad - x_patch
        if x_pad_diff < 0:
            # x_crop = x_pad_diff * (-1)
            # LN_patch = np.delete(LN_patch, np.s_[0:x_crop], axis=2)
            LN_patch = np.resize(LN_patch, (z_patch, y_patch, x_pad))
            x_pad_diff = 0

        z_pad_size_0 = z_pad_diff//2
        z_pad_size_1 = z_pad_diff - z_pad_size_0

        y_pad_size_0 = y_pad_diff//2
        y_pad_size_1 = y_pad_diff - y_pad_size_0

        x_pad_size_0 = x_pad_diff//2
        x_pad_size_1 = x_pad_diff - x_pad_size_0

        LN_patch_zero_padding = np.pad(LN_patch,
               ((z_pad_size_0, z_pad_size_1), (y_pad_size_0, y_pad_size_1), (x_pad_size_0, x_pad_size_1)),
               'constant', constant_values=(0, 0))
        x_data_zero_padding.append(LN_patch_zero_padding)

    x_data_zero_padding = np.asarray(x_data_zero_padding)
    return x_data_zero_padding


