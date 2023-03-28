# generate the heatmap by Grad-CAM
# highlight the region related to metastasis

import os
import pickle
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tf.compat.v1.disable_eager_execution()

def draw_heatmap(model, img_ori, p_name, patch_id, patches_save_path, patches_heatmap_save_path):
    img = np.expand_dims(img_ori, axis=0)
    # for sigmoid, only one output
    class_output = model.output[:, 0]
    last_conv_layer = model.get_layer("add_3")
    gap_weights = model.get_layer("global_average_pooling2d")
    # Calculating the derivative of the output of the prediction class to the output of the gap layer
    grads = K.gradients(class_output, gap_weights.output)[0]
    # Returns the output of the gradient and the last convolution layer
    iterate = K.function([model.input], [grads, last_conv_layer.output[0]])
    # Returns the gradient of the pooling layer and the output of the convolution layer
    pooled_grads_value, conv_layer_output_value = iterate([img])
    pooled_grads_value = np.squeeze(pooled_grads_value, axis=0)
    # The gradient of the GAP layer is used as the weight of the feature graph.
    # That is, each graph is multiplied by the global average pooled gradient
    for i in range(128):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    # Calculate the weighted average of the feature map, and directly calculate the average in the channel dimension
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    # Then perform relu activation: maximum compares heatmap and 0 one by one, and selects a larger value
    heatmap = np.maximum(heatmap, 0)
    # max By default, the maximum column direction is calculated,
    # that is, each value is divided by the maximum value of the column, and normalized to 0-1
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    # Adjust to the same size as the original image
    heatmap = cv2.resize(heatmap, (img_ori.shape[1], img_ori.shape[0]))
    # normalization 0-255
    heatmap = np.uint8(255 * heatmap)
    img_ori /= np.max(img_ori)
    img_ori = np.uint8(255 * img_ori)

    # display fused imgs
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed_img = cv2.addWeighted(img_ori, 0.6, heatmap, 0.4, 0)
    # plt.imshow(superimposed_img)
    # plt.show()

    # write imgs into jpg file
    cv2.imwrite(patches_save_path + p_name + '_' + str(patch_id) + '.jpg', img_ori)
    cv2.imwrite(patches_heatmap_save_path + p_name + '_' + str(patch_id) + '_CAM.jpg', heatmap)


if __name__ == '__main__':
    # basic settings
    img_type = 'T2'
    center_name = 'CenterI'
    patches_save_path = './data/' + center_name + '/patch_save/'
    patches_heatmap_save_path = './data/' + center_name + '/heatmap_save/'

    if not (os.path.exists(patches_save_path)):
        os.mkdir(patches_save_path)

    if not (os.path.exists(patches_heatmap_save_path)):
        os.mkdir(patches_heatmap_save_path)

    # load the trained I_DiagnosticNetwork
    I_DiagnosticNetwork = load_model('./best_model_save/I_DiagnosticNetwork.hdf5', compile=False)
    I_DiagnosticNetwork.summary()

    # load data for model test
    all_p_data = pickle.load(
        open('./data/' + center_name + '/patches_2d_bin_std_all/' + img_type + '_2D_patches_3channel.bin', "rb"))
    # load the info
    all_p_name = pickle.load(open('./data/' + center_name + '/patches_2d_bin_std_all/P_name.bin', "rb"))
    all_LN_meta_labels = pickle.load(open('./data/' + center_name + '/patches_2d_bin_std_all/P_label.bin', "rb"))
    all_p_N_stage = pickle.load(open('./data/' + center_name + '/patches_2d_bin_std_all/P_N_stage.bin', "rb"))
    all_p_N_stage_fine = pickle.load(open('./data/' + center_name + '/patches_2d_bin_std_all/P_N_stage_fine.bin', "rb"))

    # generate heatmap for each patch
    for p_index in range(0, len(all_p_name)):
        p_name = all_p_name[p_index]
        P_patches = all_p_data[p_index]
        patch_num_p = len(P_patches)
        for patch_id in range(0, patch_num_p):
            # patche_pred = I_DiagnosticNetwork.predict(P_patches[patch_id])
            draw_heatmap(I_DiagnosticNetwork, P_patches[patch_id], p_name, patch_id, patches_save_path, patches_heatmap_save_path)
        print('patient ', p_index, ' heatmap generated.')