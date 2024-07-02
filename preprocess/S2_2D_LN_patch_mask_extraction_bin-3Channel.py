'''
step 2
read LN masks and write in bin file
date: 2021-02-06
author: xiav
'''

import numpy as np
import SimpleITK as sitk
import pickle
import openpyxl
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':

    modality = 'mask'
    center_name = 'YUNNAN'  # 'SHANXI'  'YUNNAN'  'FUDAN'
    path_ori = './processed_data/' + center_name + '/patches_2d_all/' + modality
    path_bin = './processed_data/' + center_name + '/patches_2d_bin_std_all/'
    info_path = './processed_data/' + center_name + '/info/info_' + center_name + '_correct.xlsx'

    info = openpyxl.load_workbook(info_path)
    shenames = info.get_sheet_names()
    # data sheet
    all_sheet = info[shenames[0]]
    # number of patients
    all_num = all_sheet.max_row - 1
    # patient name in sheet
    # all_year = list(all_sheet.columns)[1]
    all_name = list(all_sheet.columns)[1]
    all_label = list(all_sheet.columns)[15]

    # year_list = []
    name_list = []
    label_list = []
    p_LN_list_vol = []

    p_num = 0

    for index in range(1,  all_num+1):

        # p_year = all_year[index].value
        p_name = str(all_name[index].value)
        p_label = all_label[index].value

        # year_list.append(p_year)
        name_list.append(p_name)
        label_list.append(p_label)

        # vol_name = modality + '_' + p_name + '_' + str(p_year) + '_hm.nii.gz'

        vol_name = modality + '_' + p_name + '.nii.gz'
        p_vol_path = path_ori + '\\' + vol_name
        p_vol = sitk.ReadImage(p_vol_path)
        p_vol_array = sitk.GetArrayFromImage(p_vol)
        # plt.imshow(p_vol_array[0,:,:])
        # plt.show()
        # repeat as three channels
        p_vol_array_std_3channel = np.repeat(p_vol_array[..., np.newaxis], 3, 3)

        p_temp = p_vol_array_std_3channel[0, :]
        # plt.subplot(131)
        # plt.imshow(p_vol_array_std_3channel[0, :, :, 0], cmap='gray')
        # plt.subplot(132)
        # plt.imshow(p_vol_array_std_3channel[0, :, :, 1], cmap='gray')
        # plt.subplot(133)
        # plt.imshow(p_vol_array_std_3channel[0, :, :, 2], cmap='gray')
        # plt.show()

        p_LN_list_vol.append(p_vol_array_std_3channel)
        print(vol_name, ' appended.')
        p_num = p_num + 1

    print(str(p_num), 'patients data appended.')
    bin_file = open(path_bin + modality + "_2D_patches_correct_3channel.bin", "wb")
    pickle.dump(p_LN_list_vol, bin_file)  # 保存list到文件
    bin_file.close()

    # label_bin_file = open(path_bin + "P_label.bin", "wb")
    # pickle.dump(label_list, label_bin_file)  # 保存list到文件
    # label_bin_file.close()
    #
    # name_bin_file = open(path_bin + "P_name.bin", "wb")
    # pickle.dump(name_list, name_bin_file)  # 保存list到文件
    # name_bin_file.close()

    # year_bin_file = open(path_bin + "P_year.bin", "wb")
    # pickle.dump(year_list, year_bin_file)  # 保存list到文件
    # year_bin_file.close()