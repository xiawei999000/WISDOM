#### [Wei Xia, et al. "Multicenter Evaluation of a Weakly Supervised Deep Learning Model for Lymph Node Diagnosis in Rectal Cancer on MRI." Radiology: Artificial Intelligence (2024): e230152.](https://doi.org/10.1148/ryai.230152)
#### Weakly supervISed model DevelOpment fraMework (WISDOM) to construct lymph node diagnosis model with preoperative MRI data coupled with postoperative patient-level pathological information.
* [0_pretraining_CIFIR10.py](https://github.com/xiawei999000/WISDOM/blob/main/0_pretraining_CIFIR10.py): use the CIFIR10 images to pretrain the intensity diagnostic network.<br>
* [1_MI_training.py](https://github.com/xiawei999000/WISDOM/blob/main/1_MI_training.py): build the intensity diagnostic model MI using the T2W-MRI image with weak supervision component.<br>
* [2_MIS_training.py](https://github.com/xiawei999000/WISDOM/blob/main/2_MIS_training.py): build the integrated diagnostic model MIS using the short, long diameters, diamter ratio, and the img predictions of MI.<br>
* [3_MISA_training.py](https://github.com/xiawei999000/WISDOM/blob/main/3_MISA_training.py): build the integrated diagnostic model MISA using the short, long diameters, diamter ratio, ADC value, and the img predictions of MI.<br>
* [4_MISA_test_output_preds.py](https://github.com/xiawei999000/WISDOM/blob/main/4_MISA_test_output_preds.py): generate the predictions of MISA for statistic analysis.<br>
* [5_diagnostic_network_extract.py](https://github.com/xiawei999000/WISDOM/blob/main/5_diagnostic_network_extract.py): extract the diagnostic networks.<br>
* [6_Heatmap_generation.py](https://github.com/xiawei999000/WISDOM/blob/main/6_Heatmap_generation.py): generate the heatmap by Grad-CAM to highlight the region related to metastasis.<br>
