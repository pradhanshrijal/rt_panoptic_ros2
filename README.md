# rt_panoptic_ros2
| :warning: Only Runs at 1.2 Hz |
|-------------------------------|

ROS 2 Wrapper for realtime_panoptic

This is a ROS 2 Wrapper for [`[Real-Time Panoptic Segmentation from Dense Detections]`](https://github.com/TRI-ML/realtime_panoptic).

**Test System**
- Computation: i9-12900K, GPU 4080
- System Setup: 
    - Ubuntu 22.04, ROS 2 Humble
    - CUDA 11.7, CuDNN 8.5.0.96
    - Python 3.10, PyTorch 2.0
- Real-Time Panoptic Version: [`[modified]`](https://github.com/pradhanshrijal/realtime_panoptic)

**Status**
- Tested with the original weigths.
- :warning: Only runs at 1.2 Hz at best.

# Content
- [Real-Time Panoptic Installation](#real-time-panoptic-installation)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Licence](#licence)

# Real-Time Panoptic Installation

## Apex
Install the latest [`[Apex]`](https://github.com/NVIDIA/apex).
[`[Original Install Instructions]`](https://github.com/NVIDIA/apex#linux)
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## RT Panoptic
Install the [`[modified version]`](https://github.com/pradhanshrijal/realtime_panoptic).
```
git clone https://github.com/TRI-ML/realtime_panoptic.git
# Must be done in each new terminal
# source {FOLDER_PATH}
```

Try the [Quick Demo](https://github.com/pradhanshrijal/realtime_panoptic#demo) to verify installations.

# Installation
**<u>Build the Packages</u>**
```console
# GOTO the ros 2 workspace
cd src/
git clone https://github.com/pradhanshrijal/rt_panoptic_ros2
cd ..
rosdep install -i --from-path src --rosdistro humble -y
colcon build --symlink-install --packages-select rt_panoptic_ros2
# Source the workspace
```

**<u>Download the weights</u>**
- Download the ResNet50 Weight
```
wget https://tri-ml-public.s3.amazonaws.com/github/realtime_panoptic/models/cvpr_realtime_pano_cityscapes_standalone_no_prefix.pth
```
- Copy the weight to the checkpoints folder of the rt_panoptic_ros2 package.

P.S. Symlink connections will save a lot of storage space and help organise the files.

# Usage
```console
ros2 launch rt_panoptic_ros2 panoptic.launch.py
```

# Parameters

| Parameter | Description |
|:-----------:|:------------|
| config_file | Local Path to the configuration file for the model. |
| model_file | Local Path to the pre-trained weight for the model. |
| allow_memory_fractioning | Boolean to activate setting a limit to the GPU Usage. <br>&emsp; Used together with `device_memory_fraction`. |
| allow_score_thresholding | Boolean to activate the removal of low scored detections. <br>&emsp; Used together with `threshold_array`. |
| device_id | ID for the GPU to be used. |
| device_memory_fraction | GPU Memory (in GB) used for the detections. <br>&emsp; Used together with `allow_memory_fractioning`. |
| threshold_array | Threshold values for low scoring detections. <br>&emsp; Used together with `allow_score_thresholding`. <br>&emsp; See the `config_file` for the list of detections. |


# License
`rt_panoptic_ros2` is released under the [MIT](LICENSE) License.

# Citation 
If you find this project useful in your research, please consider citing the original work:

*Rui Hou\*, Jie Li\*, Arjun Bhargava, Allan Raventos, Vitor Guizilini, Chao Fang, Jerome Lynch, Adrien Gaidon*, [**[paper]**](https://arxiv.org/abs/1912.01202), [**[oral presentation]**](https://www.youtube.com/watch?v=xrxaRU2g2vo), [**[teaser]**](https://www.youtube.com/watch?v=_N4kGJEg-rM)
```
@InProceedings{real-time-panoptic,
author = {Hou, Rui and Li, Jie and Bhargava, Arjun and Raventos, Allan and Guizilini, Vitor and Fang, Chao and Lynch, Jerome and Gaidon, Adrien},
title = {Real-Time Panoptic Segmentation From Dense Detections},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```