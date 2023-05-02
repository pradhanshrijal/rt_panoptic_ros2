# rt_panoptic_ros2
ROS 2 Wrapper for realtime_panoptic

This is a ROS 2 Wrapper for [`[Real-Time Panoptic Segmentation from Dense Detections]`](https://github.com/TRI-ML/realtime_panoptic).

:warning: Only Runs at 1.2 Hz

**Test System**
- Computation: i9-12900K, GPU 4080
- System Setup: 
    - Ubuntu 22.04, ROS 2 Humble
    - CUDA 11.7, CuDNN 8.5.0.96
    - Python 3.10, PyTorch 2.0
- Real-Time Panoptic Version: [`[modified]`](https://github.com/pradhanshrijal/realtime_panoptic)

**Status**
- Tested with the original weigths.
- Only runs at 1.2 Hz at best.