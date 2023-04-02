import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    package_name = 'rt_panoptic_ros2'
    package_dir = get_package_share_directory(package_name)
    config_file = 'cityscape_panoptic.param.yaml'

    namespace = LaunchConfiguration('namespace')
    params_file = LaunchConfiguration('params_file')
    input_topic = LaunchConfiguration('input_topic')
    pan_topic = LaunchConfiguration('pan_topic')
    seg_topic = LaunchConfiguration('seg_topic')
    det_topic = LaunchConfiguration('det_topic')

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites={}
    )

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(package_dir, 'config', config_file),
        description='Full path to the ROS 2 parameters file to use for the launched nodes'
    )

    declare_input_topic_cmd = DeclareLaunchArgument(
        'input_topic',
        default_value='/kitti/image',
        description='Input image'
    )

    declare_pan_topic_cmd = DeclareLaunchArgument(
        'pan_topic',
        default_value='image_panoptic',
        description='Output complete panoptic image'
    )

    declare_seg_topic_cmd = DeclareLaunchArgument(
        'seg_topic',
        default_value='image_segmentation',
        description='Output only detection segmentation masks'
    )

    declare_det_topic_cmd = DeclareLaunchArgument(
        'det_topic',
        default_value='camera_detections',
        description='Output Complete Panoptic Image'
    )

    panoptic = Node(
        package=package_name,
        executable='panoptic',
        name='panoptic',
        output='screen',
        parameters=[configured_params,
                    {'package_folder_path': package_dir}],
        remappings=[("input", input_topic), 
                    ("pan_image", pan_topic), 
                    ("seg_image", seg_topic), 
                    ("output_det", det_topic)]            
    )

    ld = LaunchDescription()

    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_input_topic_cmd)
    ld.add_action(declare_pan_topic_cmd)
    ld.add_action(declare_seg_topic_cmd)
    ld.add_action(declare_det_topic_cmd)
    ld.add_action(panoptic)

    return ld