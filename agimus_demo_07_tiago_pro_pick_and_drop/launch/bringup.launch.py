"""
Bringup launch file for the TIAGo Pro pick-and-drop demo (fixed base, left arm only).

Uses the left arm (pal-pro-gripper) for picking T-LESS objects from a table.

Usage:
    ros2 launch agimus_demo_07_tiago_pro_pick_and_drop bringup.launch.py use_gazebo:=true
    ros2 launch agimus_demo_07_tiago_pro_pick_and_drop bringup.launch.py use_gazebo:=true use_hpp_bridge:=true
"""

import os
import yaml

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from agimus_demos_common.launch_utils import (
    generate_default_tiago_pro_args,
    generate_include_launch,
    get_use_sim_time,
)
from agimus_demos_common.mpc_debugger_node import mpc_debugger_node

PKG = "agimus_demo_07_tiago_pro_pick_and_drop"

# Object initial pose (must match config/hpp_orchestrator_params.yaml)
_cfg_path = os.path.join(
    os.path.dirname(__file__), "..", "config", "hpp_orchestrator_params.yaml"
)
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

OBJ_X = _cfg["object"]["x"]
OBJ_Y = _cfg["object"]["y"]
OBJ_Z = _cfg["object"]["z"]

# Table is placed 0.85 m in front of robot; object center is at OBJ_X
TABLE_X = 0.85
TABLE_Y = 0.0
TABLE_Z = 0.0


def launch_setup(
    context: LaunchContext, *args, **kwargs
) -> list[LaunchDescriptionEntity]:

    tiago_robot_launch = generate_include_launch(
        "tiago_pro_common.launch.py",
        extra_launch_arguments={
            "tuck_arm": "False",
            "active_arm_side": "left",
            "end_effector_left": "pal-pro-gripper",
            "lfc_pkg": PKG,
            "lfc_yaml": "config/linear_feedback_controller_left_simu_params.yaml",
            "jse_yaml": "config/joint_state_estimator_left_simu_params.yaml",
            "pc_yaml": "config/dummy_controllers_left.yaml",
        },
    )

    wait_for_non_zero_joints_node = Node(
        package="agimus_demos_common",
        executable="wait_for_non_zero_joints_node",
        parameters=[get_use_sim_time()],
        output="screen",
    )

    environment_publisher_node = Node(
        package="agimus_demos_common",
        executable="string_publisher",
        name="environment_publisher",
        parameters=[
            {
                "topic_name": "environment_description",
                "string_value": "<robot name='empty'><link name='env'/></robot>",
            }
        ],
    )

    agimus_controller_node = Node(
        package="agimus_controller_ros",
        executable="agimus_controller_node",
        parameters=[
            get_use_sim_time(),
            PathJoinSubstitution(
                [FindPackageShare(PKG), "config", "agimus_controller_params.yaml"]
            ),
        ],
        remappings=[
            ("robot_description_semantic", "robot_srdf_description"),
        ],
        output="screen",
    )

    # Spawn table in Gazebo (position synced with HPP scene)
    spawn_table_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "table",
            "-string", Command([
                FindExecutable(name="xacro"), " ",
                PathJoinSubstitution([FindPackageShare(PKG), "urdf", "table.urdf"]),
            ]),
            "-x", str(TABLE_X), "-y", str(TABLE_Y), "-z", str(TABLE_Z),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_gazebo")),
    )

    # Spawn T-LESS object 23 on the table
    spawn_obj_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "obj_23",
            "-string", Command([
                FindExecutable(name="xacro"), " ",
                PathJoinSubstitution([FindPackageShare(PKG), "urdf", "obj_23.urdf"]),
            ]),
            "-x", str(OBJ_X), "-y", str(OBJ_Y), "-z", str(OBJ_Z),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_gazebo")),
    )

    mpc_debugger = mpc_debugger_node(
        "arm_left_tool_link",
        parent_frame="base_link",
        cost_plot=True,
        node_kwargs=dict(
            condition=IfCondition(LaunchConfiguration("use_mpc_debugger")),
            remappings=[
                ("robot_description_semantic", "robot_srdf_description"),
            ],
        ),
    )

    hpp_bridge_node = ExecuteProcess(
        cmd=[
            "xterm", "-hold", "-T", "HPP orchestrator", "-e",
            "bash -c '"
            "source /opt/ros/humble/setup.bash && "
            "source /home/gepetto/agimus_deps_ws/install/setup.bash && "
            "source /home/gepetto/ros2_ws/install/setup.bash && "
            "source /home/gepetto/hpp_ws/install/setup.bash && "
            f"python3 /home/gepetto/ros2_ws/install/{PKG}/share/{PKG}/hpp/orchestrator_node.py"
            "'",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_hpp_bridge")),
    )

    return [
        tiago_robot_launch,
        spawn_table_node,
        spawn_obj_node,
        wait_for_non_zero_joints_node,
        environment_publisher_node,
        mpc_debugger,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=wait_for_non_zero_joints_node,
                on_exit=[
                    agimus_controller_node,
                    hpp_bridge_node,
                ],
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_mpc_debugger",
                default_value="false",
                choices=["true", "false"],
            ),
            DeclareLaunchArgument(
                "use_hpp_bridge",
                default_value="false",
                choices=["true", "false"],
                description="Launch the HPP orchestrator shell in xterm.",
            ),
        ]
        + generate_default_tiago_pro_args()
        + [OpaqueFunction(function=launch_setup)]
    )
