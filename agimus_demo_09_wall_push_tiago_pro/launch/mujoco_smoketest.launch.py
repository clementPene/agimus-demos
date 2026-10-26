"""
MuJoCo pipeline smoke test -- NOT the wall-push demo.

Checks that MuJoCo <-> ros2_control <-> linear_feedback_controller <->
agimus_controller talk to each other correctly, using a standalone PAL
arm (no pal-atc tool, no F/T sensor, no wall). See this package's
README.md for what this does and does not validate.

Usage:
    ros2 launch agimus_demo_09_wall_push_tiago_pro mujoco_smoketest.launch.py
"""

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from controller_manager.launch_utils import (
    generate_controllers_spawner_launch_description,
)

PKG = "agimus_demo_09_wall_push_tiago_pro"
CFG = PathJoinSubstitution([FindPackageShare(PKG), "config", "mujoco_smoketest"])


def generate_launch_description():
    robot_description = ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"),
                " ",
                PathJoinSubstitution(
                    [FindPackageShare(PKG), "urdf", "mujoco_smoketest", "arm.urdf.xacro"]
                ),
            ]
        ),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
        output="screen",
    )

    mujoco_ros2_control_node = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description},
            PathJoinSubstitution([CFG, "controller_manager.yaml"]),
        ],
        output="screen",
    )

    # MujocoSystemInterface blocks in on_init waiting for an MJCF on
    # /mujoco_robot_description (the URDF's <hardware> has no `mujoco_model`
    # param); build it from /robot_description. Needs the `mujoco` python
    # package (flake.nix extraPyPackages).
    mjcf_publisher_node = ExecuteProcess(
        cmd=[
            "python3",
            PathJoinSubstitution(
                [FindPackageShare(PKG), "scripts", "publish_mjcf.py"]
            ),
        ],
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
    srdf_publisher_node = Node(
        package="agimus_demos_common",
        executable="string_publisher",
        name="robot_srdf_description_publisher",
        parameters=[
            {
                "topic_name": "robot_srdf_description",
                "string_value": "<robot name='mujoco_smoketest_arm'></robot>",
            }
        ],
    )

    joint_state_broadcaster_spawner = generate_controllers_spawner_launch_description(
        controller_names=["joint_state_broadcaster"],
        controller_params_files=[PathJoinSubstitution([CFG, "joint_state_broadcaster.yaml"])],
        extra_spawner_args=["--controller-manager-timeout", "30"],
    )
    lfc_controllers_spawner = generate_controllers_spawner_launch_description(
        controller_names=["linear_feedback_controller", "joint_state_estimator"],
        controller_params_files=[
            PathJoinSubstitution([CFG, "joint_state_estimator_params.yaml"]),
            PathJoinSubstitution([CFG, "linear_feedback_controller_params.yaml"]),
        ],
        # Load/configure only: the LFC must be activated together with the
        # joint_state_estimator (its chained predecessor) in ONE
        # switch_controllers transaction, otherwise it activates outside chain
        # mode ("Wrong activation, the controller needs to be in chain mode")
        # with NaN references. Same as tiago_pro_common.launch.py.
        extra_spawner_args=["--inactive", "--controller-manager-timeout", "30"],
    )
    activate_lfc = ExecuteProcess(
        cmd=[
            "ros2",
            "control",
            "switch_controllers",
            "--activate",
            "linear_feedback_controller",
            "joint_state_estimator",
        ],
        output="screen",
    )

    wait_for_non_zero_joints_node = Node(
        package="agimus_demos_common",
        executable="wait_for_non_zero_joints_node",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    agimus_controller_node = Node(
        package="agimus_controller_ros",
        executable="agimus_controller_node",
        parameters=[
            {"use_sim_time": True},
            PathJoinSubstitution([CFG, "agimus_controller_params.yaml"]),
        ],
        remappings=[
            ("robot_description_semantic", "robot_srdf_description"),
        ],
        output="screen",
    )

    simple_trajectory_publisher_node = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "agimus_controller_ros",
            "simple_trajectory_publisher",
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "--params-file",
            PathJoinSubstitution([CFG, "trajectory_weights_params.yaml"]),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            mjcf_publisher_node,
            mujoco_ros2_control_node,
            environment_publisher_node,
            srdf_publisher_node,
            wait_for_non_zero_joints_node,
            joint_state_broadcaster_spawner,
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=joint_state_broadcaster_spawner.entities[-1],
                    on_exit=[lfc_controllers_spawner],
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=lfc_controllers_spawner.entities[-1],
                    on_exit=[activate_lfc],
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=wait_for_non_zero_joints_node,
                    on_exit=[agimus_controller_node],
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=agimus_controller_node,
                    on_start=TimerAction(
                        period=2.0,
                        actions=[simple_trajectory_publisher_node],
                    ),
                )
            ),
        ]
    )
