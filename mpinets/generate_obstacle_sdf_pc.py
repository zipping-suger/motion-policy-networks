#!/usr/bin/env python3
"""
Generate SDF files and point clouds for obstacles (Cuboid and Cylinder) for use with Motion Policy Networks.
This script extracts obstacles from the problem pickle file based on environment type,
problem type, and problem ID.
"""

import numpy as np
from geometrout.primitive import Cuboid, Cylinder
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pickle
from scipy.spatial.transform import Rotation as R

from mpinets.mpinets_types import PlanningProblem, ProblemSet
from mpinets.geometry import construct_mixed_point_cloud
from geometrout.transform import SE3

import sys

# Add parent directory to path to import from run_inference.py
sys.path.append(str(Path(__file__).parent.parent))
from run_inference import convert_primitive_problems_to_depth


def get_orientation_as_rpy(primitive):
    """
    Extract orientation as RPY from a primitive object.
    Based on the geometrout library structure.
    """
    if (
        hasattr(primitive, "pose")
        and hasattr(primitive.pose, "so3")
        and hasattr(primitive.pose.so3, "_quat")
    ):
        quat = primitive.pose.so3._quat
        # Create a SciPy Rotation object from the quaternion [x, y, z, w]
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        # Get the RPY angles in radians
        rpy = r.as_euler("xyz", degrees=False)
        return [rpy[0], rpy[1], rpy[2]]  # Returns [roll, pitch, yaw]

    print(
        "Warning: Could not find orientation for primitive {}, using identity rotation".format(
            type(primitive)
        )
    )
    return [0.0, 0.0, 0.0]


def get_position(primitive):
    """
    Extract position from a primitive object.
    Based on the geometrout library structure.
    """
    # For geometrout primitives, position is stored in pose._xyz
    if hasattr(primitive, "pose") and hasattr(primitive.pose, "_xyz"):
        return primitive.pose._xyz

    # Default to origin
    print(
        "Warning: Could not find position for primitive {}, using origin".format(
            type(primitive)
        )
    )
    return [0.0, 0.0, 0.0]


def create_cuboid_sdf(cuboid, name="obstacle"):
    """
    Create SDF XML for a cuboid obstacle.

    :param cuboid: Cuboid object with center, dims, and orientation
    :param name: Name of the model in SDF
    :return: XML string of the SDF
    """
    # Get position and orientation
    position = get_position(cuboid)
    rpy = get_orientation_as_rpy(cuboid)

    # Create the root element
    sdf = ET.Element("sdf", version="1.6")
    model = ET.SubElement(sdf, "model", name=name)

    # Create link
    link = ET.SubElement(model, "link", name="link")

    # Set pose (position and orientation)
    pose_str = (
        f"{position[0]} {position[1]} {position[2]} " f"{rpy[0]} {rpy[1]} {rpy[2]}"
    )
    pose = ET.SubElement(link, "pose")
    pose.text = pose_str

    # Inertial (required for physics)
    inertial = ET.SubElement(link, "inertial")
    mass = ET.SubElement(inertial, "mass")
    mass.text = "10"  # Arbitrary mass

    inertia = ET.SubElement(inertial, "inertia")
    for axis in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
        elem = ET.SubElement(inertia, axis)
        elem.text = "0.166667" if axis in ["ixx", "iyy", "izz"] else "0"

    # Visual
    visual = ET.SubElement(link, "visual", name="visual")
    visual_geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(visual_geometry, "box")
    size = ET.SubElement(box, "size")
    size.text = f"{cuboid.dims[0]} {cuboid.dims[1]} {cuboid.dims[2]}"

    # Material
    material = ET.SubElement(visual, "material")
    script = ET.SubElement(material, "script")
    name_elem = ET.SubElement(script, "name")
    name_elem.text = "Gazebo/Grey"
    uri = ET.SubElement(script, "uri")
    uri.text = "file://media/materials/scripts/gazebo.material"

    # Collision
    collision = ET.SubElement(link, "collision", name="collision")
    collision_geometry = ET.SubElement(collision, "geometry")
    collision_box = ET.SubElement(collision_geometry, "box")
    collision_size = ET.SubElement(collision_box, "size")
    collision_size.text = f"{cuboid.dims[0]} {cuboid.dims[1]} {cuboid.dims[2]}"

    # Surface properties
    surface = ET.SubElement(collision, "surface")
    friction = ET.SubElement(surface, "friction")
    ode = ET.SubElement(friction, "ode")
    mu = ET.SubElement(ode, "mu")
    mu.text = "1"
    mu2 = ET.SubElement(ode, "mu2")
    mu2.text = "1"

    # Make it static
    static = ET.SubElement(model, "static")
    static.text = "1"

    # Format the XML
    rough_string = ET.tostring(sdf, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_cylinder_sdf(cylinder, name="obstacle"):
    """
    Create SDF XML for a cylinder obstacle.

    :param cylinder: Cylinder object with center, radius, height, and orientation
    :param name: Name of the model in SDF
    :return: XML string of the SDF
    """
    # Get position and orientation
    position = get_position(cylinder)
    rpy = get_orientation_as_rpy(cylinder)

    # Create the root element
    sdf = ET.Element("sdf", version="1.6")
    model = ET.SubElement(sdf, "model", name=name)

    # Create link
    link = ET.SubElement(model, "link", name="link")

    # Set pose (position and orientation)
    pose_str = (
        f"{position[0]} {position[1]} {position[2]} " f"{rpy[0]} {rpy[1]} {rpy[2]}"
    )
    pose = ET.SubElement(link, "pose")
    pose.text = pose_str

    # Inertial (required for physics)
    inertial = ET.SubElement(link, "inertial")
    mass = ET.SubElement(inertial, "mass")
    mass.text = "10"  # Arbitrary mass

    inertia = ET.SubElement(inertial, "inertia")
    for axis in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
        elem = ET.SubElement(inertia, axis)
        elem.text = "0.166667" if axis in ["ixx", "iyy", "izz"] else "0"

    # Visual
    visual = ET.SubElement(link, "visual", name="visual")
    visual_geometry = ET.SubElement(visual, "geometry")
    cylinder_elem = ET.SubElement(visual_geometry, "cylinder")
    radius = ET.SubElement(cylinder_elem, "radius")
    radius.text = str(cylinder.radius)
    length = ET.SubElement(cylinder_elem, "length")
    length.text = str(cylinder.height)

    # Material
    material = ET.SubElement(visual, "material")
    script = ET.SubElement(material, "script")
    name_elem = ET.SubElement(script, "name")
    name_elem.text = "Gazebo/Grey"
    uri = ET.SubElement(script, "uri")
    uri.text = "file://media/materials/scripts/gazebo.material"

    # Collision
    collision = ET.SubElement(link, "collision", name="collision")
    collision_geometry = ET.SubElement(collision, "geometry")
    collision_cylinder = ET.SubElement(collision_geometry, "cylinder")
    collision_radius = ET.SubElement(collision_cylinder, "radius")
    collision_radius.text = str(cylinder.radius)
    collision_length = ET.SubElement(collision_cylinder, "length")
    collision_length.text = str(cylinder.height)

    # Surface properties
    surface = ET.SubElement(collision, "surface")
    friction = ET.SubElement(surface, "friction")
    ode = ET.SubElement(friction, "ode")
    mu = ET.SubElement(ode, "mu")
    mu.text = "1"
    mu2 = ET.SubElement(ode, "mu2")
    mu2.text = "1"

    # Make it static
    static = ET.SubElement(model, "static")
    static.text = "1"

    # Format the XML
    rough_string = ET.tostring(sdf, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def save_obstacles_as_sdf(obstacles, output_dir="obstacles"):
    """
    Save a list of obstacles as individual SDF files.

    :param obstacles: List of Cuboid or Cylinder objects
    :param output_dir: Directory to save SDF files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for i, obstacle in enumerate(obstacles):
        if isinstance(obstacle, Cuboid):
            sdf_content = create_cuboid_sdf(obstacle, f"obstacle_{i}")
        elif isinstance(obstacle, Cylinder):
            sdf_content = create_cylinder_sdf(obstacle, f"obstacle_{i}")
        else:
            print(f"Unsupported obstacle type: {type(obstacle)}")
            continue

        with open(output_path / f"obstacle_{i}.sdf", "w") as f:
            f.write(sdf_content)
        print(f"Saved obstacle_{i}.sdf")


def create_combined_sdf(obstacles, name="obstacles"):
    """
    Create a single SDF file containing all obstacles.

    :param obstacles: List of Cuboid or Cylinder objects
    :param name: Name of the model in SDF
    :return: XML string of the SDF
    """
    # Create the root element
    sdf = ET.Element("sdf", version="1.6")
    model = ET.SubElement(sdf, "model", name=name)

    for i, obstacle in enumerate(obstacles):
        # Get position and orientation
        position = get_position(obstacle)
        rpy = get_orientation_as_rpy(obstacle)

        # Create link for each obstacle
        link = ET.SubElement(model, "link", name=f"link_{i}")

        # Set pose (position and orientation)
        pose_str = (
            f"{position[0]} {position[1]} {position[2]} " f"{rpy[0]} {rpy[1]} {rpy[2]}"
        )
        pose = ET.SubElement(link, "pose")
        pose.text = pose_str

        if isinstance(obstacle, Cuboid):
            # Visual
            visual = ET.SubElement(link, "visual", name=f"visual_{i}")
            visual_geometry = ET.SubElement(visual, "geometry")
            box = ET.SubElement(visual_geometry, "box")
            size = ET.SubElement(box, "size")
            size.text = f"{obstacle.dims[0]} {obstacle.dims[1]} {obstacle.dims[2]}"

            # Collision
            collision = ET.SubElement(link, "collision", name=f"collision_{i}")
            collision_geometry = ET.SubElement(collision, "geometry")
            collision_box = ET.SubElement(collision_geometry, "box")
            collision_size = ET.SubElement(collision_box, "size")
            collision_size.text = (
                f"{obstacle.dims[0]} {obstacle.dims[1]} {obstacle.dims[2]}"
            )

        elif isinstance(obstacle, Cylinder):
            # Visual
            visual = ET.SubElement(link, "visual", name=f"visual_{i}")
            visual_geometry = ET.SubElement(visual, "geometry")
            cylinder_elem = ET.SubElement(visual_geometry, "cylinder")
            radius = ET.SubElement(cylinder_elem, "radius")
            radius.text = str(obstacle.radius)
            length = ET.SubElement(cylinder_elem, "length")
            length.text = str(obstacle.height)

            # Collision
            collision = ET.SubElement(link, "collision", name=f"collision_{i}")
            collision_geometry = ET.SubElement(collision, "geometry")
            collision_cylinder = ET.SubElement(collision_geometry, "cylinder")
            collision_radius = ET.SubElement(collision_cylinder, "radius")
            collision_radius.text = str(obstacle.radius)
            collision_length = ET.SubElement(collision_cylinder, "length")
            collision_length.text = str(obstacle.height)

        # Material
        material = ET.SubElement(visual, "material")
        script = ET.SubElement(material, "script")
        name_elem = ET.SubElement(script, "name")
        name_elem.text = "Gazebo/Grey"
        uri = ET.SubElement(script, "uri")
        uri.text = "file://media/materials/scripts/gazebo.material"

        # Surface properties
        surface = ET.SubElement(collision, "surface")
        friction = ET.SubElement(surface, "friction")
        ode = ET.SubElement(friction, "ode")
        mu = ET.SubElement(ode, "mu")
        mu.text = "1"
        mu2 = ET.SubElement(ode, "mu2")
        mu2.text = "1"

        # Make it static
        static = ET.SubElement(model, "static")
        static.text = "1"

    # Format the XML
    rough_string = ET.tostring(sdf, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def save_combined_sdf(obstacles, output_path="obstacles.sdf"):
    """
    Save all obstacles as a single SDF file.

    :param obstacles: List of Cuboid or Cylinder objects
    :param output_path: Path to save the SDF file
    """
    sdf_content = create_combined_sdf(obstacles)

    with open(output_path, "w") as f:
        f.write(sdf_content)
    print(f"Saved combined obstacles to {output_path}")


def save_pointcloud(point_cloud, output_path="obstacles.npy"):
    """
    Save a point cloud to a .npy file in the expected dictionary format.
    """
    # Dummy values for missing fields; replace with actual data if available
    observation_data = {
        "pc": point_cloud,  # Nx3
        "camera_pose": np.eye(4),  # 4x4 identity (world frame)
        "pc_color": np.ones_like(point_cloud) * np.array([1.0, 1.0, 0.0]),  # Nx3, yellow color
        "label_map": {"robot": -1},  # Dummy label
        "pc_label": np.zeros(point_cloud.shape[0]),  # All zeros (not robot)
    }
    np.save(output_path, observation_data)
    print(f"Saved point cloud to {output_path}")


def list_available_problems(problems):
    """
    List all available problems in the pickle file.

    :param problems: The loaded problems dictionary
    """
    print("Available environment types:")
    for env_type in problems.keys():
        print(f"  - {env_type}")
        for prob_type in problems[env_type].keys():
            print(f"    - {prob_type}: {len(problems[env_type][prob_type])} problems")


def get_problem_by_id(problems: ProblemSet, environment_type, problem_type, problem_id):
    """
    Get a specific problem by ID using the same filtering logic as run_inference.py.

    :param problems: The loaded problems dictionary
    :param environment_type: Environment type (e.g., "tabletop", "cubby")
    :param problem_type: Problem type (e.g., "task-oriented", "neutral-start")
    :param problem_id: Problem ID (0-indexed)
    :return: The selected problem
    """
    # Filter problems based on environment_type and problem_type
    filtered_problems = []
    env_type_arg = environment_type.replace("-", "_")
    problem_type_arg = problem_type.replace("-", "_")

    for env_type, scene_sets in problems.items():
        if env_type_arg != "all" and env_type != env_type_arg:
            continue
        for prob_type, problem_list in scene_sets.items():
            if problem_type_arg != "all" and prob_type != problem_type_arg:
                continue
            filtered_problems.extend(problem_list)

    if not filtered_problems:
        raise ValueError(
            f"No problems found for environment type '{environment_type}' and problem type '{problem_type}'."
        )

    if problem_id >= len(filtered_problems) or problem_id < 0:
        raise IndexError(
            f"Problem index {problem_id} out of range for the filtered set. There are {len(filtered_problems)} problems available. Max index is {len(filtered_problems) - 1}."
        )

    return filtered_problems[problem_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SDF files and point clouds for obstacles from a problem pickle file"
    )
    parser.add_argument(
        "problems_file", type=str, help="Path to pickle file containing problems"
    )
    parser.add_argument(
        "--environment-type",
        type=str,
        default="all",
        choices=[
            "tabletop",
            "cubby",
            "merged-cubby",
            "dresser",
            "cabinet",
            "pillar",
            "all",
        ],
        help="The environment class",
    )
    parser.add_argument(
        "--problem-type",
        type=str,
        default="all",
        choices=["task-oriented", "neutral-start", "neutral-goal", "all"],
        help="The type of planning problem",
    )
    parser.add_argument(
        "--problem-id",
        type=int,
        default=0,
        help="ID of the problem to extract (0-indexed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="obstacles",
        help="Output directory or file path prefix",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Save all obstacles in a single SDF file",
    )
    parser.add_argument(
        "--save-pointcloud",
        action="store_true",
        help="Generate and save a point cloud of the obstacles to a .npy file",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help=(
            "If saving a point cloud, use a partial view pointcloud rendered in Pybullet."
            " If not set, uses pointclouds sampled from every side of the primitives in the scene."
        ),
    )
    parser.add_argument(
        "--list", action="store_true", help="List available problems and exit"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print debug information about obstacles"
    )

    args = parser.parse_args()

    # Load problems from pickle file
    with open(args.problems_file, "rb") as f:
        problems: ProblemSet = pickle.load(f)

    # List available problems if requested
    if args.list:
        list_available_problems(problems)
        exit(0)

    # Get the specific problem
    try:
        problem = get_problem_by_id(
            problems, args.environment_type, args.problem_type, args.problem_id
        )
        print(
            f"Selected problem: {args.environment_type}/{args.problem_type}/#{args.problem_id}"
        )
        print(f"Number of obstacles: {len(problem.obstacles)}")

        # Extract obstacles from the problem
        obstacles = problem.obstacles

        # Debug information
        if args.debug:
            for i, obstacle in enumerate(obstacles):
                print(f"Obstacle {i}: {type(obstacle)}")
                if hasattr(obstacle, "center"):
                    print(f"  Center: {obstacle.center}")
                if hasattr(obstacle, "dims"):
                    print(f"  Dimensions: {obstacle.dims}")
                if hasattr(obstacle, "radius"):
                    print(f"  Radius: {obstacle.radius}")
                if hasattr(obstacle, "height"):
                    print(f"  Height: {obstacle.height}")

                # Check for pose attributes
                if hasattr(obstacle, "pose"):
                    pose = obstacle.pose
                    print(f"  Pose position: {pose._xyz}")
                    if hasattr(pose, "so3") and hasattr(pose.so3, "_quat"):
                        quat = pose.so3._quat
                        print(f"  Quaternion: {quat.w}, {quat.x}, {quat.y}, {quat.z}")

                print()

        # Generate SDF files
        if args.combined:
            sdf_output_path = f"{args.output}.sdf"
            save_combined_sdf(obstacles, sdf_output_path)
        else:
            save_obstacles_as_sdf(obstacles, args.output)

        # Generate and save point cloud if requested
        if args.save_pointcloud:
            if args.use_depth:
                # Need to wrap the single problem in the ProblemSet structure
                # to pass it to convert_primitive_problems_to_depth
                problem_set = {}
                env_key = args.environment_type.replace("-", "_")
                prob_key = args.problem_type.replace("-", "_")
                if env_key not in problem_set:
                    problem_set[env_key] = {}
                if prob_key not in problem_set[env_key]:
                    problem_set[env_key][prob_key] = []
                problem_set[env_key][prob_key].append(problem)

                convert_primitive_problems_to_depth(problem_set)
                point_cloud = problem.obstacle_point_cloud
            else:
                # Default to using 4096 points, same as run_inference.py
                NUM_OBSTACLE_POINTS = 4096
                point_cloud = construct_mixed_point_cloud(
                    obstacles, NUM_OBSTACLE_POINTS
                )

            if point_cloud is not None:
                pointcloud_output_path = f"{args.output}.npy"
                save_pointcloud(point_cloud, pointcloud_output_path)
            else:
                print("Could not generate point cloud.")

    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        list_available_problems(problems)
        exit(1)
