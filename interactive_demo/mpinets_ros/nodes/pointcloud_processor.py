#!/usr/bin/env python3

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import time


class PointCloudProcessor:
    def __init__(self):
        rospy.init_node("pointcloud_processor")

        # Parameters
        self.output_rate = rospy.get_param("~output_rate", 2.0)  # Hz
        self.num_obstacle_points = 4096

        # Publishers/Subscribers
        self.processed_pub = rospy.Publisher(
            "/mpinets/processed_pointcloud", PointCloud2, queue_size=1, latch=True
        )

        self.raw_sub = rospy.Subscriber(
            "/camera_right/filtered/cropped_points",
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1,
        )

        # State
        self.latest_pointcloud = None
        self.latest_colors = None
        self.last_publish_time = rospy.Time(0)

        rospy.loginfo("PointCloudProcessor node ready")

    def pointcloud_callback(self, msg):
        """Process incoming point cloud"""
        try:
            # Extract points efficiently
            points_list = []
            rgb_list = []

            # Downsample during extraction
            for i, p in enumerate(
                pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
            ):
                if i % 5 == 0:  # Process every 5th point
                    points_list.append([p[0], p[1], p[2]])
                    rgb_list.append(p[3])

            if not points_list:
                return

            points = np.array(points_list, dtype=np.float32)
            rgb_float = np.array(rgb_list, dtype=np.float32)

            # Process colors
            rgb_int = rgb_float.astype(np.uint32)
            r = ((rgb_int >> 16) & 0xFF).astype(np.float32) / 255.0
            g = ((rgb_int >> 8) & 0xFF).astype(np.float32) / 255.0
            b = (rgb_int & 0xFF).astype(np.float32) / 255.0

            colors = np.column_stack([r, g, b, np.ones(len(points))])

            # Clean point cloud
            cleaned_points, cleaned_colors = self.clean_pointcloud(points, colors)

            self.latest_pointcloud = cleaned_points
            self.latest_colors = cleaned_colors

            # # Throttled publishing
            # current_time = rospy.Time.now()
            # if (current_time - self.last_publish_time).to_sec() >= (
            #     1.0 / self.output_rate
            # ):
            #     self.publish_processed_pointcloud()
            #     self.last_publish_time = current_time

            self.publish_processed_pointcloud()

        except Exception as e:
            rospy.logerr(f"Point cloud processing error: {e}")

    def clean_pointcloud(self, points, colors):
        """Clean and downsample point cloud"""
        if len(points) == 0:
            return np.zeros((self.num_obstacle_points, 3)), np.zeros(
                (self.num_obstacle_points, 4)
            )

        # Workspace filtering
        mask = (
            (points[:, 0] > 0.1)
            & (points[:, 0] < 1.5)
            & (points[:, 1] > -1.5)
            & (points[:, 1] < 1.5)
            & (points[:, 2] > -0.05)
            & (points[:, 2] < 1.5)
        )

        filtered_points = points[mask]
        filtered_colors = colors[mask]

        n_points = len(filtered_points)

        if n_points >= self.num_obstacle_points:
            indices = np.random.choice(
                n_points, self.num_obstacle_points, replace=False
            )
            return filtered_points[indices], filtered_colors[indices]
        elif n_points > 0:
            repeat_times = (self.num_obstacle_points + n_points - 1) // n_points
            repeated_points = np.repeat(filtered_points, repeat_times, axis=0)
            repeated_colors = np.repeat(filtered_colors, repeat_times, axis=0)
            indices = np.random.choice(
                len(repeated_points), self.num_obstacle_points, replace=False
            )
            return repeated_points[indices], repeated_colors[indices]
        else:
            return np.zeros((self.num_obstacle_points, 3)), np.zeros(
                (self.num_obstacle_points, 4)
            )

    def publish_processed_pointcloud(self):
        """Publish the processed point cloud"""
        if self.latest_pointcloud is None:
            return

        header = Header(frame_id="panda_link0", stamp=rospy.Time.now())

        # Combine points and colors
        points_colors = np.column_stack([self.latest_pointcloud, self.latest_colors])

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("r", 12, PointField.FLOAT32, 1),
            PointField("g", 16, PointField.FLOAT32, 1),
            PointField("b", 20, PointField.FLOAT32, 1),
            PointField("a", 24, PointField.FLOAT32, 1),
        ]

        pc_msg = pc2.create_cloud(header, fields, points_colors)
        self.processed_pub.publish(pc_msg)

        rospy.loginfo_throttle(
            5,
            f"Published processed point cloud with {len(self.latest_pointcloud)} points",
        )


if __name__ == "__main__":
    processor = PointCloudProcessor()
    rospy.spin()