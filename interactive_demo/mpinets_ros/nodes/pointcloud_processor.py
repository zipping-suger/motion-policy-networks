#!/usr/bin/env python3

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class PointCloudProcessor:
    def __init__(self):
        rospy.init_node("pointcloud_processor")
        
        self.num_points = 4096
        self.publish_rate = rospy.get_param('~publish_rate', 10)  # Default to 10 Hz
        
        self.processed_pub = rospy.Publisher(
            "/mpinets/processed_pointcloud", PointCloud2, queue_size=1
        )

        self.raw_sub = rospy.Subscriber(
            "/camera_right/filtered/points_no_outliers",
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1,
        )
        
        # Store the latest processed point cloud
        self.latest_processed_points = None
        self.rate = rospy.Rate(self.publish_rate)
        
        rospy.loginfo(f"PointCloudProcessor node ready, publishing at {self.publish_rate} Hz")
        
        # Start the publishing loop
        self.publish_loop()

    def pointcloud_callback(self, msg):
        """Extract and randomly sample points from incoming point cloud"""
        try:
            # Extract all points
            points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            
            if not points:
                return
            
            # Convert to numpy array
            points = np.array(points, dtype=np.float32)
            
            # Randomly sample points
            if len(points) >= self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
                sampled_points = points[indices]
            else:
                # If not enough points, repeat existing ones
                repeat_times = (self.num_points + len(points) - 1) // len(points)
                repeated_points = np.repeat(points, repeat_times, axis=0)
                indices = np.random.choice(len(repeated_points), self.num_points, replace=False)
                sampled_points = repeated_points[indices]
            
            # Store for publishing in the main loop
            self.latest_processed_points = sampled_points
            
        except Exception as e:
            rospy.logerr(f"Point cloud processing error: {e}")

    def publish_loop(self):
        """Main publishing loop that runs at fixed rate"""
        while not rospy.is_shutdown():
            if self.latest_processed_points is not None:
                self.publish_pointcloud(self.latest_processed_points)
            self.rate.sleep()

    def publish_pointcloud(self, points):
        """Publish the sampled point cloud"""
        header = Header(frame_id="panda_link0", stamp=rospy.Time.now())

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]

        pc_msg = pc2.create_cloud(header, fields, points)
        self.processed_pub.publish(pc_msg)

        rospy.loginfo_throttle(5, f"Published point cloud with {len(points)} points")


if __name__ == "__main__":
    processor = PointCloudProcessor()