#!/usr/bin/python3
"""Publish a measured, full MoveIt planning scene from site-owned YAML."""

import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import SolidPrimitive
import yaml


class PlanningSceneFromYaml(Node):
    def __init__(self):
        super().__init__("franka_site_planning_scene")
        path = self.declare_parameter("scene_config", "").value
        if not path:
            raise RuntimeError("scene_config is required")
        with Path(path).open("r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream) or {}
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        topic = self.config.get("planning_scene_topic", "/monitored_planning_scene")
        self.publisher = self.create_publisher(PlanningScene, topic, qos)
        self.scene = self._build_scene()
        if self.scene is None:
            self.get_logger().error(
                "Site scene is incomplete or contains measurement placeholders; "
                "no PlanningScene will be published, so the gateway remains fail-closed"
            )
            return
        self.timer = self.create_timer(1.0, self._publish)
        self._publish()

    @staticmethod
    def _pose(values):
        if len(values) != 7 or not all(math.isfinite(float(value)) for value in values):
            raise ValueError("pose must be [x, y, z, qx, qy, qz, qw]")
        pose = Pose()
        (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = map(float, values)
        norm = math.sqrt(
            pose.orientation.x**2
            + pose.orientation.y**2
            + pose.orientation.z**2
            + pose.orientation.w**2
        )
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("object quaternion must be normalized")
        return pose

    @staticmethod
    def _primitive(spec):
        primitive = SolidPrimitive()
        kind = str(spec["type"]).lower()
        dimensions = [float(value) for value in spec["dimensions"]]
        expected = {"box": 3, "cylinder": 2, "sphere": 1}
        if kind not in expected or len(dimensions) != expected[kind]:
            raise ValueError(f"invalid {kind!r} primitive dimensions")
        if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
            raise ValueError("primitive dimensions must be finite and positive")
        primitive.type = {
            "box": SolidPrimitive.BOX,
            "cylinder": SolidPrimitive.CYLINDER,
            "sphere": SolidPrimitive.SPHERE,
        }[kind]
        primitive.dimensions = dimensions
        return primitive

    def _build_scene(self):
        objects = self.config.get("objects", [])
        required = set(self.config.get("required_objects", []))
        by_name = {item.get("id"): item for item in objects}
        if not required or not required.issubset(by_name):
            return None
        if any(
            not by_name[name].get("enabled", False)
            or not by_name[name].get("measurements_confirmed", False)
            for name in required
        ):
            return None

        scene = PlanningScene()
        scene.name = self.config.get("scene_name", "franka-site-scene")
        scene.is_diff = False
        scene.robot_state.is_diff = True
        frame = self.config.get("frame_id", "base")
        for spec in objects:
            if not spec.get("enabled", False):
                continue
            collision = CollisionObject()
            collision.header.frame_id = spec.get("frame_id", frame)
            collision.id = spec["id"]
            collision.operation = CollisionObject.ADD
            collision.primitives = [self._primitive(spec)]
            collision.primitive_poses = [self._pose(spec["pose"])]
            attached_to = spec.get("attached_to", "")
            if attached_to:
                attached = AttachedCollisionObject()
                attached.link_name = attached_to
                attached.object = collision
                attached.touch_links = list(spec.get("touch_links", [attached_to]))
                scene.robot_state.attached_collision_objects.append(attached)
            else:
                scene.world.collision_objects.append(collision)
        if not scene.world.collision_objects:
            return None
        return scene

    def _publish(self):
        self.scene.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.scene)


def main():
    rclpy.init()
    node = PlanningSceneFromYaml()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
