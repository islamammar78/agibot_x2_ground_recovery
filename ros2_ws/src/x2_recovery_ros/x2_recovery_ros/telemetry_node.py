
#!/usr/bin/env python3
"""ROS 2 node that logs the recovery status and joint telemetry.

It only subscribes to /x2/recovery_status and /x2/joint_states (published by
recovery_node); it does not talk to the simulator itself.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState


class TelemetryNode(Node):
    """Subscribes to the recovery topics and prints what it receives."""

    def __init__(self):
        super().__init__('telemetry_node')

        self.status_sub = self.create_subscription(
            String,
            '/x2/recovery_status',
            self.status_callback,
            10,
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/x2/joint_states',
            self.joint_callback,
            10,
        )

        self.get_logger().info('Telemetry node started.')

    def status_callback(self, msg):
        """Log each status message (IDLE, RUNNING, SUCCEEDED, FAILED)."""
        self.get_logger().info(
            f'Recovery status: {msg.data}'
        )

    def joint_callback(self, msg):
        """Log only the first joint to keep the output readable."""
        if msg.name and msg.position:
            self.get_logger().info(
                f'Joint {msg.name[0]}: '
                f'{msg.position[0]:.3f} rad'
            )


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
