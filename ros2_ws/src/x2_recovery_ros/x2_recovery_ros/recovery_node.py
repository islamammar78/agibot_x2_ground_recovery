#!/usr/bin/env python3
"""ROS 2 node that starts and supervises one X2 ground-recovery episode.

Interface:
  service   /x2/start_recovery  (std_srvs/Trigger)  request a recovery
  topic     /x2/recovery_status (std_msgs/String)   IDLE | RUNNING | SUCCEEDED | FAILED
  topic     /x2/joint_states    (sensor_msgs/JointState) simulated joint angles

The MuJoCo simulation and the trained CrossQ policy run in-process inside
recovery_runner.run_recovery_episode(); this node only handles ROS 2 I/O.
"""

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState

from .recovery_runner import run_recovery_episode


class RecoveryNode(Node):
    """Owns the service, the publishers and the recovery state machine."""

    def __init__(self):
        super().__init__('recovery_node')

        # Configurable recovery timeout, in seconds
        self.declare_parameter('recovery_timeout_seconds', 30.0)
        self.recovery_timeout = float(
            self.get_parameter(
                'recovery_timeout_seconds'
            ).value
        )

        # Current recovery state. recovery_running + the lock reject a second
        # request while an episode is active (the runner uses one simulator).
        self.status = 'IDLE'
        self.recovery_running = False
        self.state_lock = threading.Lock()

        # Recovery status publisher
        self.status_pub = self.create_publisher(
            String,
            '/x2/recovery_status',
            10,
        )

        # Joint-state publisher
        self.joint_pub = self.create_publisher(
            JointState,
            '/x2/joint_states',
            10,
        )

        # Publish status every 0.5 seconds
        self.status_timer = self.create_timer(
            0.5,
            self.publish_status,
        )

        # Service to request recovery
        self.recovery_service = self.create_service(
            Trigger,
            '/x2/start_recovery',
            self.start_recovery_callback,
        )

        self.get_logger().info(
            'Recovery node started. Initial status: IDLE'
        )

    def publish_status(self):
        """Timer callback: publish the current status at 2 Hz."""
        msg = String()
        msg.data = self.status
        self.status_pub.publish(msg)

    def set_status(self, new_status):
        """Update the status if it is one of the allowed values."""
        valid_statuses = {
            'IDLE',
            'RUNNING',
            'SUCCEEDED',
            'FAILED',
        }

        if new_status not in valid_statuses:
            self.get_logger().error(
                f'Invalid recovery status: {new_status}'
            )
            return

        self.status = new_status
        self.get_logger().info(
            f'Recovery status: {self.status}'
        )

    def publish_joint_state(self, msg):
        """Publish a JointState message supplied by the runner."""
        self.joint_pub.publish(msg)

    def start_recovery_callback(self, request, response):
        """Accept a request and start recovery in a worker thread."""

        with self.state_lock:
            if self.recovery_running:
                response.success = False
                response.message = 'Recovery already running'
                self.get_logger().warning(response.message)
                return response

            self.recovery_running = True
            self.set_status('RUNNING')

        response.success = True
        response.message = 'Recovery request accepted'

        self.get_logger().info(response.message)

        # Run the episode in a thread so the service call returns immediately
        # and the status keeps being published while the robot recovers.
        worker = threading.Thread(
            target=self.execute_recovery,
            daemon=True,
        )

        try:
            worker.start()
        except Exception as exc:
            with self.state_lock:
                self.recovery_running = False
                self.set_status('FAILED')

            response.success = False
            response.message = f'Could not start recovery: {exc}'
            self.get_logger().error(response.message)

        return response

    def execute_recovery(self):
        """Run one recovery episode and update the final status."""

        try:
            self.get_logger().info(
                'Starting recovery episode...'
            )

            result = run_recovery_episode(
                publish_joint_state=self.publish_joint_state,
                timeout_seconds=self.recovery_timeout,
            )

            # The runner returns {success, timed_out, steps, total_reward, message}.
            if result.get('timed_out', False):
                self.set_status('FAILED')
                self.get_logger().error(
                    'Recovery timed out.'
                )

            elif result.get('success', False):
                self.set_status('SUCCEEDED')
                self.get_logger().info(
                    'Recovery completed successfully.'
                )

            else:
                self.set_status('FAILED')
                self.get_logger().warning(
                    'Recovery episode failed: '
                    + result.get('message', 'No details provided')
                )

        except Exception as exc:
            self.set_status('FAILED')
            self.get_logger().error(
                f'Recovery execution error: {exc}'
            )

        finally:
            # Always release the busy flag so another request can be accepted.
            with self.state_lock:
                self.recovery_running = False

            self.get_logger().info(
                'Recovery attempt finished; ready for another request.'
            )


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
