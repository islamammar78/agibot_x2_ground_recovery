# X2 Ground Recovery — Evaluation Results

## 1. Evaluation Setup

- **Policy:** CrossQ
- **Checkpoint:** `frasa/models/crossq/frasa-standup-v0.zip`
- **Environment:** `frasa-standup-v0`
- **Episodes:** 5

## 2. Success Criteria

A recovery counts as successful when the robot:
- Stands upright and remains stable for 5 seconds.
- Maintains contact with both feet.
- Does not rely on its hands, knees, torso, or other body parts for support.

## 3. Episode Results


| Episode | Result | 
| 1 | FAIL |
| 2 | FAIL |
| 3 | FAIL |
| 4 | FAIL |
| 5 | FAIL |



## 4. ROS 2 Integration Tests

| Test | Result | Notes |
|---|---|---|
| Fresh build | PASS | `x2_recovery_ros` built successfully using `colcon build --packages-select x2_recovery_ros --symlink-install`. |
| Launch node | PASS | Recovery and telemetry nodes started successfully; initial status was `IDLE`. |
| Start recovery via service | PASS | `/x2/start_recovery` accepted the request and transitioned to `RUNNING`. |
| Live joint telemetry | PASS | Joint values, including `left_hip_pitch_joint`, updated during recovery. |
| Reject second request while running | PASS | Second request returned `success=False`, message: `Recovery already running`. |
| Timeout transitions to `FAILED` | PASS | Recovery timed out after 30 seconds without success. The status changed to `FAILED`, and the node logged `Recovery timed out.` |

## 5. Commands Used

**Build:**
```bash
cd ~/Desktop/agibot_x2_ground_recovery/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select x2_recovery_ros --symlink-install
```

**Launch:**
```bash
source install/setup.bash
ros2 launch x2_recovery_ros recovery.launch.py
```

**Start recovery:**
```bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger "{}"
```

**Monitor joint telemetry:**
```bash
ros2 topic echo /x2/joint_states
```

**Monitor recovery status:**
```bash
ros2 topic echo /x2/recovery_status
```

**Recovery timeout:** 30 seconds, as configured in the recovery node.


## Failure Analysis

The project adapted the FRASA reinforcement-learning approach to train a CrossQ policy for AgiBot X2 Ultra ground recovery. Although the original FRASA reward function was retained, the adaptation introduced differences in robot dynamics, control and target-state initialisation.

The reward function is composed of three terms:

\[
R_t=R_{\mathrm{state}}+R_{\mathrm{variation}}+R_{\mathrm{collision}}
\]

The state reward measures how close the robot is to its target configuration:

\[
R_{\mathrm{state}}=
\exp\left(-20\left\|\mathbf q-\mathbf q^*\right\|^2\right)
\]

where \(\mathbf q\) is the current state and \(\mathbf q^*\) is the target state. The variation term penalises excessive action changes, encouraging smoother behaviour, while the collision term penalises self-collision.

The exponential state reward approaches 1 when the robot is close to its target and decreases rapidly as the state error increases. For example, a squared state error of 0.5 gives \(e^{-10}\approx0.000045\). The reward does not explicitly reward pelvis height, so raising the pelvis does not automatically increase the reward. When the robot starts lying on the floor, the state reward may remain very small until its configuration approaches the target.

In the original FRASA setup, this sparse-reward challenge was addressed partly through the initial-state distribution. Approximately 10% of episodes were configured to start near the target, allowing the agent to experience high-reward states and learn how to maintain the target configuration.

For the X2 adaptation, 30% of episodes were configured to start near the target using a PD controller. However, the target standing configuration was not sufficiently stable under this controller, and the robot tended to fall. Consequently, these episodes did not consistently provide stable, high-reward experiences from which the reinforcement-learning agent could learn to maintain standing.

A further difference is the control interface. The original FRASA setup used position-controlled joints, whereas the X2 simulation uses torque-controlled actuators. A PD controller converts joint-position errors and velocities into torque commands, but successful tracking does not guarantee whole-body balance. The target pose and controller must also account for the X2's mass, inertia, contact conditions and actuator torque limits.

The final diagnostic evaluation supports these findings. From supine starts, the final and best policies achieved maximum pelvis heights of only 0.10 m and 0.11 m, respectively, from an initial height of 0.08 m. Neither learned meaningful stand-up recovery. From standing starts, both policies ultimately fell, despite achieving higher returns than the zero-action baseline. The trained policies therefore failed to achieve reliable recovery.

The principal suspected issue is the instability of the target standing configuration, compounded by the sparse state reward and differences between the original FRASA setup and the X2 adaptation. The instability is supported by the PD-controller test, in which the robot tipped backwards after approximately 1.5 seconds. However, further testing is required to establish the individual contribution of each factor.

## Proposed Mitigations

- **Modify the reward function:** Introduce a separate pelvis-height term that penalises low pelvis heights and encourages the robot to lift its body during recovery. Combine it with an uprightness or orientation term to prevent the robot from maximising pelvis height without achieving a stable standing posture.

- **Improve the PD controller and target stability:** Develop a PD controller tuned to the X2's mass, inertia and actuator torque limits. Adjust the target joint configuration and controller gains until the robot can maintain a balanced standing posture. This should make the 30% of near-target episodes more useful for learning.

- **Investigate alternative torque-controlled recovery approaches:** Evaluate the [HumanUP simulation framework](https://github.com/RunpeiDong/HumanUP/tree/main/simulation) as a reference for training recovery policies on a torque-controlled humanoid. Its robot model, control limits, dynamics and training procedure would need to be adapted to the X2 before its suitability could be established.

## Conclusion

The ROS 2 integration tests passed, including service-based recovery requests, live joint telemetry, rejection of concurrent requests and timeout handling. However, all five validation episodes failed to achieve the defined recovery criteria within the configured timeout.

The next priority is to establish a stable X2 standing configuration under the PD controller before further training. The reward function and initial-state distribution can then be reviewed to ensure that the agent receives useful learning signals for both raising the pelvis and maintaining a stable standing posture.
