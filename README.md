# X2 Ground Recovery: FRASA adapted to the AgiBot X2 Ultra

This repository adapts **FRASA (Fall Recovery And Stand-up Agent)** ([paper](https://arxiv.org/pdf/2410.08655), original code by the Rhoban Team) to the **AgiBot X2 Ultra** humanoid. A policy is trained with reinforcement learning (CrossQ, via SBX/JAX and RL Baselines3 Zoo) in a *MuJoCo* environment to recover from lying on the ground (supine) to standing. A ROS 2 package wraps the trained policy behind a service/topic interface.

> **Status: the training did not succeed.** The policy does not stand the robot up (see [Results](#evaluation-results) and [Failure analysis](#failure-analysis)). The pipeline (environment, training, evaluation, ROS 2 integration) works end to end; the learned behaviour does not meet the recovery criteria.

## Summary of setup

| Item | Value |
|---|---|
| Robot | AgiBot X2 Ultra (about 42 kg, 29 joints; 12 controlled) |
| Simulator | MuJoCo (physics step 2 ms, control step 50 ms / 20 Hz) |
| RL algorithm | CrossQ (`sbx`, JAX), RL Baselines3 Zoo (`train_sbx.py`) |
| Hardware | Laptop (Lenovo LOQ 15IAX9), RTX 4060 (4 GB), about 23 GB RAM, Ubuntu |
| Software | conda env `frasa` (Python 3.10), `jax[cuda12]==0.6.2`, ROS 2 Humble (for the ROS 2 package) |

## Installation

```bash
conda create -n frasa python=3.10 && conda activate frasa
pip install rl_zoo3
pip install "jax[cuda12]==0.6.2"      # GPU training; omit for CPU-only JAX
python -m pip install opencv-python==5.0.0.93
```
then:

```bash
git clone git@github.com:islamammar78/agibot_x2_ground_recovery.git
cd ~/agibot_x2_ground_recovery/frasa/
pip install -e .                      # installs frasa_env
```

Always run the scripts from this `frasa/` folder: another copy of `frasa_env` elsewhere on the machine would otherwise be imported.

### Issue with MuJoCo using Wayland

If the MuJoCo viewer freezes under Wayland, build GLFW from source with shared libraries and point `LD_LIBRARY_PATH` and `PYGLFW_LIBRARY` to it:

```bash
sudo apt install libwayland-dev libxkbcommon-dev xorg-dev
cd path/to/glfw && cmake -S . -B build -D BUILD_SHARED_LIBS=ON && cd build && make
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:path/to/glfw/build/src/"
export PYGLFW_LIBRARY="path/to/glfw/build/src/libglfw.so"
```

## Code guide

Where to look, in suggested reading order. Paths are relative to the repository root.

| Area | File | What it contains |
|---|---|---|
| Environment | `frasa/frasa_env/env/standup_env.py` | **The core.** `__init__` (options, observation/action spaces), `step()` (PD control, terminations, the 3-term reward), `reset()` (loads a pkl pose, randomization, `q_desired = q`), `randomize_fall` (used only without the cache) |
| Physics layer | `frasa/frasa_env/mujoco_simulator/simulator.py` | MuJoCo wrapper: joint/sensor access, `apply_pd_control` (torque = kp·error − kd·velocity), `self_collisions` (used by the reward) |
| Robot model | `frasa/frasa_env/mujoco_simulator/model/x2_ultra.xml`, `scene.xml` | X2 Ultra and the floor scene, including the added `trunk` site |
| Start poses | `frasa/standup_generate_initial.py` | Builds the pkl: 70% settled supine poses, 30% near-target standing poses |
| Training | `frasa/train_sbx.py`, `frasa/hyperparams/crossq.yml` | CrossQ training via RL Zoo (mostly unchanged from FRASA) |
| Play / test | `frasa/enjoy_sbx.py`, `frasa/test_env.py`, `frasa/run_single_recovery.py` | Watch a trained model, test the env, run one supine recovery |
| ROS 2 | `ros2_ws/src/x2_recovery_ros/x2_recovery_ros/recovery_runner.py` | Runs one episode with the policy and checks for success |
| ROS 2 | `.../recovery_node.py`, `.../telemetry_node.py`, `launch/recovery.launch.py` | Service, status and joint topics, and the subscriber and launch file |
| Results | `Validation.md`, `frasa/training_reward.png` | Evaluation episodes, ROS 2 tests, and the training reward plot |

If you only have a few minutes, read `step()` and `reset()` in `standup_env.py` and `standup_generate_initial.py`: they explain most of the results.

## Environment Setup

### Generating initial positions

Initial poses are pre-generated and stored in `frasa_env/env/standup_initial_configurations.pkl`:

```bash
python standup_generate_initial.py --count 300
```

The file contains a shuffled mix of:

- **70% supine poses.** The base orientation is exactly lying on the back (`Rotation.from_euler("y", -90)`), all joints at 0, and the pelvis is dropped from a random height in 0.17-0.40 m. It then settles for 10 s with zero torque, and **only the final settled pose is saved**. Training therefore starts from a robot fully at rest on the ground.
- **30% near-target (standing) poses.** These start from the target joints plus a random hip/knee/ankle offset (±0.1 rad for 70% of them, ±0.25 rad for the rest), and are held with PD control for 0.8 s. A pose is kept only if pelvis z > 0.55 m, |tilt| < 15° and max|qvel| < 2. This mirrors FRASA's 10% near-target starts, used so the agent sees high-reward states.

Options: `--count`, `--stand-fraction`, `--z-min`, `--z-max`, `--settle-time`, `--stand-settle-time`, `--near-offset`, `--far-offset`, `--far-fraction`, `--seed`.

### Testing the environment

```bash
python test_env.py                       # zero actions, MuJoCo viewer
python test_env.py --normal --std 2      # Gaussian noise on actions
```

## Training

Command used for the last run (run E, 1M steps), launched with resource limits because an earlier 16-env CPU run froze the laptop.

Run this from the FRASA directory. Ensure `hyperparams/crossq.yml` exists before starting.

```bash
cd ~/agibot_x2_ground_recovery/frasa
mkdir -p ~/frasa_runs/E

nohup systemd-run --user --scope \
  -p MemoryMax=10G -p CPUQuota=600% \
  nice -n 19 taskset -c 0-5 \
  env OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  python train_sbx.py --algo crossq --env frasa-standup-v0 \
  --conf hyperparams/crossq.yml \
  -n 1000000 --eval-freq 25000 --eval-episodes 5 \
  -f ~/frasa_runs/E/logs \
  -params n_envs:4 buffer_size:300000 gradient_steps:64 learning_starts:20000 \
  > ~/frasa_runs/E/train.log 2>&1 &
```

### Monitor training

Follow the training log:

```bash
tail -f ~/frasa_runs/E/train.log
```

Press `Ctrl+C` to stop following the log. This does **not** stop the training process.

If TensorBoard is installed and logging is enabled:

```bash
tensorboard --logdir ~/frasa_runs/E/logs
```

Print recorded evaluation rewards:

```bash
python -c "
import numpy as np, glob, os
files = glob.glob(os.path.expanduser('~/frasa_runs/E/logs/crossq/*/evaluations.npz'))
if not files:
    print('No evaluation results found yet.')
else:
    f = max(files, key=os.path.getmtime)
    e = np.load(f)
    for t, r in zip(e['timesteps'], e['results'].mean(axis=1)):
        print(t, round(r, 1))
"
```

These are recorded evaluation results, not necessarily live training progress.

### Stop training and save the model

**Important:** Sending an interrupt requests a graceful shutdown, but it does not guarantee that the current model will be saved. Checkpoint saving depends on the training script and its interruption handling. Ideally, configure periodic checkpoints before starting a long run.

1. Find the running training process:

   ```bash
   pgrep -af 'train_sbx.py'
   ```

2. Send an interrupt to the training process:

   ```bash
   PID=$(pgrep -f '[t]rain_sbx.py' | head -n 1)

   if [ -n "$PID" ]; then
       kill -INT "$PID"
       echo "Interrupt sent to training PID $PID"
   else
       echo "No active training process found."
   fi
   ```

3. Wait for shutdown and inspect the log:

   ```bash
   tail -n 50 ~/frasa_runs/E/train.log
   ```

4. Find saved model checkpoints:

   ```bash
   find ~/frasa_runs/E/logs -type f -name '*.zip' \
     -printf '%TY-%Tm-%Td %TH:%TM  %p\n' | sort -r
   ```

The output shows the checkpoint paths and modification times. Common checkpoint names include `rl_model_*.zip` and `best_model.zip`, depending on the configured callbacks. The final model may have a different name.

To list all files saved for run E:

```bash
find ~/frasa_runs/E -type f
```

**Do not assume the model was saved just because the process stopped.** Confirm that a checkpoint exists and check its timestamp. If no recent checkpoint exists, the training code must be configured to save periodically or handle interruption before relying on graceful shutdown to preserve the model.

### Notes

- Do not launch two trainings simultaneously; memory use can increase substantially.
- The configuration keys in `hyperparams/crossq.yml` are `frasa-standup-v0-10ms` and `frasa-standup-v0-25ms`, which do not match the registered environment ID `frasa-standup-v0`. The command therefore overrides the relevant hyperparameters using `-params`.
- Run E outputs are stored under `~/frasa_runs/E/`. Use a different output directory for each new run to avoid mixing logs and checkpoints.
## Evaluation / Enjoy a Trained Agent

```bash
python enjoy_sbx.py --algo crossq --env frasa-standup-v0 --gym-packages frasa_env \
  --folder ~/frasa_runs/E/logs/ --exp-id 1 --deterministic -n 2000
# add --load-best for best_model.zip
```

`run_single_recovery.py` runs one supine recovery episode in the viewer with the checkpoint in `models/crossq/` (also used by ROS 2).

## X2 Ultra Model

The model is in `frasa_env/mujoco_simulator/model/x2_ultra.xml` (scene in `scene.xml`; source URDF in `agibot_x2_urdf/`). Torque-controlled motors with control ranges, and joint limits.

**Model changes with respect to the Sigmaban setup**

- Added a `trunk` site on the pelvis body. The environment uses it to place the robot and measure tilt (this was the original `KeyError: 'trunk'`).
- **Torque actuation instead of position servos.** Sigmaban uses position-controlled servos. The X2 is driven by torque motors, and a PD controller (kp = 120, kd = 12) converts joint targets to torques.
- **Controlled joints (12):** for each side, `elbow`, `shoulder_pitch`, `shoulder_roll`, `hip_pitch`, `knee`, `ankle_pitch`. Left and right actions are mirrored (the right shoulder roll sign is flipped). All other joints (hip roll/yaw, ankle roll, waist, wrists, head, shoulder yaw) receive **zero torque (limp)**.

## Stand Up Environment (`frasa-standup-v0`)

### Observation space (26 values)

| Index | Description | Unit |
|---|---|---|
| 0-5 | Joint positions `q` of the 6 controlled DoF (clipped to joint range) | rad |
| 6-11 | Joint velocities `q_dot` (clipped to ±10) | rad/s |
| 12-17 | Desired joint positions `q_desired` (clipped to joint range) | rad |
| 18 | Torso tilt (pitch, from the IMU; 0 = upright, about -90° = supine) | rad |
| 19 | Tilt rate | rad/s |
| 20-25 | Previous action | rad/s |

Tilt = `-arctan2(R[2,0], R[2,2])` of the IMU rotation, so only pitch is observed.

### Action space (6 values)

Desired joint **velocity** for the 6 controlled DoF (each in ±2π rad/s, mirrored to both sides). The value is integrated over the control step into `q_desired`, which the PD controller tracks (kp = 120, kd = 12) on the torque motors.

### Resets

With the pkl cache (default): choose a stored pose (seeded), zero velocities and controls, apply domain randomization (±1.5° joint offsets, friction, mass, COM, damping, etc.), randomize the shoulder roll by ±5°, set `q_desired = q`. No simulation step is run, so the policy acts from the first step. Without the cache, `randomize_fall` drops the robot from a random pose and waits `stabilization_time` (2 s).

### Termination and truncation

- Terminated when the robot is upside down (`terminate_upside_down`) or the gyro is too high (`terminate_gyro`). `terminate_shock` is off.
- Truncated after 5 s (101 steps at 20 Hz).

### Simplifications

- 6 controlled DoF per side. Pitch-only tilt; no roll/yaw observation and no pelvis height or contact observations.
- Mirrored left/right actions, so the policy cannot move the sides differently.
- Hip roll/yaw, ankle roll, waist, wrists, head and shoulder yaw are limp.
- Observation clipping to keep the Gym environment checker happy.

### Reward

`R = R_state + R_variation + R_collision` (the same form as FRASA):

| Term | Formula | Weight |
|---|---|---|
| `R_state` | `exp(-20 · ‖[q, tilt] − desired_state‖²)` | 20 (the exponent coefficient), max 1 |
| `R_variation` | `0.05 · exp(-‖a_t − a_{t-1}‖)` | 0.05 |
| `R_collision` | `0.1 · exp(-self_collisions)` | 0.1 |

`desired_state = [-0.3, 0, 0.2, -0.312, 0.669, -0.363, 0]` (elbow, shoulder pitch, shoulder roll, hip pitch, knee, ankle pitch in rad, then tilt = 0).

**Why:** `R_state` rewards reaching the standing configuration. `R_variation` encourages smooth actions (safe on hardware). `R_collision` discourages self-collision. There is no explicit action-size or height term. R_state is about 0.82 at distance 0.1, 0.45 at 0.2, 0.17 at 0.3 and about 0 beyond 0.5. A zero-action episode returns about 15, a standing robot about 115.

**Limitation:** because R_state is almost 0 far from the target, the agent gets no gradient from the floor. FRASA relies on starts near the target to find the high rewards; this failed on the X2 (see below). Earlier reward experiments (height-shaping `B`, uprightness `C`) were exploited by the policy (the pelvis raised while folded at a 117-126° tilt, or the tilt lowered with the pelvis on the floor), so the final version returned to the paper's reward.

### Environment options

Passed via `gym.make(..., options={...})` or `--env-kwargs`. Main ones:

| Option | Default | Description |
|---|---|---|
| `truncate_duration` | 5.0 | Episode length in seconds |
| `stabilization_time` | 2.0 | Settling time in `randomize_fall` (no cache only) |
| `terminate_upside_down` | True | End the episode when upside down |
| `terminate_gyro` | True | End the episode on high gyro |
| `terminate_shock` | False | End the episode on a shock |
| `vmax` | 2π | Maximum joint velocity of the action |

`reset(options={"use_cache": False})` bypasses the pkl and uses `randomize_fall`.

See `StandupEnv.__init__` in `frasa_env/env/standup_env.py` for the complete list.

## RL algorithm and training settings

- **Algorithm:** CrossQ (SBX, JAX).
- **Run E settings:** 1M steps, 4 envs, buffer 300k, 64 gradient steps per update, `learning_starts` 20k, auto entropy coefficient, evaluation every 25k steps with 5 episodes, about 90-200 steps/s on the RTX 4060.
- **Paper (for comparison):** 8-16 envs, gamma 0.995-0.998, 512 gradient steps per 4 env steps, a 2-2.5M buffer, convergence after 235-575k steps. Our setup was much lighter because of the hardware.

## Evaluation results

**Recovery success check:** the robot is upright and stable (pelvis z > 0.45 m, tilt < 35° in our offline analysis; the ROS 2 runner uses |tilt| ≤ 15°, angular speed ≤ 0.3 rad/s held for 1 s), both feet on the ground, no support from hands, knees or torso. The ROS 2 check is provisional and does not verify foot contact.

Final model of run E, 15 episodes per condition (return per 5 s episode; a zero-action baseline is the reference):

| Start | Zero action | Final model | Best model | Outcome |
|---|---|---|---|---|
| Supine | 15.2 | 11.9 | 12.2 | pelvis never above 0.11 m |
| Standing | 11.3 | 37.5 | 31.3 | robot ended lying down (z 0.08 m, tilt about 92°) |

The five validation episodes through ROS 2 all **failed** (see `../Validation.md`). Earlier runs: A (original reward) was below the zero-action baseline; B ended with the pelvis at 0.43 m and tilt 117° (gamed); C had tilt 21° at z 0.11 m (gamed); D (supine-only) crashed with a CUDA error at about 300k steps, with best eval about 40 at z 0.5 m and tilt 126°.

## Failure analysis

1. **Sparse reward:** R_state is about 0 when far from the target, so there is no gradient out of the supine pose.
2. **Dense workarounds were exploited** (height or uprightness terms): folded pelvis raise, or tilt reduction on the floor.
3. **Near-target starts did not help.** The X2 standing target is probably not a balanced pose. Held by PD on 12 joints (others limp), the robot stands about 1.5 s then tips backwards, and it also fell with all 23 joints held. So the high-reward states were not stable.
4. **Torque/joint coverage unproven:** ankle limit ±36 Nm, and the static check said standing needs about 50% of it. Limp hip roll/yaw/ankle roll and the pitch-only tilt may also limit balance.
5. **Lighter training than the paper:** 4 envs, a 300k buffer and 64 gradient steps instead of 8-16 envs, 2M+ buffer and 512 gradient steps.
6. **Position servos vs torque motors:** the PD layer does not guarantee whole-body tracking.
7. **Process issues:** mixed-start evaluation hid the supine failure, and there were no periodic checkpoints (run D crashed).

### Proposed improvements

- Find a balanced standing target (search or optimization with a PD/static check) and re-verify the near-target starts hold for several seconds.
- Add a gated dense reward (pelvis height only when tilt is small) with a foot-contact term.
- Check torque feasibility and tune PD gains to the X2's mass and actuator limits; consider controlling more joints (hip roll, ankle roll, waist).
- Train closer to the paper (more envs, 512 gradient steps, a larger buffer) and add periodic checkpoints.
- Consider a curriculum from sitting or kneeling, or a framework built for torque-controlled humanoids, e.g. [HumanUP](https://github.com/RunpeiDong/HumanUP).

## ROS 2 integration

Package: `ros2_ws/src/x2_recovery_ros` (ROS 2 Humble, `ament_python`).

### Nodes

- **`recovery_node`**: provides the `/x2/start_recovery` service (`std_srvs/Trigger`). On a request it starts one recovery episode in a background thread (a lock rejects concurrent requests), enforces a timeout (`recovery_timeout_seconds`, default 30 s), publishes the status (`IDLE`, `RUNNING`, `SUCCEEDED`, `FAILED`) on `/x2/recovery_status` every 0.5 s, and publishes joint positions on `/x2/joint_states` during the episode.
- **`telemetry_node`**: subscribes to `/x2/recovery_status` and `/x2/joint_states` and logs them.

### How the nodes connect to the simulator

`recovery_node` imports `recovery_runner.run_recovery_episode`, which runs the MuJoCo simulator **in-process** (there is no separate simulator process). It loads the CrossQ checkpoint (`frasa/models/crossq/frasa-standup-v0.zip`, override with `X2_RECOVERY_MODEL_PATH`) on CPU, creates `frasa-standup-v0`, places the robot supine at 0.16 m and lets it settle for 1 s. It then loops at 20 Hz: read the observation from the simulator, call `model.predict`, apply the action (PD torques) and step MuJoCo. Each step the runner reads the joint positions from `sim.data.qpos` and publishes them as `sensor_msgs/JointState`, and checks the success criterion (tilt, angular speed, hold time). The launch file starts `recovery_node` with the conda Python (`/home/islam/miniconda3/envs/frasa/bin/python`) so that `sbx` and `mujoco` are available.

### Build and launch

```bash
cd ~/agibot_x2_ground_recovery
export X2_RECOVERY_PROJECT_ROOT="$(pwd)"
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --packages-select x2_recovery_ros
source install/setup.bash
ros2 launch x2_recovery_ros recovery.launch.py
```
Note: Run these commands in order. The X2_RECOVERY_PROJECT_ROOT environment variable must be exported again in each new terminal session before launching the recovery node.

### Service and topic commands
on a second terminal, source the environment and run the following commands:

```bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger "{}"
ros2 topic echo /x2/recovery_status
ros2 topic echo /x2/joint_states
ros2 topic list
ros2 service list
```

ROS 2 integration tests (service-based recovery requests, live joint telemetry, rejection of concurrent requests, timeout handling) all passed; the recovery itself fails because of the policy (see above).

## Citing

This work builds on FRASA:

```
@article{duclusaud2024fall,
  title={FRASA: An End-to-End Reinforcement Learning Agent for Fall Recovery and Stand Up of Humanoid Robots},
  author={Duclusaud, Clément and Passault, Grégoire and Padois, Vincent and Ly, Olivier},
  journal={arXiv preprint arXiv:2410.08655},
  year={2024}
}
```
