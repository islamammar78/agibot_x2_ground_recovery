conda create -n x2 python=3.10.12
conda activate x2

python3 -m mujoco.viewer --mjcf X2_URDF-v1.3.0/scene.xml

