# Install all needed deps
module load python/3.10
module load cuda/12.1.1

# apt-get update
# sudo apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6  -y

python -m venv $HOME/humanoid_env
source $HOME/humanoid_env/bin/activate

pip3 install --upgrade pip

pip3 install --upgrade numpy==1.26.4
pip3 install numpy-quaternion
pip3 install --upgrade "jax[cuda12]"
pip3 install mujoco
pip3 install mujoco_mjx
pip3 install brax
pip3 install opencv-python
pip3 install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install stable-baselines3[extra]
pip3 install tensorflow
pip3 install rl_zoo3
pip3 install perlin_noise
