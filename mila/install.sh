# Install all needed deps
module load python/3.10
module load cuda/11.8

# apt-get update
# sudo apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6  -y

python -m venv $HOME/humanoid_env
source $HOME/humanoid_env/bin/activate

pip3 install --no-cache-dir --upgrade numpy==2.0.0
pip3 install --no-cache-dir numpy-quaternion
pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip3 install mujoco
pip3 install mujoco_mjx
pip3 install brax
pip3 install opencv-python
pip3 install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip3 install nvidia-cudnn-cu11==8.9.6.50
pip3 install stable-baselines3[extra]
pip3 install tensorflow
pip3 install rl_zoo3
pip3 install perlin_noise

# install dependencies for rendering with OpenCV
# sudo apt-get install -y libglfw3
# sudo apt-get install -y libglfw3-dev

pip3 install --upgrade "jax[cuda12]"

