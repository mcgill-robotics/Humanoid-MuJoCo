FROM ubuntu:22.04

# Install all needed deps
RUN apt-get update
RUN apt-get install -y python3 python3-pip python-is-python3
RUN apt-get install -y --no-install-recommends git
RUN apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6  -y
RUN apt-get install -y vim
RUN apt-get install -y screen

RUN pip3 install --no-cache-dir --upgrade numpy
RUN pip3 install --no-cache-dir numpy-quaternion
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install mujoco
RUN pip3 install mujoco_mjx
RUN pip3 install brax
RUN pip3 install opencv-python
RUN pip3 install matplotlib
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install nvidia-cudnn-cu11==8.9.6.50
RUN pip3 install stable-baselines3[extra]
RUN pip3 install tensorflow
RUN pip3 install rl_zoo3
RUN pip3 install perlin_noise

# install dependencies for rendering with OpenCV
RUN apt-get install -y libglfw3
RUN apt-get install -y libglfw3-dev

RUN apt-get install nvidia-cuda-toolkit; exit 0
RUN apt --fix-broken install; exit 0

RUN pip3 install --upgrade "jax[cuda12]"
RUN apt-get update && apt-get install -y cmake libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build zlib1g-dev clang-12

WORKDIR /Humanoid-MPC

ENV PATH $PATH:/Humanoid-MPC/mujoco_mpc/build/bin/