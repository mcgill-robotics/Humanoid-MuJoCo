FROM ubuntu

# Install all needed deps
RUN apt-get update \
    && apt-get install -y python3 python3-pip python-is-python3 \
    && apt-get install -y --no-install-recommends git \
    && apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6  -y \
    && apt-get install -y vim \
    && apt-get install -y screen

RUN pip3 install --no-cache-dir --upgrade numpy \
    && pip3 install --no-cache-dir numpy-quaternion \
    && pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && pip3 install mujoco \
    && pip3 install mujoco_mjx \
    && pip3 install brax \
    && pip3 install opencv-python \
    && pip3 install matplotlib \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install nvidia-cudnn-cu11==8.9.6.50

RUN apt-get autoremove -y  \
    && apt-get clean && rm -rf /var/lib/apt/lists/*  \
    && rm -rf /root/.cache/pip