FROM ubuntu

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
# THIS LINE IS TO PREVENT DOCKER FROM CACHING THE PIP INSTALL SINCE IT CHANGES WHEN WE PUSH TO THE REPO
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN pip3 install git+https://github.com/mcgill-robotics/Humanoid-rl-baselines3-zoo.git@master

RUN apt-get autoremove -y
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN rm -rf /root/.cache/pip