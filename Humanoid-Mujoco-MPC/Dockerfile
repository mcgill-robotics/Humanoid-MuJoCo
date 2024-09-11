FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /mujoco_mpc
COPY mujoco_mpc /mujoco_mpc

RUN apt-get update && \
    apt-get install -y \
    cmake \
    libgl1-mesa-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxi-dev \
    ninja-build \
    zlib1g-dev \
    clang-12 \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cd /mujoco_mpc

CMD ["bash", "-c", "while true; do sleep 1000; done"]
