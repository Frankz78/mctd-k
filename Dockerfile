FROM ubuntu:22.04


ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    sudo \
    bzip2 \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1-mesa-glx \
    libosmesa6 \
    libglfw3 \
    libglew-dev \
    libgl1-mesa-dev \
    libglfw3-dev \
    patchelf \
    tmux \
    xvfb \
    libosmesa6-dev \        
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN python3.10 -m venv venv
ENV PATH="/workspace/venv/bin:$PATH"

RUN git clone https://github.com/Frankz78/mctd-k.git
WORKDIR /workspace/mctd-k

RUN pip install pip==23.2.1 setuptools==65.5.0 wheel==0.38.4 packaging==21.3

RUN pip install -r requirements.txt
RUN pip install -r extra_requirements.txt
RUN pip install cython==0.29.36

RUN mkdir -p /root/.mujoco && \
    cd /root/.mujoco && \
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xvzf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

RUN python -c "import mujoco_py"
CMD [ "bash" ]


