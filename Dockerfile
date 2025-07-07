FROM ubuntu:22.04

ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    sudo \
    bzip2 \
    build-essential \
    gcc \
    g++ \
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

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create conda environment with updated packages
RUN conda create -n mctd-k python=3.10 -y

# Activate conda environment and set environment variables
ENV CONDA_DEFAULT_ENV=mctd-k
ENV PATH="/opt/conda/envs/mctd-k/bin:$PATH"

# Update conda and install latest libstdcxx-ng
RUN conda update -n base -c defaults conda -y && \
    conda install -n mctd-k libstdcxx-ng -y

# Set environment variables to use system libraries instead of conda libraries
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
ENV CONDA_BUILD_SYSROOT=""

RUN git clone https://github.com/Frankz78/mctd-k.git
WORKDIR /workspace/mctd-k

# Install pip and base packages in conda environment
RUN conda install pip==23.2.1 setuptools==65.5.0 wheel==0.38.4 packaging==21.3 -y

# Update libstdc++ to fix GLIBCXX compatibility issues
RUN conda install libstdcxx-ng -y

# Install project dependencies
RUN pip install -r requirements.txt
RUN pip install -r extra_requirements.txt
RUN pip install cython==0.29.36

# Replace conda's libstdc++ with system version to fix GLIBCXX issues
RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/envs/mctd-k/lib/libstdc++.so.6

RUN mkdir -p /root/.mujoco && \
    cd /root/.mujoco && \
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xvzf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

RUN python -c "import mujoco_py"
CMD [ "bash" ]


