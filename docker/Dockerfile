# =============================================================================
# Dockerfile for HimLoco Lab
# Base: NVIDIA Isaac Sim Container + Isaac Lab
# Target: Ubuntu 20.04 host with NVIDIA 570 driver
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base - Isaac Sim Container
# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/isaac-sim:6.0.0-dev2

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Accept EULA
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y

# -----------------------------------------------------------------------------
# Stage 2: System Dependencies
# -----------------------------------------------------------------------------
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    rsync \
    unzip \
    # Python and related
    python3.10 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-setuptools \
    # YAML and config
    libyaml-cpp-dev \
    # Linear algebra
    libeigen3-dev \
    # Boost libraries
    libboost-all-dev \
    # Logging
    libspdlog-dev \
    # Formatting
    libfmt-dev \
    # ROS dependencies (if needed)
    ros-noetic-catkin \
    # Network tools
    openssh-client \
    # Misc
    vim \
    htop \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage 3: Isaac Lab Installation
# -----------------------------------------------------------------------------
# Create non-root user for Isaac Lab
ARG ISAACLAB_USER=isaaclab
ARG ISAACLAB_UID=1234
ARG ISAACLAB_GID=1234

RUN groupadd -g ${ISAACLAB_GID} ${ISAACLAB_USER} && \
    useradd -m -u ${ISAACLAB_UID} -g ${ISAACLAB_GID} -s /bin/bash ${ISAACLAB_USER}

# Install Isaac Lab (using pip for headless server)
USER ${ISAACLAB_USER}
WORKDIR /home/${ISAACLAB_USER}

# Clone Isaac Lab repository
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    ./setup.sh --pip

# Set environment for Isaac Lab
ENV ISAACLAB_PATH=/home/${ISAACLAB_USER}/IsaacLab
ENV PATH=${ISAACLAB_PATH}/bin:$PATH
ENV PYTHONPATH=${ISAACLAB_PATH}:$PYTHONPATH

# -----------------------------------------------------------------------------
# Stage 4: HimLoco Lab Installation
# -----------------------------------------------------------------------------
# Copy HimLoco Lab source code
COPY --chown=${ISAACLAB_USER}:${ISAACLAB_GID} source/ /home/${ISAACLAB_USER}/himloco_lab/source/
COPY --chown=${ISAACLAB_USER}:${ISAACLAB_GID} scripts/ /home/${ISAACLAB_USER}/himloco_lab/scripts/
COPY --chown=${ISAACLAB_USER}:${ISAACLAB_GID} deploy/ /home/${ISAACLAB_USER}/himloco_lab/deploy/
COPY --chown=${ISAACLAB_USER}:${ISAACLAB_GID} unitree_ros/ /home/${ISAACLAB_USER}/himloco_lab/unitree_ros/
COPY --chown=${ISAACLAB_USER}:${ISAACLAB_GID} unitree_sdk2/ /home/${ISAACLAB_USER}/himloco_lab/unitree_sdk2/

# Install HimLoco Lab as pip package
RUN cd /home/${ISAACLAB_USER}/himloco_lab && \
    python3 -m pip install -e source/himloco_lab

# Install additional Python dependencies
RUN python3 -m pip install \
    mujoco \
    onnx \
    onnxruntime \
    pyyaml \
    matplotlib \
    pandas \
    numpy

# -----------------------------------------------------------------------------
# Stage 5: Unitree Robot Models
# -----------------------------------------------------------------------------
# Clone Unitree model files (if not already present)
RUN git clone https://huggingface.co/datasets/unitreerobotics/unitree_model.git /home/${ISAACLAB_USER}/unitree_model && \
    git clone https://github.com/unitreerobotics/unitree_ros.git /home/${ISAACLAB_USER}/unitree_ros

# Configure robot model paths
# Note: Update these paths in source/himloco_lab/himloco_lab/assets/robots/unitree.py
ENV UNITREE_MODEL_DIR=/home/${ISAACLAB_USER}/unitree_model
ENV UNITREE_ROS_DIR=/home/${ISAACLAB_USER}/unitree_ros/unitree_ros

# -----------------------------------------------------------------------------
# Stage 6: Working Directory and Entry Point
# -----------------------------------------------------------------------------
WORKDIR /home/${ISAACLAB_USER}/himloco_lab

# Create output directories
RUN mkdir -p /home/${ISAACLAB_USER}/himloco_lab/logs && \
    mkdir -p /home/${ISAACLAB_USER}/himloco_lab/outputs

# Default entrypoint
ENTRYPOINT ["/bin/bash"]
