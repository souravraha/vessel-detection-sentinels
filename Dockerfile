# Base Image
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04@sha256:9ccfe38d9cb31ae23f6f8f9595b450565da18399c2e71ebc9a5c079f786319e3

# Environment Variables
ENV WANDB_API_KEY=$WANDB_API_KEY \
    RUN_TAG=$RUN_TAG \
    WANDB_MODE=$WANDB_MODE \
    WANDB_START_METHOD="thread" \
    WANDB_PROJECT="vessel-detection" \
    COPERNICUS_USERNAME=$COPERNICUS_USERNAME \
    COPERNICUS_PASSWORD=$COPERNICUS_PASSWORD

# System Dependencies and Cleanup
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata && \
    apt-get install -y software-properties-common ffmpeg libsm6 libxext6 libhdf5-serial-dev netcdf-bin libnetcdf-dev && \
    add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
    apt-get update && \
    apt-get install -y \
        curl build-essential pkg-config python3-dev \
        gdal-bin libgdal-dev python3-gdal \
        proj-bin libproj-dev \
        libgeos-dev libpq-dev python3-pip apt-transport-https ca-certificates gnupg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /home/vessel_detection/requirements.txt

# Install Python Packages (use python3 -m pip to ensure the upgraded pip is used)
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -U numpy==1.23.* cython && \
    python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu113 torch==1.13.* torchvision==0.14.* && \
    python3 -m pip install --no-cache-dir -r /home/vessel_detection/requirements.txt

# Set Working Directory and Prepare App
WORKDIR /home/vessel_detection/src
COPY src /home/vessel_detection/src
COPY tests /home/vessel_detection/src/
RUN mkdir -p /root/.cache/torch/hub/checkpoints/
COPY torch_weights/swin_v2_s-637d8ceb.pth /root/.cache/torch/hub/checkpoints/swin_v2_s-637d8ceb.pth
COPY torch_weights/resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
COPY torch_weights/swin_v2_t-b137f0e2.pth /root/.cache/torch/hub/checkpoints/swin_v2_t-b137f0e2.pth

# CMD
CMD ["python3", "main.py"]