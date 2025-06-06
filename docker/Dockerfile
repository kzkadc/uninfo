FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# set timeout
RUN /bin/echo -e "Acquire::http::Timeout \"300\";\n\
Acquire::ftp::Timeout \"300\";" >> /etc/apt/apt.conf.d/99timeout

# update and install necessary tools
RUN apt update && apt upgrade -y
RUN apt install -y build-essential libbz2-dev libdb-dev \
  libreadline-dev libffi-dev libgdbm-dev liblzma-dev \
  libncursesw5-dev libsqlite3-dev libssl-dev \
  zlib1g-dev uuid-dev tk-dev wget liblapack-dev \
  cpanminus curl locales libfontconfig1 ca-certificates \
  graphviz fonts-humor-sans git libmagickwand-dev \
  libopencv-dev

# cleanup
RUN apt clean -y \
  && apt autoremove -y \
  && apt autoclean -y \
  && rm /var/lib/apt/lists/* -rf

# install python
ARG VER=3.12.10
RUN wget "https://www.python.org/ftp/python/${VER}/Python-${VER}.tgz" \
    && tar -xzf Python-${VER}.tgz \
    && cd Python-${VER} \
    && ./configure --enable-optimizations --with-lto \
        PROFILE_TASK="-m test --pgo --timeout=$(TESTTIMEOUT) -j8" \
    && make -j8 \
    && make install -j8

# install pip
RUN wget "https://bootstrap.pypa.io/get-pip.py" \
    && python3 get-pip.py

RUN pip install -U pip setuptools wheel
RUN pip install -U jupyter
RUN pip install numpy
RUN pip install matplotlib seaborn umap-learn pandas scipy \
                scikit-learn opencv-python POT \
                scikit-image pyyaml 
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install pytorch-ignite
RUN pip install timm einops
RUN pip install imagenet-c
RUN pip install transformers sentencepiece
RUN pip install nvidia-ml-py "pynvml<12"

# CLIP
RUN pip install open_clip_torch

# https://github.com/openai/CLIP
RUN pip install ftfy regex
RUN pip install git+https://github.com/openai/CLIP.git

RUN pip uninstall numpy -y
RUN pip install "numpy<2"
