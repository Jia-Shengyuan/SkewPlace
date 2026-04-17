FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
LABEL maintainer="Yibo Lin <yibolin@pku.edu.cn>"

# Rotates to the keys used by NVIDIA as of 27-APR-2022.
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# Installs system dependencies.
RUN apt-get update -o Acquire::Retries=5 -o Acquire::http::Timeout=30 \
        && apt-get install -y --fix-missing -o Acquire::Retries=5 -o Acquire::http::Timeout=30 \
            git \
            flex \
            libcairo2-dev \
            libboost-all-dev 


# Installs system dependencies from conda.
RUN conda install -y -c conda-forge bison

# Installs cmake.
ADD https://cmake.org/files/v3.21/cmake-3.21.0-linux-x86_64.sh /cmake-3.21.0-linux-x86_64.sh
RUN mkdir /opt/cmake \
        && sh /cmake-3.21.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
        && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
        && cmake --version

# Installs python dependencies. 
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
