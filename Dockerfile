FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /usr/src/
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=dialog

COPY . .

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 wget\
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get -y install cuda-toolkit-12-6 \
    && export CUDA_HOME=/usr/local/cuda \
    && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64 \
    && export PATH=$PATH:$CUDA_HOME/bin
    && nvcc --version
    && python3 -m pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps matrice \
    && pip install  -r requirements.txt


ENV OMP_NUM_THREADS=8
