FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /usr/src/
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=dialog

COPY . .

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 \
    && pip install  -r requirements.txt


ENV OMP_NUM_THREADS=4
