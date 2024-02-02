FROM ubuntu:latest

WORKDIR /usr/src/

COPY . .

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 python3 python3-pip \
    && pip install  -r requirements.txt

ENV OMP_NUM_THREADS=4
