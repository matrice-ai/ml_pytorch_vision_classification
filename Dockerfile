FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create working directory
RUN mkdir -p /usr/src/
WORKDIR /usr/src/

# Copy contents
COPY . /usr/src/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ENV OMP_NUM_THREADS=8
