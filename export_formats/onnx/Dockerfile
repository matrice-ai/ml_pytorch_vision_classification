FROM pytorch/pytorch:latest

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create working directory
RUN mkdir -p /usr/src/
WORKDIR /usr/src/

# Copy contents
COPY . /usr/src/

# Install python_sdk
RUN apt-get update
RUN apt-get install -y git
RUN git clone  https://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@github.com/matrice-ai/python-sdk.git -b dev python_sdk

ENV OMP_NUM_THREADS=8