FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create working directory
RUN mkdir -p /usr/src/
WORKDIR /usr/src/

# Copy contents
COPY . /usr/src/

# Install python_sdk
RUN python3 -m pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps matrice_sdk

ENV OMP_NUM_THREADS=8