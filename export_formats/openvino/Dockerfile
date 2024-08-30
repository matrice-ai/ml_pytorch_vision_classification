FROM openvino/ubuntu22_runtime:latest

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create working directory
RUN mkdir -p /usr/src/
WORKDIR /usr/src/

# Copy contents
COPY . /usr/src/

# Install python_sdk
RUN python3 -m pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps matrice

ENV OMP_NUM_THREADS=8