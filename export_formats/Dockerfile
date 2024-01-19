# Use the openvino/ubuntu22_runtime image as the base image or any other runtime of your choice
FROM openvino/ubuntu22_runtime:latest

USER root
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt install pip -y
SHELL ["/bin/bash","-c"]
# Install pip requirements
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD ["python", "openvino_eval.py"]
CMD ["sleep", "infinity"]
