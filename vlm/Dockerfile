# example deep learning VM
# for a full list see us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# and for details see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# pip gives a warning if you install packages as root
# set this flag to just ignore the warning
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install -U pip
WORKDIR /usr/src
ENV MODEL_PATH=/usr/src/models

# install other requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0

# copy the rest of the files into the container
COPY src /usr/src

# start model service
CMD uvicorn api_service:app --port 5004 --host 0.0.0.0