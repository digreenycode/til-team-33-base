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

RUN pip install --upgrade pip
WORKDIR /workspace

# install other requirements
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install --upgrade transformers accelerate jiwer \
    && pip uninstall -y torch_xla


# copy the rest of the files into the container
COPY src .

# Copy specific model files into the container
COPY wsp-trained/config.json \
     wsp-trained/generation_config.json \
     wsp-trained/model.safetensors \
     wsp-trained/preprocessor_config.json \
     wsp-trained/training_args.bin \
     src/wsp-trained/

EXPOSE 5001

# start model service
CMD uvicorn api_service:app --port 5001 --host 0.0.0.0