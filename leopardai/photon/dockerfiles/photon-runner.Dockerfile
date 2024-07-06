ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY . /tmp/leopardai-sdk
RUN pip install /tmp/leopardai-sdk[runtime]
RUN rm -rf /tmp/leopardai-sdk

WORKDIR /workspace