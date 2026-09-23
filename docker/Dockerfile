# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stage 1: Build XLA Serving binary
FROM ubuntu:22.04@sha256:eb29ed27b0821dca09c2e28b39135e185fc1302036427d5f4d70a41ce8fd7659 as build_image

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        automake \
        build-essential \
        ca-certificates \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng-dev \
        libtool \
        libzmq3-dev \
        openjdk-11-jdk \
        openjdk-11-jre-headless \
        pkg-config \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
        software-properties-common \
        swig \
        unzip \
        zip \
        zlib1g-dev \
        python3-setuptools \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 --no-cache-dir install \
    future>=0.17.1 \
    grpcio \
    h5py \
    keras_applications>=1.0.8 \
    keras_preprocessing>=1.1.0 \
    mock \
    numpy \
    portpicker \
    requests \
    --ignore-installed setuptools \
    --ignore-installed six>=1.12.0

# Install Bazelisk (resolves .bazelversion 7.7.0)
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel

WORKDIR /src
COPY . .

# Build the binary
RUN bazel build //runtime/serving/gcp/server:xla_serving_main_cpu

# Stage 2: Runtime image
FROM ubuntu:22.04@sha256:eb29ed27b0821dca09c2e28b39135e185fc1302036427d5f4d70a41ce8fd7659

ARG XLA_SERVING_VERSION_GIT_BRANCH=master
ARG XLA_SERVING_VERSION_GIT_COMMIT=head
ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="neptune-eng@google.com"
LABEL xla_serving_github_branchtag=${XLA_SERVING_VERSION_GIT_BRANCH}
LABEL xla_serving_github_commit=${XLA_SERVING_VERSION_GIT_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install XLA Serving pkg & dependencies
COPY --from=build_image /src/bazel-bin/runtime/serving/gcp/server/xla_serving_main_cpu /usr/bin/xla_serving_main
COPY --from=build_image /src/bazel-bin/runtime/serving/gcp/server/xla_serving_main_cpu.runfiles /usr/bin/xla_serving_main.runfiles

# Expose ports
# gRPC
EXPOSE 8500

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
ENV MODEL_NAME=default

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
RUN echo '#!/bin/bash \n\n\
xla_serving_main --port=8500 \
--model_name=${MODEL_NAME} --model_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/xla_serving_entrypoint.sh \
&& chmod +x /usr/bin/xla_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/xla_serving_entrypoint.sh"]
