# Start with a minimal base image
FROM debian:buster-slim

# Install basic packages
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh 
RUN /bin/bash ~/miniconda.sh -b -p $CONDA_DIR
RUN rm ~/miniconda.sh
RUN conda init bash

# Copy your environment file into the container
COPY environment.yaml /usr/src/app/environment.yaml
COPY requirements.txt /usr/src/app/requirements.txt

# Set the working directory
WORKDIR /usr/src/app


# Create the Conda environment
RUN conda config --add channels conda-forge
RUN conda create --name gym python=3.10

RUN conda run --name gym pip install swig
RUN conda run --name gym pip install box2d-py
RUN conda run --name gym pip install pygame
RUN conda run --name gym pip install tensorflow-cpu

RUN conda run --name gym conda install --file ./requirements.txt

VOLUME ./:/usr/src/app/mnt