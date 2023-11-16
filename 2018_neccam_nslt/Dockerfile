FROM nvidia/cuda:11.4.0-devel-ubuntu18.04

RUN apt-get update; apt-get update
RUN apt-get install git wget -y

ENV PATH="/root/miniconda2/bin:${PATH}"
ARG PATH="/root/miniconda2/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda2-py27_4.8.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda2-py27_4.8.3-Linux-x86_64.sh -b \
    && rm -f Miniconda2-py27_4.8.3-Linux-x86_64.sh

RUN git clone https://github.com/neccam/nslt.git
WORKDIR nslt

# Install dependencies
RUN pip install tensorflow==1.3.0

# Download auxiliary files
RUN wget "https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy" -O "BaseModel/bvlc_alexnet.npy"
# RUN wget "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"

# Unpack dataset
# TODO

# Resize images
# TODO: requires MATLAB to run https://github.com/neccam/nslt/blob/master/ResizeImages/resize_full_frames.m

WORKDIR /nslt/nslt
# python -m nmt --src=sign --tgt=de --train_prefix=../Data/phoenix2014T.train --dev_prefix=../Data/phoenix2014T.dev --test_prefix=../Data/phoenix2014T.test --out_dir=<your_output_dir> --vocab_prefix=../Data/phoenix2014T.vocab --source_reverse=True --num_units=1000 --num_layers=4 --num_train_steps=150000 --residual=True --attention=luong --base_gpu=<gpu_id> --unit_type=gru