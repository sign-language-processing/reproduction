FROM nvidia/cuda:11.4.0-devel-ubuntu18.04

RUN apt-get update; apt-get update
RUN apt-get install git wget -y

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.12.0-Linux-x86_64.sh

RUN git clone https://github.com/neccam/slt.git
WORKDIR slt

# ERROR: Could not find a version that satisfies the requirement conda==4.8.0
RUN sed -i 's/^conda==4.8.0//' requirements.txt
# ERROR: Could not find a version that satisfies the requirement conda-package-handling==1.6.0
RUN sed -i 's/^conda-package-handling==1.6.0//' requirements.txt
# ERROR: Could not find a version that satisfies the requirement tensorboard==2.1.2
RUN sed -i 's/^tensorboard==2.1.2/tensorboard==2.1.1/' requirements.txt
# ERROR: Could not find a version that satisfies the requirement tensorflow-estimator==2.1.2
RUN sed -i 's/^tensorflow-estimator==2.1.2/tensorflow-estimator==2.1.0/' requirements.txt
# ERROR: Could not find a version that satisfies the requirement warmup-scheduler==0.1.1
RUN sed -i 's/^warmup-scheduler==0.1.1/warmup-scheduler==0.3/' requirements.txt


RUN pip install -r requirements.txt

# Must download from the correct folder
RUN cd data && bash download.sh

# Fix config file data paths
RUN sed -i 's/PHOENIX2014T\///' configs/sign.yaml

# CUDA_VISIBLE_DEVICES=1 python -m signjoey train configs/sign.yaml

