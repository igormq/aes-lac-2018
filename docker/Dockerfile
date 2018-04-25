FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/

# install miniconda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# updating conda and pip and prevents warning messages
RUN conda update -n base conda && \
    pip install pip --upgrade

# install pytorch and some dependencies
RUN conda install -y jupyter numpy scipy scikit-learn && \
    conda install -y pytorch torchvision cuda91 -c pytorch

# install project dependencies
COPY ../requirements.txt /requirements.txt
RUN pip install -r requirements.txt && \
    rm requirements.txt

# install basics
RUN apt-get update -y && \
    apt-get install -y git cmake tree htop bmon iotop && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# install warp-CTC
RUN git clone https://github.com/SeanNaren/warp-ctc.git && \
    cd warp-ctc && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cd .. && \
    cd pytorch_binding && \
    CUDA_HOME="/usr/local/cuda" python setup.py install && \
    cd ../.. && \
    rm -rf warp-ctc;

# install pytorch audio
RUN apt-get update -y && \
    apt-get install -y sox libsox-dev libsox-fmt-all && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/pytorch/audio.git && \
    cd audio && \
    python setup.py install && \
    cd .. && \
    rm -rf audio

# install ctcdecode
RUN git clone --recursive https://github.com/parlance/ctcdecode.git && \
    cd ctcdecode && \
    pip install . && \
    cd .. && \
    rm -rf ctcdecode

ENTRYPOINT ["/bin/bash", "-c"]

EXPOSE 8888
CMD jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --no-browser