#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM nvcr.io/nvidia/pytorch:21.10-py3

# Install miniconda ----------------
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3.9 pip wget

WORKDIR /work

COPY requirements.txt /work/
COPY run_t5.sh /work/
COPY fine_tunning_t5.py /work/

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

RUN pip install -r /work/requirements.txt
RUN ls /work

ENTRYPOINT [ "./run_t5.sh" ]
