#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=true

python3 fine_tunning_t5.py
