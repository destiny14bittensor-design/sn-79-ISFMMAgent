#!/usr/bin/env bash
export PYTHONPATH=/home/administrator/sn-79
export PATH=/home/administrator/miniconda3/envs/sn79-py3109/bin:/usr/local/gcc-14.1.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/gcc-14.1.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CONDA_PREFIX=/home/administrator/miniconda3/envs/sn79-py3109

exec /home/administrator/miniconda3/envs/sn79-py3109/bin/python3 \
    /home/administrator/sn-79/agents/proxy/launcher.py \
    --config /home/administrator/sn-79/agents/proxy/config_isfmm.json
