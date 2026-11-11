#!/usr/bin/env bash
# probe
set -x
pwd; id; hostname; nproc; nvidia-smi -L; df -h ~ .
env | grep -iE '^(JOB|OUT|HOME|PATH)' 
ls ~ ~/Documents ~/Documents/drama3-pilot-20261014 ~/Documents/drama3-pilot-20261014/eq/runs 2>&1 | head -80
ls ~/Documents/drama3-pilot-20261014/eq/runs/gptq-v1 2>&1 | head; ls ~/Documents/drama3-pilot-20261014/checkpoint 2>&1 | head -30
find ~ -maxdepth 6 -name 'layer-00.x.*' 2>/dev/null | head
find ~ -maxdepth 5 -type d -name '*tensorfold*' 2>/dev/null | head
ls ~/.cache/huggingface/hub 2>&1 | head
