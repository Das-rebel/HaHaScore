#!/bin/bash
cd ~/funny-strength-predictor
exec python3 -u bridge1_pseudolabel_cpu.py >> ~/tmp/bridge1_output/bridge1_$(date +%Y%m%d_%H%M%S).log 2>&1
