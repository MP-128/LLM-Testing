#!/bin/bash
cd "/Users/mathesonparmar/Desktop/AQ Work Trial/eval_harness"
python3 run_pass3.py --dataset pc_sample > /tmp/pass3_output.log 2>&1
echo "EXIT_CODE=$?"
