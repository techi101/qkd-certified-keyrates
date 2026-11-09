#!/bin/bash
cd "$(dirname "$0")/../.."  # run from the repo root
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for nc in 8 10 12; do
  timeout 1200 ~/qkdenv/bin/python src/t2_scaling.py $nc 8 out/t2_nc$nc.json > out/t2_nc$nc.log 2>&1 &
done
wait; echo DONE
