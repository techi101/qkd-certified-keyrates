#!/bin/bash
cd "$(dirname "$0")/../.."  # run from the repo root
for k in 0 1 2 3; do
  idx=$(python3 -c "print(','.join(str(i) for i in range($k,120,4)))")
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/qkdenv/bin/python src/t1_standard_float.py $idx 0,1e-14,1e-12,1e-10,1e-8 out/t1_part$k.json > out/t1_part$k.log 2>&1 &
done
wait
echo DONE
