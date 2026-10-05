#!/bin/bash
# usage: benchmarks/jobs/submit_node_check.sh gadi_h200|gadi_v100
set -e
M=$1
case $M in gadi_h200) Q=gpuhopper; MEM=500 ;; gadi_v100) Q=gpuvolta; MEM=380 ;; *) echo "machine?"; exit 2 ;; esac
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D/../results"
sed -e "s/@M@/$M/g" -e "s/@Q@/$Q/g" -e "s/@MEM@/$MEM/g" "$D/node_check.pbs.in" > "$D/node_check_$M.pbs"
qsub "$D/node_check_$M.pbs"
