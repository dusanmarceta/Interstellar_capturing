#!/bin/sh

number_of_jobs=36

NUM_CORES=80
MEMORY_LIMIT="40G"  # Set memory limit to 40 GB
core_id=-96
# Limit memory usage to 40 GB (in kilobytes)
ulimit -v $((40 * 1024 * 1024))

for i in `seq 1 $number_of_jobs`;
do
    core_id=$(( (core_id + 1) ))  # rotira kroz raspoloživa jezgra
    nohup taskset -c ${core_id} python capture_shell_barycentric.py "$i" &
    sleep 60
done

