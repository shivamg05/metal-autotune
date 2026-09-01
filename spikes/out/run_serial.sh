#!/bin/bash
cd /Users/shivamgarg/dev/metal-autotune
for n in 01 02 03 04 05 06 07 08 09; do
  f=$(ls spikes/spike_${n}_*.py)
  echo "=== running $f"
  uv run python "$f" > "spikes/out/logs/$(basename $f .py).log" 2>&1
  echo "=== $f exit=$?"
done
echo "=== running spike_10 --full"
uv run python spikes/spike_10_measurement.py --full > spikes/out/logs/spike_10_measurement_full.log 2>&1
echo "=== spike_10 --full exit=$?"
echo ALL_SPIKES_DONE
