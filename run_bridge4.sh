#!/bin/bash
# Bridge 4 feature extraction runner with memory monitoring
cd ~/funny-strength-predictor

LOG="$HOME/tmp/bridge4_run.log"
CACHE="$HOME/tmp/bridge4_features.npz"

echo "Bridge 4 Feature Extraction" > "$LOG"
echo "Started: $(date)" >> "$LOG"

# Monitor and run
python3 -u bridge4_humor_arc_tracker.py 2>&1 | tee -a "$LOG" &
PID=$!
echo "PID: $PID" >> "$LOG"

# Memory monitoring loop
while kill -0 $PID 2>/dev/null; do
    # Check available memory
    AVAIL=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().available/1e9:.1f}')")
    echo "$(date +%H:%M:%S) PID=$PID avail=${AVAIL}GB" >> "$LOG"
    sleep 60
done

wait $PID
EXIT=$?
echo "Finished: $(date) exit=$EXIT" >> "$LOG"
