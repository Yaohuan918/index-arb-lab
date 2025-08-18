#!/usr/bin/env bash
# Very simple log rotation script: archive current log with timestamp and keep last N copies.

set -euo pipefail 
LOG_DIR="logs"
MAX_FILES=5 

mkdir -p "$LOG_DIR" 
cd "$LOG_DIR" 

if [ -f app.log ]; then
    ts=$(date +%Y%m%d_%H%M%S) 
    cp app.log "app.log.${ts}" 
    : > app.log  # Clear the current log file 
fi 

# Keep latest N archives; delete older ones 
ls -1t app.log.* 2>/dev/null | tail -n +$MAX_FILES | xargs -r rm -f 
echo "Log rotation complete." 
SH 
chmod +x scripts/rotate_logs.sh