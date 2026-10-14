#!/bin/bash

# Monitoring script for the Flask application
# Checks every 30 seconds and detects freezes or malfunctions

LOG_FILE="/tmp/flask_monitor.log"
APP_URL="http://localhost:5001"
APP_LOG="/tmp/flask_app.log"
CHECK_INTERVAL=30

echo "$(date '+%Y-%m-%d %H:%M:%S') - Monitoring started" >> "$LOG_FILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 1. Port check
    PORT_CHECK=$(lsof -ti:5001 2>/dev/null | wc -l)
    if [ "$PORT_CHECK" -eq 0 ]; then
        echo "$TIMESTAMP - ⚠️  WARNING: No process running on port 5001!" >> "$LOG_FILE"
        echo "$TIMESTAMP - ⚠️  WARNING: No process running on port 5001!"
    fi
    
    # 2. HTTP response check
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$APP_URL" 2>/dev/null)
    if [ "$HTTP_CODE" != "200" ]; then
        echo "$TIMESTAMP - ⚠️  WARNING: HTTP response code: $HTTP_CODE (expected: 200)" >> "$LOG_FILE"
        echo "$TIMESTAMP - ⚠️  WARNING: HTTP response code: $HTTP_CODE (expected: 200)"
    fi
    
    # 3. Process CPU usage check (freeze check)
    PROCESS_CPU=$(ps aux | grep -E "python.*app.py" | grep -v grep | awk '{print $3}' | head -1)
    if [ -n "$PROCESS_CPU" ]; then
        # CPU usage of 0 may indicate a freeze (but this is not fully reliable)
        CPU_INT=$(echo "$PROCESS_CPU" | cut -d. -f1)
        if [ "$CPU_INT" -eq 0 ] && [ "$CPU_INT" -lt 0 ]; then
            # Wait 5 seconds and check again
            sleep 5
            PROCESS_CPU2=$(ps aux | grep -E "python.*app.py" | grep -v grep | awk '{print $3}' | head -1)
            CPU_INT2=$(echo "$PROCESS_CPU2" | cut -d. -f1)
            if [ "$CPU_INT2" -eq 0 ]; then
                echo "$TIMESTAMP - ⚠️  WARNING: Process CPU usage is 0 - possible freeze!" >> "$LOG_FILE"
                echo "$TIMESTAMP - ⚠️  WARNING: Process CPU usage is 0 - possible freeze!"
            fi
        fi
    fi
    
    # 4. Check the last log line (is there an error?)
    LAST_ERROR=$(tail -50 "$APP_LOG" 2>/dev/null | grep -iE "error|traceback|exception|failed" | tail -1)
    if [ -n "$LAST_ERROR" ]; then
        ERROR_TIME=$(echo "$LAST_ERROR" | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" | tail -1)
        if [ -n "$ERROR_TIME" ]; then
            ERROR_EPOCH=$(date -j -f "%Y-%m-%d %H:%M:%S" "$ERROR_TIME" +%s 2>/dev/null || date -d "$ERROR_TIME" +%s 2>/dev/null)
            NOW_EPOCH=$(date +%s)
            TIME_DIFF=$((NOW_EPOCH - ERROR_EPOCH))
            # Warn if there is an error within the last 60 seconds
            if [ "$TIME_DIFF" -lt 60 ]; then
                echo "$TIMESTAMP - ⚠️  WARNING: Error detected in the latest logs!" >> "$LOG_FILE"
                echo "$TIMESTAMP - ⚠️  WARNING: $LAST_ERROR" >> "$LOG_FILE"
                echo "$TIMESTAMP - ⚠️  WARNING: Error detected in the latest logs!"
            fi
        fi
    fi
    
    # 5. Process liveness check
    PROCESS_COUNT=$(ps aux | grep -E "python.*app.py" | grep -v grep | wc -l | tr -d ' ')
    if [ "$PROCESS_COUNT" -gt 2 ]; then
        echo "$TIMESTAMP - ⚠️  WARNING: $PROCESS_COUNT Flask processes running (expected: 1-2)" >> "$LOG_FILE"
        echo "$TIMESTAMP - ⚠️  WARNING: $PROCESS_COUNT Flask processes running"
    elif [ "$PROCESS_COUNT" -eq 0 ]; then
        echo "$TIMESTAMP - ❌ ERROR: Flask process is not running!" >> "$LOG_FILE"
        echo "$TIMESTAMP - ❌ ERROR: Flask process is not running!"
    fi
    
    # 6. Normal status logu (her kontrolte)
    if [ "$HTTP_CODE" == "200" ] && [ "$PROCESS_COUNT" -gt 0 ] && [ "$PROCESS_COUNT" -le 2 ]; then
        echo "$TIMESTAMP - ✅ OK: Application running normally (HTTP: $HTTP_CODE, Process: $PROCESS_COUNT)" >> "$LOG_FILE"
    fi
    
    sleep "$CHECK_INTERVAL"
done
