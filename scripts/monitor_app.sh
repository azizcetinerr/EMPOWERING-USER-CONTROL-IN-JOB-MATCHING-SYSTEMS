#!/bin/bash

# Flask uygulaması monitoring script'i
# 30 saniyede bir kontrol eder ve donma/yanlış çalışma durumlarını tespit eder

LOG_FILE="/tmp/flask_monitor.log"
APP_URL="http://localhost:5001"
APP_LOG="/tmp/flask_app.log"
CHECK_INTERVAL=30

echo "$(date '+%Y-%m-%d %H:%M:%S') - Monitoring başlatıldı" >> "$LOG_FILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 1. Port kontrolü
    PORT_CHECK=$(lsof -ti:5001 2>/dev/null | wc -l)
    if [ "$PORT_CHECK" -eq 0 ]; then
        echo "$TIMESTAMP - ⚠️  UYARI: Port 5001'de çalışan process yok!" >> "$LOG_FILE"
        echo "$TIMESTAMP - ⚠️  UYARI: Port 5001'de çalışan process yok!"
    fi
    
    # 2. HTTP yanıt kontrolü
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$APP_URL" 2>/dev/null)
    if [ "$HTTP_CODE" != "200" ]; then
        echo "$TIMESTAMP - ⚠️  UYARI: HTTP yanıt kodu: $HTTP_CODE (beklenen: 200)" >> "$LOG_FILE"
        echo "$TIMESTAMP - ⚠️  UYARI: HTTP yanıt kodu: $HTTP_CODE (beklenen: 200)"
    fi
    
    # 3. Process CPU kullanımı kontrolü (donma kontrolü)
    PROCESS_CPU=$(ps aux | grep -E "python.*app.py" | grep -v grep | awk '{print $3}' | head -1)
    if [ -n "$PROCESS_CPU" ]; then
        # CPU kullanımı 0 ise donma olabilir (ama tam güvenilir değil)
        CPU_INT=$(echo "$PROCESS_CPU" | cut -d. -f1)
        if [ "$CPU_INT" -eq 0 ] && [ "$CPU_INT" -lt 0 ]; then
            # 5 saniye bekleyip tekrar kontrol et
            sleep 5
            PROCESS_CPU2=$(ps aux | grep -E "python.*app.py" | grep -v grep | awk '{print $3}' | head -1)
            CPU_INT2=$(echo "$PROCESS_CPU2" | cut -d. -f1)
            if [ "$CPU_INT2" -eq 0 ]; then
                echo "$TIMESTAMP - ⚠️  UYARI: Process CPU kullanımı 0 - Donma olabilir!" >> "$LOG_FILE"
                echo "$TIMESTAMP - ⚠️  UYARI: Process CPU kullanımı 0 - Donma olabilir!"
            fi
        fi
    fi
    
    # 4. Son log satırını kontrol et (hata var mı?)
    LAST_ERROR=$(tail -50 "$APP_LOG" 2>/dev/null | grep -iE "error|traceback|exception|failed" | tail -1)
    if [ -n "$LAST_ERROR" ]; then
        ERROR_TIME=$(echo "$LAST_ERROR" | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" | tail -1)
        if [ -n "$ERROR_TIME" ]; then
            ERROR_EPOCH=$(date -j -f "%Y-%m-%d %H:%M:%S" "$ERROR_TIME" +%s 2>/dev/null || date -d "$ERROR_TIME" +%s 2>/dev/null)
            NOW_EPOCH=$(date +%s)
            TIME_DIFF=$((NOW_EPOCH - ERROR_EPOCH))
            # Son 60 saniye içinde hata varsa uyar
            if [ "$TIME_DIFF" -lt 60 ]; then
                echo "$TIMESTAMP - ⚠️  UYARI: Son loglarda hata tespit edildi!" >> "$LOG_FILE"
                echo "$TIMESTAMP - ⚠️  UYARI: $LAST_ERROR" >> "$LOG_FILE"
                echo "$TIMESTAMP - ⚠️  UYARI: Son loglarda hata tespit edildi!"
            fi
        fi
    fi
    
    # 5. Process yaşama kontrolü
    PROCESS_COUNT=$(ps aux | grep -E "python.*app.py" | grep -v grep | wc -l | tr -d ' ')
    if [ "$PROCESS_COUNT" -gt 2 ]; then
        echo "$TIMESTAMP - ⚠️  UYARI: $PROCESS_COUNT adet Flask process çalışıyor (beklenen: 1-2)" >> "$LOG_FILE"
        echo "$TIMESTAMP - ⚠️  UYARI: $PROCESS_COUNT adet Flask process çalışıyor"
    elif [ "$PROCESS_COUNT" -eq 0 ]; then
        echo "$TIMESTAMP - ❌ HATA: Flask process çalışmıyor!" >> "$LOG_FILE"
        echo "$TIMESTAMP - ❌ HATA: Flask process çalışmıyor!"
    fi
    
    # 6. Normal durum logu (her kontrolte)
    if [ "$HTTP_CODE" == "200" ] && [ "$PROCESS_COUNT" -gt 0 ] && [ "$PROCESS_COUNT" -le 2 ]; then
        echo "$TIMESTAMP - ✅ OK: Uygulama normal çalışıyor (HTTP: $HTTP_CODE, Process: $PROCESS_COUNT)" >> "$LOG_FILE"
    fi
    
    sleep "$CHECK_INTERVAL"
done
