#!/bin/bash
# 每日增量更新K线数据：从前天到今天
# crontab: 0 18 * * * /mnt/f/self/program/codes/trade-yh/cron_update.sh

set -euo pipefail

PROJECT_DIR="/mnt/f/self/program/codes/trade-yh"
LOG_DIR="$PROJECT_DIR/data/logs"
LOG_FILE="$LOG_DIR/update_$(date +%Y%m%d).log"

mkdir -p "$LOG_DIR"

START=$(date -d "5 days ago" +%Y%m%d)
END=$(date +%Y%m%d)

echo "===== $(date '+%Y-%m-%d %H:%M:%S') 开始更新 $START ~ $END =====" >> "$LOG_FILE"

cd "$PROJECT_DIR"
python3 baostock_update.py --start "$START" --end "$END" >> "$LOG_FILE" 2>&1

echo "===== $(date '+%Y-%m-%d %H:%M:%S') 更新完成 =====" >> "$LOG_FILE"

echo "===== 开始扫描形态 =====" >> "$LOG_FILE"
# --plot：charts.enabled 默认 false，定时任务要留图，这里显式打开
python3 cli.py scan --plot >> "$LOG_FILE" 2>&1
echo "===== 结束扫描形态 =====" >> "$LOG_FILE"

echo "===== 开始同步观察池 =====" >> "$LOG_FILE"
python3 cli.py watch sync
echo "===== 结束同步观察池 =====" >> "$LOG_FILE"

echo "" >> "$LOG_FILE"
