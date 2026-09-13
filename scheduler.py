"""Run Aria only during the configured daily window (default: 20:00-23:00 Tehran time)."""
import os, time
from datetime import datetime
from zoneinfo import ZoneInfo
from agent import cycle

TZ = ZoneInfo(os.getenv('ARIA_TIMEZONE', 'Asia/Tehran'))
START = os.getenv('ARIA_START', '20:00')
END = os.getenv('ARIA_END', '23:00')
INTERVAL = int(os.getenv('ARIA_INTERVAL_MINUTES', '30'))

def hm(s):
    h, m = map(int, s.split(':'))
    return h * 60 + m

def in_window(now):
    cur = now.hour * 60 + now.minute
    start, end = hm(START), hm(END)
    return start <= cur < end if start < end else (cur >= start or cur < end)

while True:
    now = datetime.now(TZ)
    if in_window(now):
        try:
            cycle()
        except Exception as exc:
            print(f'[Aria] cycle error: {exc}', flush=True)
        time.sleep(INTERVAL * 60)
    else:
        time.sleep(60)
