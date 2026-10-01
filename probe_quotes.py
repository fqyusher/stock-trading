"""临时探针：核对东财/腾讯实时报价接口的字段位置。用完即删。"""

import json
import urllib.parse
import urllib.request

H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://quote.eastmoney.com/",
}
CODES = ["601127", "301632", "002001", "603345", "002028"]


def sec(c):
    return ("0." if c[0] in "0348" else "1.") + c


def probe_em():
    p = {
        "fltt": 2,
        "invt": 2,
        "fields": "f2,f3,f5,f6,f12,f14,f15,f16,f17,f18",
        "secids": ",".join(sec(c) for c in CODES),
    }
    url = "https://push2.eastmoney.com/api/qt/ulist.np/get?" + urllib.parse.urlencode(p)
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=15).read()
    d = json.loads(raw.decode())
    print("EM rc=", d.get("rc"))
    for r in (d.get("data") or {}).get("diff") or []:
        print(json.dumps(r, ensure_ascii=False))


def probe_tx():
    syms = [("sh" if c[0] in "69" else "sz") + c for c in CODES]
    url = "https://qt.gtimg.cn/q=" + ",".join(syms)
    req = urllib.request.Request(url, headers={**H, "Referer": "https://gu.qq.com/"})
    text = urllib.request.urlopen(req, timeout=15).read().decode("gbk", "ignore")
    for line in text.strip().split(";"):
        line = line.strip()
        if "=" not in line:
            continue
        parts = line.split("=", 1)[1].strip().strip('"').split("~")
        print("TX len=", len(parts))
        for i in (1, 2, 3, 4, 5, 6, 30, 31, 32, 33, 34, 36, 37, 38, 44, 45, 47):
            if i < len(parts):
                print(f"   [{i}] {parts[i]}")
        break


probe_em()
probe_tx()
