"""存储层：sqlite / csv / json 三种后端，统一 Store 接口。

内部统一用英文列名，只在 report.py 出报表时转回中文。
日期一律存 TEXT 'YYYY-MM-DD'，便于字符串比较和排序。

模块级 sqlite 函数保留为底层实现，SqliteStore 只是薄封装；
csv / json 后端按"一股一文件"存放，便于人工查看与外部工具处理。
上层一律通过 open_store(cfg) 拿后端，不直接碰连接。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

# 东财中文列名 → 内部英文列（供外部导入中文表时复用；抓取层已改直连HTTP，自行拼英文列）
COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "涨跌幅": "pct_chg",
}

KLINE_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "turnover",
    "pct_chg",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kline (
    code     TEXT NOT NULL,
    date     TEXT NOT NULL,
    open     REAL,
    close    REAL,
    high     REAL,
    low      REAL,
    volume   REAL,
    amount   REAL,
    turnover REAL,
    pct_chg  REAL,
    PRIMARY KEY (code, date)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS stock_info (
    code       TEXT PRIMARY KEY,
    name       TEXT,
    price      REAL,
    float_mv   REAL,
    amount     REAL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(db_path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    """打开连接。sqlite 连接不能跨进程共享，每个 worker 各自 connect。"""
    db_path = Path(db_path)
    if readonly:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=60)
        # /mnt/f 这类 drvfs 挂载点上 WAL 可能失败，失败就退回默认日志模式
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def normalize_kline(raw: pd.DataFrame) -> pd.DataFrame:
    """中文列原始表 → 内部列名与类型。空表原样返回空表。"""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = raw.rename(columns=COLUMN_MAP)
    missing = [c for c in KLINE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"K线数据缺少列: {missing}")

    df = df[KLINE_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in KLINE_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "close"])
    return df.sort_values("date").reset_index(drop=True)


def upsert_kline(conn: sqlite3.Connection, code: str, df: pd.DataFrame) -> int:
    """写入/覆盖K线。幂等：同 (code, date) 直接替换。"""
    if df is None or df.empty:
        return 0
    rows = [(code, *row) for row in df[KLINE_COLUMNS].itertuples(index=False, name=None)]
    conn.executemany(
        "INSERT OR REPLACE INTO kline "
        "(code, date, open, close, high, low, volume, amount, turnover, pct_chg) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def delete_code(conn: sqlite3.Connection, code: str) -> None:
    conn.execute("DELETE FROM kline WHERE code = ?", (code,))


def last_date(conn: sqlite3.Connection, code: str) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM kline WHERE code = ?", (code,)).fetchone()
    return row[0] if row and row[0] else None


def bar_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        code: n
        for code, n in conn.execute("SELECT code, COUNT(*) FROM kline GROUP BY code")
    }


def read_kline(
    conn: sqlite3.Connection, code: str, limit_bars: int | None = None
) -> pd.DataFrame:
    """返回按日期升序、DatetimeIndex 的K线。limit_bars 取最近N根。"""
    if limit_bars:
        sql = (
            "SELECT date, open, close, high, low, volume, amount, turnover, pct_chg "
            "FROM kline WHERE code = ? ORDER BY date DESC LIMIT ?"
        )
        df = pd.read_sql_query(sql, conn, params=(code, int(limit_bars)))
        df = df.iloc[::-1].copy()
    else:
        sql = (
            "SELECT date, open, close, high, low, volume, amount, turnover, pct_chg "
            "FROM kline WHERE code = ? ORDER BY date"
        )
        df = pd.read_sql_query(sql, conn, params=(code,))

    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def iter_klines(
    conn: sqlite3.Connection, codes: Iterable[str], limit_bars: int | None = None
) -> Iterator[tuple[str, pd.DataFrame]]:
    for code in codes:
        yield code, read_kline(conn, code, limit_bars)


def detect_adjust_drift(
    conn: sqlite3.Connection, code: str, recent: pd.DataFrame, tolerance: float
) -> bool:
    """检测前复权价是否整体漂移（即发生了除权除息）。

    前复权下除权会让**全部历史**价格变化，无脑增量追加会得到一段错价。
    做法：拿刚拉到的最近若干根，和库里同日期的收盘价比，超过容忍度就判定发生除权，
    调用方据此对该股全量重拉。这让缓存自愈。
    """
    if recent is None or recent.empty:
        return False

    dates = recent["date"].tolist()
    placeholders = ",".join("?" * len(dates))
    stored = dict(
        conn.execute(
            f"SELECT date, close FROM kline WHERE code = ? AND date IN ({placeholders})",
            (code, *dates),
        )
    )
    if not stored:
        return False

    for date, close in zip(recent["date"], recent["close"]):
        old = stored.get(date)
        if old is None or not old:
            continue
        if abs(close - old) / old > tolerance:
            return True
    return False


def upsert_stock_info(
    conn: sqlite3.Connection, df: pd.DataFrame, update_time: bool = True
) -> None:
    """df 需含 code / name 列，price、float_mv、amount 可选。update_time=False 时不更新时间戳。"""
    if df is None or df.empty:
        return
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") if update_time else None
    out = df.copy()
    for col in ("price", "float_mv", "amount"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if update_time:
        rows = [
            (code, name, float(price), float(float_mv), float(amount), stamp)
            for code, name, price, float_mv, amount in out[
                ["code", "name", "price", "float_mv", "amount"]
            ].itertuples(index=False, name=None)
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO stock_info (code, name, price, float_mv, amount, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
    else:
        rows = [
            (code, name, float(price), float(float_mv), float(amount))
            for code, name, price, float_mv, amount in out[
                ["code", "name", "price", "float_mv", "amount"]
            ].itertuples(index=False, name=None)
        ]
        conn.executemany(
            "INSERT INTO stock_info (code, name, price, float_mv, amount) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(code) DO UPDATE SET name=excluded.name, price=excluded.price, float_mv=excluded.float_mv, amount=excluded.amount",
            rows,
        )
    conn.commit()


def get_stock_names(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT code, name FROM stock_info"))


def all_codes(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT code FROM kline ORDER BY code")]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# 存储后端抽象：sqlite / csv / json
# ---------------------------------------------------------------------------

STORAGE_FORMATS = ("sqlite", "csv", "json")
_INFO_COLUMNS = ["code", "name", "price", "float_mv", "amount", "updated_at"]


class Store:
    """统一存储接口，各后端语义与模块级 sqlite 函数一致。"""

    def init(self) -> None: ...

    def upsert_kline(self, code: str, df: pd.DataFrame) -> int: ...

    def kline_path(self, code: str) -> Path: ...

    def delete_code(self, code: str) -> None: ...

    def last_date(self, code: str) -> str | None: ...

    def bar_counts(self) -> dict[str, int]: ...

    def read_kline(self, code: str, limit_bars: int | None = None) -> pd.DataFrame: ...

    def iter_klines(
        self, codes: Iterable[str], limit_bars: int | None = None
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        for code in codes:
            yield code, self.read_kline(code, limit_bars)

    def detect_adjust_drift(
        self, code: str, recent: pd.DataFrame, tolerance: float
    ) -> bool: ...

    def upsert_stock_info(self, df: pd.DataFrame, update_time: bool = True) -> None: ...

    def update_stock_time(self, code: str) -> None: ...

    def get_stock_names(self) -> dict[str, str]: ...

    def get_stock_info(self) -> pd.DataFrame: ...

    def all_codes(self) -> list[str]: ...

    def set_meta(self, key: str, value: str) -> None: ...

    def get_meta(self, key: str) -> str | None: ...

    def date_range(self) -> tuple[str | None, str | None]: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SqliteStore(Store):
    """sqlite 后端：单文件数据库，写入性能最好。"""

    def __init__(self, db_path: str | Path, readonly: bool = False) -> None:
        self.db_path = Path(db_path)
        self.conn = connect(self.db_path, readonly=readonly)

    def init(self) -> None:
        init_db(self.conn)

    def upsert_kline(self, code: str, df: pd.DataFrame) -> int:
        return upsert_kline(self.conn, code, df)

    def kline_path(self, code: str) -> Path:
        return self.db_path

    def delete_code(self, code: str) -> None:
        delete_code(self.conn, code)

    def last_date(self, code: str) -> str | None:
        return last_date(self.conn, code)

    def bar_counts(self) -> dict[str, int]:
        return bar_counts(self.conn)

    def read_kline(self, code: str, limit_bars: int | None = None) -> pd.DataFrame:
        return read_kline(self.conn, code, limit_bars)

    def detect_adjust_drift(
        self, code: str, recent: pd.DataFrame, tolerance: float
    ) -> bool:
        return detect_adjust_drift(self.conn, code, recent, tolerance)

    def upsert_stock_info(self, df: pd.DataFrame, update_time: bool = True) -> None:
        upsert_stock_info(self.conn, df, update_time=update_time)

    def update_stock_time(self, code: str) -> None:
        stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute(
            "UPDATE stock_info SET updated_at = ? WHERE code = ?",
            (stamp, code),
        )

    def get_stock_names(self) -> dict[str, str]:
        return get_stock_names(self.conn)

    def get_stock_info(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT code, name, price, float_mv, amount FROM stock_info",
            self.conn,
            dtype={"code": str},
        )

    def all_codes(self) -> list[str]:
        return all_codes(self.conn)

    def set_meta(self, key: str, value: str) -> None:
        set_meta(self.conn, key, value)

    def get_meta(self, key: str) -> str | None:
        return get_meta(self.conn, key)

    def date_range(self) -> tuple[str | None, str | None]:
        row = self.conn.execute("SELECT MIN(date), MAX(date) FROM kline").fetchone()
        return (row[0], row[1]) if row else (None, None)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class FileStore(Store):
    """文件型后端基类：一股一个K线文件 + 共享的 stock_info / meta 文件。

    写入策略是读旧数据合并后整体重写，幂等且无需锁；扫描端只读文件，
    天然支持多进程并发读。
    """

    kind = ""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.kline_dir = self.root / "kline"

    def init(self) -> None:
        self.kline_dir.mkdir(parents=True, exist_ok=True)

    # ---- 序列化由子类实现 ----
    def _kline_suffix(self) -> str: ...

    def _read_kline(self, path: Path) -> pd.DataFrame: ...

    def _write_kline(self, path: Path, df: pd.DataFrame) -> None: ...

    def _read_info(self) -> pd.DataFrame: ...

    def _write_info(self, df: pd.DataFrame) -> None: ...

    def _read_meta(self) -> dict: ...

    def _write_meta(self, meta: dict) -> None: ...

    # ---- 通用逻辑 ----
    def _kline_path(self, code: str) -> Path:
        return self.kline_dir / f"{code}.{self._kline_suffix()}"

    def kline_path(self, code: str) -> Path:
        return self._kline_path(code)

    def _exists(self, code: str) -> bool:
        return self._kline_path(code).exists()

    def upsert_kline(self, code: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        new = df[KLINE_COLUMNS]
        path = self._kline_path(code)
        if path.exists():
            old = self._read_kline(path)
            merged = (
                pd.concat([old, new], ignore_index=True)
                .drop_duplicates(subset="date", keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            merged = new.sort_values("date").reset_index(drop=True)
        self._write_kline(path, merged)
        return len(new)

    def delete_code(self, code: str) -> None:
        path = self._kline_path(code)
        if path.exists():
            path.unlink()

    def last_date(self, code: str) -> str | None:
        if not self._exists(code):
            return None
        df = self._read_kline(self._kline_path(code))
        return str(df["date"].iloc[-1]) if not df.empty else None

    def bar_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in sorted(self.kline_dir.glob(f"*.{self._kline_suffix()}")):
            counts[path.stem] = self._count_rows(path)
        return counts

    def _count_rows(self, path: Path) -> int:
        return len(self._read_kline(path))

    def read_kline(self, code: str, limit_bars: int | None = None) -> pd.DataFrame:
        path = self._kline_path(code)
        if not path.exists():
            return pd.DataFrame(columns=KLINE_COLUMNS[1:]).rename_axis("date")
        df = self._read_kline(path)
        if df.empty:
            return df
        if limit_bars:
            df = df.iloc[-int(limit_bars):].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    def detect_adjust_drift(
        self, code: str, recent: pd.DataFrame, tolerance: float
    ) -> bool:
        """与 sqlite 版同语义：重叠日收盘价偏差超容忍度即判定除权漂移。"""
        if recent is None or recent.empty or not self._exists(code):
            return False
        stored_df = self._read_kline(self._kline_path(code))
        if stored_df.empty:
            return False
        stored = dict(
            zip(stored_df["date"].astype(str), pd.to_numeric(stored_df["close"], errors="coerce"))
        )
        for date, close in zip(recent["date"], recent["close"]):
            old = stored.get(str(date))
            if old is None or not old or pd.isna(old):
                continue
            if abs(close - old) / old > tolerance:
                return True
        return False

    def upsert_stock_info(self, df: pd.DataFrame, update_time: bool = True) -> None:
        if df is None or df.empty:
            return
        stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") if update_time else None
        out = df.copy()
        out["code"] = out["code"].astype(str)
        for col in ("price", "float_mv", "amount"):
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        if "name" not in out.columns:
            out["name"] = ""

        if update_time:
            out["updated_at"] = stamp
        else:
            out["updated_at"] = None

        out = out[_INFO_COLUMNS]

        old = self._read_info()
        if not old.empty:
            merged = (
                pd.concat([old, out], ignore_index=True)
                .drop_duplicates(subset="code", keep="last")
                .reset_index(drop=True)
            )
            if not update_time:
                for idx, row in out.iterrows():
                    code = row["code"]
                    old_row = old[old["code"] == code]
                    if not old_row.empty:
                        merged.loc[merged["code"] == code, "updated_at"] = (
                            old_row.iloc[0]["updated_at"]
                        )
        else:
            merged = out
        self._write_info(merged)

    def update_stock_time(self, code: str) -> None:
        stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        info = self._read_info()
        if info.empty:
            return
        # 确保 updated_at 列是 object 类型，避免类型不兼容警告
        if "updated_at" in info.columns and info["updated_at"].dtype != object:
            info["updated_at"] = info["updated_at"].astype(object)
        info.loc[info["code"].astype(str) == str(code), "updated_at"] = stamp
        self._write_info(info)

    def get_stock_names(self) -> dict[str, str]:
        info = self._read_info()
        if info.empty:
            return {}
        return dict(zip(info["code"].astype(str), info["name"].astype(str)))

    def get_stock_info(self) -> pd.DataFrame:
        info = self._read_info()
        if info.empty:
            return pd.DataFrame(columns=["code", "name", "price", "float_mv", "amount"])
        return info[["code", "name", "price", "float_mv", "amount"]].copy()

    def all_codes(self) -> list[str]:
        return sorted(p.stem for p in self.kline_dir.glob(f"*.{self._kline_suffix()}"))

    def set_meta(self, key: str, value: str) -> None:
        meta = self._read_meta()
        meta[key] = value
        self._write_meta(meta)

    def get_meta(self, key: str) -> str | None:
        return self._read_meta().get(key)

    def date_range(self) -> tuple[str | None, str | None]:
        """K线文件按日期升序存，取每个文件首尾行即可，不用全量解析。"""
        lo: str | None = None
        hi: str | None = None
        for path in self.kline_dir.glob(f"*.{self._kline_suffix()}"):
            first, last = self._edge_dates(path)
            if first is None:
                continue
            if lo is None or first < lo:
                lo = first
            if hi is None or last > hi:
                hi = last
        return lo, hi

    def _edge_dates(self, path: Path) -> tuple[str | None, str | None]:
        df = self._read_kline(path)
        if df.empty:
            return None, None
        return str(df["date"].iloc[0]), str(df["date"].iloc[-1])

    def commit(self) -> None:
        pass  # 每次写入即落盘，无事务概念

    def close(self) -> None:
        pass


class CsvStore(FileStore):
    """csv 后端：可读性最好，方便 Excel / 外部工具直接消费。"""

    kind = "csv"

    def _kline_suffix(self) -> str:
        return "csv"

    def _read_kline(self, path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, dtype={"date": str})
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return pd.DataFrame(columns=KLINE_COLUMNS)
        if df.empty:
            return df
        df["date"] = df["date"].astype(str)
        # 外部工具/旧版本写出的文件可能缺列（如只有 OHLCV），补 NaN 与 json 后端对齐。
        # read_csv 已把正常数值列推断成数值型，只有真正落成 object 的列才需要 coerce：
        # 原来无差别对 8 列各跑一次 pd.to_numeric，每只股票白建 8 个 Series。
        # 隔离实测省 ~1.2ms/只（1.52→0.29ms），全市场串行约 6.6s；输出完全等价。
        for col in KLINE_COLUMNS[1:]:
            if col not in df.columns:
                df[col] = float("nan")
            elif not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[KLINE_COLUMNS]

    def _write_kline(self, path: Path, df: pd.DataFrame) -> None:
        df.to_csv(path, index=False, encoding="utf-8")

    def _count_rows(self, path: Path) -> int:
        # 快路径：数行即可，不解析内容（一行一根K线）
        try:
            with open(path, "rb") as fh:
                return max(sum(1 for _ in fh) - 1, 0)
        except OSError:
            return 0

    def _edge_dates(self, path: Path) -> tuple[str | None, str | None]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                header = fh.readline()
                first = fh.readline()
                if not first:
                    return None, None
                last = first
                for line in fh:
                    if line.strip():
                        last = line
        except OSError:
            return None, None
        first_date = first.split(",", 1)[0].strip()
        last_date = last.split(",", 1)[0].strip()
        return first_date or None, last_date or None

    _info_path_name = "stock_info.csv"
    _meta_path_name = "meta.json"

    def _read_info(self) -> pd.DataFrame:
        path = self.root / self._info_path_name
        if not path.exists():
            return pd.DataFrame(columns=_INFO_COLUMNS)
        try:
            return pd.read_csv(path, dtype={"code": str})
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=_INFO_COLUMNS)

    def _write_info(self, df: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.root / self._info_path_name, index=False, encoding="utf-8")

    def _read_meta(self) -> dict:
        # meta 两种文件后端统一用 json 存，避免 csv 转义麻烦
        path = self.root / self._meta_path_name
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_meta(self, meta: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self._meta_path_name).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class JsonStore(FileStore):
    """json 后端：与 csv 同布局，只是换成 json 序列化。"""

    kind = "json"

    def _kline_suffix(self) -> str:
        return "json"

    def _read_kline(self, path: Path) -> pd.DataFrame:
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return pd.DataFrame(columns=KLINE_COLUMNS)
        if not records:
            return pd.DataFrame(columns=KLINE_COLUMNS)
        df = pd.DataFrame(records)
        df["date"] = df["date"].astype(str)
        for col in KLINE_COLUMNS[1:]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = float("nan")
        return df[KLINE_COLUMNS]

    def _write_kline(self, path: Path, df: pd.DataFrame) -> None:
        path.write_text(
            df.to_json(orient="records", force_ascii=False), encoding="utf-8"
        )

    def _read_info(self) -> pd.DataFrame:
        path = self.root / "stock_info.json"
        if not path.exists():
            return pd.DataFrame(columns=_INFO_COLUMNS)
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return pd.DataFrame(columns=_INFO_COLUMNS)
        if not records:
            return pd.DataFrame(columns=_INFO_COLUMNS)
        df = pd.DataFrame(records)
        df["code"] = df["code"].astype(str)
        return df

    def _write_info(self, df: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "stock_info.json").write_text(
            df.to_json(orient="records", force_ascii=False), encoding="utf-8"
        )

    def _read_meta(self) -> dict:
        path = self.root / "meta.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_meta(self, meta: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def stock_codes(cfg) -> list[str]:
    """本地库里的股票代码（不含指数 idx_xxx）。只读、不联网。

    csv 后端是一次 glob，sqlite 是一次 DISTINCT 查询，都很快。用作"全市场代码清单"
    的兜底：东财列表接口挂掉时，至少还能拿本地库去批量取实时报价。
    """
    st = open_store(cfg, readonly=True)
    try:
        return sorted(c for c in st.all_codes() if not c.startswith("idx_"))
    finally:
        st.close()


def market_dir(cfg) -> Path:
    """市场环境辅助文件目录（daily_summary.csv / 行业热度历史 / 涨停跟踪）。

    从 data_dir 反推而不是用相对路径：data_dir 在 config.load_config 里已解析成
    绝对路径，写死 Path("data/store") 会跟着 CWD 跑偏。
    """
    d = Path(cfg.data.data_dir).parent / "market"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_tail_rows_csv(path: Path, n: int) -> list[dict[str, str]]:
    """只读一个 K 线 CSV 的最后 n 行，返回 [{列名: 原文字符串}]。

    从文件尾按块倒着读，不加载整文件、不用 pandas（read_kline 会先全量解析再切片，
    全市场几千只做回填时太慢）。兼容 CRLF：逐行 strip 掉 \\r。
    """
    if n <= 0:
        return []
    block = 64 * 1024
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            if size == 0:
                return []
            # 表头单独读，不参与倒读
            fh.seek(0)
            header_line = fh.readline()
            header = [c.strip() for c in header_line.decode("utf-8").rstrip("\r\n").split(",")]
            if not header or header == [""]:
                return []
            header_end = len(header_line)
            if header_end >= size:
                return []

            pos = size
            tail = b""
            while pos > header_end:
                read_size = min(block, pos - header_end)
                pos -= read_size
                fh.seek(pos)
                chunk = fh.read(read_size)
                tail = chunk + tail
                lines = tail.split(b"\n")
                # 第一段可能是被截断的行，块还没读到文件头就丢弃它
                if pos > header_end:
                    lines = lines[1:]
                complete = [ln for ln in lines if ln.strip()]
                if len(complete) >= n:
                    break
            else:
                lines = tail.split(b"\n")
                complete = [ln for ln in lines if ln.strip()]

        rows: list[dict[str, str]] = []
        for raw in complete[-n:]:
            text = raw.decode("utf-8", "ignore").strip()
            if not text:
                continue
            values = text.split(",")
            rows.append(dict(zip(header, values)))
        return rows
    except OSError:
        return []


def storage_path(cfg) -> Path:
    """当前后端的存储位置：sqlite 是库文件，csv / json 是数据目录。"""
    kind = cfg.data.storage
    if kind == "sqlite":
        return Path(cfg.data.db_path)
    return Path(cfg.data.data_dir) / kind


def storage_exists(cfg) -> bool:
    """本地库是否已建过（用于 info 命令判断要不要先 update --full）。"""
    path = storage_path(cfg)
    return path.exists()


def open_store(cfg, readonly: bool = False) -> Store:
    """按 cfg.data.storage 打开对应后端。写模式顺带建好表/目录。"""
    kind = cfg.data.storage
    if kind == "sqlite":
        st: Store = SqliteStore(cfg.data.db_path, readonly=readonly)
    elif kind == "csv":
        st = CsvStore(Path(cfg.data.data_dir) / "csv")
    elif kind == "json":
        st = JsonStore(Path(cfg.data.data_dir) / "json")
    else:
        raise ValueError(f"未知存储格式 '{kind}'，可选: {', '.join(STORAGE_FORMATS)}")
    if not readonly:
        st.init()
    return st
