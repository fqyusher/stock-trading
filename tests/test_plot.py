"""出图：报表里的每条命中都应产出对应的标注图。

守的是一个曾经静默失效的 bug：重扫缓存按 code 做键、但每次只跑一个检测器时，
同一只票的第二个形态开始会复用到第一个形态的命中列表，被判成"未复现"跳过，
出图数悄悄少于命中数。现在按 code 缓存的 `_rescan_code` 会把该票命中的检测器
一次性跑完，所以按 code 缓存重新变得安全——本测试守的就是"每条命中一张图"。
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest
import synth

from tradeyh import plot, scan, store
from tradeyh.patterns import channel as channel_mod
from tradeyh.patterns.base import PatternHit
from tradeyh.swings import Pivot

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # 无显示环境，必须在 pyplot 之前设定
pytest.importorskip("mplfinance")

# 同一只票命中 4 个形态（box_range / break_bottom_b2 / cup_handle / w_bottom），
# 正好覆盖"一个 code 多个 pattern"这条路径
CASES = {"000001": ("多形态股", synth.ideal_break_bottom_b2)}


@pytest.fixture
def seeded_cfg(cfg, tmp_path):
    """把合成K线写进临时 sqlite 库，并把 cfg 指向它。"""
    db_path = tmp_path / "kline.db"
    cfg.data["storage"] = "sqlite"
    cfg.data["db_path"] = str(db_path)
    cfg.data["out_dir"] = str(tmp_path / "out")
    cfg.universe["min_bars"] = 100  # 合成序列只有 100~190 根
    cfg.charts["max_charts"] = 0  # 显式不限数量，验证"每条命中都出图"

    conn = store.connect(db_path)
    store.init_db(conn)
    rows = []
    for code, (name, builder) in CASES.items():
        df = builder().reset_index().rename(columns={"index": "date"})
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        store.upsert_kline(conn, code, df[store.KLINE_COLUMNS])
        rows.append({"code": code, "name": name})
    store.upsert_stock_info(conn, pd.DataFrame(rows))
    conn.commit()
    conn.close()
    return cfg


def test_generate_charts_emits_one_image_per_hit(seeded_cfg, tmp_path):
    hits = scan.scan_market(seeded_cfg, workers=1)
    assert not hits.empty

    per_code = hits.groupby("代码")["形态key"].nunique()
    assert (per_code > 1).any(), "种子数据应包含同一只票的多次命中，否则守不住缓存键"

    base = tmp_path / "charts"
    paths = plot.generate_charts(seeded_cfg, hits, base_dir=base, stamp="20260918")

    assert len(paths) == len(hits), (
        f"命中 {len(hits)} 条却只出了 {len(paths)} 张图："
        f"{sorted(hits['形态key'])}"
    )
    day = base / "20260918"
    for path in paths:
        assert path.exists() and path.stat().st_size > 0
        assert path.parent.parent == day, "图必须写进 <根>/<stamp>/<形态key>/"
        assert path.parent.name in set(hits["形态key"])
    assert (day / "index.html").exists()


def test_generate_charts_groups_by_pattern(seeded_cfg, tmp_path):
    """日期之下再按形态key分一层：同类候选放同一目录，文件名不含名次。"""
    hits = scan.scan_market(seeded_cfg, workers=1)
    base = tmp_path / "charts"
    paths = plot.generate_charts(seeded_cfg, hits, base_dir=base, stamp="20260918")

    day = base / "20260918"
    # 每个命中的形态都要有自己的子目录，且目录里正好是对应的图
    by_pattern: dict[str, set[str]] = {}
    for path in paths:
        by_pattern.setdefault(path.parent.name, set()).add(path.name)
    assert set(by_pattern) == set(hits["形态key"])
    for pattern, names in by_pattern.items():
        assert (day / pattern).is_dir()
        assert names == {p.name for p in (day / pattern).glob("*.png")}
        for name in names:
            assert name.endswith(f"_{pattern}.png"), "文件名末尾应是形态key"
    # 日期目录本身只放索引页，图不再平铺在这一层
    assert not list(day.glob("*.png"))

    # 索引页链接必须带上形态子目录，否则点开是 404
    index = (day / "index.html").read_text(encoding="utf-8")
    for path in paths:
        assert f'"{path.parent.name}/{path.name}"' in index


def test_generate_charts_unlimited_by_default(seeded_cfg, tmp_path):
    """不配 charts.max_charts 时默认不限张数：所有命中都要出图。

    以前回落默认值是 60，命中几百条时后面的形态直接不出图。
    """
    seeded_cfg.charts.pop("max_charts", None)
    hits = scan.scan_market(seeded_cfg, workers=1)
    paths = plot.generate_charts(
        seeded_cfg, hits, base_dir=tmp_path / "charts", stamp="20260918"
    )
    assert len(paths) == len(hits)


def test_generate_charts_groups_by_date(seeded_cfg, tmp_path):
    """不同日期的图落在不同子目录，互不覆盖。"""
    hits = scan.scan_market(seeded_cfg, workers=1)
    base = tmp_path / "charts"

    first = plot.generate_charts(seeded_cfg, hits, base_dir=base, stamp="20260918")
    second = plot.generate_charts(seeded_cfg, hits, base_dir=base, stamp="20260919")

    assert {p.name for p in first} == {p.name for p in second}
    assert all(p.parent.parent.name == "20260918" for p in first)
    assert all(p.parent.parent.name == "20260919" for p in second)
    assert (base / "20260918" / "index.html").exists()
    assert (base / "20260919" / "index.html").exists()


def test_generate_charts_overwrites_same_day(seeded_cfg, tmp_path):
    """同一天重跑：同一只票的同一个形态覆盖同名文件，不做累加。

    文件名以前带榜单名次（`001_代码_形态_状态.png`），同一天重跑只要名次变了
    就是新文件、旧的还留着，图目录越堆越多。所以第二次特意用**倒序**的报表
    模拟名次变化。
    """
    hits = scan.scan_market(seeded_cfg, workers=1)
    base = tmp_path / "charts"
    day = base / "20260918"

    first = plot.generate_charts(seeded_cfg, hits, base_dir=base, stamp="20260918")
    before = sorted(p.relative_to(day).as_posix() for p in day.glob("*/*.png"))
    assert len(before) == len(first)

    reranked = hits.iloc[::-1].reset_index(drop=True)
    second = plot.generate_charts(seeded_cfg, reranked, base_dir=base, stamp="20260918")
    after = sorted(p.relative_to(day).as_posix() for p in day.glob("*/*.png"))

    assert after == before, "同日重跑（名次变化）不应新增文件"
    assert {p.name for p in second} == {p.name for p in first}


def test_generate_charts_filenames_are_stable_and_unique(seeded_cfg, tmp_path):
    """文件名只由 代码 + 形态key 决定，且互不覆盖。"""
    hits = scan.scan_market(seeded_cfg, workers=1)
    paths = plot.generate_charts(
        seeded_cfg, hits, base_dir=tmp_path / "charts", stamp="20260918"
    )

    names = sorted(p.name for p in paths)
    assert len(set(names)) == len(names), f"文件名应互不覆盖：{names}"
    for name, (_, row) in zip(names, hits.sort_values(["代码", "形态key"]).iterrows()):
        assert name == f"{row['代码']}_{row['形态key']}.png"


def test_charts_dir_defaults_to_out_dir_charts(cfg):
    """charts.dir 留空时，根目录是 out_dir/charts，日期 + 形态再各套一层。"""
    assert plot.charts_dir(cfg, "20260918") == Path(cfg.data.out_dir) / "charts" / "20260918"
    assert (
        plot.charts_pattern_dir(cfg, "break_bottom", "20260918")
        == Path(cfg.data.out_dir) / "charts" / "20260918" / "break_bottom"
    )
    cfg.charts["dir"] = "/tmp/custom_charts"
    assert plot.charts_dir(cfg, "20260918") == Path("/tmp/custom_charts/20260918")
    assert (
        plot.charts_pattern_dir(cfg, "break_bottom", "20260918")
        == Path("/tmp/custom_charts/20260918/break_bottom")
    )


def test_generate_charts_handles_empty_report(cfg, tmp_path):
    assert (
        plot.generate_charts(cfg, pd.DataFrame(), base_dir=tmp_path / "charts", stamp="20260918")
        == []
    )


def test_style_uses_chinese_up_down_colors():
    """A股习惯涨红跌绿，而 mplfinance 的 yahoo 预设是绿涨红跌，必须显式覆盖。"""
    import mplfinance as mpf

    mc = plot._mpf_style(mpf)["marketcolors"]
    assert mc["candle"]["up"] == plot._COLOR_UP
    assert mc["candle"]["down"] == plot._COLOR_DOWN
    # 影线和成交量柱要跟随同一套涨跌色，否则一根K线的红绿对不上
    assert mc["wick"]["up"] == plot._COLOR_UP
    assert mc["volume"]["down"] == plot._COLOR_DOWN


def test_cjk_font_is_available_or_warned():
    """中文标签能不能画出来：至少要有一种中文字体，否则整张图都是方框。"""
    fonts = plot._cjk_fonts()
    if not fonts:
        pytest.skip("本机没有中文字体，已由 _cjk_fonts 记 warning")
    assert all(isinstance(name, str) and name for name in fonts)


# --------------------------------------------------------------------------- #
# 右侧价位标签排版 / 趋势线外推
# --------------------------------------------------------------------------- #


def test_label_layout_keeps_min_gap():
    """挤在一起的价位要推开到至少一个间距，且顺序不能乱。

    以前 `_spread_labels` 有 max_shift=gap*3 的上限，密集簇里最后一个标签位移被
    截断，仍然叠在一起；顺手把"顺序"也守住——推歪一点可以，上下颠倒就读反了。
    """
    prices = [10.00, 10.05, 10.10, 10.15, 10.20]
    gap = 0.5
    lo, hi = 9.0, 12.0

    out = plot._layout_labels(prices, lo, hi, gap)

    assert len(out) == len(prices)
    ordered = sorted(out)
    assert all(b - a >= gap * 0.999 for a, b in zip(ordered, ordered[1:])), out
    assert all(lo <= y <= hi for y in out), out
    # 价格升序 → 显示位置升序，标签不会上下颠倒
    by_price = [out[i] for i in sorted(range(len(prices)), key=lambda i: prices[i])]
    assert by_price == sorted(by_price), by_price


def test_label_layout_falls_back_to_even_spacing_when_band_too_small():
    """区间根本容不下时按价格顺序等距铺开——宁可位置不准，也不能叠成一张。"""
    prices = [10.0] * 5
    out = plot._layout_labels(prices, 10.0, 11.0, gap=0.5)  # 需要 2.0 的区间，只有 1.0

    assert sorted(out) == pytest.approx([10.0, 10.25, 10.5, 10.75, 11.0])


def _fake_hit(df: pd.DataFrame, pivots: list[Pivot]) -> PatternHit:
    return PatternHit(
        code="000001",
        name="测试",
        pattern="w_bottom",
        pattern_cn="W底",
        status="forming",
        score=50.0,
        geom_score=50.0,
        vol_score=50.0,
        trend_score=50.0,
        signal_date=df.index[-1],
        pivots=list(pivots),
    )


def test_trend_line_extrapolation_is_capped(make_ctx):
    """趋势线只外推锚点跨度的一小截，不能一路拉到最后一根K线。

    两个挨得很近的锚点（这里相距 10 根）如果外推到 69 根之后的最后一根K线，
    斜率被放大成一条飞出去的陡线——300413 的压力线就是这么飞到 30+ 的（现价 20）。
    """
    df = synth.make_ohlcv(synth.ramp(10.0, 20.0, 200))
    ctx = make_ctx(df)
    hit = _fake_hit(
        df,
        [
            Pivot(idx=120, date=df.index[120], price=float(df["high"].iloc[120]), kind="H"),
            Pivot(idx=130, date=df.index[130], price=float(df["high"].iloc[130]), kind="H"),
            Pivot(idx=110, date=df.index[110], price=float(df["low"].iloc[110]), kind="L"),
            Pivot(idx=160, date=df.index[160], price=float(df["low"].iloc[160]), kind="L"),
        ],
    )

    segments = plot._trend_lines(ctx, hit, lo_idx=0)
    assert segments, "两个锚点就该连出一条线"

    last_idx = len(df) - 1
    anchors = {"压力线": (120, 130), "支撑线": (110, 160)}
    for points, _color, name in segments:
        i0, i1 = anchors[name]
        ext = min(last_idx - i1, max(5, int((i1 - i0) * plot._TREND_MAX_EXTEND)))
        end_idx = df.index.get_loc(points[-1][0])
        assert end_idx == i1 + ext, f"{name} 的外推长度不对"
        assert end_idx < last_idx, f"{name} 不该一路外推到最后一根K线"


# --------------------------------------------------------------------------- #
# 画出来的「价位线」必须真像个价格
# --------------------------------------------------------------------------- #

# 合成样本上实测：合法价位的取值落在窗口的 [0.90×, 1.29×]；量比这类指标键是 [0.04×, 0.3×]。
# 取 [0.5, 2.5] 两边都留足余量——不会误伤真价格，又能把量级差一档的指标键全兜住。
_LEVEL_LO_FACTOR = 0.5
_LEVEL_HI_FACTOR = 2.5


def _synth_ohlcv_samples() -> list[tuple[str, pd.DataFrame]]:
    """`synth` 里所有「不需要参数、返回 OHLCV DataFrame」的样本。"""
    out: list[tuple[str, pd.DataFrame]] = []
    for name, fn in vars(synth).items():
        if name.startswith("_") or not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):  # 内建/不可内省的 callable
            continue
        if any(
            p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in sig.parameters.values()
        ):
            continue
        try:
            df = fn()
        except Exception:  # noqa: BLE001 - 造不出来的样本直接跳过
            continue
        if hasattr(df, "columns") and {"open", "high", "low", "close", "volume"} <= set(df.columns):
            out.append((name, df))
    return out


def test_drawn_levels_are_plausible_prices(cfg):
    """画成价位线的值必须落在可见窗口价格的合理倍数内。

    守的是一类反复出现的 bug：检测器往 `key_levels` 里塞了**非价格**键（量比、比率、
    根数、计数），而 `collect_levels` 是黑名单制（默认画、指标键显式排除），漏登记就
    被当成价位画成横线。量比量级只有 0.5~3，画上去会把 y 轴从「十几块」一路拉到 0
    附近，K线只剩画面上方三分之一——`突破量比` / `收复量比` 曾让**每一个**
    break_bottom 命中都中招（配 `%` 结尾的 `破底深度%` 靠那条后缀规则幸免）。

    这条刻意**不查键名、只查数值像不像价格**，所以以后任何形态再加比率键都会被抓住，
    不必记得回来登记黑名单。
    """
    samples = _synth_ohlcv_samples()
    assert len(samples) > 20, f"合成样本太少（{len(samples)}），这条守不住东西"

    offenders: list[str] = []
    for name, df in samples:
        ctx = scan.build_context("000001", "测试", df, cfg)
        window = ctx.df.iloc[-180:]
        lo, hi = float(window["low"].min()), float(window["high"].max())
        for hit in scan.scan_context(ctx, scan.resolve_detectors(None)):
            is_channel = hit.pattern in channel_mod.PATTERN_CN
            for row in plot.collect_levels(hit, is_channel):
                if not (_LEVEL_LO_FACTOR * lo <= row["value"] <= _LEVEL_HI_FACTOR * hi):
                    offenders.append(
                        f"{name} {hit.pattern} 画出 {row['display']}={row['value']:.2f} "
                        f"（窗口[{lo:.2f}, {hi:.2f}]）"
                    )
    assert not offenders, "非价格键被当成价位线画出来了:\n  " + "\n  ".join(offenders)
