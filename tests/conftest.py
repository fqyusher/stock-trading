from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tradeyh import config, scan  # noqa: E402


@pytest.fixture
def cfg():
    """项目默认配置。测试里可以直接改字段来验证某个阈值真的在起作用。"""
    return config.load_config(ROOT / "config.yaml")


@pytest.fixture
def make_ctx(cfg):
    def _make(df, code="000001", name="测试", explain=True):
        return scan.build_context(code, name, df, cfg, explain=explain)

    return _make


@pytest.fixture
def http_stub(monkeypatch):
    """把 fetch 的网络出口换成可编程桩。

    默认测试不联网（`pytest.ini` 排除 network 标记），但指数报价 / 全市场快照的
    解析逻辑必须被覆盖，所以这里提供注册式桩：

        http_stub.add_json("ulist.np", {...})        # 按 URL 子串路由
        http_stub.add_text("qt.gtimg.cn", "v_...;")

    未注册的 URL 一律抛 `FetchError`（不是静默返回空），这样"全渠道失败"的回退
    逻辑也能被真实覆盖，用例不会假通过。同时把重试等待与随机 sleep 打平，保持
    测试快速稳定。
    """
    from tradeyh import fetch

    class Stub:
        def __init__(self):
            self.json_routes: list[tuple[str, object]] = []
            self.text_routes: list[tuple[str, object]] = []
            self.json_calls: list[tuple[str, dict]] = []
            self.text_calls: list[tuple[str, dict]] = []

        def add_json(self, url_substr: str, payload) -> None:
            self.json_routes.append((url_substr, payload))

        def add_text(self, url_substr: str, text) -> None:
            self.text_routes.append((url_substr, text))

        @staticmethod
        def _match(routes, url):
            for sub, payload in routes:
                if sub in url:
                    return payload() if callable(payload) else payload
            raise fetch.FetchError(f"http_stub 未注册该 URL: {url}")

        def get_json(self, url, params, timeout):
            self.json_calls.append((url, params))
            return self._match(self.json_routes, url)

        def get_text(self, url, **kwargs):
            self.text_calls.append((url, kwargs))
            return self._match(self.text_routes, url)

    stub = Stub()
    monkeypatch.setattr(fetch, "_get_json", stub.get_json)
    monkeypatch.setattr(fetch, "_http_get_text", stub.get_text)
    monkeypatch.setattr(fetch, "_with_retry", lambda fn, *a, **kw: fn(*a))
    monkeypatch.setattr(fetch, "_sleep_jitter", lambda: None)
    return stub


def hits_of(ctx, module):
    """跑单个检测器，失败时把 trace 一起报出来，方便定位是哪条规则挡住的。"""
    return module.detect(ctx)


def assert_hit(ctx, module, status=None):
    hits = module.detect(ctx)
    assert hits, "未命中，判定过程:\n  " + "\n  ".join(ctx.trace)
    hit = hits[0]
    if status is not None:
        assert hit.status == status, (
            f"状态应为 {status} 实际 {hit.status}\n  " + "\n  ".join(ctx.trace)
        )
    return hit


def assert_no_hit(ctx, module):
    hits = module.detect(ctx)
    assert not hits, f"不该命中，却得到 {[h.detail for h in hits]}"
