import os
import json
import glob
from datetime import datetime
from typing import Tuple

import pandas as pd


def _get_project_root() -> str:
    """返回 risk_control_news_project 根目录路径。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_all_batches() -> pd.DataFrame:
    """加载 result/json 目录下所有按日聚合的 JSON，拉平成一个 DataFrame。

    - 每行对应一条分析记录（与 Excel 行字段一致）；
    - 追加技术字段：
      - __date: 文件名中的 YYYYMMDD（新闻所属日期）
      - __batch_time_range, __batch_run_time, __is_trading_day
      - __crawler_start_date, __crawler_end_date, __crawler_start_hour, __crawler_end_hour
      - __news_time_dt: 解析后的“新闻时间” datetime（若存在该列）
    """
    project_root = _get_project_root()
    json_dir = os.path.join(project_root, "result", "json")

    if not os.path.isdir(json_dir):
        return pd.DataFrame()

    rows = []
    pattern = os.path.join(json_dir, "*.json")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        date_str, _ = os.path.splitext(fname)
        # 简单校验文件名是否形如 YYYYMMDD
        if not (len(date_str) == 8 and date_str.isdigit()):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                batches = json.load(f) or []
        except Exception:
            continue
        if not isinstance(batches, list):
            continue
        for b in batches:
            if not isinstance(b, dict):
                continue
            batch_time_range = b.get("batch_time_range")
            batch_run_time = b.get("batch_run_time")
            is_trading_day = b.get("is_trading_day")
            cw = b.get("crawler_window") or {}
            results = b.get("results") or []
            if not isinstance(results, list):
                continue
            for r in results:
                if not isinstance(r, dict):
                    continue
                row = dict(r)  # 复制一份，避免污染原对象
                row["__date"] = date_str
                row["__batch_time_range"] = batch_time_range
                row["__batch_run_time"] = batch_run_time
                row["__is_trading_day"] = bool(is_trading_day) if is_trading_day is not None else None
                row["__crawler_start_date"] = cw.get("start_date")
                row["__crawler_end_date"] = cw.get("end_date")
                row["__crawler_start_hour"] = cw.get("start_hour")
                row["__crawler_end_hour"] = cw.get("end_hour")
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 尝试解析“新闻时间”为 datetime，便于时间筛选
    news_time_col_candidates = ["新闻时间", "news_time"]
    news_time_col = None
    for col in news_time_col_candidates:
        if col in df.columns:
            news_time_col = col
            break
    if news_time_col is not None:
        df["__news_time_dt"] = pd.to_datetime(df[news_time_col], errors="coerce")
    else:
        # 若没有新闻时间列，可以退而求其次使用批次运行时间
        if "__batch_run_time" in df.columns:
            df["__news_time_dt"] = pd.to_datetime(df["__batch_run_time"], errors="coerce")

    return df


def get_date_range(df: pd.DataFrame) -> Tuple[datetime, datetime]:
    """从 DataFrame 推导全局最小/最大日期，用于时间选择默认值。"""
    if df.empty:
        today = datetime.today()
        return today, today

    # 优先使用 __news_time_dt 的日期范围
    if "__news_time_dt" in df.columns:
        s = df["__news_time_dt"].dropna()
        if not s.empty:
            return s.min().to_pydatetime(), s.max().to_pydatetime()

    # 回退到 __date 字段
    if "__date" in df.columns:
        try:
            dts = pd.to_datetime(df["__date"], format="%Y%m%d", errors="coerce").dropna()
            if not dts.empty:
                return dts.min().to_pydatetime(), dts.max().to_pydatetime()
        except Exception:
            pass

    today = datetime.today()
    return today, today
