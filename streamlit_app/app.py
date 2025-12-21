import io
from datetime import datetime, date

import pandas as pd
import streamlit as st

from data_loader import load_all_batches, get_date_range


HIDDEN_COLUMNS = {
    "风控类型",
    "风险打分",
    "策略1阈值",
    "策略2阈值",
    "有效期（天）",
    "风险有效期起始时间",
    "风险有效期终止时间",
    "结论(黑白名单等)",
    "源数据类型",
    "用途",
    "可信度（数据源）",
}


# 导出到 Excel 时允许的业务列（按期望顺序）
EXPORT_COLUMNS = [
    "时间",
    "新闻时间",
    "标的所属交易所",
    "交易所所属国家",
    "标的类型1",
    "标的类型2",
    "标的类型3",
    "标的类型4",
    "标的名称",
    "标的代码",
    "标的分类(人工打标)",
    "风险分类1",
    "风险分类2",
    "风险类型",
    "风险分类(人工打标)",
    "影响方向",
    "可信度（大模型生成）",
    "名称/代码命中比(1/n)",
    "（标的分类+风险分类）与事件原因相似度",
    "事件原因与源新闻相似度",
    "风险事件或原因",
    "风险发生时间",
    "数据频率",
    "数据源类型1",
    "数据源类型2",
    "出处：源内容",
    "分析模型或方法",
    "过滤方法或模型",
    "备注",
    "可信度（人工打标）",
]


st.set_page_config(page_title="风险控制新闻分析可视化", layout="wide")


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_all_batches()


def _filter_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """根据侧边栏选择的日期与小时范围过滤数据。"""
    if df.empty:
        return df

    min_dt, max_dt = get_date_range(df)
    st.sidebar.markdown("### 时间范围筛选")
    start_date = st.sidebar.date_input("起始日期", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
    end_date = st.sidebar.date_input("结束日期", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
    if isinstance(start_date, list):
        start_date = start_date[0]
    if isinstance(end_date, list):
        end_date = end_date[0]

    start_hour = st.sidebar.slider("开始小时", 0, 23, 0)
    end_hour = st.sidebar.slider("结束小时", 0, 23, 23)

    if "__news_time_dt" in df.columns:
        dt = df["__news_time_dt"]
        mask = (
            (dt.dt.date >= start_date)
            & (dt.dt.date <= end_date)
            & (dt.dt.hour >= start_hour)
            & (dt.dt.hour <= end_hour)
        )
        return df[mask].copy(), start_date, end_date, start_hour, end_hour

    # 若没有精确时间，只按 __date 做粗略过滤
    if "__date" in df.columns:
        dts = pd.to_datetime(df["__date"], format="%Y%m%d", errors="coerce")
        mask = (
            (dts.dt.date >= start_date)
            & (dts.dt.date <= end_date)
        )
        return df[mask].copy(), start_date, end_date, start_hour, end_hour

    return df.copy(), start_date, end_date, start_hour, end_hour


def page_intro(df: pd.DataFrame) -> None:
    st.title("风险控制新闻分析可视化平台")

    st.markdown("""
本平台基于 `risk_control_news_project` 的分析结果构建：

- 后台由 `timeScheduler_main.py` 周期性运行完整流水线，将结果写入 `result/json/YYYYMMDD.json`；
- 每个 JSON 文件代表某一自然日的所有分析批次（`YYYYMMDD` 为新闻所属日期）；
- 本页面基于这些 JSON 文件进行可视化展示与交互检索。
""")

    st.markdown("""---""")
    st.subheader("字段说明（部分常用字段）")
    st.markdown("""
- **时间类**  
  - `时间`：结果生成时间。  
  - `新闻时间`：原新闻发布时间。  
  - `风险发生时间`：风险事件发生时间（如有）。

- **标的信息**  
  - `标的名称`、`标的代码`。  
  - `标的类型1~4`：不同层级的标的分类。  
  - `交易所`、`交易所所在国家`。

- **风险分类**  
  - `风险分类1`、`风险分类2`、`风险类型`。  
  - `风险分类(人工打标)`：人工或规则打标的风险分类。  
  - `标的分类(人工打标)`：人工或规则打标的标的分类。

- **相似度与得分**  
  - `名称/代码命中比(1/n)`。  
  - `（标的分类+风险分类）与事件原因相似度`。  
  - `事件原因与源新闻相似度`。  
  - `风险得分`、`大模型生成可信度(1-10)` 等。

- **说明类字段**  
  - `风险事件或原因`、`结论`、`用途`、`备注` 等。
""")

    st.markdown("""---""")
    st.subheader("标的代码规范与检索说明")
    st.markdown("""
在质检阶段（`core/dataRes_check/dataRes_check_main.py`）会对标的代码做统一规范，核心要点是：

- 将原始代码清洗为“**数字部分 + 交易所后缀**”的规范形式；
- 不同市场可能使用不同后缀约定，例如：
  - A股/北交所/B股：以 6 位数字为基础，根据前缀推断交易所与板块，再附加相应后缀（如 `.XSHG` / `.XSHE` / `.BJ` 等形式，具体以实际结果文件中的 `标的代码` 为准）；  
  - 港股：将数字部分左补零为 5 位，并统一附加 `.HK` 后缀（如 `00700.HK`）。

本可视化中，“标的代码检索”是直接基于 **最终结果中的 `标的代码` 列** 做匹配的：

- 推荐使用与结果表中一致的规范形式进行搜索，例如：`600019.XSHG`、`000001.XSHE`、`00700.HK` 等；
- 也支持输入代码片段进行包含匹配（如只输入 `600019` 或 `XSHG`），系统会在 `标的代码` 列中做不区分大小写的模糊检索。
""")

    st.markdown("""---""")
    st.subheader("使用说明")
    st.markdown("""
- 通过左侧栏选择不同页面：介绍 / 总览 / 总结。  
- 在总览页：
  - 使用左侧栏选择日期与小时范围；
  - 使用顶部“高级筛选”选择标的/风险维度、相似度阈值、备注有/无；
  - 表格中 NaN/缺失值将直接显示为空；
  - 左侧有下载按钮，可导出当前时间范围内所有分析记录为 Excel。  
- 在总结页：
  - 查看交易所所属国家、标的类型、风险分类等维度的整体分布；
  - 可按某一类别下钻查看更详细的分布结构。
""")


def _render_sidebar_latest_update(df: pd.DataFrame) -> None:
    """在侧边栏底部展示基于 result/json 的最新更新时间。"""

    st.sidebar.markdown("---")
    st.sidebar.markdown("**最新更新时间**")

    if df.empty or "__date" not in df.columns:
        st.sidebar.write("暂无数据")
        return

    try:
        latest_date = str(df["__date"].max())
        df_latest = df[df["__date"] == latest_date]
        ts_str = ""

        # 首选：使用结果表中的“时间”列（批次/生成时间）
        if "时间" in df_latest.columns:
            ts_parsed = pd.to_datetime(df_latest["时间"], errors="coerce")
            if ts_parsed.notna().any():
                ts = ts_parsed.max()
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        # 退回：使用解析好的新闻时间
        elif "__news_time_dt" in df_latest.columns:
            ts = df_latest["__news_time_dt"].max()
            if pd.notna(ts):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        st.sidebar.write(f"最新日期：{latest_date}")
        if ts_str:
            st.sidebar.write(f"最新时间：{ts_str}")
    except Exception:
        st.sidebar.write("暂无数据")


def _build_multiselect(df: pd.DataFrame, label: str, col_name: str):
    if col_name not in df.columns:
        return None
    series = df[col_name].fillna("(空)")
    vc = series.value_counts()
    options = [f"{v}（{c}）" for v, c in vc.items()]
    value_map = {f"{v}（{c}）": v for v, c in vc.items()}
    selected = st.multiselect(label, options)
    if not selected:
        return df
    selected_values = [value_map[s] for s in selected]
    return df[series.isin(selected_values)]


def page_overview(df: pd.DataFrame) -> None:
    if df.empty:
        st.title("总览：分析结果表格视图")
        st.warning("当前没有可用数据，请先运行调度任务生成 result/json 下的 JSON 文件。")
        return

    # 时间筛选（在侧边栏），返回时间过滤后的 DataFrame
    df_time, start_date, end_date, start_hour, end_hour = _filter_by_time(df)

    # 将元素筛选折叠框放在页面顶部，标题之上
    with st.expander("元素筛选", expanded=False):
        # 标的代码检索（支持部分匹配）
        code_query = st.text_input("标的代码检索（支持部分匹配，如 '600519' 或 '600519.SH'）")
        if code_query and "标的代码" in df_time.columns:
            q = code_query.strip()
            if q:
                series_code = df_time["标的代码"].fillna("").astype(str)
                df_time = df_time[series_code.str.contains(q, case=False, na=False)]

        # 元素筛选
        cols_top = st.columns(4)
        with cols_top[0]:
            tmp = _build_multiselect(df_time, "标的类型1", "标的类型1")
            if tmp is not None:
                df_time = tmp
        with cols_top[1]:
            tmp = _build_multiselect(df_time, "标的类型2", "标的类型2")
            if tmp is not None:
                df_time = tmp
        with cols_top[2]:
            tmp = _build_multiselect(df_time, "标的类型3", "标的类型3")
            if tmp is not None:
                df_time = tmp
        with cols_top[3]:
            tmp = _build_multiselect(df_time, "标的类型4", "标的类型4")
            if tmp is not None:
                df_time = tmp

        cols_mid = st.columns(3)
        with cols_mid[0]:
            tmp = _build_multiselect(df_time, "标的分类(人工打标)", "标的分类(人工打标)")
            if tmp is not None:
                df_time = tmp
        with cols_mid[1]:
            tmp = _build_multiselect(df_time, "风险分类1", "风险分类1")
            if tmp is not None:
                df_time = tmp
        with cols_mid[2]:
            tmp = _build_multiselect(df_time, "风险分类2", "风险分类2")
            if tmp is not None:
                df_time = tmp

        cols_mid2 = st.columns(2)
        with cols_mid2[0]:
            tmp = _build_multiselect(df_time, "风险类型", "风险类型")
            if tmp is not None:
                df_time = tmp
        with cols_mid2[1]:
            tmp = _build_multiselect(df_time, "风险分类(人工打标)", "风险分类(人工打标)")
            if tmp is not None:
                df_time = tmp

        # 相似度过滤
        sim_cols = [
            "（标的分类+风险分类）与事件原因相似度",
            "事件原因与源新闻相似度",
        ]
        sim_sliders = {}
        for col in sim_cols:
            if col in df_time.columns:
                vals = pd.to_numeric(df_time[col], errors="coerce")
                vals = vals.dropna()
                if not vals.empty:
                    min_v, max_v = float(vals.min()), float(vals.max())
                    sim_sliders[col] = st.slider(col, min_v, max_v, (min_v, max_v), step=0.01)
        for col, (lo, hi) in sim_sliders.items():
            vals = pd.to_numeric(df_time[col], errors="coerce")
            mask = (vals.isna()) | ((vals >= lo) & (vals <= hi))
            df_time = df_time[mask]

        # 备注有/无
        remark_col = "备注"
        if remark_col in df_time.columns:
            opt = st.radio("备注过滤", ["全部", "仅有备注", "仅无备注"], horizontal=True)
            remarks = df_time[remark_col].fillna("").astype(str).str.strip()
            if opt == "仅有备注":
                df_time = df_time[remarks != ""]
            elif opt == "仅无备注":
                df_time = df_time[remarks == ""]

    # 页面主标题
    st.title("总览：分析结果表格视图")

    # 下载按钮（基于当前视图导出）
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 导出当前结果")
    if not df_time.empty:
        # 若有编辑后的表格，优先使用编辑结果（包含勾选列）
        df_export = None
        edited = st.session_state.get("overview_table")
        if isinstance(edited, pd.DataFrame) and not edited.empty:
            df_export = edited.copy()
        else:
            # 否则基于当前视图导出（按新闻时间降序）
            if "__news_time_dt" in df_time.columns:
                df_export = df_time.sort_values("__news_time_dt", ascending=False)
            else:
                df_export = df_time.copy()

        # 导出前处理“可信度（人工打标）”列：仅对显式勾选/标记为“是”的行输出“是”，其余为空
        if "可信度（人工打标）" in df_export.columns:
            col = df_export["可信度（人工打标）"]
            # 先统一成布尔：仅当值为 True 或 字符串"是" 视为 True，其余一律 False
            if col.dtype == bool:
                col_bool = col
            else:
                col_bool = col.astype(str).str.strip().eq("是")

            df_export["可信度（人工打标）"] = col_bool.map(lambda v: "是" if bool(v) else "")

        # 仅保留业务需要的导出列，去掉 __* 技术列及其他多余字段
        cols_ordered = [c for c in EXPORT_COLUMNS if c in df_export.columns]
        if cols_ordered:
            df_export = df_export[cols_ordered]
        else:
            # 兜底：去掉 __ 前缀和隐藏列
            tech_cols = [c for c in df_export.columns if c.startswith("__")]
            drop_cols = tech_cols + [c for c in HIDDEN_COLUMNS if c in df_export.columns]
            if drop_cols:
                df_export = df_export.drop(columns=drop_cols)

        # 导出时将 NaN 替换为空字符串
        df_export_excel = df_export.fillna("")
        buf = io.BytesIO()
        df_export_excel.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        fname = f"final_risk_control_results_{start_date.strftime('%Y%m%d')}_{start_hour:02d}-{end_date.strftime('%Y%m%d')}_{end_hour:02d}.xlsx"
        st.sidebar.download_button(
            label="Download",
            data=buf,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # 展示表格（NaN 显示为空）
    if df_time.empty:
        st.info("在当前筛选条件下没有记录。")
        return

    df_display = df_time.copy()
    # 页面展示按新闻时间降序
    if "__news_time_dt" in df_display.columns:
        df_display = df_display.sort_values("__news_time_dt", ascending=False)
    df_display = df_display.fillna("")
    # 隐藏技术列（__ 前缀）
    tech_cols = [c for c in df_display.columns if c.startswith("__")]
    hidden_cols = [c for c in df_display.columns if c in HIDDEN_COLUMNS]
    show_cols = [c for c in df_display.columns if c not in tech_cols and c not in hidden_cols]

    # 确保存在布尔型的“可信度（人工打标）”列供勾选
    if "可信度（人工打标）" not in df_display.columns:
        df_display["可信度（人工打标）"] = False
    else:
        col = df_display["可信度（人工打标）"]
        # 将非空字符串视为已勾选
        if col.dtype != bool:
            df_display["可信度（人工打标）"] = col.astype(str).str.strip().eq("是")

    # 调整列顺序：把“可信度（人工打标）”放到最后一列
    base_cols = [c for c in show_cols if c != "可信度（人工打标）"]
    if "可信度（人工打标）" in df_display.columns:
        base_cols.append("可信度（人工打标）")

    edited = st.data_editor(
        df_display[base_cols],
        use_container_width=True,
        height=700,
        hide_index=True,
        key="overview_table",
        column_config={
            "可信度（人工打标）": st.column_config.CheckboxColumn(
                "可信度（人工打标）",
                help="勾选表示该分析条目在导出的 Excel 中记为 '是'",
                default=False,
            )
        },
    )


def _pie_series(df: pd.DataFrame, col: str, empty_label: str = "(空)"):
    if col not in df.columns:
        return None
    series = df[col].fillna(empty_label)
    vc = series.value_counts().reset_index()
    vc.columns = [col, "count"]
    vc["pct"] = vc["count"] / vc["count"].sum()
    return vc


def _render_summary_overview(df_time: pd.DataFrame) -> None:
    import plotly.express as px

    st.markdown("#### 顶层分布概览")

    cols1 = st.columns(2)
    with cols1[0]:
        vc_country = _pie_series(df_time, "交易所所属国家", empty_label="未知")
        if vc_country is not None and not vc_country.empty:
            fig = px.pie(vc_country, names="交易所所属国家", values="count", title="交易所所属国家分布")
            st.plotly_chart(fig, use_container_width=True)
            # 下钻入口：选择国家
            labels = [f"{row['交易所所属国家']}（{row['count']}）" for _, row in vc_country.iterrows()]
            if labels:
                sel = st.selectbox("选择国家查看交易所分布", labels, key="country_sel")
                country = sel.split("（", 1)[0] if sel else None
                if country and st.button("查看所选国家详情", key="btn_country_drill"):
                    st.session_state["summary_mode"] = "drilldown"
                    st.session_state["drill_type"] = "country"
                    st.session_state["drill_path"] = {"country": country}
                    st.rerun()
    with cols1[1]:
        vc_t1 = _pie_series(df_time, "标的类型1", empty_label="未知")
        if vc_t1 is not None and not vc_t1.empty:
            fig = px.pie(vc_t1, names="标的类型1", values="count", title="标的类型1 分布")
            st.plotly_chart(fig, use_container_width=True)
            labels = [f"{row['标的类型1']}（{row['count']}）" for _, row in vc_t1.iterrows()]
            if labels:
                sel = st.selectbox("选择标的类型1查看下一级", labels, key="t1_sel")
                t1 = sel.split("（", 1)[0] if sel else None
                if t1 and st.button("查看所选标的类型1详情", key="btn_t1_drill"):
                    st.session_state["summary_mode"] = "drilldown"
                    st.session_state["drill_type"] = "target_type"
                    st.session_state["drill_path"] = {"t1": t1}
                    st.rerun()

    cols2 = st.columns(2)
    with cols2[0]:
        # 第三个图：风险分类1
        vc_risk1 = _pie_series(df_time, "风险分类1", empty_label="未知")
        if vc_risk1 is not None and not vc_risk1.empty:
            fig = px.pie(vc_risk1, names="风险分类1", values="count", title="风险分类1 分布")
            st.plotly_chart(fig, use_container_width=True)
            labels = [f"{row['风险分类1']}（{row['count']}）" for _, row in vc_risk1.iterrows()]
            if labels:
                sel = st.selectbox("选择风险分类1查看下一级", labels, key="r1_sel")
                r1 = sel.split("（", 1)[0] if sel else None
                if r1 and st.button("查看所选风险分类1详情", key="btn_r1_drill"):
                    st.session_state["summary_mode"] = "drilldown"
                    st.session_state["drill_type"] = "risk_category"
                    st.session_state["drill_path"] = {"r1": r1}
                    st.rerun()
    with cols2[1]:
        # 第四个图：标的分类(人工打标)
        vc_target_manual = _pie_series(df_time, "标的分类(人工打标)", empty_label="无打标")
        if vc_target_manual is not None and not vc_target_manual.empty:
            fig = px.pie(vc_target_manual, names="标的分类(人工打标)", values="count", title="标的分类(人工打标) 分布")
            st.plotly_chart(fig, use_container_width=True)

    vc_risk_manual = _pie_series(df_time, "风险分类(人工打标)", empty_label="无打标")
    if vc_risk_manual is not None and not vc_risk_manual.empty:
        fig = px.pie(vc_risk_manual, names="风险分类(人工打标)", values="count", title="风险分类(人工打标) 分布")
        st.plotly_chart(fig, use_container_width=True)


def _render_summary_drilldown(df_time: pd.DataFrame) -> None:
    import plotly.express as px

    st.markdown("#### 下钻视图")

    if st.button("返回总结总览"):
        st.session_state["summary_mode"] = "overview"
        st.rerun()

    drill_type = st.session_state.get("drill_type")
    drill_path = st.session_state.get("drill_path", {}) or {}

    if drill_type == "country":
        country = drill_path.get("country")
        if not country:
            st.info("未指定国家。")
            return
        st.subheader(f"交易所所属国家：{country} 的交易所分布")
        series = df_time["交易所所属国家"].fillna("未知") if "交易所所属国家" in df_time.columns else None
        if series is None:
            st.info("当前数据中不存在 '交易所所在国家' 字段。")
            return
        df_sub = df_time[series == country]
        if df_sub.empty:
            st.info("在当前时间范围内该国家下没有记录。")
            return
        # 使用标的所属交易所列
        vc_exch = _pie_series(df_sub, "标的所属交易所", empty_label="未知")
        if vc_exch is None or vc_exch.empty:
            st.info("该国家下没有可用于绘图的交易所信息。")
            return
        fig = px.pie(vc_exch, names="标的所属交易所", values="count", title=f"{country} 内交易所分布")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vc_exch, use_container_width=True)

    elif drill_type == "target_type":
        t1 = drill_path.get("t1")
        t2 = drill_path.get("t2")
        series_t1 = df_time["标的类型1"].fillna("未知") if "标的类型1" in df_time.columns else None
        if series_t1 is None:
            st.info("当前数据中不存在 '标的类型1' 字段。")
            return
        if not t1:
            st.info("未指定标的类型1。")
            return
        df_lvl1 = df_time[series_t1 == t1]
        if df_lvl1.empty:
            st.info("在当前时间范围内该标的类型1下没有记录。")
            return

        if not t2:
            # 第二层：展示标的类型2 分布，并可进一步选择某个类型2 下钻到类型3
            st.subheader(f"标的类型1 = {t1} 下的 标的类型2 分布")
            vc_t2 = _pie_series(df_lvl1, "标的类型2", empty_label="未知")
            if vc_t2 is None or vc_t2.empty:
                st.info("该标的类型1 下没有可用的 '标的类型2' 信息。")
                return
            fig = px.pie(vc_t2, names="标的类型2", values="count", title=f"{t1} 内 标的类型2 分布")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(vc_t2, use_container_width=True)

            labels = [f"{row['标的类型2']}（{row['count']}）" for _, row in vc_t2.iterrows()]
            if labels:
                sel = st.selectbox("选择标的类型2查看标的类型3 分布", labels, key="t2_sel")
                t2_val = sel.split("（", 1)[0] if sel else None
                if t2_val and st.button("查看所选标的类型2详情", key="btn_t2_drill"):
                    st.session_state["drill_path"] = {"t1": t1, "t2": t2_val}
                    st.rerun()
        else:
            # 第三层：在 t1,t2 下展示 标的类型3 分布
            series_t2 = df_lvl1["标的类型2"].fillna("未知") if "标的类型2" in df_lvl1.columns else None
            if series_t2 is None:
                st.info("当前数据中不存在 '标的类型2' 字段。")
                return
            df_lvl2 = df_lvl1[series_t2 == t2]
            if df_lvl2.empty:
                st.info("在当前时间范围内该标的类型2下没有记录。")
                return
            st.subheader(f"标的类型1 = {t1}，标的类型2 = {t2} 下的 标的类型3 分布")
            vc_t3 = _pie_series(df_lvl2, "标的类型3", empty_label="未知")
            if vc_t3 is None or vc_t3.empty:
                st.info("该组合下没有可用的 '标的类型3' 信息。")
                return
            fig = px.pie(vc_t3, names="标的类型3", values="count", title=f"{t1} / {t2} 内 标的类型3 分布")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(vc_t3, use_container_width=True)

    elif drill_type == "risk_category":
        r1 = drill_path.get("r1")
        r2 = drill_path.get("r2")
        series_r1 = df_time["风险分类1"].fillna("未知") if "风险分类1" in df_time.columns else None
        if series_r1 is None:
            st.info("当前数据中不存在 '风险分类1' 字段。")
            return
        if not r1:
            st.info("未指定风险分类1。")
            return
        df_lvl1 = df_time[series_r1 == r1]
        if df_lvl1.empty:
            st.info("在当前时间范围内该风险分类1下没有记录。")
            return

        if not r2:
            # 第二层：展示 风险分类2 分布
            st.subheader(f"风险分类1 = {r1} 下的 风险分类2 分布")
            vc_r2 = _pie_series(df_lvl1, "风险分类2", empty_label="未知")
            if vc_r2 is None or vc_r2.empty:
                st.info("该风险分类1 下没有可用的 '风险分类2' 信息。")
                return
            fig = px.pie(vc_r2, names="风险分类2", values="count", title=f"{r1} 内 风险分类2 分布")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(vc_r2, use_container_width=True)

            labels = [f"{row['风险分类2']}（{row['count']}）" for _, row in vc_r2.iterrows()]
            if labels:
                sel = st.selectbox("选择风险分类2查看风险类型分布", labels, key="r2_sel")
                r2_val = sel.split("（", 1)[0] if sel else None
                if r2_val and st.button("查看所选风险分类2详情", key="btn_r2_drill"):
                    st.session_state["drill_path"] = {"r1": r1, "r2": r2_val}
                    st.rerun()
        else:
            # 第三层：在 r1,r2 下展示 风险类型 分布
            series_r2 = df_lvl1["风险分类2"].fillna("未知") if "风险分类2" in df_lvl1.columns else None
            if series_r2 is None:
                st.info("当前数据中不存在 '风险分类2' 字段。")
                return
            df_lvl2 = df_lvl1[series_r2 == r2]
            if df_lvl2.empty:
                st.info("在当前时间范围内该风险分类2下没有记录。")
                return
            st.subheader(f"风险分类1 = {r1}，风险分类2 = {r2} 下的 风险类型 分布")
            vc_rt = _pie_series(df_lvl2, "风险类型", empty_label="未知")
            if vc_rt is None or vc_rt.empty:
                st.info("该组合下没有可用的 '风险类型' 信息。")
                return
            fig = px.pie(vc_rt, names="风险类型", values="count", title=f"{r1} / {r2} 内 风险类型 分布")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(vc_rt, use_container_width=True)

    else:
        st.info("当前下钻类型未定义或不受支持。")


def page_summary(df: pd.DataFrame) -> None:
    st.title("总结：多维度分布概览")

    if df.empty:
        st.warning("当前没有可用数据，请先运行调度任务生成 result/json 下的 JSON 文件。")
        return

    df_time, _, _, _, _ = _filter_by_time(df)
    if df_time.empty:
        st.info("在当前时间范围内没有记录。")
        return

    mode = st.session_state.get("summary_mode", "overview")
    if mode == "drilldown":
        _render_summary_drilldown(df_time)
    else:
        st.session_state["summary_mode"] = "overview"
        _render_summary_overview(df_time)


def main():
    df = get_data()

    st.sidebar.title("导航")
    page = st.sidebar.radio("选择页面", ("介绍", "总览", "总结"))

    if page == "介绍":
        page_intro(df)
    elif page == "总览":
        page_overview(df)
    elif page == "总结":
        page_summary(df)

    # 侧边栏底部展示最新更新时间
    _render_sidebar_latest_update(df)


if __name__ == "__main__":
    main()
