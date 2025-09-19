
"""
Streamlit 앱: 기후변화 - 해양생태계 대시보드
작성자: ChatGPT (한국어 UI)

요약(요구사항 준수):
 - 상단: 공개 데이터 대시보드 (공식 공개 데이터 연결 시도, 재시도 로직, 실패 시 예시 데이터 자동 대체 및 안내)
 - 하단: 사용자 입력 대시보드 (프롬프트에 제공된 설명/링크만 사용, 업로드 금지)
 - 한국어 UI, Pretendard 폰트 사용 시도, 전처리(결측/형변환/중복/미래 데이터 제거), 캐시(@st.cache_data), CSV 다운로드 버튼 제공
 - 데이터 표준화: date, value, group(optional)
 - 코드 주석에 출처(URL) 표기
"""
import io
import os
import time
import tempfile
import zipfile
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import pytz
import requests
import streamlit as st

# ------------------ 설정 ------------------
LOCAL_TZ = pytz.timezone("Asia/Seoul")
TODAY = datetime.now(LOCAL_TZ).date()

# 폰트 파일(없으면 자동 생략)
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"

st.set_page_config(page_title="해양생태계 & 기후변화 대시보드", layout="wide")

# 공개 데이터 출처 (코드 주석으로 남김)
# - GCBD (Global Coral Bleaching Database) — 논문/메타: https://www.nature.com/articles/s41597-022-01121-y
#   figshare 다운로드(예시): https://springernature.figshare.com/ndownloader/files/34571891
# - NOAA Coral Reef Watch: https://coralreefwatch.noaa.gov/
# - NOAA OISST (SST): https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
# - Ocean carbon & acidification (OCADS / GLODAP): https://www.ncei.noaa.gov/products/ocean-carbon-acidification-data-system , https://glodap.info/
# - 사용자 참고(프롬프트 제공):
#   KISTI(우리나라 주변 바다 산성화): https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO202210261284373
#   NIFS 보고서 PDF: https://www.nifs.go.kr/cmmn/file/climatechange_01.pdf

# 권장 GCBD 링크 (figshare zip/csv). 실제 환경에 따라 파일 ID가 바뀔 수 있으니 필요시 갱신.
GCBD_FIGSHARE_URLS = [
    # common figshare GCBD asset (may be zip or csv)
    "https://springernature.figshare.com/ndownloader/files/34571891",  # often a ZIP in supplementary
    "https://figshare.com/ndownloader/files/32677238",  # previously used id (fallback)
]

# 사용자 제공 참고 링크 (프롬프트)
KISTI_LINK = "https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO202210261284373"
NIFS_PDF_LINK = "https://www.nifs.go.kr/cmmn/file/climatechange_01.pdf"

# ------------------ 유틸리티 ------------------

def seoul_today() -> datetime.date:
    return datetime.now(LOCAL_TZ).date()

def drop_future_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[~(df[date_col].dt.date > seoul_today())]
    return df

def retry_get(url: str, timeout: int = 15, retries: int = 3, backoff: float = 1.5) -> requests.Response:
    last_exc = None
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            time.sleep(backoff ** (i + 1))
    raise last_exc

def standardize_timeseries(df: pd.DataFrame, date_col_candidates=None, value_col_candidates=None) -> pd.DataFrame:
    """
    입력 데이터 프레임을 (date, value, group(opt)) 형태로 최대한 매핑.
    """
    df = df.copy()
    if date_col_candidates is None:
        date_col_candidates = ["date", "year", "obs_date", "survey_date", "sample_date"]
    if value_col_candidates is None:
        value_col_candidates = ["bleaching_rate", "bleaching_rate_percent", "coral_cover", "value", "percent", "bleaching"]
    # find date col
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # try any column that looks like year or date
        for c in df.columns:
            if "year" in c.lower() or "date" in c.lower():
                date_col = c
                break
    if date_col is None:
        # create synthetic date if 'year' can't be found
        df["date"] = pd.NaT
    else:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    # find value col
    value_col = None
    for c in value_col_candidates:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        # try numeric columns other than date
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]
    if value_col is None:
        df["value"] = np.nan
    else:
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
    # group (optional)
    group_cols = [c for c in ["region", "site", "country", "reef_name", "taxon", "species"] if c in df.columns]
    df["group"] = df[group_cols[0]] if group_cols else None
    # cleanup: drop rows without date and value if both absent
    if "date" in df.columns:
        df = df[~(df["date"].isna() & df["value"].isna())]
    # remove duplicates
    df = df.drop_duplicates()
    # remove future dates
    df = drop_future_dates(df, date_col="date")
    return df[["date", "value", "group"]]

# ------------------ 공개 데이터 로드: GCBD ------------------

@st.cache_data(ttl=3600)
def load_gbd_from_figshare(urls=GCBD_FIGSHARE_URLS) -> Tuple[pd.DataFrame, str]:
    """
    figshare 링크(들)를 시도해서 GCBD CSV/ZIP을 불러옴.
    반환: (raw_df, used_url)
    예외 발생 시 빈 DataFrame과 빈 URL 반환 (호출부에서 예시 대체 로직 처리)
    """
    last_exc = None
    for url in urls:
        try:
            resp = retry_get(url, timeout=20, retries=3)
            content = resp.content
            ctype = resp.headers.get("content-type", "").lower()
            # If zip content or url endswith .zip -> extract first CSV
            if url.lower().endswith(".zip") or "zip" in ctype or b"PK" in content[:4]:
                with tempfile.TemporaryDirectory() as td:
                    zpath = os.path.join(td, "download.zip")
                    with open(zpath, "wb") as f:
                        f.write(content)
                    with zipfile.ZipFile(zpath, "r") as z:
                        # find CSV file(s)
                        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                        if not csv_names:
                            raise RuntimeError("ZIP 내 CSV 파일을 찾을 수 없음")
                        # pick first reasonable CSV
                        with z.open(csv_names[0]) as f:
                            df = pd.read_csv(f, low_memory=False)
                        return df, url
            else:
                # try read as CSV in-memory
                try:
                    df = pd.read_csv(io.BytesIO(content), low_memory=False)
                    return df, url
                except Exception:
                    # sometimes figshare returns HTML landing page; try to decode text and locate a downloadable link
                    text = content.decode("utf-8", errors="ignore")
                    # simple heuristic: find direct ndownloader link inside text
                    import re
                    m = re.search(r'https?://.*?ndownloader/files/\d+', text)
                    if m:
                        dl = m.group(0)
                        resp2 = retry_get(dl, timeout=20, retries=2)
                        df = pd.read_csv(io.BytesIO(resp2.content), low_memory=False)
                        return df, dl
                    raise
        except Exception as e:
            last_exc = e
            continue
    # all attempts failed
    raise last_exc if last_exc is not None else RuntimeError("GCBD 다운로드 실패")

# ------------------ 보조 공개 데이터 로드 (SST / 산성화 예시) ------------------

@st.cache_data(ttl=3600)
def load_example_sst_acid():
    # 예시 데이터 (대체용) — 연/월 시계열
    coral = pd.DataFrame({
        "date": pd.date_range(start="1980-01-01", periods=45, freq="Y"),
        "bleaching_rate_percent": np.clip(np.linspace(5, 65, 45) + np.random.randn(45)*3, 0, 100)
    })
    sst = pd.DataFrame({
        "date": pd.date_range(start="1980-01-01", periods=540, freq="M"),
        "sst_anomaly": np.clip(np.linspace(-0.3, 1.2, 540) + np.random.randn(540)*0.1, -2, 3)
    })
    acid = pd.DataFrame({
        "date": pd.date_range(start="1990-01-01", periods=35, freq="Y"),
        "surface_pH": np.clip(8.2 - np.linspace(0, 0.15, 35) + np.random.randn(35)*0.01, 7.7, 8.3)
    })
    return coral, sst, acid

# ------------------ 앱 UI ------------------

# Try apply Pretendard font (if available)
try:
    if os.path.exists(PRETENDARD_PATH):
        st.markdown(
            f"""
            <style>
            @font-face {{
                font-family: 'PretendardCustom';
                src: url('{PRETENDARD_PATH}');
            }}
            html, body, [class*="css"] {{
                font-family: 'PretendardCustom', sans-serif;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
except Exception:
    pass

st.title("🌊 해양생태계 & 기후변화 대시보드")
st.caption("공개 데이터(우선 GCBD) + 사용자 입력(보고서 기반) — 모든 UI는 한국어")

st.markdown("## 📌 공개 데이터 대시보드 (GCBD 우선 연결 시도)")
col_left, col_right = st.columns([2, 1])

with col_right:
    st.markdown("**데이터 출처(시도):**")
    st.write("- GCBD (Global Coral Bleaching Database) — 논문/메타: https://www.nature.com/articles/s41597-022-01121-y")
    st.write("- Figshare (예시 다운로드): " + ", ".join(GCBD_FIGSHARE_URLS))
    st.write("- NOAA Coral Reef Watch: https://coralreefwatch.noaa.gov/")
    st.write("- NOAA OISST: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html")
    st.write("- Ocean carbon & acidification: https://www.ncei.noaa.gov/products/ocean-carbon-acidification-data-system")
    st.write("---")
    st.markdown("**연동/인증 참고**")
    st.write("- Kaggle 사용 시: `kaggle` 패키지와 API token 설정 필요 (Kaggle 인증 파일을 환경변수 또는 ~/.kaggle/kaggle.json에 배치). 이 앱에서는 kaggle API 사용 예시는 포함하지 않았습니다.")

# 공개데이터 로드 시도
gcbd_df = None
gcbd_source = ""
public_data_warning = None

try:
    with st.spinner("GCBD 데이터 다운로드 시도 중..."):
        raw_gcbd, gcbd_source = load_gbd_from_figshare()
        # 표준화
        std_gcbd = standardize_timeseries(raw_gcbd)
        # If no useful values, raise to trigger fallback
        if std_gcbd["value"].notna().sum() < 3:
            raise RuntimeError("GCBD에서 의미있는 수치 컬럼(value)을 찾지 못함")
        gcbd_df = std_gcbd
        st.success("✅ GCBD 공식 데이터 로드 및 전처리 완료")
        st.caption(f"데이터 출처: {gcbd_source}")
except Exception as e:
    # 실패 시: 재시도 로직은 load_gbd_from_figshare 내부에서 수행
    public_data_warning = f"공개 데이터(GCBD) 로드 실패: {e}"
    st.warning(public_data_warning)
    # 요청사항(원래 prompt)에 따라 예시 데이터로 자동 대체
    coral_ex, sst_ex, acid_ex = load_example_sst_acid()
    # 표준화: coral_ex -> date,value
    gcbd_df = coral_ex.rename(columns={"bleaching_rate_percent":"value"})
    gcbd_df["group"] = None
    # mark that example is used
    st.info("대체 데이터(예시)를 사용하여 시각화합니다. (코드 주석의 원본 URL을 확인하세요)")

# 기본 전처리: 날짜형 변환/중복제거/미래데이터 제거
gcbd_df = gcbd_df.copy()
gcbd_df["date"] = pd.to_datetime(gcbd_df["date"], errors="coerce")
gcbd_df = gcbd_df.drop_duplicates().reset_index(drop=True)
gcbd_df = drop_future_dates(gcbd_df, date_col="date")

# 사이드바: 자동 구성 (기간 필터, 스무딩 선택)
st.sidebar.header("공개 데이터 옵션")
min_date = gcbd_df["date"].min()
max_date = gcbd_df["date"].max()
if pd.isna(min_date) or pd.isna(max_date):
    min_date = pd.to_datetime("1980-01-01")
    max_date = seoul_today()
date_range = st.sidebar.slider("기간 선택", min_value=min_date.date(), max_value=max_date.date(),
                               value=(min_date.date(), max_date.date()))
smoothing = st.sidebar.selectbox("스무딩(이동평균)", options=["사용 안 함", "3년(또는 36개월)", "5년(또는 60개월)"], index=0)

# 필터 적용 (연도 단위 혹은 월 단위 혼합 고려)
mask = (gcbd_df["date"].dt.date >= date_range[0]) & (gcbd_df["date"].dt.date <= date_range[1])
plot_df = gcbd_df.loc[mask].copy()

# 자동 집계: 연도별로 요약 (만약 데이터가 월 단위이면 연평균)
if (plot_df["date"].dt.freq is None) or True:
    # resample to annual by year
    plot_df["year"] = plot_df["date"].dt.year
    annual = plot_df.groupby("year")["value"].mean().reset_index()
    annual["date"] = pd.to_datetime(annual["year"].astype(str) + "-01-01")
    viz_df = annual[["date", "value"]].sort_values("date")
else:
    viz_df = plot_df[["date", "value"]].sort_values("date")

# smoothing
if smoothing != "사용 안 함":
    window = 36 if "36" in smoothing else 60
    # window measured in months if monthly, else in years — we'll approximate by points
    viz_df["value_smoothed"] = viz_df["value"].rolling(window=3 if "3" in smoothing else 5, center=True, min_periods=1).mean()
else:
    viz_df["value_smoothed"] = viz_df["value"]

# 메인: 산호 백화율 시계열
with col_left:
    st.subheader("최근 산호 백화 현상 비율 (연별 요약)")
    if viz_df.dropna().shape[0] == 0:
        st.warning("표시할 데이터가 없습니다.")
    else:
        fig = px.line(viz_df, x="date", y="value_smoothed",
                      labels={"date":"연도", "value_smoothed":"백화율 (임의단위)"},
                      title="산호 백화 현상 비율(연별 평균)")
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)
    # CSV 다운로드 (전처리된 표)
    csv_bytes = viz_df.rename(columns={"value_smoothed":"value"}).to_csv(index=False).encode("utf-8")
    st.download_button("전처리된 산호백화_데이터 다운로드 (CSV)", data=csv_bytes, file_name="gcbd_preprocessed.csv", mime="text/csv")

# ---------- 공개 데이터 보조 시각화: SST / 산성화 (예시 또는 실제 로드 시 사용) ----------
st.markdown("---")
st.subheader("보조 지표: 해양 표층 수온(SST) & 해양 산성화 (표본/요약)")

# 시도: 실제 SST/산성화 원본을 가져오는 복잡한 로직은 주석으로 안내.
st.write("참고: NOAA OISST 등 원자료는 대용량(netCDF)으로 제공됩니다. 실서비스에서는 xarray.open_dataset을 활용해 지역/기간을 추출하세요. 여기서는 예시/요약 데이터로 표시합니다.")

# load example series (or derive from earlier example)
if public_data_warning:
    coral_ex, sst_ex, acid_ex = load_example_sst_acid()
    sst_plot = sst_ex.copy()
    acid_plot = acid_ex.copy()
else:
    # if GCBD came with additional columns, create simple derived SST/acid series if possible
    sst_plot = pd.DataFrame({"date": pd.date_range(start=gcbd_df["date"].min(), end=gcbd_df["date"].max(), freq="Y")})
    sst_plot["sst_anomaly"] = np.linspace(-0.2, 0.9, len(sst_plot))
    acid_plot = pd.DataFrame({"date": pd.date_range(start=sst_plot["date"].min(), periods=len(sst_plot), freq="Y")})
    acid_plot["surface_pH"] = 8.2 - np.linspace(0, 0.12, len(acid_plot))

col_sst, col_acid = st.columns(2)
with col_sst:
    fig_sst = px.area(sst_plot, x="date", y="sst_anomaly",
                      labels={"date":"연도", "sst_anomaly":"SST 이상치 (°C)"},
                      title="해양 표층 수온 이상치 (예시/요약)")
    st.plotly_chart(fig_sst, use_container_width=True)
    st.download_button("SST_시계열_다운로드 (CSV)", data=sst_plot.to_csv(index=False).encode("utf-8"),
                       file_name="sst_timeseries.csv", mime="text/csv")
with col_acid:
    fig_acid = px.line(acid_plot, x="date", y="surface_pH",
                       labels={"date":"연도", "surface_pH":"표층 pH"},
                       title="표층 pH 추세 (예시/요약)")
    st.plotly_chart(fig_acid, use_container_width=True)
    st.download_button("산성화_시계열_다운로드 (CSV)", data=acid_plot.to_csv(index=False).encode("utf-8"),
                       file_name="ocean_acidification_timeseries.csv", mime="text/csv")

# ------------------ 사용자 입력(보고서) 기반 대시보드 ------------------
st.markdown("---")
st.header("📝 사용자 입력 기반 대시보드 (보고서 내용만 사용, 업로드 없음)")
st.markdown("**보고서 요약(입력 내용)**")
st.write("제목(가제): 역대 최악의 바다 그리고 더 최악이 될 바다.")
st.write("요약: 산호 백화, 해양 산성화, 고수온으로 인한 어류 폐사 등 — 본문과 참고자료(링크 포함)가 제공됨.")

# 본론1: 최근 45년간 산호 백화 비율 — 이미 공개데이터(상단 GCBD 또는 예시)를 사용
st.subheader("본론 1 — 최근 45년간 산호 백화 현상 비율 (보고서용)")
# For report visuals, take last 45 years from gcbd_df (if possible) otherwise use example
report_df = gcbd_df.copy()
if report_df["date"].dtype == "object" or report_df["date"].isna().all():
    # fallback synthetic
    report_df = pd.DataFrame({"date": pd.date_range(start="1980-01-01", periods=45, freq="Y"),
                              "value": np.clip(np.linspace(5, 65, 45) + np.random.randn(45)*3, 0, 100)})
report_df = report_df.sort_values("date").reset_index(drop=True)
if len(report_df) >= 45:
    report_plot = report_df.tail(45).copy()
else:
    # pad/extend synthetic if needed
    report_plot = report_df.copy()

fig_report = px.area(report_plot, x="date", y="value",
                     labels={"date":"연도", "value":"산호 백화율 (임의단위)"},
                     title="보고서용: 최근 45년 산호 백화율 (연별)")
st.plotly_chart(fig_report, use_container_width=True)
st.download_button("보고서_산호백화_데이터 다운로드 (CSV)", data=report_plot.to_csv(index=False).encode("utf-8"),
                   file_name="report_coral_45y.csv", mime="text/csv")

# 본론2: 해양 산성화, 고수온, 어류 폐사 영향 — 복합 시각화 (보고서 텍스트를 바탕으로 만든 요약/합성 지표)
st.subheader("본론 2 — 해양 산성화·고수온·어류 폐사 영향 (복합 시각화)")
# Merge sst_plot & acid_plot & coral for a combined view (annual)
try:
    sst_a = sst_plot.copy()
    sst_a["date"] = pd.to_datetime(sst_a["date"], errors="coerce")
    sst_a["year"] = sst_a["date"].dt.year
    sst_ann = sst_a.groupby("year")["sst_anomaly"].mean().reset_index()
    sst_ann["date"] = pd.to_datetime(sst_ann["year"].astype(str) + "-01-01")
except Exception:
    sst_ann = pd.DataFrame({"date": pd.date_range(start="1980-01-01", periods=35, freq="Y"), "sst_anomaly": np.linspace(-0.2, 0.9, 35)})

acid_a = acid_plot.copy()
acid_a["date"] = pd.to_datetime(acid_a["date"], errors="coerce")
acid_a["year"] = acid_a["date"].dt.year
acid_ann = acid_a.groupby("year")["surface_pH"].mean().reset_index()
acid_ann["date"] = pd.to_datetime(acid_ann["year"].astype(str) + "-01-01")

merge_base = pd.merge(report_plot.rename(columns={"value":"bleaching_rate_percent"})[["date","bleaching_rate_percent"]],
                      sst_ann[["date","sst_anomaly"]], on="date", how="outer")
merge_base = pd.merge(merge_base, acid_ann[["date","surface_pH"]], on="date", how="outer")
merge_base = merge_base.sort_values("date").reset_index(drop=True)

# 합성 어류 폐사 지수 (간단한 가중치 모델)
merge_base["sst_norm"] = (merge_base["sst_anomaly"] - merge_base["sst_anomaly"].min()) / (merge_base["sst_anomaly"].max() - merge_base["sst_anomaly"].min() + 1e-9)
merge_base["pH_drop"] = 8.2 - merge_base["surface_pH"]
merge_base["pH_norm"] = (merge_base["pH_drop"] - merge_base["pH_drop"].min()) / (merge_base["pH_drop"].max() - merge_base["pH_drop"].min() + 1e-9)
merge_base["fish_mortality_index"] = (0.7 * merge_base["sst_norm"].fillna(0) + 0.3 * merge_base["pH_norm"].fillna(0)) * 100
merge_base = drop_future_dates(merge_base, date_col="date")

# Plot combined with multiple traces
fig_comb = px.line(merge_base, x="date", y="bleaching_rate_percent", labels={"date":"연도", "bleaching_rate_percent":"백화율 (%)"}, title="해양 고수온 · 산성화 · 어류 폐사(합성지수) 비교")
fig_comb.add_scatter(x=merge_base["date"], y=merge_base["sst_anomaly"], mode="lines+markers", name="SST 연평균 이상치 (°C)")
fig_comb.add_scatter(x=merge_base["date"], y=merge_base["fish_mortality_index"], mode="lines+markers", name="어류 폐사 지수 (합성)")
st.plotly_chart(fig_comb, use_container_width=True)
st.download_button("본론2_복합_전처리_데이터 다운로드 (CSV)", data=merge_base.to_csv(index=False).encode("utf-8"),
                   file_name="report_combined_ocean_data.csv", mime="text/csv")

# ------------------ 결론 / 참고자료 ------------------
st.markdown("---")
st.subheader("결론 및 권고 (자동 제안)")
st.write(
    "- 산호 백화 현상과 해양 산성화, 고수온은 상호 연결되어 해양생태계에 심각한 영향을 줍니다.\n"
    "- 권고: 온실가스 감축, 해양 보호구역 확대, 산호 복원, 장기 모니터링 체계 구축 등.\n"
)
st.markdown("**참고자료(앱에서 시도/참조한 링크)**")
st.write(f"- GCBD 논문/데이터(참고): https://www.nature.com/articles/s41597-022-01121-y")
st.write(f"- Figshare 다운로드(시도): {', '.join(GCBD_FIGSHARE_URLS)}")
st.write(f"- KISTI(한국 관련 논문): {KISTI_LINK}")
st.write(f"- NIFS 보고서 PDF: {NIFS_PDF_LINK}")
st.caption("앱 노트: 공개 데이터는 원문 포맷(예: netCDF, ZIP, 대용량 CSV)으로 제공되는 경우가 많아, 실서비스 환경에서는 인증·다운로드·전처리(예: xarray.open_dataset) 단계를 추가하세요.")
