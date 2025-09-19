# streamlit_app.py
"""
Streamlit 앱: 기후변화 - 해양생태계 대시보드
작성자: ChatGPT (한국어 UI)
- 공개 데이터 우선 시도: NOAA Coral Reef Watch (ERDDAP / CSV), NOAA OISST (ERDDAP) 등
- 실패 시: 예시(합성/샘플) 데이터로 자동 대체 및 화면 안내
- 한국어 UI, Pretendard 시도, 전처리(결측/형변환/중복/미래데이터 제거), 캐시(@st.cache_data), CSV 다운로드 제공
- 데이터 표준화: date, value, group(optional)
주의: 실제 서비스 환경에서는 netCDF/대용량 자료 처리를 위해 xarray + dask 사용을 권장합니다.
출처(참고 URL - 코드 주석에 명시):
 - NOAA Coral Reef Watch (CRW) data resources / ERDDAP instructions:
   https://coralreefwatch.noaa.gov/  and https://coralreefwatch.noaa.gov/instructions/Accessing_Coral_Reef_Watch_Data_via_Data_Servers_at_CoastWatch_20240403.pdf
 - NOAA OISST (Optimum Interpolation SST) / ERDDAP access:
   https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html  and ERDDAP CSV endpoints (example): https://coastwatch.pfeg.noaa.gov/erddap/
 - Ocean Carbon & Acidification (OCADS) / GLODAP general pages:
   https://www.ncei.noaa.gov/products/ocean-carbon-acidification-data-system
   https://glodap.info/
"""

import io
import os
import time
import zipfile
import tempfile
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
    전처리: 결측 처리, 형변환, 중복 제거, 미래 데이터 제거
    """
    df = df.copy()
    if date_col_candidates is None:
        date_col_candidates = ["date", "time", "datetime", "year", "obs_date", "survey_date", "sample_date"]
    if value_col_candidates is None:
        value_col_candidates = ["bleaching_rate", "bleaching_rate_percent", "coral_cover", "value", "percent", "sst", "sst_anomaly", "surface_pH"]
    # find date col
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if "date" in c.lower() or "time" in c.lower() or "year" in c.lower():
                date_col = c
                break
    if date_col is None:
        df["date"] = pd.NaT
    else:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    # find value col
    value_col = None
    for c in value_col_candidates:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
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
    # drop rows without date & value both
    if "date" in df.columns:
        df = df[~(df["date"].isna() & df["value"].isna())]
    df = df.drop_duplicates().reset_index(drop=True)
    df = drop_future_dates(df, date_col="date")
    # ensure columns order
    cols = ["date", "value", "group"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[["date", "value", "group"]]

# ------------------ 공개 데이터 로드 시도: NOAA CRW (ERDDAP) 및 OISST (ERDDAP) ------------------

@st.cache_data(ttl=3600)
def load_crw_from_erddap(sample_point_lat: float = 0.0, sample_point_lon: float = 0.0) -> Tuple[pd.DataFrame, str]:
    """
    NOAA Coral Reef Watch (CRW) ERDDAP 접근을 시도하여 'Degree Heating Week' 또는 'bleaching' 관련 시계열을 CSV로 가져옵니다.
    구현은 ERDDAP의 /csv endpoint 사용을 시도합니다. (사용자 환경/네트워크/ERDDAP 구성에 따라 실패 가능)
    반환: (raw_df, used_url) 또는 예외 발생
    참고(사용자 읽기용): CRW ERDDAP 접근 방법: https://coralreefwatch.noaa.gov/instructions/Accessing_Coral_Reef_Watch_Data_via_Data_Servers_at_CoastWatch_20240403.pdf
    """
    # ERDDAP 서비스 예시 엔드포인트 (환경에 따라 변경 필요)
    # 아래는 ERDDAP의 시간-위치 기반 CSV 추출 예시 포맷입니다.
    # - 실제로는 관심 지점(위도/경도)이나 영역, 변수명을 정확히 알고 요청해야 합니다.
    # - 여기서는 'virtual station'식 단일 픽셀 시계열 요청을 시도하는 예시 URL을 구성합니다.
    # NOTE: 여러 ERDDAP 인스턴스가 존재하므로 아래 URL은 환경/시점에 따라 동작하지 않을 수 있습니다.
    # 예시 (CoastWatch/NOAA): CSV 형식으로 요청하면 CSV 응답을 얻을 수 있습니다.
    # (사용 실패 시 예외를 발생시키고 상위에서 대체 데이터 사용)
    erddap_examples = [
        # NOAA CRW data server (public): use the 'satellite' time series virtual station CSV example
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/noaa_oisst_v2_avhrr.csv?sst[1982-01-01T00:00:00Z:1:2024-12-31T00:00:00Z][({lat}):1:({lat})][({lon}):1:({lon})]".format(lat=sample_point_lat, lon=sample_point_lon),
        # Generic CRW ERDDAP endpoints may exist; attempt a common host (may 404)
        "https://coastwatch.pfeg.noaa.gov/erddap/tabledap/coralreefwatch.csv",
    ]
    last_exc = None
    for url in erddap_examples:
        try:
            resp = retry_get(url, timeout=25, retries=2)
            # some ERDDAP CSVs include header/units lines; read with pandas
            df = pd.read_csv(io.BytesIO(resp.content), low_memory=False)
            return df, url
        except Exception as e:
            last_exc = e
            continue
    raise last_exc if last_exc is not None else RuntimeError("CRW ERDDAP 접근 실패")

@st.cache_data(ttl=3600)
def load_oisst_timeseries_via_erddap(bbox=None, start="1982-01-01", end=None) -> Tuple[pd.DataFrame, str]:
    """
    NOAA OISST(예: monthly or daily) 데이터를 ERDDAP CSV로 시도해 원격에 접근.
    bbox: (min_lat, max_lat, min_lon, max_lon) 혹은 단일 픽셀 (lat, lon) 튜플
    반환: (df, url) 또는 예외 발생
    출처: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
    """
    if end is None:
        end = seoul_today().isoformat()
    # ERDDAP griddap 예시 (dataset id, 변수명 등은 서비스에 따라 달라짐)
    # 여기서는 'noaa_oisst_v2_avhrr' (예시) 의 .csv 요청 포맷을 사용
    try:
        if bbox and len(bbox) == 4:
            min_lat, max_lat, min_lon, max_lon = bbox
            # CSV request (일별/월별 시계열 추출) - 이 URL은 ERDDAP 인스턴스에 따라 달라질 수 있음.
            url = (
                f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/noaa_oisst_v2_avhrr.csv?"
                f"sst[{start}T00:00:00Z:1:{end}T00:00:00Z][({min_lat}):1:({max_lat})][({min_lon}):1:({max_lon})]"
            )
        else:
            # fallback: global timeseries aggregated example (may fail)
            url = f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/noaa_oisst_v2_avhrr.csv?sst[{start}T00:00:00Z:1:{end}T00:00:00Z]"
        resp = retry_get(url, timeout=25, retries=2)
        df = pd.read_csv(io.BytesIO(resp.content), low_memory=False)
        return df, url
    except Exception as e:
        raise

@st.cache_data(ttl=3600)
def load_public_ocean_data():
    """
    공개 데이터 로드 시도: 1) NOAA Coral Reef Watch via ERDDAP 2) NOAA OISST via ERDDAP
    재시도/예외 처리 포함. 실패 시 예시 데이터(합성/샘플)를 반환.
    """
    # 시도 1: Coral Reef Watch (point-based or table)
    try:
        # sample virtual station (적당한 위도/경도는 임의로 0,0 사용 — 실제로는 관심 지점 지정 필요)
        raw1, src1 = load_crw_from_erddap(sample_point_lat=0.0, sample_point_lon=0.0)
        df1 = standardize_timeseries(raw1)
        # require at least some numeric values
        if df1["value"].notna().sum() >= 3:
            return {"type": "crw", "df": df1, "source": src1}
    except Exception:
        pass
    # 시도 2: OISST (SST) via ERDDAP
    try:
        raw2, src2 = load_oisst_timeseries_via_erddap(bbox=(-10, 10, 120, 150), start="1982-01-01")
        # OISST CSV 구조가 (time, lat, lon, sst) 일 수 있으므로 간단히 평균 시계열 생성
        df2 = raw2.copy()
        # try common column names
        time_col = None
        for c in ["time", "date", "t"]:
            if c in df2.columns:
                time_col = c
                break
        if time_col is None:
            time_col = df2.columns[0]
        df2["date"] = pd.to_datetime(df2[time_col], errors="coerce")
        numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        valcol = None
        for v in ["sst", "sst_anomaly", "sea_surface_temperature"]:
            if v in df2.columns:
                valcol = v
                break
        if valcol is None and numeric_cols:
            valcol = numeric_cols[0]
        df2["value"] = pd.to_numeric(df2[valcol], errors="coerce")
        df2 = df2[["date", "value"]]
        df2 = drop_future_dates(df2, date_col="date")
        df2 = standardize_timeseries(df2)
        if df2["value"].notna().sum() >= 3:
            return {"type": "oisst", "df": df2, "source": src2}
    except Exception:
        pass
    # 모든 공개 데이터 시도 실패 -> 예시 데이터 반환
    coral = pd.DataFrame({
        "date": pd.date_range(start="1980-01-01", periods=45, freq="Y"),
        "value": np.clip(np.linspace(5, 65, 45) + np.random.randn(45)*3, 0, 100)
    })
    sst = pd.DataFrame({
        "date": pd.date_range(start="1980-01-01", periods=540, freq="M"),
        "value": np.clip(np.linspace(-0.3, 1.2, 540) + np.random.randn(540)*0.1, -2, 3)
    })
    acid = pd.DataFrame({
        "date": pd.date_range(start="1990-01-01", periods=35, freq="Y"),
        "value": np.clip(8.2 - np.linspace(0, 0.15, 35) + np.random.randn(35)*0.01, 7.7, 8.3)
    })
    return {"type": "example", "df_coral": coral, "df_sst": sst, "df_acid": acid, "source": "예시 데이터 사용"}

# ------------------ UI 시작 ------------------

# Pretendard 적용 시도 (있으면 사용)
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
st.caption("공개 데이터(우선 NOAA 계열 시도) + 사용자 입력(보고서 기반) — 한국어 UI")

st.markdown("## 📌 공개 데이터 대시보드 (공식 공개 데이터 우선 시도)")
col_left, col_right = st.columns([2, 1])

with col_right:
    st.markdown("**시도한 공개 데이터 출처(예시)**")
    st.write("- NOAA Coral Reef Watch (CRW) — ERDDAP / 시간-위치 기반 시계열. (문서/ERDDAP 접근 권장)")
    st.write("- NOAA OISST (Optimum Interpolation SST) — ERDDAP / netCDF 대형 시계열 (요약/월별 사용 가능)")
    st.write("- Ocean Carbon & Acidification (OCADS) / GLODAP (pH·산성화 관련 관측)")
    st.write("---")
    st.markdown("**참고/권고**")
    st.write("- 대용량 원자료(netCDF)는 xarray.open_dataset + 지역/기간 서브셋 사용 권장")
    st.write("- 만약 Kaggle 데이터 사용 시: kaggle CLI 인증(https://www.kaggle.com/docs/api) 필요")

# 공개 데이터 로드 시도
public_result = load_public_ocean_data()
public_data_warning = None

if public_result.get("type") == "example":
    st.warning("공개 데이터 자동 연결 시도에 실패하여 예시(대체) 데이터를 사용합니다. (앱은 공개 데이터 연결을 다시 시도하도록 구성되어 있습니다.)")
    coral_df = public_result["df_coral"].rename(columns={"value":"value"})
    sst_df = public_result["df_sst"].rename(columns={"value":"value"})
    acid_df = public_result["df_acid"].rename(columns={"value":"value"})
    gc_source = public_result.get("source", "예시")
else:
    if public_result["type"] == "crw":
        st.success("✅ NOAA Coral Reef Watch(ERDDAP) 데이터 로드 성공 (가능한 경우)")
    else:
        st.success("✅ NOAA OISST(ERDDAP) 데이터 로드 성공 (가능한 경우)")
    gc_source = public_result.get("source", "")
    base_df = public_result["df"].copy()
    # 표준화
    std = standardize_timeseries(base_df)
    # split into plausible series: if variable looks like SST or pH, route to sst/acid, else coral
    if std["value"].notna().mean() >= 0:
        # simple heuristic: if values usually between 0-40 -> assume bleaching %; if around 7-9 -> pH; if -2..3 -> sst
        vmean = std["value"].median()
        if vmean >= -1 and vmean <= 4:
            sst_df = std.rename(columns={"value":"value"})[["date","value","group"]]
            coral_df = pd.DataFrame({"date":[], "value":[]})
            acid_df = pd.DataFrame({"date":[], "value":[]})
        elif vmean >= 6 and vmean <= 9:
            acid_df = std.rename(columns={"value":"value"})[["date","value","group"]]
            coral_df = pd.DataFrame({"date":[], "value":[]})
            sst_df = pd.DataFrame({"date":[], "value":[]})
        else:
            coral_df = std.rename(columns={"value":"value"})[["date","value","group"]]
            sst_df = pd.DataFrame({"date":[], "value":[]})
            acid_df = pd.DataFrame({"date":[], "value":[]})

# 기본 전처리: 날짜형, 중복 제거, 미래 데이터 제거
def finalize_df(df: pd.DataFrame):
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)
    df = drop_future_dates(df, date_col="date")
    return df

coral_df = finalize_df(coral_df)
sst_df = finalize_df(sst_df)
acid_df = finalize_df(acid_df)

# 사이드바: 공개 데이터 옵션 (자동 구성)
st.sidebar.header("공개 데이터 옵션")
# determine global min/max across available series
all_dates = pd.concat([coral_df["date"], sst_df["date"], acid_df["date"]]).dropna()
if all_dates.empty:
    min_date = pd.to_datetime("1980-01-01")
    max_date = seoul_today()
else:
    min_date = all_dates.min()
    max_date = all_dates.max()
date_range = st.sidebar.slider("기간 선택", min_value=min_date.date(), max_value=max_date.date(),
                               value=(min_date.date(), max_date.date()))
smoothing = st.sidebar.selectbox("스무딩(이동평균)", options=["사용 안 함", "3년(연계)", "5년(연계)"], index=0)

# 메인: 산호 백화 시계열 (연별 요약)
with col_left:
    st.subheader("최근 산호 백화 현상 비율 (연별 요약)")
    if coral_df.empty:
        st.info("공개데이터에서 직접 산호 백화 비율 시계열을 확보하지 못했습니다. 예시/대체 데이터를 사용하거나, NOAA Coral Reef Watch에서 'virtual station' 또는 QCed 관측 파일을 ERDDAP로 추출해 보세요.")
        # create example for visualization
        viz_df = pd.DataFrame({"date": pd.date_range(start="1980-01-01", periods=45, freq="Y"),
                               "value": np.clip(np.linspace(5, 65, 45) + np.random.randn(45)*3, 0, 100)})
    else:
        # filter by date_range
        mask = (coral_df["date"].dt.date >= date_range[0]) & (coral_df["date"].dt.date <= date_range[1])
        plot_df = coral_df.loc[mask].copy()
        # aggregate to annual if mixed freq
        plot_df["year"] = plot_df["date"].dt.year
        annual = plot_df.groupby("year")["value"].mean().reset_index()
        annual["date"] = pd.to_datetime(annual["year"].astype(str) + "-01-01")
        viz_df = annual[["date","value"]].sort_values("date")
    # smoothing
    if smoothing != "사용 안 함":
        window = 3 if "3" in smoothing else 5
        viz_df["value_smoothed"] = viz_df["value"].rolling(window=window, center=True, min_periods=1).mean()
    else:
        viz_df["value_smoothed"] = viz_df["value"]
    if viz_df.dropna().shape[0] == 0:
        st.warning("표시할 산호 데이터가 없습니다.")
    else:
        fig = px.line(viz_df, x="date", y="value_smoothed",
                      labels={"date":"연도", "value_smoothed":"산호 백화율 (임의단위)"},
                      title="산호 백화 현상 비율 (연별 평균, 전처리된 값)")
        fig.update_traces(mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)
    st.download_button("전처리된 산호백화_데이터 다운로드 (CSV)",
                       data=viz_df.rename(columns={"value_smoothed":"value"}).to_csv(index=False).encode("utf-8"),
                       file_name="coral_preprocessed.csv", mime="text/csv")

# 보조 지표: SST 및 해양 산성화 (pH)
st.markdown("---")
st.subheader("보조 지표: 해양 표층 수온(SST) & 표층 pH(산성화)")

col_sst, col_acid = st.columns(2)
with col_sst:
    if sst_df.empty:
        st.info("SST(해수온) 공개데이터를 확보하지 못했습니다. 예시 시계열을 사용합니다.")
        sst_plot = pd.DataFrame({"date": pd.date_range(start="1980-01-01", periods=540, freq="M"),
                                 "sst_anomaly": np.clip(np.linspace(-0.3, 1.2, 540) + np.random.randn(540)*0.1, -2, 3)})
    else:
        mask = (sst_df["date"].dt.date >= date_range[0]) & (sst_df["date"].dt.date <= date_range[1])
        tmp = sst_df.loc[mask].copy()
        # if monthly/daily -> convert to annual mean for plotting consistency
        tmp["year"] = tmp["date"].dt.year
        sst_plot = tmp.groupby("year")["value"].mean().reset_index()
        sst_plot["date"] = pd.to_datetime(sst_plot["year"].astype(str) + "-01-01")
        sst_plot = sst_plot.rename(columns={"value":"sst_anomaly"})
    fig_sst = px.area(sst_plot, x="date", y=sst_plot.columns[1],
                      labels={"date":"연도", sst_plot.columns[1]:"SST 이상치 (임의단위, °C)"},
                      title="해양 표층 수온 이상치 (예시/요약)")
    st.plotly_chart(fig_sst, use_container_width=True)
    st.download_button("SST_시계열_다운로드 (CSV)", data=sst_plot.to_csv(index=False).encode("utf-8"),
                       file_name="sst_timeseries.csv", mime="text/csv")

with col_acid:
    if acid_df.empty:
        st.info("표층 pH 공개데이터를 확보하지 못했습니다. 예시 시계열을 사용합니다.")
        acid_plot = pd.DataFrame({"date": pd.date_range(start="1990-01-01", periods=35, freq="Y"),
                                  "surface_pH": np.clip(8.2 - np.linspace(0, 0.15, 35) + np.random.randn(35)*0.01, 7.7, 8.3)})
    else:
        mask = (acid_df["date"].dt.date >= date_range[0]) & (acid_df["date"].dt.date <= date_range[1])
        tmp = acid_df.loc[mask].copy()
        tmp["year"] = tmp["date"].dt.year
        acid_plot = tmp.groupby("year")["value"].mean().reset_index()
        acid_plot["date"] = pd.to_datetime(acid_plot["year"].astype(str) + "-01-01")
        acid_plot = acid_plot.rename(columns={"value":"surface_pH"})
    fig_acid = px.line(acid_plot, x="date", y=acid_plot.columns[1],
                       labels={"date":"연도", acid_plot.columns[1]:"표층 pH"},
                       title="표층 pH 추세 (예시/요약)")
    st.plotly_chart(fig_acid, use_container_width=True)
    st.download_button("산성화_시계열_다운로드 (CSV)", data=acid_plot.to_csv(index=False).encode("utf-8"),
                       file_name="ocean_acidification_timeseries.csv", mime="text/csv")

# ------------------ 사용자 입력(보고서) 기반 대시보드 (업로드 없음: 프롬프트 내 텍스트만 사용) ------------------
st.markdown("---")
st.header("📝 사용자 입력 기반 대시보드 (보고서 내용만 사용, 업로드 없음)")

st.markdown("**보고서 원문(요약)**")
st.write("제목(가제): 역대 최악의 바다 그리고 더 최악이 될 바다.")
st.write("서론: 최근 수십년간 지구 온난화가 가속화됨에 따라 해수온 상승, 산호 백화, 해양 산성화 등이 해양생태계에 심각한 영향을 주고 있습니다.")
st.write("본론 요약: 산호 백화, 고수온에 의한 어류 폐사, 해양 산성화의 악화 등 — 권고: 온실가스 감축, 해양보호구역 확대, 산호 복원, 장기 모니터링 등.")

# 본론1: 최근 45년간 산호 백화 비율 (보고서용) — 상단 공개데이터(또는 예시) 사용
st.subheader("본론 1 — 최근 45년간 산호 백화 비율 (보고서용)")
# generate report_df from coral_df or example
report_df = coral_df.copy()
if report_df.empty or report_df["date"].isna().all():
    report_df = pd.DataFrame({"date": pd.date_range(start="1980-01-01", periods=45, freq="Y"),
                              "value": np.clip(np.linspace(5, 80, 45) + np.random.randn(45)*4, 0, 100)})
report_df = report_df.sort_values("date").reset_index(drop=True)
report_plot = report_df.tail(45).copy()
fig_report = px.area(report_plot, x="date", y="value",
                     labels={"date":"연도", "value":"산호 백화율 (임의단위)"},
                     title="보고서용: 최근 45년 산호 백화율 (연별)")
st.plotly_chart(fig_report, use_container_width=True)
st.download_button("보고서_산호백화_데이터 다운로드 (CSV)", data=report_plot.to_csv(index=False).encode("utf-8"),
                   file_name="report_coral_45y.csv", mime="text/csv")

# 본론2: 해양 산성화·고수온·어류 폐사 영향 (복합 시각화)
st.subheader("본론 2 — 해양 산성화·고수온·어류 폐사 영향 (복합 시각화)")
# 병합: report_plot + sst_plot(연평균) + acid_plot(연평균)
def to_annual(df, value_name):
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date"])
    df2["year"] = df2["date"].dt.year
    ann = df2.groupby("year").agg({value_name: "mean"}).reset_index()
    ann["date"] = pd.to_datetime(ann["year"].astype(str) + "-01-01")
    return ann[["date", value_name]]

r = report_plot.rename(columns={"value":"bleaching_rate_percent"})[["date","bleaching_rate_percent"]].copy()
sst_ann = to_annual(sst_plot.rename(columns={sst_plot.columns[1]:"sst_anomaly"}), "sst_anomaly") if not sst_plot.empty else pd.DataFrame({"date":[],"sst_anomaly":[]})
acid_ann = to_annual(acid_plot.rename(columns={acid_plot.columns[1]:"surface_pH"}), "surface_pH") if not acid_plot.empty else pd.DataFrame({"date":[],"surface_pH":[]})

merge_base = pd.merge(r, sst_ann, on="date", how="outer")
merge_base = pd.merge(merge_base, acid_ann, on="date", how="outer")
merge_base = merge_base.sort_values("date").reset_index(drop=True)
merge_base = drop_future_dates(merge_base, date_col="date")

# 합성 어류 폐사 지수 (간단 가중치 모델): SST 상승 및 pH 하락을 이용
merge_base["sst_norm"] = (merge_base["sst_anomaly"] - merge_base["sst_anomaly"].min()) / (merge_base["sst_anomaly"].max() - merge_base["sst_anomaly"].min() + 1e-9) if "sst_anomaly" in merge_base.columns else 0
merge_base["pH_drop"] = 8.2 - merge_base.get("surface_pH", 8.2)
merge_base["pH_norm"] = (merge_base["pH_drop"] - merge_base["pH_drop"].min()) / (merge_base["pH_drop"].max() - merge_base["pH_drop"].min() + 1e-9) if "surface_pH" in merge_base.columns else 0
merge_base["fish_mortality_index"] = (0.7 * merge_base["sst_norm"].fillna(0) + 0.3 * merge_base["pH_norm"].fillna(0)) * 100

fig_comb = px.line(merge_base, x="date", y="bleaching_rate_percent", labels={"date":"연도", "bleaching_rate_percent":"백화율 (%)"},
                   title="해양 고수온 · 산성화 · 어류 폐사(합성지수) 비교")
if "sst_anomaly" in merge_base.columns:
    fig_comb.add_scatter(x=merge_base["date"], y=merge_base["sst_anomaly"], mode="lines+markers", name="SST 연평균 이상치 (임의단위 °C)")
fig_comb.add_scatter(x=merge_base["date"], y=merge_base["fish_mortality_index"], mode="lines+markers", name="어류 폐사 지수 (합성)")
st.plotly_chart(fig_comb, use_container_width=True)
st.download_button("본론2_복합_전처리_데이터 다운로드 (CSV)", data=merge_base.to_csv(index=False).encode("utf-8"),
                   file_name="report_combined_ocean_data.csv", mime="text/csv")

# ------------------ 결론 / 참고자료 ------------------
st.markdown("---")
st.subheader("결론 및 권고 (자동 제안)")
st.write(
    "- 산호 백화, 해양 산성화, 고수온은 상호 연계되어 해양생태계에 심각한 영향을 줍니다.\n"
    "- 권고: 온실가스 감축, 해양 보호구역 확대, 산호 복원, 장기 모니터링 체계 구축 등.\n"
)
st.markdown("**참고자료(앱에서 시도/참조한 링크)**")
st.write("- NOAA Coral Reef Watch: https://coralreefwatch.noaa.gov/ . (CRW ERDDAP 접근 안내 문서 권장)")
st.write("- NOAA OISST (SST): https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html")
st.write("- Ocean Carbon & Acidification (OCADS): https://www.ncei.noaa.gov/products/ocean-carbon-acidification-data-system")
st.write("- GLODAP: https://glodap.info/")
st.caption("앱 노트: 공개 데이터는 종종 netCDF/ZIP/대용량 CSV 형식으로 제공됩니다. 실서비스 환경에서는 인증·다운로드·전처리(xarray.open_dataset 등)를 추가하세요.")
