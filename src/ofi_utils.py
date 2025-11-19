# src/ofi_utils.py
from __future__ import annotations
import os, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict
try:
    import pyreadr
except Exception:
    pyreadr=None
from statsmodels.api import OLS, add_constant
try:
    from zoneinfo import ZoneInfo
    _NY_TZ = ZoneInfo("America/New_York")
except Exception:
    try:
        import pytz
        _NY_TZ = pytz.timezone("America/New_York")
    except Exception:
        _NY_TZ = None

@dataclass
class ColumnMap:
    symbol: str; bid: str; ask: str; bidsz: str; asksz: str; time_m: str

SYMBOL_CANDIDATES=["sym_root","symbol","sym","ticker"]
BID_CANDIDATES=["best_bid","bid","nbbo_bid"]
ASK_CANDIDATES=["best_ask","ask","nbbo_ask"]
BIDSZ_CANDIDATES=["best_bidsiz","bidsiz","bid_size","nbbo_bidsiz","bid_sz"]
ASKSZ_CANDIDATES=["best_asksiz","asksz","ask_size","nbbo_asksiz","ask_sz"]
TIMECOL_CANDIDATES=["time_m","timeM","time","ts_m","seconds"]

def _choose(cols: List[str], cands: List[str])->Optional[str]:
    lower={c.lower():c for c in cols}
    for x in cands:
        if x.lower() in lower: return lower[x.lower()]
    return None

def resolve_columns(df: pd.DataFrame)->ColumnMap:
    cols=list(df.columns)
    symbol=_choose(cols,SYMBOL_CANDIDATES); time_m=_choose(cols,TIMECOL_CANDIDATES)
    bid=_choose(cols,BID_CANDIDATES); ask=_choose(cols,ASK_CANDIDATES)
    bidsz=_choose(cols,BIDSZ_CANDIDATES); asksz=_choose(cols,ASKSZ_CANDIDATES)
    missing=[x for x in [symbol,time_m,bid,ask,bidsz,asksz] if x is None]
    if missing: raise ValueError(f"Column resolution failed. Missing {missing}.")
    return ColumnMap(symbol,bid,ask,bidsz,asksz,time_m)

def detect_time_unit(maxv:int)->str:
    if maxv<1_000_000: return "s"
    elif maxv<1_000_000_000: return "ms"
    else: return "us"

def _localize(ts: pd.Series)->pd.Series:
    if isinstance(ts.dt.tz,type(None)):
        if _NY_TZ is not None: return ts.dt.tz_localize(_NY_TZ)
        return ts.dt.tz_localize("America/New_York")
    return ts.dt.tz_convert("America/New_York")

def time_m_to_timedelta(df: pd.DataFrame, col:str)->pd.Series:
    maxv=int(pd.Series(df[col]).max()); unit=detect_time_unit(maxv)
    if unit=="s":   return pd.to_timedelta(df[col].astype("int64"),unit="s")
    if unit=="ms":  return pd.to_timedelta(df[col].astype("int64"),unit="ms")
    return pd.to_timedelta(df[col].astype("int64"),unit="us")

def read_rda(path:str)->pd.DataFrame:
    if pyreadr is None: raise ImportError("pyreadr not installed")
    res=pyreadr.read_r(path); name,df=next(iter(res.items()))
    if not isinstance(df,pd.DataFrame): raise ValueError("Top-level object not a DataFrame")
    return df

def filter_crossed(df: pd.DataFrame,bid_col:str,ask_col:str)->pd.DataFrame:
    return df.loc[df[ask_col]>=df[bid_col]].copy()

def parse_trading_day_from_filename(path:str)->pd.Timestamp:
    stem=os.path.splitext(os.path.basename(path))[0]
    for fmt in ("%Y-%m-%d","%Y%m%d"):
        try:
            dt=pd.to_datetime(stem,format=fmt); return pd.Timestamp(dt.date(),tz="America/New_York")
        except Exception: pass
    ts=pd.Timestamp(os.path.getmtime(path),unit="s",tz="UTC").tz_convert("America/New_York")
    return pd.Timestamp(ts.date(),tz="America/New_York")

def build_tob_series_1s(df: pd.DataFrame,cmap:ColumnMap,trading_day:pd.Timestamp,freq:str="1s")->pd.DataFrame:
    offsets=time_m_to_timedelta(df,cmap.time_m)
    day_midnight=pd.Timestamp(trading_day.year,trading_day.month,trading_day.day,tz="America/New_York")
    ts=day_midnight+offsets; ts=_localize(pd.Series(ts)).astype("datetime64[ns, America/New_York]")
    df=df.copy(); df["ts"]=ts.values
    df=filter_crossed(df,cmap.bid,cmap.ask).set_index("ts").sort_index()
    # Remove duplicate timestamps, keeping the last occurrence
    df = df[~df.index.duplicated(keep='last')]
    start=pd.Timestamp(trading_day.date(),tz="America/New_York")+pd.Timedelta(hours=9,minutes=30)
    end  =pd.Timestamp(trading_day.date(),tz="America/New_York")+pd.Timedelta(hours=16)
    # Convert index to timezone-aware if it's not already
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    df=df.loc[(df.index>=start)&(df.index<=end)]
    grid=pd.date_range(start=start,end=end,freq=freq,tz="America/New_York")
    df=df[[cmap.bid,cmap.ask,cmap.bidsz,cmap.asksz]].reindex(grid).ffill()
    df=df.dropna(subset=[cmap.bid,cmap.ask,cmap.bidsz,cmap.asksz])
    return df.rename(columns={cmap.bid:"bid",cmap.ask:"ask",cmap.bidsz:"bid_sz",cmap.asksz:"ask_sz"})

def compute_ofi_depth_mid(df: pd.DataFrame)->pd.DataFrame:
    bP,aP=df["bid"],df["ask"]; bS,aS=df["bid_sz"],df["ask_sz"]
    dbP,daP=bP.diff(),aP.diff(); dbS,daS=bS.diff(),aS.diff()
    ofi=pd.Series(np.zeros(len(df)),index=df.index,dtype="float64")
    # Bid side: price up = aggressive buy (+), price down = bid withdrawn (-), size change at same price
    ofi+=np.where(dbP>0,bS,0.0); ofi+=np.where(dbP<0,-bS.shift(1),0.0); ofi+=np.where(dbP==0,dbS.fillna(0.0),0.0)
    # Ask side: price down = aggressive sell (-), price up = ask withdrawn (+), size change at same price (-)
    ofi+=np.where(daP>0,-aS.shift(1),0.0); ofi+=np.where(daP<0,-aS,0.0); ofi+=np.where(daP==0,-daS.fillna(0.0),0.0)
    depth=bS+aS; mid=0.5*(bP+aP); d_mid_bps=1e4*mid.pct_change()
    
    # Filter out unrealistic price jumps (>10% move in 1 second = >1000 bps)
    # These are likely data errors or symbol mix-ups
    d_mid_bps = d_mid_bps.where(abs(d_mid_bps) < 1000, np.nan)
    
    return pd.DataFrame({"bid":bP,"ask":aP,"bid_sz":bS,"ask_sz":aS,"depth":depth,"ofi":ofi,"mid":mid,"d_mid_bps":d_mid_bps},index=df.index)

def normalize_ofi(df: pd.DataFrame,window_secs:int=600,min_periods:int=50)->pd.DataFrame:
    roll=df["depth"].rolling(window=window_secs,min_periods=min_periods).mean()
    out=df.copy(); out["depth_roll_10m"]=roll; out["normalized_OFI"]=out["ofi"]/roll.replace(0,np.nan)
    return out

def run_ols_xy(x: pd.Series, y: pd.Series):
    d=pd.concat([x,y],axis=1).dropna(); n=len(d)
    if n<10: return dict(alpha=np.nan,beta=np.nan,se_beta=np.nan,r2=np.nan,n=n,notes="n<10")
    X=add_constant(d.iloc[:,0].values); Y=d.iloc[:,1].values
    try:
        res=OLS(Y,X).fit(cov_type="HC1")
        return dict(alpha=float(res.params[0]),beta=float(res.params[1]),se_beta=float(res.bse[1]),r2=float(res.rsquared),n=int(n),notes="")
    except Exception as e:
        return dict(alpha=np.nan,beta=np.nan,se_beta=np.nan,r2=np.nan,n=n,notes=f"ols_error:{e}")

def run_ols_symbol_day(ts_df: pd.DataFrame):
    st=run_ols_xy(ts_df["normalized_OFI"],ts_df["d_mid_bps"])
    st["mean_depth"]=float(ts_df["depth"].mean())
    with np.errstate(invalid="ignore",divide="ignore"):
        st["ofi_scale"]=float(np.nanstd(ts_df["ofi"])/np.nanmean(ts_df["depth_roll_10m"]))
    return st

def resample_to(df: pd.DataFrame,freq:str)->pd.DataFrame:
    agg=df[["bid","ask","bid_sz","ask_sz"]].resample(freq).last().dropna()
    agg=compute_ofi_depth_mid(agg); agg=normalize_ofi(agg,window_secs=600,min_periods=10 if freq!="1s" else 50)
    return agg

def save_timeseries_parquet(ts_df: pd.DataFrame,outdir:str,day:str,symbol:str):
    dd=os.path.join(outdir,"timeseries",day); os.makedirs(dd,exist_ok=True)
    ts_df.to_parquet(os.path.join(dd,f"{symbol}.parquet"),index=True)

def append_panel_row(row:Dict,outdir:str,name:str):
    path=os.path.join(outdir,"regressions",name); os.makedirs(os.path.dirname(path),exist_ok=True)
    if os.path.exists(path):
        pan=pd.read_parquet(path); pan=pd.concat([pan,pd.DataFrame([row])],ignore_index=True)
        keys=[k for k in ["symbol","day","half_hour_start"] if k in pan.columns]
        if keys: pan=pan.drop_duplicates(subset=keys,keep="last")
    else:
        pan=pd.DataFrame([row])
    pan.to_parquet(path,index=False)

def process_day_rda(path:str,outdir:str,freq:str="1s",do_halfhour_10s:bool=True)->pd.DataFrame:
    df=read_rda(path); cmap=resolve_columns(df); day=parse_trading_day_from_filename(path); day_str=str(day.date())
    rows=[]
    for symbol,g in df.groupby(cmap.symbol):
        symbol=str(symbol)
        ts1s_raw=build_tob_series_1s(g,cmap,trading_day=day,freq=freq)
        ts1s=normalize_ofi(compute_ofi_depth_mid(ts1s_raw),window_secs=600,min_periods=50)
        save_timeseries_parquet(ts1s,outdir,day_str,symbol)
        st=run_ols_symbol_day(ts1s); row=dict(symbol=symbol,day=day_str,**st); append_panel_row(row,outdir,"by_symbol_day.parquet"); rows.append(row)
        if do_halfhour_10s:
            ts10=resample_to(ts1s_raw, "10s")
            bins=ts10.index.floor("30min")
            for hstart,sub in ts10.groupby(bins):
                st=run_ols_xy(sub["normalized_OFI"],sub["d_mid_bps"])
                rowh=dict(symbol=symbol,day=day_str,half_hour_start=str(hstart),mean_depth=float(sub["depth"].mean()),**st)
                append_panel_row(rowh,outdir,"by_symbol_day_halfhour.parquet")
    return pd.DataFrame(rows)

def make_scatter(ts_df: pd.DataFrame,symbol:str,day:str,figdir:str):
    import matplotlib.pyplot as plt, numpy as np, os
    os.makedirs(figdir,exist_ok=True); d=ts_df.dropna(subset=["d_mid_bps","normalized_OFI"]).copy(); n=len(d)
    plt.figure()
    if n==0:
        plt.text(0.5,0.5,"No valid points",ha="center",va="center"); plt.axis("off")
    else:
        X=d["normalized_OFI"].values; y=d["d_mid_bps"].values
        if n>20000:
            idx=np.random.choice(n,20000,replace=False); X=X[idx]; y=y[idx]; n=len(X)
        plt.scatter(X,y,s=4,alpha=0.3)
        title=""
        if n>=10:
            try:
                res=OLS(y,add_constant(X)).fit(cov_type="HC1")
                xg=np.linspace(np.nanpercentile(X,1),np.nanpercentile(X,99),100)
                yg=res.params[0]+res.params[1]*xg; plt.plot(xg,yg,linewidth=2)
                title=f"  β={float(res.params[1]):.3g}, R²={float(res.rsquared):.3f}, n={n}"
            except Exception as e:
                title=f"  fit error: {e}"
        else: title=f"  n={n} (no fit)"
        plt.xlabel("normalized_OFI"); plt.ylabel("d_mid_bps"); plt.title(f"{symbol} {day}{title}")
    out=os.path.join(figdir,f"scatter_{symbol}_{day}.png"); plt.tight_layout(); plt.savefig(out,dpi=150); plt.close()

def beta_histogram(panel_path:str,figdir:str):
    import matplotlib.pyplot as plt, os, pandas as pd
    os.makedirs(figdir,exist_ok=True); plt.figure()
    if not os.path.exists(panel_path):
        plt.text(0.5,0.5,"No panel parquet found",ha="center",va="center"); plt.axis("off")
    else:
        pan=pd.read_parquet(panel_path).dropna(subset=["beta"])
        if len(pan)==0:
            plt.text(0.5,0.5,"No valid β to plot",ha="center",va="center"); plt.axis("off")
        else:
            plt.hist(pan["beta"].values,bins=30); plt.xlabel("β"); plt.ylabel("count"); plt.title("β across symbol×day")
    out=os.path.join(figdir,"beta_hist.png"); plt.tight_layout(); plt.savefig(out,dpi=150); plt.close()

def intraday_beta_vs_depth(panel_halfhour:str,timeseries_root:str,figdir:str):
    import matplotlib.pyplot as plt, os, pandas as pd, numpy as np
    if not os.path.exists(panel_halfhour): return
    pan=pd.read_parquet(panel_halfhour).dropna(subset=["beta"])
    if len(pan)==0: return
    pan["hh"]=pd.to_datetime(pan["half_hour_start"])
    beta_prof=pan.groupby(pan["hh"].dt.strftime("%H:%M")).agg(beta_med=("beta","median")).reset_index()
    depths=[]
    if os.path.isdir(timeseries_root):
        for day in os.listdir(timeseries_root):
            ddir=os.path.join(timeseries_root,day)
            if not os.path.isdir(ddir): continue
            for pq in os.listdir(ddir):
                if not pq.endswith(".parquet"): continue
                ts=pd.read_parquet(os.path.join(ddir,pq))
                hh=pd.to_datetime(ts.index).floor("30min").strftime("%H:%M")
                df=pd.DataFrame({"hh":hh,"depth":ts["depth"].values})
                depths.append(df.groupby("hh").agg(depth_med=("depth","median")))
    if depths:
        depth_prof=pd.concat(depths).groupby(level=0).median().reset_index()
    else:
        depth_prof=pd.DataFrame({"hh":beta_prof["hh"],"depth_med":np.nan})
    merged=beta_prof.merge(depth_prof,on="hh",how="left")
    plt.figure(); plt.plot(merged["hh"],merged["beta_med"],label="median β")
    if "depth_med" in merged:
        try:
            d=merged["depth_med"].astype(float); d=(d-np.nanmedian(d))/(np.nanstd(d)+1e-9)
            plt.plot(merged["hh"],-d,linestyle="--",label="-scaled depth (median)")
        except Exception: pass
    plt.xticks(rotation=45); plt.title("Intraday β vs depth"); plt.legend()
    os.makedirs(figdir,exist_ok=True); out=os.path.join(figdir,"intraday_beta_vs_depth.png")
    plt.tight_layout(); plt.savefig(out,dpi=150); plt.close()
