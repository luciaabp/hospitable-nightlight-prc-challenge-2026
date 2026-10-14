"""Own off-block detector from adsb.lol (ob2).

Per departure: aircraft by callsign (+/- 1 h of MVT) and its ground segment ending at the last ground point before
MVT + 120 s. After a stop >= LONG_STOP (stay at the stand), off-block = first moving point after the last one; else the
first moving point if the segment starts stopped; else the first point (censored: the hidden gap is left to the model).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import RAW
from components.base_model.adsb_relaxed import (LOOKBACK, MATCH_SEC as MATCH, MAX_LAST_GAP,  # noqa: F401
                                           SEG_GAP, departures, load_cut)

STOP_KT = 1.0
LONG_STOP = 900.0
COLS=['ob2_gs0','ob2_taxi','ob2_first','ob2_obs','ob2_long','ob2_last_gap','ob2_n','ob2_lat','ob2_lon','ob2_dwell','ob2_mlat','ob2_gapmax','ob2_stop_pre']

def detect(cut, deps):
    vis=cut.groupby(['apt','callsign','icao'],observed=True)['t'].agg(['min','max']).reset_index()
    m=deps.merge(vis,on=['apt','callsign'])
    m['d']=np.maximum(0,np.maximum(m['min']-m['mvt'],m['mvt']-m['max']))
    m=m[m.d<MATCH].sort_values('d').drop_duplicates('MVT_ID_mvt')[['MVT_ID_mvt','mvt','icao']]
    g=cut[cut.chao].sort_values(['icao','t'])
    by={k:(v.t.to_numpy(),v.lat.to_numpy(),v.lon.to_numpy(),v.gs.to_numpy(float),(v.fonte.astype(str)=='mlat').to_numpy()) for k,v in g.groupby('icao',observed=True)}
    out=[]
    for mid,mvt,icao in m.itertuples(index=False):
        if icao not in by: continue
        t,lat,lon,gs,ml=by[icao]
        k=np.searchsorted(t,mvt+120,side='right')-1
        if k<0 or mvt-t[k]>MAX_LAST_GAP: continue
        j=k
        while j>0 and t[j]-t[j-1]<SEG_GAP and mvt-t[j-1]<LOOKBACK: j-=1
        tt=t[j:k+1]; gg=np.nan_to_num(gs[j:k+1],nan=0.0); n=len(tt)
        mov=gg>STOP_KT
        # stops: runs of ~mov; duration until the next moving point (or the last point)
        ob=None; longf=0; dwell=0.0; stop_pre=0.0
        idx_mov=np.flatnonzero(mov)
        if idx_mov.size==0:
            ob=tt[0]; obs=0
        else:
            # stationary runs
            starts=[]; i=0
            while i<n:
                if not mov[i]:
                    s=i
                    while i<n and not mov[i]: i+=1
                    e=i  # first movement after the run (or n)
                    end_t=tt[e] if e<n else tt[n-1]
                    starts.append((s,e,end_t-tt[s]))
                else: i+=1
            longs=[r for r in starts if r[2]>=LONG_STOP and r[1]<n]
            if longs:
                s,e,dur=longs[-1]; ob=tt[e]; obs=1; longf=1; dwell=dur
            elif not mov[0]:
                e=idx_mov[0]; ob=tt[e]; obs=1; dwell=tt[e]-tt[0]
            else:
                ob=tt[0]; obs=0
            stop_pre=sum(r[2] for r in starts if r[0]>0)  # time stopped after off-block (queues)
        out.append((mid,float(gs[j]),mvt-ob,mvt-tt[0],obs,longf,mvt-tt[-1],n,lat[j],lon[j],dwell,float(ml[j:k+1].mean()),float(np.diff(tt).max()) if n>1 else 0.,stop_pre))
    return pd.DataFrame(out,columns=['MVT_ID_mvt',*COLS])

def _job(a):
    cut_dir,day,deps=a
    return detect(load_cut(cut_dir,day),deps)

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    cut_dir=Path(sys.argv[1]); out=Path(sys.argv[2]); procs=int(sys.argv[3]); selected_days=sys.argv[4] if len(sys.argv)>4 else None
    deps=departures(RAW)
    days=sorted(p.stem for p in cut_dir.glob('*.parquet'))
    if selected_days: days=[d for d in days if d in selected_days.split(',')]
    jobs=[(cut_dir,d,g.drop(columns='dia')) for d,g in deps[deps.dia.isin(days)].groupby('dia')]
    with ProcessPoolExecutor(procs) as ex: r=pd.concat(ex.map(_job,jobs),ignore_index=True).drop_duplicates('MVT_ID_mvt')
    r.to_parquet(out,index=False); print(len(r),'->',out)
