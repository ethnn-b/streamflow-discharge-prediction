"""Recomputes the paper's headline statistics (nRMSE/nMAE/|PBIAS| contrasts per network, main effects) straight from res_main/.
Used to cross-check the hand-adapted tables in paper_latex_v10/tables/, which make_tables.py does not emit for the US and Kaveri nMAE columns.
Usage: PYTHONPATH=. python rerun_expts/headline_stats.py
"""
import sys, json, glob, os, io, contextlib, itertools, warnings
import numpy as np
from scipy import stats
warnings.simplefilter('ignore')
sys.path.insert(0,'.')
from rerun_expts.common import load_network, connected
GAPS=[3,7,30,100]
def load(net, d='res_main'):
    R={}
    for f in glob.glob(f'rerun_expts/{d}/*.json'):
        r=json.load(open(f))
        if r['net']==net: R[(r['arm'],r['seed'],r['gap'])]=r
    return R
def setup(net):
    with contextlib.redirect_stdout(io.StringIO()):
        df,st,_,D,C,recv=load_network(net)
    return df,st,D,C,recv
def g(ps,sc,keep,metric):
    rows=[(s,v) for s,v in ps.items() if v and sc.get(s) and s in keep]
    if metric=='PBIAS': return np.mean([v['PBIAS'] for s,v in rows])
    return np.mean([100*v[metric]/sc[s] for s,v in rows]) if rows else np.nan
def con(R,a,b,f,absv=False):
    pg={};d=[]
    for gap in GAPS:
        for s in sorted({k[1] for k in R}):
            if (a,s,gap) in R and (b,s,gap) in R:
                xa,xb=f(R[(a,s,gap)]),f(R[(b,s,gap)])
                if absv: xa,xb=abs(xa),abs(xb)
                x=100*(xa-xb)/abs(xb); d.append(x); pg.setdefault(gap,[]).append(x)
    t,p=stats.ttest_1samp([np.mean(v) for v in pg.values()],0)
    return round(np.mean(d),2),f'{sum(x<0 for x in d)}/{len(d)}',round(p,4)
for net,pimf,base in [('mahanadi','ARLanc','AR'),('kaveri','ALanc','A'),('us','ARLanc','AR')]:
    R=load(net); df,st,D,C,recv=setup(net)
    sc={s:float(df[s].mean()) for s in st}
    keep=set(connected(st,C,df))
    print(net,len(st),'gauges; connected',len(keep),'recv',len(recv),recv, 'cells',len(R))
    for m in ('RMSE','MAE'):
        print(' ',m,'PIMF vs vanilla',con(R,pimf,'vanilla',lambda r:g(r['per_station'],sc,keep,m)))
    print('  PBIAS abs',con(R,pimf,'vanilla',lambda r:g(r['per_station'],sc,keep,'PBIAS'),True))
    if net=='kaveri':
        print('  anchor vs {A,S} nMAE',con(R,'ALanc','A',lambda r:g(r['per_station'],sc,keep,'MAE')))
        print('  anchor vs {A,S} nRMSE',con(R,'ALanc','A',lambda r:g(r['per_station'],sc,keep,'RMSE')))
    if net=='mahanadi':
        rc=set(recv)
        for m in ('RMSE','MAE'): print('  recv',m,con(R,pimf,'vanilla',lambda r:g(r['per_station'],sc,rc,m)))
        print('  recv pbias',con(R,pimf,'vanilla',lambda r:g(r['per_station'],sc,rc,'PBIAS'),True))
        for m in ('RMSE','MAE'): print('  pooled',m,con(R,pimf,'vanilla',lambda r:r['pooled'][m]))
        print('  pooled pbias',con(R,pimf,'vanilla',lambda r:r['pooled']['PBIAS'],True))
        # N: PIMF vs AR
        print('  N nMAE PIMF vs AR',con(R,'ARLanc','AR',lambda r:g(r['per_station'],sc,keep,'MAE')))
        for gap in (3,7):
            xs=[100*(g(R[('ARLanc',s,gap)]['per_station'],sc,keep,'MAE')-g(R[('AR',s,gap)]['per_station'],sc,keep,'MAE'))/g(R[('AR',s,gap)]['per_station'],sc,keep,'MAE') for s in (1,2,3,4,42)]
            print('   gap',gap,np.mean(xs))
        # main effects w/ 3 features
        arms={(a,r,n):('A' if a else '')+('R' if r else '')+('N' if n else '') for a in(0,1) for r in(0,1) for n in(0,1)}
        def nm(a,r,n):
            s=('A' if a else '')+('R' if r else '')+('L' if n else '')+('anc' if n else '')
            return s or 'base'
        for i,name in enumerate('ARN'):
            diffs={}
            for a,r,n in itertools.product((0,1),repeat=3):
                x=[a,r,n]
                if x[i]==1:
                    y=list(x);y[i]=0
                    A_,B_=nm(*x),nm(*y)
                    for gap in GAPS:
                        for s in sorted({k[1] for k in R}):
                            xa=g(R[(A_,s,gap)]['per_station'],sc,keep,'RMSE');xb=g(R[(B_,s,gap)]['per_station'],sc,keep,'RMSE')
                            diffs.setdefault((gap,s),[]).append(100*(xa-xb)/xb)
            per={k:np.mean(v) for k,v in diffs.items()}
            pg={}
            for (gap,s),v in per.items(): pg.setdefault(gap,[]).append(v)
            t,p=stats.ttest_1samp([np.mean(v) for v in pg.values()],0)
            print('  main',name,round(np.mean(list(per.values())),2),sum(v<0 for v in per.values()),len(per),round(p,4))
