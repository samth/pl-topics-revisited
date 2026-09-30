"""Collect every number the report shows into out/report_data.json."""
import json, re, numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.metrics import normalized_mutual_info_score as nmi
import sys; sys.path.insert(0, 'scripts')
from stm_labels import STM40
from bertopic_labels import L as BTL
K = 40; R = {}
df = pd.read_pickle('data/corpus.pkl'); c4 = df[df.core4].reset_index(drop=True)
th = pd.read_csv('out/stm_K40_theta.csv').set_index('doc_id').loc[c4.doc_id].values
V = ['popl', 'pldi', 'icfp', 'oopsla']
raw = pd.DataFrame([json.loads(l) for l in __import__('gzip').open('data/papers.jsonl.gz')])
R['coverage'] = dict(total=len(raw), with_abs=int(raw.abstract.notna().sum()), corpus=len(df), core4=len(c4),
    by_venue=df.groupby('venue').size().to_dict(),
    core4_year={v: c4[c4.venue == v].groupby('year').size().to_dict() for v in V},
    other_year=df[~df.core4].groupby('year').size().to_dict(),
    core4_raw={v: int((raw.venue == v).sum()) for v in V})
R['searchK'] = pd.read_csv('out/stm_searchK.csv').to_dict('list')
R['lda_sweep'] = pd.read_csv('out/lda_ksweep.csv').groupby('k')[['npmi', 'c_v', 'diversity']].mean().reset_index().to_dict('list')
R['lda20'] = [l.split(': ', 1)[1] for l in open('out/lda20_topics.txt').read().strip().split('\n')]
lab = pd.read_csv('out/stm_K40_labels.csv')
AREA = {'Types & semantics': [7, 9, 15, 17, 23, 27, 38, 39], 'Verification & logic': [2, 28, 31, 32, 34],
        'Static analysis': [5, 6, 37], 'Compilers & runtimes': [12, 14, 19, 21, 36], 'Parallelism & concurrency': [10, 20, 35],
        'Software engineering & testing': [8, 11, 13, 24, 26], 'Language design & DSLs': [1, 22, 25, 30, 33],
        'Emerging: probabilistic, synthesis, security, quantum': [3, 4, 16, 18, 29, 40]}
t2a = {t: a for a, ts in AREA.items() for t in ts}
top = {}
for k in range(K):
    idx = np.argsort(-th[:, k])[:3]
    top[k + 1] = [f"{c4.title[i]} ({c4.venue[i].upper()} {c4.year[i]})" for i in idx]
R['topics'] = [dict(id=int(r.topic), name=STM40[r.topic], area=t2a[r.topic], prob=r.prob.split(', ')[:8], frex=r.frex.split(', ')[:8],
                    prev=round(r.prevalence, 4), papers=top[r.topic]) for r in lab.itertuples()]
R['areas'] = list(AREA)
tr = pd.read_csv('out/stm_K40_trajectories.csv')
R['traj'] = {int(k): {v: g2[['year', 'mean', 'lo', 'hi']].round(4).values.tolist() for v, g2 in g.groupby('venue')} for k, g in tr.groupby('topic')}
# holistic: empirical area shares per venue-year (3-year centred window)
hol = {}
for v in V:
    rows = []
    for y in range(c4[c4.venue == v].year.min(), c4[c4.venue == v].year.max() + 1):
        w = ((c4.venue == v) & (c4.year.between(y - 1, y + 1))).values
        if w.sum() < 10: continue
        m = th[w].mean(0)
        rows.append([y] + [round(float(sum(m[t - 1] for t in ts)), 4) for ts in AREA.values()])
    hol[v] = rows
R['holistic'] = hol
# OOPSLA: OO share gap (topics 21+33) OOPSLA minus other three, bootstrap
rng = np.random.default_rng(0); oo = th[:, 20] + th[:, 32]; ev = []
for y in range(1987, 2027):
    a = oo[((c4.venue == 'oopsla') & (c4.year == y)).values]; b = oo[((c4.venue != 'oopsla') & (c4.year == y)).values]
    if len(a) < 5 or len(b) < 5: continue
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(1000)]
    ev.append([y, round(a.mean(), 4), round(b.mean(), 4), round(a.mean() - b.mean(), 4), *np.round(np.percentile(bs, [2.5, 97.5]), 4)])
R['oopsla_event'] = ev
d = pd.DataFrame({'oo': oo, 'oopsla': (c4.venue == 'oopsla').astype(int), 'year': c4.year, 'vy': c4.venue + c4.year.astype(str)})
d = d[d.year.between(1997, 2019)]; d['post07'] = (d.year >= 2007).astype(int); d['post10'] = (d.year >= 2010).astype(int); d['t'] = d.year - 2007
did = {}
for name, f in [('DiD, year fixed effects', 'oo ~ C(year) + oopsla + oopsla:post07 + oopsla:post10'),
                ('DiD + OOPSLA-specific linear trend', 'oo ~ C(year) + oopsla + oopsla:t + oopsla:post07 + oopsla:post10')]:
    m = smf.ols(f, d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.vy)[0]})
    did[name] = {k: [round(m.params[k], 4), round(m.bse[k], 4), float(m.pvalues[k])] for k in ['oopsla:post07', 'oopsla:post10'] + (['oopsla:t'] if 'oopsla:t' in m.params else [])}
R['oopsla_did'] = did
dd = pd.read_csv('out/stm_K40_oopsla_did.csv'); dd['name'] = dd.topic.map(STM40)
R['oopsla_did_topics'] = dd[dd.p < 0.01].sort_values('est').round(4).to_dict('records')
R['oopsla_cp'] = pd.read_csv('out/oopsla_changepoints.csv').assign(name=lambda x: x.topic.map(STM40)).to_dict('records')
s = pd.read_csv('out/venue_jsd_diversity.csv')
R['diversity'] = {v: g[['year', 'val', 'lo', 'hi']].round(3).values.tolist() for v, g in s[s.kind == 'diversity'].groupby('a')}
j = s[s.kind == 'jsd'].copy(); j['pair'] = j.a + '–' + j.b_
R['jsd'] = {p: g[['year', 'val', 'lo', 'hi']].round(4).values.tolist() for p, g in j.groupby('pair')}
c = pd.read_csv('out/venue_classifier.csv'); acc = []
for y in range(1988, 2027):
    w = c.year.between(y - 2, y + 2); x = c.correct[w].values
    bs = [rng.choice(x, len(x)).mean() for _ in range(500)]
    acc.append([y, round(x.mean(), 4), *np.round(np.percentile(bs, [2.5, 97.5]), 4)])
R['venue_acc'] = acc; R['venue_acc_by_venue'] = c.groupby('venue').correct.mean().round(3).to_dict()
R['popl_pldi_auc'] = pd.read_csv('out/popl_pldi_auc.csv').round(4).values.tolist()
# POPL vs PLDI log-ratio early vs late (empirical, from theta)
pp = []
for k in range(K):
    row = dict(id=k + 1, name=STM40[k + 1])
    for era, (a, b) in {'early': (1988, 2004), 'late': (2012, 2026)}.items():
        mp = th[((c4.venue == 'popl') & c4.year.between(a, b)).values, k].mean(); ml = th[((c4.venue == 'pldi') & c4.year.between(a, b)).values, k].mean()
        row[era] = round(float(np.log2(mp / ml)), 3)
    pp.append(row)
R['popl_pldi'] = pp
# separation logic
sl = pd.read_csv('out/seplogic_by_venue_year.csv')
R['sl'] = {v: g[['year', 'n', 'cites', 'mentions', 'topic']].round(4).values.tolist() for v, g in sl[sl.venue.isin(V)].groupby('venue')}
R['sl_cp'] = pd.read_csv('out/seplogic_changepoint.csv').round(4).values.tolist()
tm = pd.read_csv('out/seplogic_topicmix.csv'); tm['name'] = tm.topic.map(STM40)
R['sl_mix'] = tm.sort_values('diff').round(4).to_dict('records')
# related work
R['rw'] = pd.read_csv('out/related_work_summary.csv').round(4).to_dict('records')
f9 = pd.read_csv('out/fig9_cases.csv')
R['fig9'] = f9.assign(g=f9.set.where(f9.set == 'Citations', 'Random')).groupby(['paper', 'method', 'g']).pct.describe()[['count', '25%', '50%', '75%']].round(3).reset_index().to_dict('records')
R['nix'] = {'SPECTER2 (cosine)': 1315, 'TF-IDF (cosine)': 138, 'LDA k=20, Euclidean (paper)': 2907, 'LDA k=20, Hellinger': 1333, 'STM k=40, Hellinger': 173, 'pool': 4024}
# BERTopic across all venues
bt = pd.read_csv('out/bertopic_assign.csv').merge(df[['doc_id', 'venue', 'year']])
bt['area'] = bt.bt.map(lambda t: BTL[t][1]); bt['name'] = bt.bt.map(lambda t: BTL[t][0])
ven = bt.groupby(['venue', 'area']).size().unstack(fill_value=0); ven = ven.div(ven.sum(1), axis=0)
order = ['popl', 'pldi', 'icfp', 'oopsla', 'esop', 'ecoop', 'toplas', 'jfp', 'haskell', 'dls', 'onward', 'sle', 'gpce', 'hopl', 'cc', 'cgo', 'ppopp', 'ismm', 'sas', 'vmcai', 'aplas', 'cav', 'tacas', 'cpp']
ven = ven.loc[[o for o in order if o in ven.index]]
acols = ven.max().sort_values(ascending=False).index.tolist()
R['bt_heat'] = dict(venues=ven.index.tolist(), areas=acols, z=ven[acols].round(4).values.tolist(), n=bt.groupby('venue').size().loc[ven.index].tolist())
early = bt[bt.year.between(2006, 2015)].name.value_counts(normalize=True); late = bt[bt.year.between(2021, 2026)].name.value_counts(normalize=True)
g = pd.DataFrame({'early': early, 'late': late}).fillna(0); g['n'] = bt.name.value_counts()
g['ratio'] = (g.late + 1e-3) / (g.early + 1e-3); g = g[g.n >= 60]
R['bt_growth'] = dict(up=g.sort_values('ratio', ascending=False).head(12).round(4).reset_index(names='name').to_dict('records'),
                      down=g.sort_values('ratio').head(12).round(4).reset_index(names='name').to_dict('records'))
R['bt_ntopics'] = int(bt.bt.nunique()); R['bt_outlier_frac'] = 0.553
# agreement between models on core-4 (hard assignments)
lda = np.load('out/lda20_theta_core4.npy').argmax(1); stm = th.argmax(1)
btc = bt.set_index('doc_id').loc[c4.doc_id].bt.values
R['agree'] = {'LDA20–STM40': round(nmi(lda, stm), 3), 'LDA20–BERTopic': round(nmi(lda, btc), 3), 'STM40–BERTopic': round(nmi(stm, btc), 3)}
json.dump(R, open('out/report_data.json', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps({k: R[k] for k in ['agree', 'oopsla_did', 'venue_acc_by_venue']}, indent=1, default=str))
print(R['bt_growth']['up'][:8]); print(R['bt_growth']['down'][:8])
print(sorted(pp, key=lambda r: r['late'] - r['early'])[:4], sorted(pp, key=lambda r: r['late'] - r['early'])[-4:])
