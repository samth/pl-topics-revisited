"""Has separation logic shifted the kinds of papers at POPL?  Three independent signals per venue-year:
(1) share of papers citing a founding separation-logic paper (OpenAlex), (2) share whose abstract says
'separation logic', (3) STM prevalence of the separation-logic topic.  Plus a Bayesian segmented (hinge)
logistic regression for the citing share at POPL, and the topic mix of SL-citing vs other POPL papers."""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
K = int(sys.argv[1]); SLT = int(sys.argv[2])     # STM topic number (1-based) that is separation logic
df = pd.read_pickle('data/corpus.pkl'); cit = pd.read_pickle('out/citations.pkl')
oa = {json.loads(l)['doi']: json.loads(l) for l in open('data/openalex.jsonl')}
refs = df.doi.str.lower().map(lambda d: set(oa.get(d, {}).get('refs', []) or []) if isinstance(d, str) else set())
founding = {'W2137628566': 'Reynolds LICS02', 'W1495368328': "O'Hearn-Reynolds-Yang CSL01", 'W4256027310': "Ishtiaq-O'Hearn POPL01",
            'W2132761501': "O'Hearn CONCUR04/TCS07"}
bi = df.index[df.key == 'conf/popl/IshtiaqO01']
if len(bi): founding[cit.oa[bi[0]]] = "Ishtiaq-O'Hearn POPL01"
df['cites_sl'] = refs.map(lambda r: bool(r & set(founding)))
df['has_refs'] = refs.map(len) > 0
df['mentions_sl'] = df.text.str.contains(r'separation[ -]logic', case=False)
th = pd.read_csv(f'out/stm_K{K}_theta.csv').set_index('doc_id')
df['sl_topic'] = df.doc_id.map(th[f'T{SLT:02d}'])
g = df[df.has_refs].groupby(['venue', 'year']).agg(n=('doc_id', 'size'), cites=('cites_sl', 'mean'), mentions=('mentions_sl', 'mean'), topic=('sl_topic', 'mean')).reset_index()
g.to_csv('out/seplogic_by_venue_year.csv', index=False)
# segmented logistic regression at POPL, 1995-2026: logit p = a + b*year + c*max(0, year - tau); profile tau
p = df[(df.venue == 'popl') & df.has_refs & (df.year >= 1995)]
best = []
for tau in range(1998, 2012):
    X = sm.add_constant(np.c_[p.year - 2000, np.maximum(0, p.year - tau)])
    m = sm.GLM(p.cites_sl.astype(float), X, family=sm.families.Binomial()).fit()
    best.append((m.llf, tau, m))
ll = np.array([b[0] for b in best]); post = np.exp(ll - ll.max()); post /= post.sum()
pd.DataFrame({'tau': [b[1] for b in best], 'posterior': post}).to_csv('out/seplogic_changepoint.csv', index=False)
print('changepoint posterior (flat prior over tau):', dict(zip([b[1] for b in best], post.round(3))))
# which topics do SL-citing POPL papers occupy, versus other POPL papers of the same era (2005+)?
q = df[(df.venue == 'popl') & (df.year >= 2005) & df.has_refs & df.core4]
T = th.loc[q.doc_id].values
diff = T[q.cites_sl.values].mean(0) - T[~q.cites_sl.values].mean(0)
pd.DataFrame({'topic': np.arange(1, K + 1), 'sl_citing': T[q.cites_sl.values].mean(0), 'other': T[~q.cites_sl.values].mean(0), 'diff': diff}).to_csv('out/seplogic_topicmix.csv', index=False)
print('POPL 2005+: n SL-citing', q.cites_sl.sum(), 'of', len(q))
print(g[g.venue == 'popl'].tail(25).round(3).to_string())
