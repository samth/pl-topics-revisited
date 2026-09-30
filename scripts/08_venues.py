"""How do the four conferences differ, and are they converging?  (a) Jensen-Shannon divergence between venue
topic mixtures per year, (b) topic diversity (effective number of topics), (c) cross-fitted classifier on SPECTER2
embeddings: how identifiable is a paper's venue from its abstract, per year, (d) changepoints in OOPSLA's OO share."""
import sys, numpy as np, pandas as pd, itertools, ruptures as rpt
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
K = int(sys.argv[1])
df = pd.read_pickle('data/corpus.pkl')
c4 = df[df.core4].reset_index(drop=True)
th = pd.read_csv(f'out/stm_K{K}_theta.csv').set_index('doc_id').loc[c4.doc_id].values
rng = np.random.default_rng(1)
V = ['icfp', 'oopsla', 'pldi', 'popl']
# (a)+(b) with 3-year windows and bootstrap over papers
rows = []
for y in range(1975, 2027):
    w = (c4.year >= y - 1) & (c4.year <= y + 1)
    groups = {v: np.where(w & (c4.venue == v))[0] for v in V}
    groups = {v: g for v, g in groups.items() if len(g) >= 15}
    for b in range(200):
        mix = {v: th[g[rng.integers(0, len(g), len(g))] if b else g].mean(0) for v, g in groups.items()}
        for v, m in mix.items():
            rows.append(dict(year=y, b=b, kind='diversity', a=v, b_=v, val=np.exp(-(m * np.log(m)).sum())))
        for a, c in itertools.combinations(sorted(mix), 2):
            rows.append(dict(year=y, b=b, kind='jsd', a=a, b_=c, val=jensenshannon(mix[a], mix[c], base=2) ** 2))
r = pd.DataFrame(rows)
s = r[r.b == 0].merge(r[r.b > 0].groupby(['year', 'kind', 'a', 'b_']).val.quantile([.025, .975]).unstack().reset_index().rename(columns={0.025: 'lo', 0.975: 'hi'}), on=['year', 'kind', 'a', 'b_'])
s.drop(columns='b').to_csv('out/venue_jsd_diversity.csv', index=False)

# (c) venue identifiability from SPECTER2 embeddings, 10-fold cross-fitted, with year as no feature
E = normalize(np.load('data/emb_specter2.npy'))[c4.doc_id.values]
y = c4.venue.values; P = np.zeros((len(c4), 4))
for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(E, y):
    clf = LogisticRegression(C=2.0, max_iter=3000, class_weight='balanced').fit(E[tr], y[tr])
    P[te] = clf.predict_proba(E[te])
c4['p_true'] = P[np.arange(len(c4)), [list(clf.classes_).index(v) for v in y]]
c4['correct'] = np.array(clf.classes_)[P.argmax(1)] == y
for i, v in enumerate(clf.classes_): c4['p_' + v] = P[:, i]
c4[['doc_id', 'venue', 'year', 'p_true', 'correct'] + ['p_' + v for v in clf.classes_]].to_csv('out/venue_classifier.csv', index=False)
print('overall 4-way accuracy', c4.correct.mean().round(3), '(chance with balanced classes = .25)')
print(c4.groupby('venue').correct.mean().round(3).to_dict())
# pairwise POPL-vs-PLDI separability per 3-year window (AUC of cross-fitted classifier)
from sklearn.metrics import roc_auc_score
pp = []
m = c4.venue.isin(['popl', 'pldi']).values
Ep, yp = E[m], (c4.venue.values[m] == 'popl').astype(int); yr = c4.year.values[m]; sc = np.zeros(m.sum())
for tr, te in StratifiedKFold(10, shuffle=True, random_state=0).split(Ep, yp):
    sc[te] = LogisticRegression(C=2.0, max_iter=3000).fit(Ep[tr], yp[tr]).decision_function(Ep[te])
for y0 in range(1990, 2026, 1):
    w = (yr >= y0 - 2) & (yr <= y0 + 2)
    aucs = []
    idx = np.where(w)[0]
    for b in range(200):
        ii = idx if b == 0 else rng.choice(idx, len(idx))
        if len(set(yp[ii])) == 2: aucs.append(roc_auc_score(yp[ii], sc[ii]))
    pp.append(dict(year=y0, auc=aucs[0], lo=np.percentile(aucs[1:], 2.5), hi=np.percentile(aucs[1:], 97.5)))
pd.DataFrame(pp).to_csv('out/popl_pldi_auc.csv', index=False)

# (d) changepoints (PELT, BIC-style penalty) in OOPSLA's yearly topic shares for every topic
oo = c4.venue == 'oopsla'
cp = []
yrs = sorted(c4.year[oo].unique())
for k in range(K):
    ser = np.array([th[(oo & (c4.year == yy)).values, k].mean() for yy in yrs])
    sig = np.std(np.diff(ser)) / np.sqrt(2)
    bk = rpt.Pelt(model='l2', min_size=3, jump=1).fit(ser).predict(pen=2 * np.log(len(ser)) * sig ** 2 * 3)
    cp.append(dict(topic=k + 1, breaks=[int(yrs[b]) for b in bk[:-1]]))
pd.DataFrame(cp).to_csv('out/oopsla_changepoints.csv', index=False); print(pd.DataFrame(cp).to_string())
