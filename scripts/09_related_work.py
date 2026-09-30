"""'Finding your friends' at scale: are cited papers closer than random ones, and which representation
retrieves a paper's actual (in-corpus) references best?  Citation links come from OpenAlex."""
import json, sys, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.stats import rankdata
K = int(sys.argv[1])
df = pd.read_pickle('data/corpus.pkl')
oa = {}
for l in open('data/openalex.jsonl'):
    r = json.loads(l)
    if r.get('oa'): oa[r['doi']] = r
df['oa'] = df.doi.str.lower().map(lambda d: oa.get(d, {}).get('oa') if isinstance(d, str) else None)
df['refs'] = df.doi.str.lower().map(lambda d: oa.get(d, {}).get('refs', []) if isinstance(d, str) else [])
df['cited_by'] = df.doi.str.lower().map(lambda d: oa.get(d, {}).get('cited_by', np.nan) if isinstance(d, str) else np.nan)
id2row = {o: i for i, o in enumerate(df.oa) if o}
df['cites'] = [[id2row[x] for x in refs if x in id2row and id2row[x] != i] for i, refs in enumerate(df.refs)]
df[['doc_id', 'oa', 'cited_by', 'cites']].to_pickle('out/citations.pkl')
print('docs with OpenAlex id', df.oa.notna().sum(), '; in-corpus citation edges', sum(map(len, df.cites)))

# ---- representations (all rows aligned with df) ----
reps = {}
E = normalize(np.load('data/emb_specter2.npy')); reps['SPECTER2 (cosine)'] = ('cos', E)
T = TfidfVectorizer(token_pattern=r'\S+', sublinear_tf=True, min_df=2).fit_transform(df.tokens); reps['TF-IDF (cosine)'] = ('cos', T)
c4 = np.where(df.core4)[0]
lda = np.full((len(df), 20), np.nan); lda[c4] = np.load('out/lda20_theta_core4.npy')
reps['LDA k=20, Euclidean (paper)'] = ('euc', lda)
reps['LDA k=20, Hellinger'] = ('hel', lda)
st = pd.read_csv(f'out/stm_K{K}_theta.csv').set_index('doc_id')
stm = np.full((len(df), K), np.nan); stm[df.set_index('doc_id').index.get_indexer(st.index)] = st.values
reps[f'STM k={K}, Hellinger'] = ('hel', stm)

def sims(kind, X, q, cand):
    if kind == 'cos': s = X[cand] @ X[q].T; return np.asarray(s.todense()).ravel() if hasattr(s, 'todense') else np.ravel(s)
    if kind == 'euc': return -np.linalg.norm(X[cand] - X[q], axis=1)
    if kind == 'hel': return -np.linalg.norm(np.sqrt(X[cand]) - np.sqrt(X[q]), axis=1)

def evaluate(pool_mask, names, label):
    rows = []
    pool = np.where(pool_mask)[0]
    qs = [i for i in pool if len([j for j in df.cites[i] if pool_mask[j] and df.year[j] <= df.year[i]]) >= 5]
    print(label, 'queries:', len(qs))
    for i in qs:
        cand = pool[(df.year.values[pool] <= df.year[i]) & (pool != i)]
        rel = np.isin(cand, df.cites[i])
        for n in names:
            kind, X = reps[n]
            s = sims(kind, X, i, cand)
            r = rankdata(-s, method='average')
            rr = r[rel]; npos = rel.sum(); nneg = len(cand) - npos
            auc = 1 - (rr.sum() - npos * (npos + 1) / 2) / (npos * nneg)   # P(cited ranks above random non-cited)
            rows.append(dict(eval=label, method=n, q=i, auc=auc, r20=(rr <= 20).sum() / npos, r100=(rr <= 100).sum() / npos,
                             mrr=1 / rr.min(), npos=npos, ncand=len(cand)))
    return rows
rows = evaluate(df.core4.values, list(reps), 'ICFP/OOPSLA/PLDI/POPL')
rows += evaluate(np.ones(len(df), bool), ['SPECTER2 (cosine)', 'TF-IDF (cosine)'], 'all 24 venues')
res = pd.DataFrame(rows); res.to_csv('out/related_work_eval.csv', index=False)
rng = np.random.default_rng(0)
summ = []
for (e, m), g in res.groupby(['eval', 'method'], sort=False):
    d = dict(eval=e, method=m, n=len(g))
    for c in ['auc', 'r20', 'r100', 'mrr']:
        bs = [g[c].values[rng.integers(0, len(g), len(g))].mean() for _ in range(1000)]
        d[c] = g[c].mean(); d[c + '_lo'], d[c + '_hi'] = np.percentile(bs, [2.5, 97.5])
    summ.append(d)
summ = pd.DataFrame(summ); summ.to_csv('out/related_work_summary.csv', index=False); print(summ.round(3).to_string())

# ---- Figure-9 replication: the paper's four case studies ----
cases = {'CDRS': 'conf/pldi/HawkinsAFRS12', 'PCC': 'conf/popl/Necula97', 'TAL': 'conf/popl/MorrisettWCG98', 'SEMC': 'conf/popl/Greenberg15'}
out = []
for name, key in cases.items():
    i = int(np.where(df.key == key)[0][0])
    cites = [j for j in df.cites[i] if df.core4[j]]
    pool = np.where(df.core4.values & (np.arange(len(df)) != i))[0]
    for n in ['LDA k=20, Euclidean (paper)', 'SPECTER2 (cosine)']:
        kind, X = reps[n]
        s_all = sims(kind, X, i, pool); pct = rankdata(-s_all) / len(pool)     # percentile rank of each candidate
        for j in cites: out.append(dict(paper=name, method=n, set='Citations', pct=pct[np.searchsorted(pool, j)]))
        for rs in range(5):
            for j in rng.choice(len(pool), max(len(cites), 10), replace=False): out.append(dict(paper=name, method=n, set=f'Random {rs+1}', pct=pct[j]))
    print(name, 'in-corpus core-4 citations:', len(cites))
pd.DataFrame(out).to_csv('out/fig9_cases.csv', index=False)
# the Nix 'Editing by example' anecdote: where does it rank for Osera & Zdancewic (PLDI'15)?
q = int(np.where(df.key == 'conf/pldi/OseraZ15')[0][0]); nix = int(np.where(df.key == 'conf/popl/Nix84')[0][0])
pool = np.where(df.core4.values & (df.year.values <= 2015) & (np.arange(len(df)) != q))[0]
for n in reps:
    kind, X = reps[n]
    if np.isnan(np.asarray(X[q].todense() if hasattr(X, 'todense') else X[q])).any(): continue
    s = sims(kind, X, q, pool); print('Nix84 rank for OseraZ15 under', n, ':', int(rankdata(-s)[np.searchsorted(pool, nix)]), 'of', len(pool))
