"""(a) Replicate the paper: LDA k=20 on ICFP/OOPSLA/PLDI/POPL abstracts.
(b) Model selection the paper lacked: k sweep scored by held-out perplexity and NPMI coherence."""
import pandas as pd, numpy as np, json, pickle
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel
df = pd.read_pickle('data/corpus.pkl')
c4 = df[df.core4].reset_index(drop=True)
texts = [t.split() for t in c4.tokens]
dic = Dictionary(texts); dic.filter_extremes(no_below=5, no_above=1.0)
bow = [dic.doc2bow(t) for t in texts]
rng = np.random.default_rng(0); perm = rng.permutation(len(bow)); ntest = len(bow) // 10
test = [bow[i] for i in perm[:ntest]]; train = [bow[i] for i in perm[ntest:]]
res = []
for k in ([] if len(__import__("sys").argv) > 1 else [10, 20, 30, 40, 50, 60, 80, 100]):
    for seed in [1, 2]:
        m = LdaMulticore(train, id2word=dic, num_topics=k, passes=15, iterations=200, random_state=seed, workers=2, alpha='asymmetric', eta='auto' if False else None, chunksize=2000)
        pp = m.log_perplexity(test)
        top = [[w for w, _ in m.show_topic(i, 10)] for i in range(k)]
        npmi = CoherenceModel(topics=top, texts=texts, dictionary=dic, coherence='c_npmi').get_coherence()
        cv = CoherenceModel(topics=top, texts=texts, dictionary=dic, coherence='c_v').get_coherence()
        uniq = len({w for t in top for w in t}) / (10 * k)   # topic diversity
        res.append(dict(k=k, seed=seed, heldout_bound_per_word=pp, npmi=npmi, c_v=cv, diversity=uniq))
        print(res[-1], flush=True)
if res: pd.DataFrame(res).to_csv('out/lda_ksweep.csv', index=False)
# the replication model: k=20 on all core-4 docs
m = LdaMulticore(bow, id2word=dic, num_topics=20, passes=20, iterations=400, random_state=7, workers=2)
theta = np.zeros((len(bow), 20))
for i, b in enumerate(bow):
    for t, p in m.get_document_topics(b, minimum_probability=0): theta[i, t] = p
np.save('out/lda20_theta_core4.npy', theta)
m.save('out/lda20.model'); dic.save('out/lda20.dict')
with open('out/lda20_topics.txt', 'w') as f:
    for i in range(20): f.write(f"{i}: " + ', '.join(w for w, _ in m.show_topic(i, 12)) + '\n')
print(open('out/lda20_topics.txt').read())
