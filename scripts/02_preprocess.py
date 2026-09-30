"""Build the analysis corpus: one document per paper with an abstract (title + abstract).
Tokenisation mirrors Greenberg/Fisher/Walker (WordNet lemmatiser, hyphen-joining, English stopwords
+ words in >25% of docs + hand-picked filler words), plus automatic bigram phrase detection."""
import json, gzip, re, html, collections, pickle
import numpy as np, pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

rows = [json.loads(l) for l in gzip.open('data/papers.jsonl.gz')]
df = pd.DataFrame(rows)
df = df[df.abstract.notna() & (df.abstract.str.len() > 100)].copy()
def clean(s):
    s = html.unescape(s)
    s = re.sub(r'<[^>]+>', ' ', s)               # jats/html tags
    s = re.sub(r'\$[^$]*\$', ' ', s)             # inline latex
    s = re.sub(r'^\s*abstract[:.\s]+', '', s, flags=re.I)
    return re.sub(r'\s+', ' ', s).strip()
df['abstract'] = df.abstract.map(clean)
df['title'] = df.title.map(lambda t: clean(t).rstrip('.'))
# ~170 'abstracts' are scraped ACM DL page chrome (citation alerts, 'View Profile', ...) rather than text
junk = df.abstract.str.contains('Citation Alert|View Profile|Save to Binder|Get Access|Authors Info', case=False, regex=True)
print('dropping', junk.sum(), 'scraped-page pseudo-abstracts'); df = df[~junk]
df = df.drop_duplicates('title').reset_index(drop=True)
TOP4 = {'popl', 'pldi', 'icfp', 'oopsla'}
df['core4'] = df.venue.isin(TOP4)
df['text'] = df.title + '. ' + df.abstract
print(len(df), 'docs;', df.core4.sum(), 'in ICFP/OOPSLA/PLDI/POPL')

lem = WordNetLemmatizer()
stop = set(stopwords.words('english')) | set('''easily sophisticated interesting via also paper present show use
using used new approach result propose based well however work provide study one two three first many
may often such different make allow e g i e et al can thus several within without additional well known
novel way number describe existing technique method demonstrate introduce well-known moreover'''.split())
def toks(t):
    t = t.lower()
    t = re.sub(r'(\w)-(\w)', r'\1\2', t); t = re.sub(r'(\w)-(\w)', r'\1\2', t)   # object-oriented -> objectoriented
    out = []
    for w in re.findall(r'[a-z][a-z0-9]+', t):
        w = lem.lemmatize(w)
        if w not in stop and len(w) > 2 and not w.isdigit(): out.append(w)
    return out
docs = [toks(t) for t in df.text]
bigram = Phrases(docs, min_count=15, threshold=0.5, scoring='npmi', connector_words=ENGLISH_CONNECTOR_WORDS)
docs = [bigram[d] for d in docs]
N = len(docs); dfreq = collections.Counter(w for d in docs for w in set(d))
vocab = sorted(w for w, c in dfreq.items() if 5 <= c <= 0.25 * N)
print('vocab', len(vocab), 'dropped >25%:', [w for w, c in dfreq.items() if c > 0.25 * N])
V = set(vocab)
df['tokens'] = [' '.join(w for w in d if w in V) for d in docs]
df['ntok'] = df.tokens.str.split().map(len)
df = df[df.ntok >= 10].reset_index(drop=True)
df['doc_id'] = np.arange(len(df))
print(len(df), 'docs after min-length filter')
print(df.groupby('venue').size().to_string())
df.drop(columns=['opentoc', 'url']).to_pickle('data/corpus.pkl')
df[['doc_id', 'key', 'venue', 'year', 'core4', 'title', 'tokens']].to_csv('data/corpus_tokens.csv', index=False)
