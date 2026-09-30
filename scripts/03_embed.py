"""SPECTER2 (proximity adapter) embeddings of title [SEP] abstract, plus a MiniLM embedding for BERTopic's KeyBERT-style reps."""
import numpy as np, pandas as pd, torch, time
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
torch.set_num_threads(3)
df = pd.read_pickle('data/corpus.pkl')
tok = AutoTokenizer.from_pretrained('allenai/specter2_base')
m = AutoAdapterModel.from_pretrained('allenai/specter2_base')
m.load_adapter('allenai/specter2', source='hf', load_as='proximity', set_active=True)
m.eval()
texts = (df.title + tok.sep_token + df.abstract).tolist()
order = np.argsort([len(t) for t in texts])   # length-bucketing for speed
E = np.zeros((len(texts), 768), dtype=np.float32)
bs = 32
with torch.inference_mode():
    for i in range(0, len(texts), bs):
        idx = order[i:i+bs]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512, return_tensors='pt', return_token_type_ids=False)
        E[idx] = m(**enc).last_hidden_state[:, 0, :].numpy()
        if i % 320 == 0: print(i, time.strftime("%X"), flush=True)
np.save('data/emb_specter2.npy', E)
print('done')
import pickle; pickle.dump(dict(zip(df.key, E)), open('data/emb_specter2_bykey.pkl', 'wb'))
