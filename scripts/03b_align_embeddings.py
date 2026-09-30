"""Re-align cached SPECTER2 embeddings (keyed by dblp key) to the current corpus row order; embed any missing."""
import numpy as np, pandas as pd, pickle, torch
df = pd.read_pickle('data/corpus.pkl'); d = pd.read_pickle('data/emb_specter2_bykey.pkl')
miss = df[~df.key.isin(d)]
print('missing', len(miss))
if len(miss):
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
    tok = AutoTokenizer.from_pretrained('allenai/specter2_base')
    m = AutoAdapterModel.from_pretrained('allenai/specter2_base')
    m.load_adapter('allenai/specter2', source='hf', load_as='proximity', set_active=True); m.eval()
    with torch.inference_mode():
        for k, t, a in zip(miss.key, miss.title, miss.abstract):
            enc = tok([t + tok.sep_token + a], truncation=True, max_length=512, return_tensors='pt', return_token_type_ids=False)
            d[k] = m(**enc).last_hidden_state[:, 0, :].numpy()[0]
    pickle.dump(d, open('data/emb_specter2_bykey.pkl', 'wb'))
np.save('data/emb_specter2.npy', np.stack([d[k] for k in df.key]))
