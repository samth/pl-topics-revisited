"""Embedding-based topic model over all 24 venues: SPECTER2 -> UMAP -> HDBSCAN -> class-TF-IDF (BERTopic).
Outlier documents are re-assigned by embedding similarity to topic centroids so every paper has a topic."""
import numpy as np, pandas as pd, pickle
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_pickle('data/corpus.pkl')
E = np.load('data/emb_specter2.npy')
En = E / np.linalg.norm(E, axis=1, keepdims=True)
umap = UMAP(n_neighbors=30, n_components=8, min_dist=0.0, metric='cosine', random_state=11)
hdb = HDBSCAN(min_cluster_size=40, min_samples=10, metric='euclidean', cluster_selection_method='leaf', prediction_data=True)
vec = CountVectorizer(token_pattern=r'\S+', min_df=5)            # use our lemmatised tokens for c-TF-IDF
tm = BERTopic(umap_model=umap, hdbscan_model=hdb, vectorizer_model=vec,
              representation_model=MaximalMarginalRelevance(diversity=0.3), top_n_words=12, calculate_probabilities=False, verbose=True)
topics, _ = tm.fit_transform(df.tokens.tolist(), embeddings=En)
print('raw topics', len(set(topics)) - 1, 'outliers', (np.array(topics) == -1).mean())
topics = tm.reduce_outliers(df.tokens.tolist(), topics, strategy='embeddings', embeddings=En)
tm.update_topics(df.tokens.tolist(), topics=topics, vectorizer_model=vec, representation_model=MaximalMarginalRelevance(diversity=0.3), top_n_words=12)
df['bt'] = topics
info = tm.get_topic_info(); info.to_csv('out/bertopic_info.csv', index=False)
# hierarchical merge tree -> coarse level with ~25 super-topics for plotting
hier = tm.hierarchical_topics(df.tokens.tolist()); hier.to_csv('out/bertopic_hierarchy.csv', index=False)
pd.DataFrame({'doc_id': df.doc_id, 'bt': topics}).to_csv('out/bertopic_assign.csv', index=False)
xy = UMAP(n_neighbors=30, n_components=2, min_dist=0.1, metric='cosine', random_state=11).fit_transform(En)
np.save('out/umap2d.npy', xy)
tm.save('out/bertopic_model', serialization='pickle', save_ctfidf=True)
print(info.head(60).to_string())
