"""Fetch OpenAlex ids + referenced_works for every pldb paper with a DOI (batched, 50 DOIs/request)."""
import json, gzip, time, urllib.request, urllib.parse, os
rows = [json.loads(l) for l in gzip.open('data/papers.jsonl.gz')]
dois = sorted({r['doi'].lower() for r in rows if r['doi']})
out = 'data/openalex.jsonl'
done = set()
if os.path.exists(out):
    for l in open(out): done.add(json.loads(l)['doi'])
todo = [d for d in dois if d not in done]
print(len(dois), 'dois;', len(todo), 'to fetch')
f = open(out, 'a')
for i in range(0, len(todo), 50):
    batch = todo[i:i+50]
    flt = 'doi:' + '|'.join(batch)
    url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({
        'filter': flt, 'per-page': 50, 'mailto': 'pldb-reanalysis@example.org',
        'select': 'id,doi,referenced_works,cited_by_count,publication_year'})
    for attempt in range(5):
        try:
            res = json.load(urllib.request.urlopen(url, timeout=60)); break
        except Exception as e:
            print('retry', e); time.sleep(2 ** attempt)
    else:
        continue
    got = set()
    for w in res['results']:
        d = (w.get('doi') or '').replace('https://doi.org/', '').lower()
        got.add(d)
        f.write(json.dumps({'doi': d, 'oa': w['id'].rsplit('/', 1)[-1], 'refs': [x.rsplit('/', 1)[-1] for x in w['referenced_works']],
                            'cited_by': w['cited_by_count']}) + '\n')
    for d in batch:
        if d not in got: f.write(json.dumps({'doi': d, 'oa': None}) + '\n')
    f.flush()
    if i % 2000 == 0: print(i, flush=True)
    time.sleep(0.15)
