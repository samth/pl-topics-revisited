"""Inject out/report_data.json into report/template.html -> report/pl-topics-revisited.html"""
import json
d = json.load(open('out/report_data.json'))
t = open('report/template.html').read()
open('report/pl-topics-revisited.html', 'w').write(t.replace('/*DATA*/null', json.dumps(d, separators=(',', ':')).replace('</', '<\\/')))
