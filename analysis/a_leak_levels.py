"""Mục A: tách "rò rỉ" khỏi "trùng khớp hợp lệ".

Với mỗi câu của mỗi lượt có RAG, phân loại ngữ cảnh truy xuất (prompt thực tế trong file kết quả):
  (a) có prefix trùng cả chuỗi hoạt động lẫn toàn bộ giá trị thuộc tính với câu hỏi;
  (b) không có (a), nhưng có prefix trùng chuỗi hoạt động kèm Next activity = đáp án đúng;
  (c) còn lại.
Báo cáo số câu, accuracy của LLM và CI Wilson theo từng mức.
Đầu ra: analysis/out/a_leak_levels.json, analysis/out/a_leak_levels.md
"""
import json
import os

from common import OUT, RAG_RUNS, context_level, fmt_ci, parse_run

res = {}
lines = ['| Lượt | N | (a) n | acc (a) | (b) n | acc (b) | (c) n | acc (c) | (a)+(b) | trùng chuỗi nhưng sai nhãn |',
         '|---|---|---|---|---|---|---|---|---|---|']
for run in RAG_RUNS:
    _, _, recs, _ = parse_run(run)
    lv = {'a': [], 'b': [], 'c': []}
    wrong_label_seq = 0
    a_wrong_label = 0
    for r in recs:
        L = context_level(r)
        lv[L].append(r['correct'])
        same = [d for d in r['docs'] if d['seq'] == r['q_seq']]
        if same and not any(d['label'] == r['expected'].strip() for d in same):
            wrong_label_seq += 1
        if L == 'a' and not any(d['label'] == r['expected'].strip() and d['vals'] == r['q_vals'] for d in same):
            a_wrong_label += 1
    n = len(recs)
    res[run] = {'N': n, 'wrong_label_same_seq': wrong_label_seq, 'a_with_other_label': a_wrong_label}
    row = [run, str(n)]
    for L in 'abc':
        k, x = len(lv[L]), sum(lv[L])
        res[run][L] = {'n': k, 'correct': x}
        row += [str(k), fmt_ci(x, k) if k else '-']
    ab = len(lv['a']) + len(lv['b'])
    res[run]['a+b'] = ab
    row += [f'{ab} ({ab/n:.0%})', str(wrong_label_seq)]
    lines.append('| ' + ' | '.join(row) + ' |')

# Accuracy không RAG trên đúng các câu của từng mức (so sánh theo cặp, cùng câu hỏi)
PAIRS = {'2.5': '2.6', '3.1a': '3.1b', '3.3a': '3.3b', '3.2a': '3.2b'}
lines += ['', 'Không RAG trên cùng câu hỏi, theo mức ngữ cảnh của lượt có RAG (n đúng/n; McNemar có RAG vs không RAG)', '',
          '| Cặp | (a) có RAG | (a) không RAG | p | (b) có RAG | (b) không RAG | p | (c) có RAG | (c) không RAG | p |',
          '|---|---|---|---|---|---|---|---|---|---|']
from common import mcnemar_exact  # noqa: E402
for rag, norag in PAIRS.items():
    _, _, ra, _ = parse_run(rag)
    _, _, rb, _ = parse_run(norag)
    assert all(x['question'] == y['question'] and x['expected'] == y['expected'] for x, y in zip(ra, rb))
    row = [f'{rag}/{norag}']
    res[rag]['norag_by_level'] = {}
    for L in 'abc':
        idx = [i for i, r in enumerate(ra) if context_level(r) == L]
        k1 = sum(ra[i]['correct'] for i in idx)
        k0 = sum(rb[i]['correct'] for i in idx)
        b = sum(1 for i in idx if ra[i]['correct'] and not rb[i]['correct'])
        c = sum(1 for i in idx if rb[i]['correct'] and not ra[i]['correct'])
        p = mcnemar_exact(b, c)
        res[rag]['norag_by_level'][L] = {'n': len(idx), 'rag': k1, 'norag': k0, 'only_rag': b, 'only_norag': c, 'p': p}
        row += [f'{k1}/{len(idx)}', f'{k0}/{len(idx)}', f'{p:.2g}' if idx else '-']
    lines.append('| ' + ' | '.join(row) + ' |')

json.dump(res, open(os.path.join(OUT, 'a_leak_levels.json'), 'w', encoding='utf-8'), indent=1)
open(os.path.join(OUT, 'a_leak_levels.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(json.dumps({k: (v['wrong_label_same_seq'], v['a_with_other_label']) for k, v in res.items()}))
