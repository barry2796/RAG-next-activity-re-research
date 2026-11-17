"""Mục C: baseline tra cứu không dùng LLM, trên đúng ngữ cảnh truy xuất của từng câu.

  (i)  lấy "Next activity" của prefix truy xuất đầu tiên (top-1);
  (ii) lấy "Next activity" của prefix trùng chuỗi hoạt động với câu hỏi nếu có, nếu không thì theo (i).
So với accuracy của LLM trên cùng câu hỏi: CI Wilson và McNemar chính xác theo cặp.
Đúng/sai của quy tắc = nhãn quy tắc trùng khớp chính xác với Expected Answer.
Đầu ra: analysis/out/c_lookup_baselines.json, analysis/out/c_lookup_baselines.md
"""
import json
import os

from common import OUT, RAG_RUNS, fmt_ci, mcnemar_exact, parse_run


def rule_top1(r):
    return r['docs'][0]['label'] if r['docs'] else None


def rule_exact(r):
    for d in r['docs']:
        if d['seq'] == r['q_seq']:
            return d['label']
    return rule_top1(r)


res = {}
lines = ['| Lượt | N | LLM | (i) top-1 | p (i) vs LLM [b/c] | (ii) trùng chuỗi, nếu không thì top-1 | p (ii) vs LLM [b/c] |',
         '|---|---|---|---|---|---|---|']
for run in RAG_RUNS:
    _, _, recs, _ = parse_run(run)
    n = len(recs)
    llm = [r['correct'] for r in recs]
    out = {'N': n, 'llm': sum(llm)}
    row = [run, str(n), fmt_ci(sum(llm), n)]
    for name, rule in [('top1', rule_top1), ('exact', rule_exact)]:
        ok = [rule(r) == r['expected'].strip() for r in recs]
        b = sum(1 for x, y in zip(ok, llm) if x and not y)   # chỉ quy tắc đúng
        c = sum(1 for x, y in zip(ok, llm) if y and not x)   # chỉ LLM đúng
        p = mcnemar_exact(b, c)
        out[name] = {'correct': sum(ok), 'only_rule': b, 'only_llm': c, 'p': p}
        row += [fmt_ci(sum(ok), n), f'{p:.2g} [{b}/{c}]']
    res[run] = out
    lines.append('| ' + ' | '.join(row) + ' |')

json.dump(res, open(os.path.join(OUT, 'c_lookup_baselines.json'), 'w', encoding='utf-8'), indent=1)
open(os.path.join(OUT, 'c_lookup_baselines.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
