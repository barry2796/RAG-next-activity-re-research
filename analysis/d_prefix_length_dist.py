"""Mục D: phân bố độ dài prefix của toàn bộ tập kiểm thử (300 câu) so với N câu đầu đã đánh giá.

Nguồn: tests/test_sets/*evaluation-attributes.csv (tập kiểm thử đầy đủ mà các lượt dùng) và file kết quả.
Script kiểm tra N câu trong file kết quả trùng đúng N dòng đầu của CSV (cùng thứ tự) trước khi đếm.
Độ dài prefix = số hoạt động trước " - Values:" (trùng với "Prefix Length" ghi trong file kết quả).
Đầu ra: analysis/out/d_prefix_length_dist.json, analysis/out/d_prefix_length_dist.md
"""
import csv
import json
import os
from collections import Counter

from common import BUCKETS, OUT, TESTSETS, TESTSET_OF_LOG, bucket, parse_run

PAIRS = [('3.1a', 'bpic20', 'BPIC 2020 Int. Decl.'), ('3.3a', 'bpic13', 'BPIC 2013 Closed Prob.'),
         ('3.2a', 'synthetic13', 'synthetic-13-multivariant'), ('2.5', 'sepsis', 'Sepsis Cases')]
res = {}
lines = ['| Nhật ký | Tập | N | ' + ' | '.join(BUCKETS) + ' |', '|---|---|---|' + '---|' * len(BUCKETS)]
for run, log, name in PAIRS:
    rows = list(csv.reader(open(os.path.join(TESTSETS, TESTSET_OF_LOG[log]), encoding='cp1252', newline='')))[1:]
    _, _, recs, _ = parse_run(run)
    n = len(recs)
    same_order = all(rows[i][0].strip() == recs[i]['question'].strip() and rows[i][1].strip() == recs[i]['expected'].strip()
                     for i in range(n))
    full = [len(r[0].rsplit(' - Values: ', 1)[0].split(',')) for r in rows]
    plen_ok = all(full[i] == recs[i]['plen'] for i in range(n))
    cf = Counter(bucket(x) for x in full)
    cn = Counter(bucket(r['plen']) for r in recs)
    res[log] = {'run': run, 'N_eval': n, 'N_full': len(full), 'first_N_match_csv': same_order,
                'plen_match': plen_ok, 'full': dict(cf), 'firstN': dict(cn)}
    lines.append(f'| {name} | toàn bộ | {len(full)} | ' + ' | '.join(str(cf.get(b, 0)) for b in BUCKETS) + ' |')
    if n != len(full):
        lines.append(f'| {name} | {n} câu đầu | {n} | ' + ' | '.join(str(cn.get(b, 0)) for b in BUCKETS) + ' |')

json.dump(res, open(os.path.join(OUT, 'd_prefix_length_dist.json'), 'w', encoding='utf-8'), indent=1)
open(os.path.join(OUT, 'd_prefix_length_dist.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print({k: (v['first_N_match_csv'], v['plen_match']) for k, v in res.items()})
