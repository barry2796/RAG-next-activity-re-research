"""Mục E: kiểm chứng định lượng bản sửa B1.

Với mỗi nhật ký, tính tỉ lệ câu kiểm thử có hoạt động cuối của prefix trùng nhãn:
  - tập kiểm thử TRƯỚC khi sửa: file CSV tác giả commit kèm mã công khai (git show 0af4ffe:tests/test_sets/...;
    các file này giống hệt bản trong rag_next_activity.zip);
  - tập kiểm thử SAU khi sửa: file CSV hiện tại (tập mà mọi lượt chạy của nhóm dùng).
So với tỉ lệ lặp tự nhiên trong nhật ký, tính bằng chính các hàm của src/log_preprocessing.py (đã sửa B1):
  - trên quần thể prefix của kho (build_prefixes b=1, g=3 + process_prefixes trên mọi trace),
  - và trên mọi cặp hoạt động liên tiếp (directly-follows) trong nhật ký.
Nguồn dữ liệu: logs/*.xes (dữ liệu gốc), tests/test_sets/*.csv, lịch sử git. Không gọi LLM.
Đầu ra: analysis/out/e_b1_check.json, analysis/out/e_b1_check.md
"""
import contextlib
import csv
import io
import json
import os
import subprocess
import sys

from common import OUT, ROOT, TESTSETS, TESTSET_OF_LOG

sys.path.insert(0, os.path.join(ROOT, 'src'))
import log_preprocessing as lp  # noqa: E402

XES = {'sepsis': 'sepsis.xes', 'bpic20': 'bpic20_international_declarations.xes',
       'bpic13': 'BPIC_2013_closed_problems.xes', 'synthetic13': 'synthetic-13-multivariant.xes'}


def rate_csv_text(text):
    rows = list(csv.reader(io.StringIO(text)))[1:]
    same = sum(1 for p, lab in rows if p.rsplit(' - Values: ', 1)[0].split(',')[-1].strip() == lab.strip())
    return same, len(rows)



res = {}
lines = ['| Nhật ký | Trước sửa: hoạt động cuối = nhãn | Sau sửa: hoạt động cuối = nhãn | Lặp tự nhiên trong kho (prefix duy nhất) | Lặp tự nhiên, mọi cặp liên tiếp |',
         '|---|---|---|---|---|']
for log, csvname in TESTSET_OF_LOG.items():
    pre_txt = subprocess.run(['git', '-C', ROOT, 'show', f'0af4ffe:tests/test_sets/{csvname}'],
                             capture_output=True).stdout.decode('cp1252')
    pre = rate_csv_text(pre_txt)
    post = rate_csv_text(open(os.path.join(TESTSETS, csvname), encoding='cp1252', newline='').read())
    with contextlib.redirect_stdout(io.StringIO()):
        content = lp.read_event_log(XES[log])
        traces = lp.extract_traces(content)
        store, _, _ = lp.process_prefixes(lp.build_prefixes(traces, 1, 3), XES[log])
    st_same = sum(1 for k, v in store.items() if k.split(',')[-1] == v.split('| Next activity: ')[1].strip())
    # mọi cặp liên tiếp: dùng process_prefixes trên từng trace hoàn chỉnh (thêm một sự kiện giả ở cuối)
    pair_same = pair_n = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for trace in traces:
            r, _, _ = lp.process_prefixes([trace + '<event><string key="concept:name" value="__END__"/></event>'], XES[log])
            for k in r:
                acts = k.split(',')
                pair_n += len(acts) - 1
                pair_same += sum(1 for a, b in zip(acts, acts[1:]) if a == b)
    res[log] = {'pre_fix': pre, 'post_fix': post, 'store_same': [st_same, len(store)], 'pairs_same': [pair_same, pair_n]}
    lines.append(f'| {log} | {pre[0]}/{pre[1]} ({pre[0]/pre[1]:.1%}) | {post[0]}/{post[1]} ({post[0]/post[1]:.1%}) | '
                 f'{st_same}/{len(store)} ({st_same/len(store):.1%}) | {pair_same}/{pair_n} ({pair_same/pair_n:.1%}) |')
    print(lines[-1], flush=True)

json.dump(res, open(os.path.join(OUT, 'e_b1_check.json'), 'w', encoding='utf-8'), indent=1)
open(os.path.join(OUT, 'e_b1_check.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
