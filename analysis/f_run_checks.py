"""Phần 1, mục 4, 6, 7: các kiểm tra trên file kết quả.

Mục 4: so sánh lượt 2.5 (150 câu đầu, Qwen2.5-3B) và lượt 4.1 (Qwen2.5-1.5B) trên cùng 150 câu:
        số câu có ngữ cảnh chứa prefix trùng chuỗi kèm đáp án ở mỗi lượt, số câu có ngữ cảnh khác nhau giữa hai
        lượt, accuracy của mỗi mô hình trên giao của hai tập.
Mục 6: lượt 2.1 - có đọc được bao nhiêu bản ghi, tỉ lệ hoạt động cuối = nhãn (để xác nhận chạy sau B1),
        thứ tự danh sách hoạt động hợp lệ trong prompt của các lượt Sepsis, và các trường header có sẵn.
Mục 7: file kết quả có thời gian của từng câu hay không.
Đầu ra: analysis/out/f_run_checks.json, analysis/out/f_run_checks.md
"""
import hashlib
import json
import os
import re

from common import OUT, VALID, context_level, parse_run, run_file
from b_long_prefix import ids, max_seq  # tokenizer của mô hình nhúng

res, lines = {}, []

# ---- Mục 4
_, _, r25, _ = parse_run('2.5')
_, _, r41, _ = parse_run('4.1')
r25 = r25[:150]
assert all(a['question'] == b['question'] and a['expected'] == b['expected'] for a, b in zip(r25, r41))
hit25 = [context_level(r) in 'ab' for r in r25]
hit41 = [context_level(r) in 'ab' for r in r41]
ctx_diff = [i for i in range(150) if [d['seq'] for d in r25[i]['docs']] != [d['seq'] for d in r41[i]['docs']]]
ctx_diff_long = sum(1 for i in ctx_diff if len(ids(r25[i]['question'].split('-')[0].strip())) > max_seq)
both = [i for i in range(150) if hit25[i] and hit41[i]]
res['item4'] = {
    'hit_2.5_first150': sum(hit25), 'correct_2.5_on_hits': sum(r25[i]['correct'] for i in range(150) if hit25[i]),
    'hit_4.1': sum(hit41), 'correct_4.1_on_hits': sum(r41[i]['correct'] for i in range(150) if hit41[i]),
    'context_differs': len(ctx_diff), 'context_differs_query_over_limit': ctx_diff_long,
    'both_hit': len(both), 'correct_3B_on_both': sum(r25[i]['correct'] for i in both),
    'correct_1.5B_on_both': sum(r41[i]['correct'] for i in both),
}
lines.append('Mục 4: ' + json.dumps(res['item4']))

# ---- Mục 6
h21, m21, r21, part21 = parse_run('2.1')
same21 = sum(1 for r in r21 if r['q_seq'].split(',')[-1] == r['expected'].strip())
order = {}
for run in ['2.1', '2.3', '2.5', '2.6', '4.1', '4.3a']:
    _, _, recs, _ = parse_run(run)
    m = re.search(r'Allowed concept:names of activities for your predictions are:\s*(.*?)\n', recs[0]['prompt'])
    order[run] = hashlib.md5(m.group(1).encode()).hexdigest()[:8] if m else None
raw21 = open(run_file('2.1'), 'rb').read()
try:
    raw21.decode('utf-8'); enc21 = 'utf-8'
except UnicodeDecodeError:
    enc21 = 'không phải UTF-8 (cp1252)'
res['item6'] = {'records_2.1': len(r21), 'partial_blocks_2.1': part21, 'last_act_eq_label_2.1': same21,
                'encoding_2.1': enc21, 'has_do_sample_field': {r: ('Do Sample' in parse_run(r)[0]) for r in ['2.1', '2.2', '2.3', '2.5']},
                'activity_list_md5_first_prompt': order,
                'metrics_2.1': m21, 'metrics_2.3': parse_run('2.3')[1]}
lines.append('Mục 6: ' + json.dumps(res['item6'], ensure_ascii=False))

# ---- Mục 7
per_q = {}
for f in sorted(os.listdir(VALID)):
    if f.endswith(('.txt', '_stdout.log')):
        txt = open(os.path.join(VALID, f), 'rb').read().decode('utf-8', errors='replace')
        per_q[f] = bool(re.search(r'(?im)^(prediction time|time per (question|sample)|elapsed for sample)', txt))
res['item7'] = {'files_with_per_question_time': [f for f, v in per_q.items() if v], 'files_checked': len(per_q)}
lines.append('Mục 7: ' + json.dumps(res['item7']))

json.dump(res, open(os.path.join(OUT, 'f_run_checks.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
open(os.path.join(OUT, 'f_run_checks.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

# ---- Bổ sung: số dự đoán khác nhau giữa 2.3 và 2.5; câu không có \boxed{} của lượt 4.1
import ast
import difflib
from b_long_prefix import capped


def oracle_pred(answer, classes):
    """Tái hiện bước trích dự đoán của src/oracle.py (regex \\boxed{}, dự phòng, so khớp mờ 0,6)."""
    m = re.search(r'\\boxed\{([^}]*)\}', answer)
    if not m:
        m = re.search(r'### Assistant:\s*(.*)', answer)
    if not m:
        ms = list(re.finditer(r'\| Next activity:\s*(.*)', answer))
        m = ms[-1] if ms else None
    if not m:
        return 'Wrong format'
    c = m.group(1).strip("'")
    if 'None' in c:
        return 'Wrong format'
    c = re.sub(r'\\text\{(.*?)\}', r'\1', c)
    c = re.sub(r'[^a-zA-Z0-9\s]', '', c).strip()
    cm = difflib.get_close_matches(c, classes, n=1, cutoff=0.6)
    return cm[0] if cm else 'Wrong format'


h23, _, r23, _ = parse_run('2.3')
h25, _, r25all, _ = parse_run('2.5')
cls = list(ast.literal_eval(h25['Activities']))
p23 = [oracle_pred(r['answer'], cls) for r in r23]
p25 = [oracle_pred(r['answer'], cls) for r in r25all]
raw_box = lambda a: (re.search(r'\\boxed\{([^}]*)\}', a) or [None, None])[1]
res['extra'] = {
    'pred_label_differs_2.3_vs_2.5': sum(a != b for a, b in zip(p23, p25)),
    'raw_boxed_string_differs_2.3_vs_2.5': sum(raw_box(a['answer']) != raw_box(b['answer']) for a, b in zip(r23, r25all)),
    'oracle_reimpl_matches_verdict_2.5': sum((p == r['expected'].strip()) == r['correct'] for p, r in zip(p25, r25all)),
    '4.1_no_boxed': sum(1 for r in r41 if '\\boxed{' not in r['answer']),
    '4.1_no_boxed_capped_2048': sum(1 for r in r41 if capped(r['answer'])),
    '4.1_no_boxed_bold_label': sum(1 for r in r41 if '\\boxed{' not in r['answer'] and re.search(r'\*\*[^*]+\*\*', r['answer'])),
    '2.5_first150_no_boxed': sum(1 for r in r25 if '\\boxed{' not in r['answer']),
    '2.5_first150_capped': sum(1 for r in r25 if capped(r['answer'])),
}
lines.append('Bổ sung: ' + json.dumps(res['extra']))
json.dump(res, open(os.path.join(OUT, 'f_run_checks.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
open(os.path.join(OUT, 'f_run_checks.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(lines[-1])
