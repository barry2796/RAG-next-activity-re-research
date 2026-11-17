"""Dùng chung cho các script phân tích: đọc file kết quả trong tests/validation/
(định dạng do src/oracle.py ghi), khoảng tin cậy Wilson và kiểm định McNemar chính xác.

Mọi script chỉ ĐỌC file kết quả/tập kiểm thử; không gọi LLM, không cần GPU.
"""
import glob
import math
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VALID = os.path.join(ROOT, 'tests', 'validation')
TESTSETS = os.path.join(ROOT, 'tests', 'test_sets')
OUT = os.path.join(os.path.dirname(__file__), 'out')
os.makedirs(OUT, exist_ok=True)

SEP = "\n#####################################################################################\n"
REC_RE = re.compile(r"^Prompt: (?P<prompt>.*?)\nPrefix Length: (?P<plen>\d+)\nEarlyness Bucket: (?P<bucket>[^\n]*)\n"
                    r"Model Answer: (?P<answer>.*)\nExpected Answer: (?P<expected>[^\n]*)\n"
                    r"Verification Result: (?P<verif>True|False|None)\n?$", re.DOTALL)
DOC_RE = re.compile(r"^\d+\. \[(?P<seq>.*)\] - (?P<vals>Values: \{.*\}) \| Next activity: (?P<label>.*)$")

RUNS = ['2.1', '2.2', '2.3', '2.5', '2.6', '3.1a', '3.1b', '3.2a', '3.2b', '3.3a', '3.3b', '4.1', '4.3a']
RAG_RUNS = ['2.5', '2.3', '3.1a', '3.3a', '3.2a', '4.3a', '4.1']
LOG_OF_RUN = {'2.1': 'sepsis', '2.2': 'sepsis', '2.3': 'sepsis', '2.5': 'sepsis', '2.6': 'sepsis',
              '3.1a': 'bpic20', '3.1b': 'bpic20', '3.2a': 'synthetic13', '3.2b': 'synthetic13',
              '3.3a': 'bpic13', '3.3b': 'bpic13', '4.1': 'sepsis', '4.3a': 'sepsis'}
TESTSET_OF_LOG = {
    'sepsis': 'test_set_sepsis_evaluation-attributes.csv',
    'bpic20': 'test_set_bpic20_international_declarations_evaluation-attributes.csv',
    'bpic13': 'test_set_BPIC_2013_closed_problems_evaluation-attributes.csv',
    'synthetic13': 'test_set_synthetic-13-multivariant_evaluation-attributes.csv',
}
BUCKETS = ['1-5', '6-10', '11-20', '21-30', '31+']


def run_file(run, suffix='.txt'):
    files = [f for f in glob.glob(os.path.join(VALID, f'{run}_*{suffix}'))
             if suffix != '.txt' or not f.endswith(('_summary.md',))]
    files = [f for f in files if not f.endswith(('_stdout.log', '_stderr.log'))] if suffix == '.txt' else files
    if len(files) != 1:
        raise FileNotFoundError(f'run {run}: {files}')
    return files[0]


def _decode(raw):
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:  # lượt 2.1 được ghi trước khi sửa B3 (cp1252)
        return raw.decode('cp1252', errors='replace')


def parse_run(run):
    """Trả về (header dict, metrics dict, danh sách bản ghi, số khối không đọc được)."""
    txt = _decode(open(run_file(run), 'rb').read()).replace('\r\n', '\n')
    head, _, rest = txt.partition('\n-----------------------------------\nOVERALL METRICS\n')
    header = {}
    for line in head.splitlines():
        if ': ' in line:
            k, v = line.split(': ', 1)
            header[k.strip()] = v.strip()
    metrics_block, _, rest2 = rest.partition('\n-----------------------------------\nEARLYNESS ANALYSIS\n')
    metrics = {}
    for line in metrics_block.splitlines():
        if ': ' in line:
            k, v = line.split(': ', 1)
            metrics[k] = v
    _, _, body = rest2.partition('-----------------------------------\n\n')
    records, partial = [], 0
    for ch in body.split(SEP):
        if not ch.strip():
            continue
        m = REC_RE.match(ch)
        if not m:
            partial += 1
            continue
        d = m.groupdict()
        d['plen'] = int(d['plen'])
        d['correct'] = d['verif'] == 'True'
        d.update(parse_prompt(d['prompt']))
        records.append(d)
    return header, metrics, records, partial


def parse_prompt(prompt):
    user = prompt.split('<|im_start|>user\n', 1)[1] if '<|im_start|>user\n' in prompt else prompt
    q = user.rsplit('Prefix to predict: ', 1)[1].split('<|im_end|>', 1)[0]
    seq, vals = q.rsplit(' - Values: ', 1)
    docs = []
    if 'Here is the past traces:' in user:
        ctx = user.split('Here is the past traces:', 1)[1].rsplit('Prefix to predict: ', 1)[0]
        for line in ctx.split('\n'):
            m = DOC_RE.match(line.strip())
            if m:
                docs.append({k: v.strip() for k, v in m.groupdict().items()})
    return {'question': q, 'q_seq': seq.strip(), 'q_vals': vals.strip(), 'docs': docs}


def bucket(plen):
    for lo, hi in [(1, 5), (6, 10), (11, 20), (21, 30)]:
        if plen <= hi:
            return f'{lo}-{hi}'
    return '31+'


def has_boxed(answer):
    return '\\boxed{' in answer


def context_level(r):
    """Mức (a)/(b)/(c) của ngữ cảnh truy xuất cho một câu hỏi có RAG.
    (a) có prefix trùng cả chuỗi hoạt động lẫn toàn bộ giá trị thuộc tính với câu hỏi;
    (b) không có (a) nhưng có prefix trùng chuỗi hoạt động và có Next activity = đáp án đúng;
    (c) còn lại (không có prefix trùng chuỗi kèm đáp án)."""
    exp = r['expected'].strip()
    same_seq = [d for d in r['docs'] if d['seq'] == r['q_seq']]
    if any(d['vals'] == r['q_vals'] for d in same_seq):
        return 'a'
    if any(d['label'] == exp for d in same_seq):
        return 'b'
    return 'c'


def wilson(x, n, z=1.959963984540054):
    if n == 0:
        return (float('nan'), float('nan'))
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return c - h, c + h


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def fmt_ci(x, n):
    lo, hi = wilson(x, n)
    return f'{x/n:.3f} [{lo:.3f}; {hi:.3f}]' if n else '-'
