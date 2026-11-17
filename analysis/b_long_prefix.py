"""Mục B: vì sao một số câu không có prefix trùng khớp trong ngữ cảnh, và nguyên nhân sụt giảm ở nhóm 31+.

1. Đọc max_seq_length thực tế của all-MiniLM-L12-v2 từ sentence_bert_config.json trong bộ nhớ đệm HuggingFace.
2. Văn bản được nhúng khi truy xuất (theo src/vector_store.py): tài liệu = chuỗi hoạt động của prefix
   (khoá t trong store_traces); truy vấn = question.split('-')[0].strip() (retrieve_context).
   Đếm số token của truy vấn bằng tokenizer của chính mô hình nhúng (kể cả [CLS], [SEP]).
3. Với câu không truy xuất được prefix có cùng chuỗi hoạt động: kiểm tra xem 3 tài liệu truy xuất có cùng
   max_seq_length token đầu với truy vấn hay không (tức là giống hệt sau khi bị cắt).
4. Bảng chéo lượt 2.5: nhóm độ dài × mức ngữ cảnh (a)/(b)/(c) × bị cắt ở giới hạn sinh, kèm accuracy.
   "Bị cắt" = câu trả lời không có \\boxed{} và dài >= 2047 token theo tokenizer của Qwen2.5-3B-Instruct.
Đầu ra: analysis/out/b_long_prefix.json, analysis/out/b_long_prefix.md
"""
import glob
import json
import os
from collections import defaultdict

from tokenizers import Tokenizer

from common import BUCKETS, OUT, bucket, context_level, has_boxed, parse_run

HF = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
emb_dir = glob.glob(os.path.join(HF, 'models--sentence-transformers--all-MiniLM-L12-v2', 'snapshots', '*'))[0]
max_seq = json.load(open(os.path.join(emb_dir, 'sentence_bert_config.json')))['max_seq_length']
emb_tok = Tokenizer.from_file(os.path.join(emb_dir, 'tokenizer.json'))
emb_tok.no_truncation()
qwen_dir = glob.glob(os.path.join(HF, 'models--Qwen--Qwen2.5-3B-Instruct', 'snapshots', '*'))[0]
qwen_tok = Tokenizer.from_file(os.path.join(qwen_dir, 'tokenizer.json'))


def ids(text):
    return emb_tok.encode(text).ids  # gồm [CLS] ... [SEP]


def capped(answer):
    return (not has_boxed(answer)) and len(qwen_tok.encode(answer).ids) >= 2047


out = {'max_seq_length': max_seq, 'embedding_snapshot': os.path.basename(emb_dir)}
lines = [f'max_seq_length (sentence_bert_config.json) = {max_seq}', '']

for run in ['2.5', '3.3a', '3.1a']:
    _, _, recs, _ = parse_run(run)
    info = []
    for r in recs:
        query = r['question'].split('-')[0].strip()
        q_ids = ids(query)
        same_seq = any(d['seq'] == r['q_seq'] for d in r['docs'])
        same_trunc = sum(1 for d in r['docs'] if ids(d['seq'])[:max_seq - 1] == q_ids[:max_seq - 1])
        info.append({'plen': r['plen'], 'tok': len(q_ids), 'over': len(q_ids) > max_seq,
                     'query_cut_by_hyphen': query != r['q_seq'], 'same_seq': same_seq,
                     'docs_identical_after_trunc': same_trunc})
    miss = [x for x in info if not x['same_seq']]
    hit = [x for x in info if x['same_seq']]
    s = {
        'N': len(info),
        'miss': len(miss),
        'miss_over_limit': sum(x['over'] for x in miss),
        'miss_query_cut_by_hyphen': sum(x['query_cut_by_hyphen'] for x in miss),
        'miss_min_plen': min((x['plen'] for x in miss), default=None),
        'miss_min_tok': min((x['tok'] for x in miss), default=None),
        'miss_over_with_all3_docs_identical_after_trunc': sum(1 for x in miss if x['over'] and x['docs_identical_after_trunc'] == len(recs[0]['docs'])),
        'hit_over_limit': sum(x['over'] for x in hit),
        'hit_max_tok': max((x['tok'] for x in hit), default=None),
        'all_over_limit': sum(x['over'] for x in info),
        'min_plen_over_limit': min((x['plen'] for x in info if x['over']), default=None),
    }
    out[run] = s
    lines.append(f'{run}: ' + json.dumps(s, ensure_ascii=False))

# Bảng chéo cho lượt 2.5
_, _, recs, _ = parse_run('2.5')
cell = defaultdict(lambda: [0, 0])
for r in recs:
    key = (bucket(r['plen']), context_level(r), capped(r['answer']))
    cell[key][0] += 1
    cell[key][1] += r['correct']
cross = {f'{b}|{l}|{"cut" if c else "ok"}': v for (b, l, c), v in sorted(cell.items())}
out['crosstab_2.5'] = cross
lines += ['', 'Bảng chéo lượt 2.5 (n đúng/n): nhóm độ dài × mức ngữ cảnh × bị cắt', '',
          '| Nhóm | (a) không cắt | (a) bị cắt | (b) không cắt | (b) bị cắt | (c) không cắt | (c) bị cắt | Tổng |',
          '|---|---|---|---|---|---|---|---|']
for b in BUCKETS:
    row, tn, tc = [b], 0, 0
    for l in 'abc':
        for c in (False, True):
            n, k = cell.get((b, l, c), [0, 0])
            tn += n; tc += k
            row.append(f'{k}/{n}' if n else '-')
    row.append(f'{tc}/{tn}')
    lines.append('| ' + ' | '.join(row) + ' |')
# tổng theo cột
row = ['Tổng']
for l in 'abc':
    for c in (False, True):
        n = sum(v[0] for (b, ll, cc), v in cell.items() if ll == l and cc == c)
        k = sum(v[1] for (b, ll, cc), v in cell.items() if ll == l and cc == c)
        row.append(f'{k}/{n}')
row.append(f'{sum(v[1] for v in cell.values())}/{sum(v[0] for v in cell.values())}')
lines.append('| ' + ' | '.join(row) + ' |')

json.dump(out, open(os.path.join(OUT, 'b_long_prefix.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
open(os.path.join(OUT, 'b_long_prefix.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
