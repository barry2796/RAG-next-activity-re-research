max_seq_length (sentence_bert_config.json) = 128

2.5: {"N": 300, "miss": 32, "miss_over_limit": 32, "miss_query_cut_by_hyphen": 0, "miss_min_plen": 34, "miss_min_tok": 133, "miss_over_with_all3_docs_identical_after_trunc": 32, "hit_over_limit": 12, "hit_max_tok": 506, "all_over_limit": 44, "min_plen_over_limit": 34}
3.3a: {"N": 100, "miss": 8, "miss_over_limit": 1, "miss_query_cut_by_hyphen": 7, "miss_min_plen": 1, "miss_min_tok": 128, "miss_over_with_all3_docs_identical_after_trunc": 1, "hit_over_limit": 2, "hit_max_tok": 201, "all_over_limit": 3, "min_plen_over_limit": 13}
3.1a: {"N": 100, "miss": 0, "miss_over_limit": 0, "miss_query_cut_by_hyphen": 0, "miss_min_plen": null, "miss_min_tok": null, "miss_over_with_all3_docs_identical_after_trunc": 0, "hit_over_limit": 0, "hit_max_tok": 128, "all_over_limit": 0, "min_plen_over_limit": null}

Bảng chéo lượt 2.5 (n đúng/n): nhóm độ dài × mức ngữ cảnh × bị cắt

| Nhóm | (a) không cắt | (a) bị cắt | (b) không cắt | (b) bị cắt | (c) không cắt | (c) bị cắt | Tổng |
|---|---|---|---|---|---|---|---|
| 1-5 | 6/6 | - | 9/9 | - | 1/4 | - | 16/19 |
| 6-10 | 47/55 | - | 18/19 | - | 1/13 | - | 66/87 |
| 11-20 | 80/110 | - | 3/3 | - | 0/1 | - | 83/114 |
| 21-30 | 25/29 | 0/2 | - | - | - | - | 25/31 |
| 31+ | 6/10 | 0/7 | - | - | 1/8 | 0/24 | 7/49 |
| Tổng | 164/210 | 0/9 | 30/31 | 0/0 | 3/26 | 0/24 | 197/300 |
