# analysis/

Các script tính các số liệu bổ sung vào báo cáo ngày 17/11/2026. Script chỉ **đọc** dữ liệu:
- file kết quả trong `tests/validation/`;
- tập kiểm thử trong `tests/test_sets/`;
- riêng `e_b1_check.py`: nhật ký `logs/*.xes` và lịch sử git.

Không script nào gọi LLM, và không cần GPU.

## Cách chạy

Chạy trong thư mục này, dùng môi trường Python của dự án (cần gói `tokenizers` và bộ nhớ đệm HuggingFace có sẵn hai tokenizer):

```bash
cd analysis
export PYTHONIOENCODING=utf-8
../.venv/Scripts/python.exe a_leak_levels.py
../.venv/Scripts/python.exe b_long_prefix.py
../.venv/Scripts/python.exe c_lookup_baselines.py
../.venv/Scripts/python.exe d_prefix_length_dist.py
../.venv/Scripts/python.exe e_b1_check.py
../.venv/Scripts/python.exe f_run_checks.py
```

Mỗi script ghi một file `.json` và một file `.md` vào `out/`.

## Danh sách script

| Script | Mục trong yêu cầu | Bảng trong báo cáo |
|---|---|---|
| `common.py` | Dùng chung: đọc file kết quả, phân loại mức (a)/(b)/(c), CI Wilson, McNemar chính xác | – |
| `a_leak_levels.py` | Phần 2, mục A | 7.16, 7.17 |
| `b_long_prefix.py` | Phần 2, mục B | 7.14 và đoạn văn Mục 7.2.5 |
| `c_lookup_baselines.py` | Phần 2, mục C | 7.18 |
| `d_prefix_length_dist.py` | Phần 2, mục D | 6.5 |
| `e_b1_check.py` | Phần 2, mục E | 6.3 |
| `f_run_checks.py` | Phần 1, mục 4, 6, 7; thêm phép đếm 99/300 và câu trả lời của lượt 4.1 | Đoạn văn các Mục 6.4, 7.2.3, 7.2.6, 7.3.3 |

Số bảng theo bản biên dịch `docs/exp_report.pdf`.
