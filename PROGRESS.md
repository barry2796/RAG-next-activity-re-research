# Lộ trình & Theo dõi tiến độ — RAG-next-activity-re-research

> Lộ trình từ hiện tại đến khi hoàn thành báo cáo nghiên cứu cuối kì. Cập nhật thủ công sau mỗi lần fix code, mỗi lần chạy `eval.py` xong, hoặc khi đổi kế hoạch.
> Tham chiếu: [CODEBASE_AUDIT_REPORT.md](CODEBASE_AUDIT_REPORT.md) · [DATA_LEAKAGE_BUG.md](DATA_LEAKAGE_BUG.md) · [src/cmd4tests.sh](src/cmd4tests.sh) (bộ thực nghiệm gốc của tác giả)

**Cập nhật lần cuối:** 2026-11-16 (chốt dữ liệu thực nghiệm)
**Hạn nộp báo cáo:** _(điền)_
**Ký hiệu:** ✅ xong · 🔄 đang làm · ⬜ chưa làm · ⏸️ tạm hoãn · ❌ bỏ

---

## 0. Mục tiêu nghiên cứu

Tái lập (re-research) phương pháp RAG cho bài toán dự đoán next activity của bài báo gốc trên phần cứng hạn chế (RTX 4050 Laptop 6GB, Qwen2.5-3B/1.5B thay cho 7B), và trả lời các câu hỏi:

- **RQ1.** RAG có cải thiện dự đoán so với LLM không có RAG (no-RAG) không, và cải thiện bao nhiêu?
- **RQ2.** RAG trên phần cứng nhỏ có cạnh tranh được với các baseline state-of-the-art (deep learning) đã công bố không? *(Định nghĩa lại 2026-11-15 cho khớp RQ2 của bài báo gốc. Baseline lấy nguyên con số tác giả trích dẫn trên slide, không chạy lại. Xem bảng ở G2.)*
- **RQ3.** Hiệu năng thay đổi thế nào theo độ dài prefix (earlyness), và vì sao nhóm prefix dài (31+) kém?
- **RQ4.** Kích thước LLM (3B vs 1.5B) và các siêu tham số (`prefix_gap`, embedding model, số documents trong context) ảnh hưởng thế nào?
- **RQ5.** Kết quả trên phần cứng nhỏ so với con số trong bài báo gốc (7B) chênh lệch ra sao?

---

## 1. Tổng quan tiến độ

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| G0 | Kiểm toán codebase & sửa lỗi data leakage | ✅ |
| G1 | Ổn định môi trường, xử lý các lỗi còn tồn đọng | 🔄 |
| G2 | Thực nghiệm cốt lõi trên `sepsis.xes` (RQ1, RQ2, độ dao động) | ✅ 2026-11-15 · [tổng hợp](tests/validation/G2_sepsis_synthesis.md) |
| G3 | Mở rộng sang các event log khác | ✅ 2026-11-16 (tập con N câu đầu) · [tổng hợp](tests/validation/G3_G4_synthesis.md) |
| G4 | Ablation: LLM size, prefix_gap, embedding, num_docs (RQ4) | ✅ một phần (4.1, 4.3a) · [tổng hợp](tests/validation/G3_G4_synthesis.md) |
| G5 | Phân tích kết quả & phân tích lỗi (RQ3, RQ5) | ⬜ |
| G6 | Viết báo cáo, slide, đóng gói repo | ⬜ |

**Ước lượng thời gian chạy:** trên sepsis (300 mẫu), lượt RAG mất **2.76 giờ ≈ 33 giây/mẫu**, lượt no-RAG mất **1.28 giờ ≈ 15 giây/mẫu**. Dùng các con số này để ước lượng mọi lượt chạy khác: `thời gian ≈ số mẫu test × 33s` (RAG) hoặc `× 15s` (no-RAG). Log lớn (BPIC_2012, bpic20, hospital_billing) có thể cần hàng chục giờ → xem quyết định ở G3.

---

## G0. Kiểm toán & sửa lỗi data leakage — ✅

- [x] Audit toàn bộ codebase so với repo gốc của tác giả (2026-11-14)
- [x] Phát hiện bug: prefix chứa chính activity mục tiêu (`log_preprocessing.py`)
- [x] Fix bug, diff với bản gốc rỗng tuyệt đối
- [x] Xác minh bằng `live_demo_qwen.py` (4/5 đúng, prefix không còn chứa đáp án)
- [x] Chạy evaluation chính thức đầu tiên sau fix: sepsis / concept_names / RAG (Accuracy 0.6767)

---

## G1. Ổn định môi trường & xử lý lỗi còn tồn đọng — 🔄

Mục tiêu: đảm bảo mọi lượt chạy ở G2–G4 tái lập được và không bị gián đoạn giữa chừng.

| # | Công việc | Ưu tiên | Trạng thái | Ghi chú |
|---|---|---|---|---|
| 1.1 | Khởi động lại Qdrant (`docker start qdrant_nap`) | CRITICAL | ✅ | Đã chạy được eval 2026-11-15 |
| 1.2 | `pip freeze > requirements-lock.txt` để chốt version thực tế (`transformers 5.x`, `langchain 1.x`...) | HIGH | ⬜ | Audit #3. Làm **trước** khi chạy tiếp để mọi kết quả cùng một môi trường |
| 1.3 | Viết hướng dẫn cài torch/bitsandbytes cho CUDA 12.6 trên Windows vào README, giải thích `requirements-core.txt` | HIGH | ⬜ | Audit #4 |
| 1.4 | Sửa `max_memory` trong [pipeline.py:95-99](src/pipeline.py#L95-L99) dùng `torch.cuda.mem_get_info()` (VRAM trống) thay vì tổng VRAM | MEDIUM | ⬜ | Audit #5. Tránh OOM giữa lượt chạy nhiều giờ |
| 1.5 | Cập nhật `.gitignore` loại trừ `tests/outputs/`, `tests/validation/` *hoặc* quyết định giữ lại kết quả trong git làm minh chứng | MEDIUM | ⬜ | Audit #8. Cần thống nhất với nhóm |
| 1.6 | Ghi chú `--evaluation_modality` trong `main.py` chỉ có tác dụng với `eval.py` | LOW | ⬜ | Audit #7 |
| 1.7 | Commit trạng thái hiện tại (bug fix + tài liệu) làm mốc trước khi chạy hàng loạt | HIGH | ⬜ | Để mỗi kết quả gắn được với một commit |
| 1.8 | Sửa `UnicodeEncodeError` khi ghi file kết quả: thêm `encoding='utf-8'` vào [oracle.py:205](src/oracle.py#L205) | HIGH | ✅ 2026-11-15 | Lỗi làm cắt phần chi tiết từng mẫu của lượt 2.1 (44/300) và 2.2 (33/300). Metrics không bị ảnh hưởng |
| 1.9 | `--rebuild_db_and_tests False` gây `KeyError: 'Activities'` trong `oracle.py` (lỗi có sẵn trong code gốc) | MEDIUM | ⏸️ Né lỗi | Luôn chạy với `True` như tác giả. Seed cố định nên test set sinh lại giống hệt (đã kiểm tra bằng hash) |
| 1.10 | `--evaluation_modality` không có tác dụng: test set, vector store và prompt giống hệt giữa concept_names và attributes | HIGH | ✅ Quyết định (b) | Không sửa code. Bài báo gốc cũng không có chế độ concept_names (prompt mẫu ở slide 15 luôn có `Values`). RQ2 đã định nghĩa lại theo bài báo (so với baseline SOTA). 2.1 và 2.3 dùng làm hai lần lặp. Mọi lượt sau dùng `--evaluation_modality evaluation-attributes`. Xem [summary 2.3](tests/validation/2.3_sepsis_RAG_qwen3b_sampling_k3_n300_2026-11-15_11-32-51_summary.md) mục 4 |
| 1.11 | Kết quả dao động giữa các lần chạy (`do_sample=True`, `temperature=0.1`): cùng thí nghiệm lệch 0.043 accuracy | HIGH | ✅ 2026-11-15 | Thêm cờ `--do_sample` vào [eval.py](src/eval.py) (parse bằng `u.str2bool`, mặc định `True` = hành vi gốc). `RAGPipeline(..., do_sample=...)`: `True` thì truyền `temperature=0.1` như gốc, `False` thì greedy, không truyền temperature. Header kết quả có dòng `Do Sample`. Đã kiểm tra: parser đúng với True/False/false/no, `generate()` nhận `do_sample=False, repetition_penalty=1.1, max_new_tokens=2048`, greedy lặp lại giống hệt từng ký tự. **Lượt greedy khác code gốc**, phải ghi rõ trong báo cáo |
| 1.13 | Vector store chứa cả trace của test set, trái với giao thức 70/30 của bài báo: 250/300 câu ở lượt 2.5 có đúng prefix kèm đáp án trong top-3 | CRITICAL | ⏸️ Quyết định không sửa | Giữ giao thức code gốc. Mọi kết quả RAG phải báo cáo kèm hạn chế này. No-RAG không bị ảnh hưởng. Xem [BUGFIXES.md](BUGFIXES.md) B6 |
| 1.12 | Vòng lặp sinh văn bản: mọi câu trả lời thiếu `\boxed{}` đều bị cắt ở đúng 2048 token vì mô hình lặp `\text{Leucocytes}, \text{CRP}, …` | MEDIUM | ⏸️ Ghi nhận, không sửa | 2.5 (greedy) có 33 câu, 2.3 (sampling) có 26, tập trung ở nhóm 31+. Giữ nguyên `max_new_tokens`/`repetition_penalty` để so sánh được. Dùng cho RQ3 (5.3). Xem [summary 2.5](tests/validation/2.5_sepsis_RAG_qwen3b_greedy_k3_n300_2026-11-15_18-42-54_summary.md) mục 5 |

**Tiêu chí hoàn thành G1:** có `requirements-lock.txt`, đã commit bản fix, chạy một lượt ngắn không bị OOM.

---

## G2. Thực nghiệm cốt lõi trên `sepsis.xes` — ✅

Mục tiêu: trả lời RQ1 (RAG vs no-RAG) và RQ2 (so với baseline SOTA) trên cùng một log, cùng LLM, kèm độ dao động giữa các lần chạy.

Cấu hình chung (giữ cố định để so sánh công bằng): `--log sepsis.xes --llm_id Qwen/Qwen2.5-3B-Instruct --prefix_gap 3` (mặc định của `eval.py` là `prefix_gap 1` và `llm_id gpt-4.1`, **phải truyền tay**).

| # | Modality | RAG | Lệnh bổ sung | Ước lượng | Trạng thái | Kết quả |
|---|---|---|---|---|---|---|
| 2.1 | concept_names | ✅ | `--evaluation_modality evaluation-concept_names --rag True` | 2.76h | ✅ 2026-11-15 | Acc 0.6767 · F1 0.4640 · [summary](tests/validation/2.1_sepsis_RAG_qwen3b_sampling_k3_conceptnames_n300_2026-11-15_03-29-45_summary.md) |
| 2.2 | concept_names | ❌ | `--evaluation_modality evaluation-concept_names --rag False` | 1.28h | ✅ 2026-11-15 | Acc 0.1133 · F1 0.0316 · [summary](tests/validation/2.2_sepsis_noRAG_qwen3b_sampling_conceptnames_n300_2026-11-15_07-50-00_summary.md) |
| 2.3 | attributes | ✅ | `--evaluation_modality evaluation-attributes --rag True` | 2.83h | ✅ 2026-11-15 | Acc 0.6333 · F1 0.5582 · [summary](tests/validation/2.3_sepsis_RAG_qwen3b_sampling_k3_n300_2026-11-15_11-32-51_summary.md) |
| 2.4 | attributes | ❌ | `--evaluation_modality evaluation-attributes --rag False` | — | ❌ Bỏ | Trùng với 2.2 (xem 1.10) |

Lượt 2.1–2.3 chạy với **sampling** (`do_sample=True`, như code gốc). Hai modality là một (1.10), nên 2.1 và 2.3 là hai lần lặp của cùng một thí nghiệm: **RAG accuracy 0.655 ± 0.022**, macro-F1 0.511 ± 0.047. Độ dao động này là kết quả dùng cho báo cáo.

**Baseline greedy (`do_sample=False`)**, dùng làm mốc so sánh cho G3 và G4:

| # | RAG | Lệnh bổ sung | Ước lượng | Trạng thái | Kết quả |
|---|---|---|---|---|---|
| 2.5 | ✅ | `--rag True --do_sample False` | 2.91h | ✅ 2026-11-15 | Acc 0.6567 · F1 0.6223 · 33 câu bị cắt ở 2048 token · [summary](tests/validation/2.5_sepsis_RAG_qwen3b_greedy_k3_n300_2026-11-15_18-42-54_summary.md) |
| 2.6 | ❌ | `--rag False --do_sample False` | 1.24h | ✅ 2026-11-15 | Acc 0.1633 · F1 0.0452 · [summary](tests/validation/2.6_sepsis_noRAG_qwen3b_greedy_n300_2026-11-15_20-00-04_summary.md) |

**Tổng hợp G2:** [tests/validation/G2_sepsis_synthesis.md](tests/validation/G2_sepsis_synthesis.md), gồm bảng toàn bộ lượt chạy, RQ1, RQ2, sampling vs greedy và khác biệt so với code gốc.

Tất cả các lượt đều thêm `--rebuild_db_and_tests True` (xem 1.9). Lệnh đầy đủ cho 2.5:

```bash
python src/eval.py --log sepsis.xes --evaluation_modality evaluation-attributes \
  --llm_id Qwen/Qwen2.5-3B-Instruct --prefix_gap 3 \
  --rebuild_db_and_tests True --rag True --do_sample False
```

Lưu ý quan trọng:
- Seed cố định nên test set sinh lại giống hệt giữa các lượt. Trước mỗi lượt no-RAG, sao lưu CSV test set và so hash sau khi sinh lại để chắc chắn.
- Kiểm tra Docker Desktop và container `qdrant_nap` đang chạy trước **mọi** lượt, kể cả no-RAG (`eval.py` luôn kết nối Qdrant).
- Sau mỗi lượt: viết file `*_summary.md` trong `tests/validation/` và điền kết quả vào bảng ở mục 3.
- Số mẫu thực tế là **300** (tổng các bucket). "Test Set Size: 315" trong header là số trace.

**Kết quả RQ1:**
- **Greedy** (2.6 → 2.5): accuracy 0.163 → 0.657 (+0.49), macro-F1 0.045 → 0.622. So cặp McNemar: 167 câu chỉ RAG đúng, 19 câu chỉ no-RAG đúng, p < 0.0001.
- **Sampling** (2.2 → 2.1/2.3): +0.52 đến +0.56.
- **Ngoại lệ ở nhóm 31+:** RAG 0.143 so với no-RAG 0.204 (p = 0.58), do vòng lặp sinh văn bản (1.12).

**RQ2: so với baseline SOTA.** Không chạy lại baseline. Dùng nguyên con số trong slide của tác giả ([slides-BPM26-journal-first.pdf](slides-BPM26-journal-first.pdf), trang 10), cũng là cách bài báo làm:

| Event log | Baseline SOTA (theo bài báo) | RAG LLM của bài báo | Của mình (Qwen2.5-3B) |
|---|---|---|---|
| Sepsis Cases | Oved et al.: 0.66 | 0.70 (phi-4) | Sampling: 0.655 ± 0.022 (2.1 + 2.3) · **Greedy: 0.657**, 95% CI [0.601; 0.708] (2.5) |
| BPIC 2012 | Zhao et al. & Hinkka et al.: 0.86 | 0.76 | — |
| BPIC 2013 (Closed) | Oved et al.: 0.69 | 0.90 | Greedy: 0.620, 95% CI [0.522; 0.709] (3.3a, **100 câu đầu**) |
| BPIC 2013 (Incidents) | Zhao et al.: 0.79 | 0.70 | — |
| BPIC 2020 (Int. Decl.) | Impedovo et al.: 0.84 | 0.98 | Greedy: 0.560, 95% CI [0.462; 0.653] (3.1a, **100 câu đầu**) |
| Hospital Billing | Bukhsh et al.: 0.85 | 0.73 | — |

Nhận xét sơ bộ (sepsis): khoảng tin cậy 95% (n=300) là [0.60; 0.71] và chứa 0.66, nghĩa là **không phát hiện được khác biệt** với baseline. Điều này **không** chứng minh hai bên tương đương: CI rộng khoảng ±0.05, và không có dự đoán từng câu của baseline để kiểm định. Không được viết "thắng" hay "tương đương".

Hạn chế phải ghi trong báo cáo:
- Baseline dùng giao thức chia dữ liệu riêng, không cùng 300 prefix, nên chỉ so được con số tổng, không kiểm định theo cặp được. Bài báo gốc cũng như vậy.
- Con số RAG của bài báo có thể chịu ảnh hưởng bug data leakage (xem G0). Kết quả của mình là sau khi sửa. So với baseline thì không bị ảnh hưởng.
- Khác mô hình: 3B 4-bit so với phi-4 14B.

**Tiêu chí hoàn thành G2:** xong 2.5 và 2.6. Có bảng so sánh và nhận xét cho RQ1 (RAG vs no-RAG), RQ2 (so với baseline SOTA) và độ dao động sampling so với greedy.

---

## G3. Mở rộng sang các event log khác — ✅

Mục tiêu: kiểm tra kết quả G2 có tổng quát hoá sang log khác không.

Do giới hạn thời gian (chốt dữ liệu 05:00 ngày 16/11), mỗi lượt chỉ đánh giá **N dòng đầu** của test set qua cờ mới `--max_test_samples N`. Cặp RAG/no-RAG cùng log dùng cùng N câu, nên so cặp được (McNemar). Mọi lượt dùng greedy, `prefix_gap 3`, `--rebuild_db_and_tests True`. Chạy bằng hàng đợi tự động qua đêm 15–16/11.

| # | Log | N | RAG | no-RAG | McNemar p | Trạng thái |
|---|---|---|---|---|---|---|
| 3.1 | `bpic20_international_declarations.xes` | 100 | 0.560 [0.462; 0.653] | 0.010 | 5.6e-17 | ✅ 2026-11-15 |
| 3.2 | `synthetic-13-multivariant.xes` | 60 | 0.900 [0.799; 0.953] | 0.017 | 2.2e-16 | ✅ 2026-11-16 |
| 3.3 | `BPIC_2013_closed_problems.xes` | 100 | 0.620 [0.522; 0.709] | 0.010 | 8.7e-19 | ✅ 2026-11-16 |

Chi tiết, phân tích theo bucket và đo rò rỉ B6 trên từng log: [G3_G4_synthesis.md](tests/validation/G3_G4_synthesis.md).

**Phát hiện chính:** rò rỉ B6 có ở mọi log. 83–100% câu có đúng prefix kèm đáp án trong ngữ cảnh truy xuất. Ở các câu không có, RAG chỉ đúng 0–6%, ngang mức no-RAG. Mức tăng của RAG gắn gần như hoàn toàn với các câu bị rò rỉ.

---

## G4. Ablation study — ✅ một phần

Mục tiêu: trả lời RQ4. Chạy trên **một log cố định** (sepsis hoặc log nhỏ nhất ở G3) để tiết kiệm thời gian, chỉ thay đổi **một** tham số mỗi lượt so với cấu hình chuẩn của G2.

| # | Biến thay đổi | Các giá trị | Lệnh bổ sung | Trạng thái |
|---|---|---|---|---|
| 4.1 | Kích thước LLM | 3B (đã có) vs **1.5B** | `--llm_id Qwen/Qwen2.5-1.5B-Instruct --max_test_samples 150` | ✅ 2026-11-15 · 0.247 so với 0.653 (150 câu đầu của 2.5), p = 7.8e-13 |
| 4.2 | `prefix_gap` | 1, **3** (đã có), 5 | `--prefix_gap 1` / `--prefix_gap 5` | ❌ Không đủ thời gian → hướng phát triển |
| 4.3a | Số documents trong context | **1** vs 3 (đã có) | `--num_documents_in_context 1 --max_test_samples 150` | ✅ 2026-11-15 · 0.600 so với 0.653, p = 0.29 (không phát hiện được khác biệt) |
| 4.3b | Số documents trong context | 5 | `--num_documents_in_context 5` | ❌ Không đủ thời gian |
| 4.4 | Embedding model | **all-MiniLM-L12-v2** (đã có), all-MiniLM-L6-v2, all-mpnet-base-v2 | `--embed_model_id ...` (nhớ `--vector_dimension` tương ứng, mpnet = 768) | ❌ Không đủ thời gian |

Ưu tiên: 4.1 > 4.3 > 4.2 > 4.4. Thay đổi `prefix_gap` và embedding model đều đổi test set/vector store → phải `--rebuild_db_and_tests True`.

**Tiêu chí hoàn thành G4:** tối thiểu 4.1 và 4.3 xong.

---

## G5. Phân tích kết quả — ⬜

| # | Công việc | Trả lời | Trạng thái |
|---|---|---|---|
| 5.1 | Bảng tổng hợp tất cả lượt chạy (mục 3) + biểu đồ so sánh RAG vs no-RAG | RQ1 | ⬜ |
| 5.1b | Bảng và biểu đồ so với baseline SOTA (bảng RQ2 ở G2), kèm khoảng tin cậy 95% cho accuracy | RQ2 | ⬜ |
| 5.2 | Biểu đồ accuracy/F1 theo earlyness bucket cho từng cấu hình | RQ3 | ⬜ |
| 5.3 | Phân tích lỗi nhóm "Very Late (31+)" (acc 0.33): đọc các dự đoán sai, kiểm tra lỗi do lặp `Leucocytes/CRP/LacticAcid`, do output bị cắt cụt (`max_new_tokens`), hay do model không tuân thủ format `\boxed{}` | RQ3 | ⬜ |
| 5.4 | Confusion matrix theo activity — lớp nào hay bị nhầm (giải thích macro-F1 thấp hơn accuracy nhiều) | RQ1–3 | ⬜ |
| 5.5 | Đọc Methodology bài báo gốc: test set và vector store dùng chung nguồn trace có phải chủ đích không? Có thể đo tỷ lệ self-retrieval trong `eval.py` | Audit #6 | ⬜ |
| 5.6 | Đối chiếu với con số trong bài báo gốc (7B), giải thích chênh lệch | RQ5 | ⬜ |

---

## G6. Viết báo cáo & đóng gói — ⬜

| # | Công việc | Trạng thái |
|---|---|---|
| 6.1 | Dàn ý báo cáo: Giới thiệu → Bài báo gốc & phương pháp → Thiết lập tái lập (phần cứng, thay đổi so với gốc) → Bug data leakage & cách phát hiện → Thực nghiệm → Kết quả → Thảo luận & hạn chế → Kết luận | ⬜ |
| 6.2 | Viết phần thiết lập: ghi rõ đổi 7B → 3B/1.5B, quantization 4-bit, version thư viện (từ `requirements-lock.txt`) | ⬜ |
| 6.3 | Viết phần bug data leakage (tóm tắt từ `DATA_LEAKAGE_BUG.md`) — đây là đóng góp đáng giá của nghiên cứu | ⬜ |
| 6.4 | Viết phần kết quả & thảo luận từ G2–G5 | ⬜ |
| 6.5 | Hạn chế: phần cứng, số log chạy được, test/vector-store chung nguồn, một seed duy nhất | ⬜ |
| 6.6 | Slide thuyết trình | ⬜ |
| 6.7 | Dọn repo: cập nhật README (cách tái lập), commit kết quả cuối, xoá file tạm | ⬜ |
| 6.8 | Rà soát cuối & nộp | ⬜ |

---

## 2. Rủi ro & phương án dự phòng

| Rủi ro | Phương án |
|---|---|
| Không đủ thời gian GPU cho G3/G4 | Ưu tiên G2 (bắt buộc) → G4.1, 4.3 → G3.1. Phần còn lại ghi vào "hướng phát triển" |
| OOM / máy tắt giữa lượt chạy nhiều giờ | Làm 1.4 trước; tắt ứng dụng dùng GPU khác; cân nhắc thêm cơ chế lưu kết quả từng phần |
| Qdrant container dừng sau khi khởi động lại máy | Kiểm tra `docker ps` trước mỗi lượt chạy RAG |
| Kết quả không ổn định giữa các lần chạy | Seed đã cố định; nếu còn thời gian, chạy lại 2.1 một lần để kiểm tra độ lặp lại |

---

## 3. Bảng tổng hợp kết quả

| Ngày | Log | Modality | LLM | RAG | prefix_gap | num_docs | Test size | Accuracy | F1 (macro) | Thời gian | File |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-15 | sepsis | concept_names | Qwen2.5-3B | ✅ | 3 | 3 | 300 | 0.6767 | 0.4640 | 2.76h | [log](tests/validation/2.1_sepsis_RAG_qwen3b_sampling_k3_conceptnames_n300_2026-11-15_03-29-45.txt) |
| 2026-11-15 | sepsis | concept_names | Qwen2.5-3B | ❌ | 3 | — | 300 | 0.1133 | 0.0316 | 1.28h | [log](tests/validation/2.2_sepsis_noRAG_qwen3b_sampling_conceptnames_n300_2026-11-15_07-50-00.txt) |
| 2026-11-15 | sepsis | attributes (= concept_names, xem 1.10) | Qwen2.5-3B | ✅ | 3 | 3 | 300 | 0.6333 | 0.5582 | 2.83h | [log](tests/validation/2.3_sepsis_RAG_qwen3b_sampling_k3_n300_2026-11-15_11-32-51.txt) |
| 2026-11-15 | sepsis | attributes · **greedy** | Qwen2.5-3B | ✅ | 3 | 3 | 300 | 0.6567 | 0.6223 | 2.91h | [log](tests/validation/2.5_sepsis_RAG_qwen3b_greedy_k3_n300_2026-11-15_18-42-54.txt) |
| 2026-11-15 | sepsis | attributes · **greedy** | Qwen2.5-3B | ❌ | 3 | — | 300 | 0.1633 | 0.0452 | 1.24h | [log](tests/validation/2.6_sepsis_noRAG_qwen3b_greedy_n300_2026-11-15_20-00-04.txt) |
| 2026-11-15 | bpic20 int. decl. (3.1a) | attributes · greedy | Qwen2.5-3B | ✅ | 3 | 3 | 100 đầu | 0.5600 | 0.3348 | 16 phút | [log](tests/validation/3.1a_bpic20-intdecl_RAG_qwen3b_greedy_k3_n100_2026-11-15_21-06-31.txt) |
| 2026-11-15 | bpic20 int. decl. (3.1b) | attributes · greedy | Qwen2.5-3B | ❌ | 3 | — | 100 đầu | 0.0100 | 0.0128 | 30 phút | [log](tests/validation/3.1b_bpic20-intdecl_noRAG_qwen3b_greedy_n100_2026-11-15_21-36-38.txt) |
| 2026-11-15 | sepsis (4.1) | attributes · greedy | **Qwen2.5-1.5B** | ✅ | 3 | 3 | 150 đầu | 0.2467 | 0.2861 | 64 phút | [log](tests/validation/4.1_sepsis_RAG_qwen1.5b_greedy_k3_n150_2026-11-15_22-40-11.txt) |
| 2026-11-15 | sepsis (4.3a) | attributes · greedy | Qwen2.5-3B | ✅ | 3 | **1** | 150 đầu | 0.6000 | 0.5531 | 63 phút | [log](tests/validation/4.3a_sepsis_RAG_qwen3b_greedy_k1_n150_2026-11-15_23-43-00.txt) |
| 2026-11-16 | BPIC 2013 closed (3.3a) | attributes · greedy | Qwen2.5-3B | ✅ | 3 | 3 | 100 đầu | 0.6200 | 0.4689 | 127 phút | [log](tests/validation/3.3a_bpic13-closed_RAG_qwen3b_greedy_k3_n100_2026-11-16_01-50-02.txt) |
| 2026-11-16 | BPIC 2013 closed (3.3b) | attributes · greedy | Qwen2.5-3B | ❌ | 3 | — | 100 đầu | 0.0100 | 0.0080 | 55 phút | [log](tests/validation/3.3b_bpic13-closed_noRAG_qwen3b_greedy_n100_2026-11-16_02-45-15.txt) |
| 2026-11-16 | synthetic-13 (3.2a) | attributes · greedy | Qwen2.5-3B | ✅ | 3 | 3 | 60 đầu | 0.9000 | 0.6044 | 14 phút | [log](tests/validation/3.2a_synthetic13_RAG_qwen3b_greedy_k3_n60_2026-11-16_02-59-19.txt) |
| 2026-11-16 | synthetic-13 (3.2b) | attributes · greedy | Qwen2.5-3B | ❌ | 3 | — | 60 đầu | 0.0167 | 0.0104 | 33 phút | [log](tests/validation/3.2b_synthetic13_noRAG_qwen3b_greedy_n60_2026-11-16_03-32-25.txt) |

Tên file kết quả đã được đổi (2026-11-16) theo dạng `<lượt>_<log>_<RAG|noRAG>_<mô hình>_<decoding>[_k<số documents>][_conceptnames]_n<số câu>_<thời điểm kết thúc>.txt`. File `_summary.md`, `_stdout.log` và `_stderr.log` của cùng lượt đi kèm cùng tên.

---

## 4. Nhật ký cập nhật

| Ngày | Sự kiện |
|---|---|
| 2026-11-14 | Audit codebase, phát hiện bug data leakage CRITICAL |
| 2026-11-14/15 | Fix bug, xác minh bằng `live_demo_qwen.py` |
| 2026-11-15 03:29 | Lượt eval chính thức đầu tiên sau fix: sepsis / concept_names / RAG — Acc 0.6767 |
| 2026-11-15 | Lập lộ trình G0–G6 |
| 2026-11-15 07:50 | Lượt 2.2: sepsis / concept_names / no-RAG — Acc 0.1133. Sửa lỗi encoding khi ghi kết quả trong `oracle.py` |
| 2026-11-15 11:32 | Lượt 2.3: sepsis / attributes / RAG — Acc 0.6333. Phát hiện `--evaluation_modality` không có tác dụng (1.10) và kết quả dao động giữa các lần chạy (1.11) |
| 2026-11-15 | Quyết định: bỏ RQ2, dùng 2.1 và 2.3 làm hai lần lặp. Đổi sang greedy decoding (`do_sample=False`), đã kiểm tra kết quả lặp lại được. Bỏ 2.4, thêm baseline greedy 2.5, 2.6 |
| 2026-11-15 | Định nghĩa lại RQ2 theo bài báo gốc: so với baseline SOTA, dùng con số trích từ slide của tác giả (trang 10), không chạy lại baseline. Câu hỏi concept_names vs attributes không còn là RQ |
| 2026-11-15 | Thêm cờ `--do_sample` (thay cho sửa cứng), kiểm tra parser và tham số thực tế của `generate()` |
| 2026-11-15 18:42 | Lượt 2.5: sepsis / RAG / greedy — Acc 0.6567, F1 0.6223. So cặp với 2.3: McNemar p = 0.51. Phát hiện vòng lặp sinh văn bản ở prefix dài (1.12) |
| 2026-11-15 20:00 | Lượt 2.6: sepsis / no-RAG / greedy — Acc 0.1633, F1 0.0452. RAG vs no-RAG (greedy): McNemar p < 0.0001. **G2 hoàn thành**, tổng hợp ở [G2_sepsis_synthesis.md](tests/validation/G2_sepsis_synthesis.md) |
| 2026-11-15 | Quyết định không sửa B6 (vector store chứa trace của test set). Thêm cờ `--max_test_samples N` (đánh giá N dòng đầu của test set) để chạy G3/G4 kịp hạn chốt dữ liệu 05:00 ngày 16/11 |
| 2026-11-15 20:50 → 2026-11-16 03:32 | Hàng đợi qua đêm (greedy, gap 3): 3.1a/b bpic20, 4.1 1.5B, 4.3a k = 1, 3.3a/b BPIC 2013 closed, 3.2a/b synthetic-13 (giảm còn 60 câu để kịp giờ). Mọi lượt exit 0. Tổng hợp ở [G3_G4_synthesis.md](tests/validation/G3_G4_synthesis.md) |
| 2026-11-16 | Đo B6 trên mọi log: 83–100% câu RAG có đúng prefix kèm đáp án trong ngữ cảnh. Đổi tên file kết quả theo cấu hình chạy, chép log stdout vào `tests/validation`. **Chốt dữ liệu** |
