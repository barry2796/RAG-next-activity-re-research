# Codebase Audit Report — RAG-next-activity-re-research

**Phạm vi:** So sánh repository nghiên cứu (`RAG-next-activity-re-research`) với repository gốc của tác giả (`angelo-casciani/rag_next_activity`, clone shallow ngày kiểm toán từ GitHub, nhánh `main`).
**Ngày kiểm toán:** 2026-11-14
**Phương pháp:** Đọc trực tiếp mã nguồn hai repository, `diff -u` theo từng file tương ứng, kiểm chứng bằng dữ liệu thực tế (CSV test set, log JSON/TXT đã chạy), kiểm tra môi trường thực thi (`pip`/`import`/`nvidia-smi`/Docker), kiểm tra cú pháp và CLI. **Không** chạy evaluation toàn bộ, không rebuild vector DB/test set, không sửa/xóa file trong hai repository.

Chú thích ký hiệu dùng xuyên suốt báo cáo:
- ✅ **Đã xác minh** — quan sát trực tiếp từ code, diff, hoặc output thực tế.
- 🔎 **Suy luận từ code** — kết luận logic dựa trên đọc code, chưa chạy thực nghiệm để chứng minh tác động cuối cùng.
- ❓ **Chưa thể xác minh** — cần thêm thông tin (vd. đọc paper gốc) mới kết luận được.

---

## 1. Executive Summary

Repository nghiên cứu về cơ bản là một bản copy trung thực của repository gốc, cộng thêm các điều chỉnh để chạy trên GPU laptop RTX 4050 6GB (thêm biến thể model Qwen2.5 nhỏ hơn, quản lý VRAM, vài fix mã hóa file cho Windows). Phần lớn các file lõi (`eval.py`, `oracle.py`, `vector_store.py`, `cmd4tests.sh`, `prompts*.json`, toàn bộ `llm_sft/`) **giống hệt byte-by-byte** với bản gốc.

Tuy nhiên, kiểm toán phát hiện **một lỗi nghiêm trọng (CRITICAL) do nhóm nghiên cứu vô tình gây ra** trong `src/log_preprocessing.py`: hàm `process_prefixes()` hiện tại đưa **chính activity mục tiêu (ground truth)** vào trong chuỗi "prefix" được hiển thị cho LLM và được embed vào Qdrant, thay vì loại trừ nó như bản gốc. Lỗi này đã được **xác minh bằng dữ liệu thực tế** (CSV test set đã sinh ra, và log một lần chạy demo thực tế ngày 2026-11-12) — không phải suy đoán. Hệ quả: bài toán "dự đoán activity tiếp theo" bị biến thành "đọc activity cuối cùng đã cho sẵn trong câu hỏi", khiến **mọi kết quả thực nghiệm tạo ra bởi code hiện tại của repo nghiên cứu đều không có giá trị khoa học** cho đến khi fix.

Ngoài ra có hai vấn đề đáng chú ý khác:
- Qdrant container hiện **đang dừng** (Exited), cần khởi động lại trước khi chạy bất kỳ script nào.
- Các dependency đã cài đặt thực tế (`transformers 5.23.0`, `langchain 1.5.0`) **lệch rất xa** so với mức tối thiểu ghi trong `requirements.txt` (`>=4.53.2`, `>=0.3.25`) — là các phiên bản major khác nhau có breaking changes đã biết. Import vẫn thành công, nhưng hành vi runtime chưa được xác nhận tương đương.

Khung đánh giá tổng thể: **framework CHƯA sẵn sàng cho thực nghiệm chính thức**. Tin tốt: chưa có kết quả evaluation chính thức nào (`tests/validation/` rỗng) nên chưa có kết quả nào bị "nhiễm" cần phải loại bỏ — chỉ cần fix lỗi trước khi chạy `eval.py` lần đầu.

---

## 2. Repository Structure Comparison

### 2.1 Cây thư mục — file giống hệt / đã sửa / mới thêm / bị thiếu

| Trạng thái | File / thư mục | Ghi chú |
|---|---|---|
| ✅ Giống hệt (byte-by-byte) | `requirements.txt`, `src/eval.py`, `src/vector_store.py`, `src/oracle.py`, `src/cmd4tests.sh`, `src/prompts.json`, `src/prompts_no_rag.json`, `src/llm_sft/README.md`, `src/llm_sft/preprocessing_dataset.py`, `LICENSE`, `appendix.pdf`, `images/architecture_nap.png`, `slides-BPM26-journal-first.pdf` | Xác minh bằng `diff -u` cho các file Python/JSON; file nhị phân xác minh qua cây thư mục khớp tên/kích thước thư mục gốc. |
| 🟡 Đã chỉnh sửa | `src/main.py`, `src/pipeline.py`, `src/log_preprocessing.py`, `src/utility.py`, `README.md` (chỉ badge DOI/version) | Chi tiết ở Phần 3. |
| 🆕 Nhóm bổ sung (không có trong bản gốc) | `src/live_demo_qwen.py`, `requirements-core.txt`, `Enhance-next-activity-prediction.pdf`, `logs/*.xes` (9 file, thay cho `logs.zip`), `tests/outputs/live_demo_results_*.json`, `tests/outputs/output_*.txt` | Script demo mới và dữ liệu log đã giải nén; xem Phần 3.7–3.8. |
| ⚠️ Có trong gốc nhưng thiếu ở repo nghiên cứu | `logs.zip`, `.env` (bản gốc có file `.env` **được track trong git** với toàn bộ giá trị rỗng — xem Phần 4.4) | `logs.zip` bị xóa và thay bằng thư mục `logs/` đã giải nén — đây là thay đổi tổ chức, không ảnh hưởng thuật toán. |

### 2.2 Thống kê theo loại file

| Loại | Repo gốc | Repo nghiên cứu |
|---|---|---|
| File Python (`src/`) | 9 | 10 (+`live_demo_qwen.py`) |
| File cấu hình (`prompts*.json`, `.gitignore`, `.env`) | 3 (+`.env` track trong git) | 3 (`.env` không track — đúng thực hành) |
| File dữ liệu (`logs/`, `tests/test_sets/*.csv`) | 1 zip + 10 CSV | 9 `.xes` đã giải nén + 10 CSV (giống hệt nội dung cấu trúc, khác nội dung do bug — xem 3.5) |
| File test/kết quả (`tests/outputs`, `tests/validation`) | README placeholder only (bị `.gitignore` loại trừ) | README placeholder + 4 file kết quả demo thực tế đã bị **commit vào git** |
| Tài liệu/hình ảnh | `README.md`, `appendix.pdf`, `slides-*.pdf`, `images/*.png` | Giống hệt + thêm `Enhance-next-activity-prediction.pdf` |

---

## 3. Source Code Differences

### 3.1 `src/main.py` — ✅ Đã xác minh

```diff
     parser.add_argument('--rag', type=u.str2bool, ..., default=True)
+    parser.add_argument('--evaluation_modality', type=str, default='evaluation-attributes',
+                        help='Evaluation modality to use (e.g., evaluation-concept_names, evaluation-attributes)')
```
- **Mục đích:** đồng bộ CLI của `main.py` với `eval.py` (vốn đã có cờ này trong cả hai repo).
- **Tác động:** 🔎 Giá trị chỉ được ghi vào `run_data['Evaluation Modality']` (dòng 114) để log — **không được truyền** cho `process_prefixes` hay bất kỳ logic nào khác trong `main.py`. Đây là **tham số CLI "chết"** (inert) trong chế độ live-prompting.
- **Phân loại:** Điều chỉnh giao diện/cấu hình, không ảnh hưởng thuật toán.
- **Rủi ro kết quả:** Không — chỉ gây hiểu nhầm nhỏ (người dùng có thể nghĩ cờ này đổi hành vi nhưng không).

### 3.2 `src/eval.py` — ✅ Đã xác minh: **không có khác biệt nào**

`diff` giữa hai file rỗng tuyệt đối. Điều này có một hệ quả quan trọng: lỗi `TypeError: int() argument must be ... not 'NoneType'` tại dòng `GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT'))` mà người dùng gặp phải ở đầu phiên làm việc **là hành vi gốc của tác giả, không phải lỗi do nhóm nghiên cứu gây ra** — chỉ là thiếu file `.env` cục bộ (đã khắc phục trong phiên trước).

### 3.3 `src/pipeline.py` — ✅ Đã xác minh, phân loại: **Điều chỉnh tương thích phần cứng**

| Thay đổi | Trước (gốc) | Sau (nghiên cứu) | Mục đích | Rủi ro |
|---|---|---|---|---|
| Thêm `import torch` | — | dòng 5 | Cần cho logic `max_memory` mới | Không |
| Mở rộng danh sách model Qwen (dòng 29) | `['Qwen/Qwen2.5-7B-Instruct']` | `['...7B...', '...3B-Instruct', '...1.5B-Instruct']` | **Thay đổi cốt lõi cho phần cứng**: cho phép dùng model nhỏ vừa VRAM 6GB thay vì 7B | Không ảnh hưởng thuật toán, nhưng **đổi độ mạnh của LLM** → kết quả không so sánh trực tiếp 1:1 với bài báo gốc (vốn dùng 7B) |
| `BitsAndBytesConfig` thêm `llm_int8_enable_fp32_cpu_offload=True` (dòng 91) | không có | có | Cho phép offload phần không vừa VRAM sang CPU ở fp32 | 🔎 Lớp nào bị offload sẽ tính toán ở độ chính xác khác (fp32 CPU) so với phần chạy 4-bit trên GPU — có thể ảnh hưởng nhẹ đến kết quả sinh văn bản nếu offload thực sự xảy ra |
| Thêm tính `max_memory` động (dòng 95-99): `total_gb = torch.cuda.get_device_properties(0).total_memory`, trừ 1GiB, set `max_memory={0: f"{gpu_budget_gb}GiB", "cpu": "32GiB"}` | không có | có | Tránh OOM khi load model trên GPU nhỏ | ⚠️ **Dùng `total_memory` (tổng VRAM) chứ không phải VRAM còn trống thực tế.** Khi kiểm tra bằng `nvidia-smi`, GPU của máy hiện có 6141 MiB tổng nhưng chỉ 4823 MiB trống (1099 MiB đã bị tiến trình khác chiếm). Logic này có thể **đặt budget GPU cao hơn VRAM thực sự khả dụng**, gây nguy cơ CUDA OOM khi có ứng dụng khác (trình duyệt, IDE, v.v.) đang dùng GPU cùng lúc. 🔎 Suy luận từ code; chưa gây lỗi trong các lần chạy demo trước đó theo log hiện có, nhưng là rủi ro tiềm ẩn chưa được phòng vệ. |
| `open(path_prompts, 'r')` → `open(path_prompts, 'r', encoding='utf-8')` (dòng 150, 170) | không có `encoding` | có `encoding='utf-8'` | **Fix tương thích Windows**: mặc định Python trên Windows dùng codepage hệ thống (thường cp1252), không phải UTF-8, nên đọc JSON có ký tự ngoài ASCII có thể lỗi/đọc sai trên Windows | Không ảnh hưởng thuật toán |

**Kết luận 3.3:** Các thay đổi này đều hợp lý cho mục tiêu "chạy trên RTX 4050 6GB", ngoại trừ điểm `max_memory` dùng tổng VRAM thay vì VRAM trống — nên theo dõi khi chạy evaluation dài hơi.

### 3.4 `src/utility.py` — ✅ Đã xác minh, không ảnh hưởng thuật toán

Chỉ thêm `encoding='utf-8'` vào `open(filepath, 'a')` trong `log_to_file`. Cùng lý do Windows compatibility như 3.3.

### 3.5 `src/log_preprocessing.py` — 🔴 CRITICAL — Thay đổi logic xử lý, **phá vỡ phương pháp nghiên cứu gốc**

Đây là phát hiện quan trọng nhất của toàn bộ audit. So sánh hàm `process_prefixes()`:

**Bản gốc** (`original_repo/src/log_preprocessing.py`, dòng 52-114):
```python
cl_event_idx = []  # index in events of the event behind each entry of cl_list
for event_idx, event in enumerate(events):
    ...
    cl_list.append(event_name)
    cl_event_idx.append(event_idx)
    ...
# The last activity is the target: the prefix is everything before it, so it must not be part of the
# prefix nor contribute its attribute values
if len(cl_list) < 2:
    continue
last_evt = cl_list[-1]
cl_list_string = ','.join(cl_list[:-1])          # ⬅ LOẠI target khỏi prefix
...
for evt in events[:cl_event_idx[-2] + 1]:         # ⬅ LOẠI event của target khỏi attribute extraction
    ...
results[cl_list_string] = f'Values: {str(attr_vals)} | Next activity: {last_evt}'
```

**Repo nghiên cứu** (`src/log_preprocessing.py`, dòng 52-109):
```python
for event in events:          # ⬅ không còn track cl_event_idx
    ...
    cl_list.append(event_name)
    ...
if not cl_list:
    continue
cl_list_string = ','.join(cl_list)                # ⬅ GIỮ LẠI target trong prefix!
...
last_evt = cl_list[-1]
attr_vals = {}
for evt in events:                                 # ⬅ KHÔNG loại event của target
    ...
results[cl_list_string] = f'Values: {str(attr_vals)} | Next activity: {last_evt}'
```

**Mục đích thay đổi:** ❓ Không rõ ràng từ code — không có comment giải thích. Có khả năng đây là một refactor vô ý khi đơn giản hóa code (loại bỏ biến `cl_event_idx` "không dùng tới" nhưng thực ra nó quan trọng để cắt bỏ phần tử cuối).

**Tác động — đã xác minh bằng dữ liệu thực tế, không suy đoán:**

1. **CSV test set bị nhiễm nhãn.** So sánh `tests/test_sets/test_set_sepsis_evaluation-attributes.csv` giữa hai repo:
   - Repo gốc, dòng 2: `prefix` kết thúc bằng `...,LacticAcid`, cột `prediction` = `Leucocytes` (**khác nhau** — đúng thiết kế).
   - Repo nghiên cứu, dòng 2: `prefix` kết thúc bằng `...,Release A`, cột `prediction` = `Release A` (**giống hệt** — bị leak). Kiểm tra cả 4 dòng đầu của file nghiên cứu: **100% các dòng đều có activity cuối của chuỗi "prefix" trùng khớp chính xác với cột "prediction"**.

2. **Lan truyền vào prompt gửi cho LLM.** Truy vết luồng gọi: `eval.py` dòng 136-137 (`lp.build_prefixes` → `lp.process_prefixes`) → cột `prefix` của CSV được nạp bởi `u.load_csv_questions` → `eval.py` dòng 32-35 truyền thẳng `el[0]` (chuỗi prefix đầy đủ, **bao gồm activity mục tiêu**) vào `pipeline_instance.produce_answer(prefix, ...)` → `pipeline.py` dòng 183-212: biến `question` (chính là `prefix`) được đưa nguyên văn vào prompt: `f'Here is the prefix to predict: {question}\nAnswer: '`. **Không có bước nào cắt bỏ token cuối trước khi hiển thị cho LLM** (hàm `retrieve_context` trong `vector_store.py` dòng 120 có `query.split('-')[0]` nhưng chỉ dùng cho truy vấn retrieval nội bộ, **không** áp dụng cho chuỗi `question` gửi vào prompt chính).

3. **Bằng chứng thực nghiệm đã xảy ra:** File `tests/outputs/live_demo_results_2026-11-12_21-42-33.json` (đã chạy thật trên máy, không phải giả lập) — mục đầu tiên:
   ```json
   "prefix": "...,Admission NC,Admission IC,Leucocytes,LacticAcid,CRP - Values: {...}",
   "expected_next_activity": "CRP",
   ```
   Activity cuối trong `prefix` (`CRP`) trùng khớp 100% với `expected_next_activity`. Trong phần suy luận của model (log TXT cùng timestamp), model thậm chí tự nhận diện "Past Trace 1" trong ngữ cảnh retrieval **chính là cùng một case với câu hỏi** (cùng giá trị thuộc tính `'La': '0.6'`) — tức **self-retrieval đã xảy ra thật**, không phải lý thuyết.

4. **Lan truyền vào vector store.** `vector_store.py::store_traces()` (dòng 91-116, không đổi so với gốc) embed chính `cl_list_string` (khóa dict) làm vector — vì khóa này giờ **chứa activity mục tiêu**, toàn bộ corpus được embed/lưu trong Qdrant hiện nay bị nhiễm theo cùng cơ chế.

**Phân loại:** **Thay đổi logic xử lý** — không phải điều chỉnh phần cứng hay giao diện, đây là thay đổi ngữ nghĩa cốt lõi của bước tiền xử lý dữ liệu.

**Có làm sai lệch phương pháp nghiên cứu gốc không?** ✅ Có, đã xác minh. Phương pháp gốc định nghĩa rõ ràng "prefix = chuỗi activity trước activity cuối; target = activity cuối" (thể hiện qua comment gốc "the prefix is everything before it, so it must not be part of the prefix"). Code hiện tại vi phạm trực tiếp định nghĩa này.

**Nguy cơ ảnh hưởng kết quả thực nghiệm?** ✅ Có, nghiêm trọng (CRITICAL). Bất kỳ accuracy/precision/recall/F1 nào tính ra từ test set/vector store hiện tại đều không phản ánh khả năng dự đoán thực sự của hệ thống — vì câu trả lời đã nằm sẵn trong câu hỏi.

**Vị trí code cần sửa:**
- `src/log_preprocessing.py`, dòng 91: `cl_list_string = ','.join(cl_list)` → cần `','.join(cl_list[:-1])`.
- `src/log_preprocessing.py`, dòng 98: `for evt in events:` → cần loại trừ event cuối (khôi phục lại cơ chế `cl_event_idx` của bản gốc).
- `src/log_preprocessing.py`, dòng 89: điều kiện `if not cl_list: continue` nên khôi phục thành `if len(cl_list) < 2: continue` để tránh trace có đúng 1 activity (không có target hợp lệ) lọt qua.

### 3.6 Vấn đề kế thừa từ bản gốc (không phải lỗi của nhóm) — ❓ Chưa thể kết luận là "lỗi"

Trong cả hai repo (file `eval.py` giống hệt, hàm `generate_test_set` trong `log_preprocessing.py` giống hệt), tập test được lấy mẫu **từ cùng một pool `traces`** dùng để xây vector store (`eval.py` dòng 134-140: `traces` được dùng để build cả `prefixes`/vector-store lẫn `test_set`, không có bước tách train/test riêng biệt trước khi index). Điều này có nghĩa là **kể cả khi đã sửa lỗi ở mục 3.5**, một case trong test set vẫn có thể được chính nó (hoặc một prefix-con của chính trace đó) truy xuất lại từ vector store, vì cùng trace đã được index.

- 🔎 Đây **không phải thay đổi của nhóm nghiên cứu** — thiết kế này tồn tại y hệt trong bản gốc của tác giả.
- ❓ Việc đây có phải là lỗi phương pháp của chính bài báo gốc hay là chủ đích (ví dụ: đánh giá kiểu case-based reasoning cho phép truy xuất case tương tự/giống) thì **không thể kết luận nếu không đọc phần Methodology của bài báo** (`Enhance-next-activity-prediction.pdf`). Báo cáo này không giả định đây là lỗi, chỉ ghi nhận để nhóm nghiên cứu lưu ý và đối chiếu với bài báo.

### 3.7 `src/live_demo_qwen.py` — 🆕 File mới, do nhóm nghiên cứu bổ sung hoàn toàn

- **Mục đích:** ✅ (ghi rõ trong code, dòng 39-105) script demo nhanh — load model, build lại vector store, chạy 5 câu hỏi mẫu, in kết quả, không dùng oracle/metrics chính thức.
- **Tác động:** Script lấy mẫu demo **trực tiếp từ chính dict `prefixes` vừa dùng để build vector store** (dòng 52-54: `items = list(prefixes.items()); random.shuffle(items); samples = items[:N_SAMPLES]`) — nghĩa là mỗi câu hỏi demo **chắc chắn 100% trùng với một document đã index** (self-retrieval theo thiết kế, không phải xác suất). Kết hợp với lỗi 3.5, đây là lý do log demo ở mục 3.5(3) cho thấy leak rõ ràng đến vậy.
- **Phân loại:** Công cụ bổ sung tiện ích (không phải eval chính thức) — nhưng **kế thừa nguyên lỗi 3.5**, không nên dùng kết quả của script này làm căn cứ đánh giá chất lượng model.

### 3.8 File cấu hình/dữ liệu bổ sung khác

| File | Trạng thái |
|---|---|
| `requirements-core.txt` | 🆕 Nhóm thêm — xem Phần 6. |
| `logs/*.xes` (thay `logs.zip`) | 🆕 Giải nén sẵn 9 file log XES — tổ chức lại, không đổi nội dung dữ liệu. |
| `Enhance-next-activity-prediction.pdf` | 🆕 Bản PDF bài báo — tài liệu tham khảo, không ảnh hưởng code. |
| `README.md` | 🟡 Chỉ đổi 2 dòng badge DOI/version (`zenodo.21293307`→`zenodo.21807802`, `v1.0.2`→`v1.0.1`) — có vẻ là do clone tại thời điểm khác, không phải chỉnh sửa chủ đích. Không ảnh hưởng nội dung hướng dẫn sử dụng. |

---

## 4. Pipeline Integrity Analysis

Đối chiếu từng bước pipeline nêu trong yêu cầu kiểm toán:

| Bước | Trạng thái | Ghi chú |
|---|---|---|
| 1. Đọc XES | ✅ Giống gốc | `log_preprocessing.read_event_log` không đổi. |
| 2. Tiền xử lý | 🔴 **Đã sửa, có lỗi** | Xem Phần 3.5. |
| 3. Tạo prefix traces | 🔴 **Đã sửa, có lỗi** | `build_prefixes` không đổi, nhưng `process_prefixes` lỗi làm "prefix" chứa cả target. |
| 4. Tạo embeddings | ✅ Giống gốc (`pipeline.initialize_embedding_model`) | Không đổi logic, nhưng input (text) bị nhiễm bởi lỗi #2-3. |
| 5. Lưu vào Qdrant | ✅ Giống gốc (`vector_store.store_traces`) | Logic lưu không đổi; dữ liệu lưu vào bị nhiễm. |
| 6. Truy xuất prefix tương đồng | ✅ Giống gốc (`vector_store.retrieve_context`) | Không đổi; nhưng xem mục 3.6 về nguy cơ self-retrieval kế thừa từ thiết kế gốc. |
| 7. Xây dựng prompt | ✅ Giống gốc (`pipeline.produce_answer`, `_generate_prompt_template`) | Không đổi logic ghép prompt; vấn đề là **nội dung đưa vào** (`question`) đã bị nhiễm từ bước 2-3. |
| 8. Gọi LLM dự đoán | 🟡 Đã sửa (tương thích phần cứng) | Xem Phần 3.3 — đổi danh sách model + quản lý VRAM. |
| 9. Xử lý kết quả dự đoán | ✅ Giống gốc (`oracle.verify_answer`) | Không đổi. |
| 10. Tính metric & lưu kết quả | ✅ Giống gốc (`oracle.compute_stats`, `write_results_to_file`) | Không đổi — `sklearn.precision_score/recall_score/f1_score/accuracy_score` dùng `average='macro'`, đúng chuẩn. Earlyness bucket tính theo độ dài prefix (`oracle.py` dòng 42-45, đếm số activity tách bằng dấu phẩy) — **bị lệch +1 gián tiếp** vì chuỗi prefix hiện chứa thêm 1 activity (target) do lỗi #2-3. |

### 4.1 Data leakage — tổng kết theo yêu cầu kiểm toán

- **Cơ chế leak đã xác minh (CRITICAL, do nhóm gây ra):** `log_preprocessing.py` dòng 91 & 98 — target activity bị giữ lại trong chuỗi prefix và trong attribute extraction. Bằng chứng: CSV test set + log chạy thật (Phần 3.5).
- **Khả năng một prefix tự truy xuất chính nó:** ✅ Đã **quan sát thấy thật** trong `live_demo_qwen.py` (do thiết kế lấy mẫu từ chính corpus) và 🔎 **về lý thuyết cũng có thể xảy ra** trong `eval.py` chính thức vì test set và vector-store traces cùng nguồn (Phần 3.6) — nhưng đặc điểm này **kế thừa từ bản gốc**, chưa chạy `eval.py` thật trên repo nghiên cứu để đo tỷ lệ xảy ra cụ thể.
- **evaluation-concept_names vs evaluation-attributes:** ✅ Cả hai là tên file test set khác nhau (`test_set_{log}_{modality}.csv`), được chọn qua cờ `--evaluation_modality` trong `eval.py` (mặặc định `evaluation-concept_names`). Khác biệt về nội dung: "concept_names" chỉ dùng tên activity, "attributes" bao gồm thêm bảng `Values: {...}` thuộc tính sự kiện (dòng 103-104, 136 của `log_preprocessing.py`, không đổi giữa hai repo). Cơ chế sinh file giống hệt nhau về code — cả hai modality đều bị ảnh hưởng bởi lỗi 3.5 như nhau.
- **Cách tính Accuracy/Precision/Recall/F1:** ✅ Giống hệt bản gốc, dùng `sklearn`, `average='macro'`, không đổi.
- **Earlyness analysis:** ✅ Logic bucket giống hệt bản gốc; 🔎 nhưng do lỗi 3.5, độ dài prefix dùng để phân bucket bị tính dư 1 (bao gồm cả target).

---

## 5. Research Methodology Impact

| Thay đổi | Phân loại | Làm sai lệch phương pháp gốc? |
|---|---|---|
| Thêm model Qwen 3B/1.5B (pipeline.py:29) | Điều chỉnh tương thích phần cứng | Không sai lệch *phương pháp*, nhưng **đổi biến độc lập** (chọn LLM yếu hơn 7B) — kết quả không so sánh ngang hàng với con số trong bài báo gốc nếu không nêu rõ. |
| `max_memory`/CPU-offload (pipeline.py:91-99) | Điều chỉnh tương thích phần cứng | Không sai lệch thuật toán; rủi ro kỹ thuật (OOM) đã nêu ở 3.3. |
| `encoding='utf-8'` (pipeline.py, utility.py) | Không ảnh hưởng thuật toán | Không. |
| `--evaluation_modality` thêm vào main.py | Điều chỉnh giao diện/cấu hình | Không (tham số chưa có tác dụng thực sự). |
| **Lỗi `cl_list_string`/`attr_vals` (log_preprocessing.py)** | **Thay đổi logic xử lý** | ✅ **Có, nghiêm trọng** — phá vỡ định nghĩa prefix/target của phương pháp gốc. |
| Thiết kế test-set/vector-store chung nguồn | Kế thừa từ gốc | ❓ Chưa thể kết luận — cần đối chiếu bài báo. |

---

## 6. Environment Compatibility

### 6.1 Dependencies khai báo vs. đã cài đặt (venv thực tế)

| Package | `requirements.txt` (giống hệt 2 repo) | `requirements-core.txt` (chỉ repo nghiên cứu) | Đã cài trong `.venv` |
|---|---|---|---|
| torch | `>=2.7.0` | **không có** | `2.14.1+cu126` ✅ cài qua kênh khác |
| bitsandbytes | `>=0.45.5` | **không có** | `0.50.2` ✅ đã cài dù không có trong file core |
| xformers | `>=0.0.25.post1` | **không có** | **không cài** (nhất quán với việc bị loại khỏi core) |
| torchvision | `>=0.22.1` | **không có** | `0.29.1+cu126` ✅ |
| transformers | `>=4.53.2` | `>=4.53.2` | `5.23.0` ⚠️ lệch major version |
| langchain | `>=0.3.25` | `>=0.3.25` | `1.5.0` ⚠️ lệch major version (0.x → 1.x có breaking changes đã biết của LangChain) |
| accelerate | `>=1.6.0` | `>=1.6.0` | `1.15.0` (cùng major, ít rủi ro hơn) |
| sentence-transformers | `>=3.2.0` | `>=3.2.0` | `6.3.0` ⚠️ lệch nhiều minor/major |
| pm4py | `>=2.7.11.8` | `>=2.7.11.8` | `2.7.23.12` (cùng nhánh 2.7.x, rủi ro thấp) |

**Nhận xét:** 🔎 `requirements-core.txt` loại bỏ đúng 4 package nặng/khó cài trên Windows (torch, bitsandbytes, xformers, torchvision) — hợp lý vì các package này thường cần cài riêng theo URL index phù hợp với CUDA của máy (ở đây là CUDA 12.6). **Nhưng không có tài liệu nào** (README không cập nhật) giải thích vì sao có 2 file requirements và cách cài `torch`/`bitsandbytes` đúng cho GPU RTX 4050 — ảnh hưởng khả năng tái lập môi trường cho người khác. ⚠️ Việc `transformers`/`langchain`/`sentence-transformers` cài phiên bản **major mới hơn nhiều** so với mức tối thiểu ghi trong `requirements.txt` là rủi ro **chưa được xác nhận an toàn bằng evaluation thật** — import thành công (xem Phần 7) không đảm bảo hành vi inference giống hệt.

### 6.2 Python / CUDA / GPU — ✅ Đã xác minh

- Python: `3.13.6`
- `torch.cuda.is_available()`: `True`, GPU nhận diện: `NVIDIA GeForce RTX 4050 Laptop GPU`
- VRAM: `nvidia-smi` báo **6141 MiB tổng, 4823 MiB trống** (1099 MiB đã bị chiếm bởi tiến trình khác tại thời điểm kiểm tra) — xác nhận rủi ro đã nêu ở mục 3.3 về `max_memory` tính theo tổng thay vì VRAM trống.

### 6.3 Docker / Qdrant — ⚠️ Vấn đề chặn thực thi

- Container `qdrant_nap` tồn tại, mapping đúng cổng `6333-6334:6333-6334`.
- Trạng thái hiện tại: **`Exited (255)`**, lần chạy cuối cách đây vài phút tại thời điểm audit; log container không cho thấy lỗi crash rõ ràng (log dừng đột ngột, nhiều khả năng do máy tắt/khởi động lại).
- Thử kết nối thực tế bằng `QdrantClient` (gRPC, không in giá trị `.env`): **thất bại** với `DEADLINE_EXCEEDED` — xác nhận Qdrant hiện không sẵn sàng phục vụ.
- **Hành động cần thiết trước khi chạy bất kỳ script nào:** `docker start qdrant_nap`.

### 6.4 `.env` — kiểm tra tên biến, không đọc giá trị (theo đúng ràng buộc)

| Biến | Có trong `.env` gốc (đã track trong git, giá trị rỗng) | Có trong `.env` repo nghiên cứu (không track, tạo thủ công) |
|---|---|---|
| `HF_TOKEN` | ✅ | ✅ |
| `GOOGLE_API_KEY` | ✅ | ✅ |
| `OPENAI_API_KEY` | ✅ | ✅ |
| `DEEPSEEK_API_KEY` | ✅ | ✅ |
| `QDRANT_URL` | ✅ | ✅ |
| `QDRANT_GRPC_PORT` | ✅ | ✅ |

Tên biến khớp 100% giữa hai repo. ✅ Điểm tích cực: `.env` của repo nghiên cứu **không bị track trong git** (đúng thực hành bảo mật), trong khi bản gốc của tác giả có track file `.env` trên GitHub công khai — nhưng chỉ với giá trị rỗng nên không rò rỉ secret thật.

---

## 7. Testing Results

Chỉ thực hiện kiểm tra an toàn (không rebuild DB/test set, không chạy eval toàn bộ), theo đúng yêu cầu.

| Kiểm tra | Kết quả | Ghi chú |
|---|---|---|
| `python -m py_compile` toàn bộ `src/*.py` + `src/llm_sft/*.py` (9 file) | ✅ PASS tất cả | Không có lỗi cú pháp. |
| Import `log_preprocessing`, `oracle`, `utility`, `vector_store`, `pipeline` | ✅ PASS | Thành công dù dùng `transformers 5.23.0`/`langchain 1.5.0` (lệch xa `requirements.txt`). |
| `python src/main.py --help` | ✅ PASS | Hiển thị đúng, bao gồm cờ `--evaluation_modality` mới. |
| `python src/eval.py --help` | ✅ PASS | Hiển thị đúng. (Trước đó lỗi do thiếu `.env`, nay đã có `.env` nên qua.) |
| Kết nối Qdrant (`QdrantClient(...).get_collections()`) | ❌ FAIL | `DEADLINE_EXCEEDED` — container đang dừng. |
| Load model Qwen2.5-3B-Instruct 4-bit thật (inference) | **Không thực hiện** | Chủ động bỏ qua để tránh tải GPU/tải mạng (download) không cần thiết, theo đúng giới hạn "an toàn, không gây tải lớn". Bằng chứng gián tiếp: log `tests/outputs/output_2026-11-12_21-42-33.txt` và `live_demo_results_*.json` cho thấy pipeline **đã từng load và infer thành công** trên máy này trước đây. |
| Chạy evaluation chính thức (`eval.py`) | **Không thực hiện** (theo yêu cầu) | `tests/validation/` hiện chỉ có `README.md` — xác nhận **chưa từng có lần chạy `eval.py` hoàn chỉnh nào** trên repo này. |

---

## 8. Identified Issues (tổng hợp theo mức độ ưu tiên)

| # | Vấn đề | File/dòng | Mức độ |
|---|---|---|---|
| 1 | Prefix chứa activity mục tiêu (data leakage) | `src/log_preprocessing.py:91,98` | **CRITICAL** |
| 2 | Qdrant container đang dừng, không kết nối được | Docker container `qdrant_nap` | **CRITICAL** (nhưng dễ khắc phục — chặn mọi thực thi) |
| 3 | Lệch version lớn `transformers`/`langchain`/`sentence-transformers` so với `requirements.txt`, chưa kiểm chứng hành vi runtime | `requirements.txt`, venv thực tế | **HIGH** |
| 4 | `requirements-core.txt` loại bỏ torch/bitsandbytes/xformers/torchvision nhưng không có tài liệu hướng dẫn cài đặt riêng | `requirements-core.txt`, `README.md` | **HIGH** |
| 5 | `max_memory` tính theo tổng VRAM thay vì VRAM trống → nguy cơ OOM khi có tiến trình GPU khác chạy song song | `src/pipeline.py:95-99` | **MEDIUM** |
| 6 | Test set và vector-store traces lấy từ cùng nguồn, không tách train/test tường minh (kế thừa từ bản gốc) | `src/eval.py:134-142`, `src/log_preprocessing.py` (hàm `generate_test_set`) | **MEDIUM** (pre-existing, cần đối chiếu bài báo) |
| 7 | `--evaluation_modality` trong `main.py` không có tác dụng thực tế (dead parameter) | `src/main.py:73-74,114` | **MEDIUM** |
| 8 | `.gitignore` của repo nghiên cứu không loại trừ `tests/outputs/`, `tests/validation/` như bản gốc → artifact thực nghiệm (bao gồm file chứa bằng chứng leak) đã bị commit vào git history | `.gitignore`, commit `0af4ffe` | **MEDIUM** |
| 9 | Thiếu tài liệu hóa việc đổi model 7B→3B/1.5B ảnh hưởng đến khả năng so sánh kết quả với bài báo gốc | `README.md` | **LOW** |
| 10 | `"Values: Values:"` bị lặp trong cột `prefix` của CSV test set | `src/log_preprocessing.py` (hàm `generate_csv_from_test_set`) | **LOW** (lỗi mỹ quan, **kế thừa từ bản gốc**, không phải do nhóm gây ra) |

---

## 9. Recommended Fixes

1. **(CRITICAL)** Sửa `process_prefixes()` trong `src/log_preprocessing.py`:
   - Khôi phục việc loại activity cuối khỏi `cl_list_string` (`cl_list[:-1]` thay vì `cl_list`).
   - Khôi phục cơ chế `cl_event_idx` để `attr_vals` chỉ lấy thuộc tính tới trước event mục tiêu.
   - Khôi phục điều kiện `if len(cl_list) < 2: continue`.
   - Sau khi sửa, **phải xóa và tạo lại** vector store (`--rebuild_db_and_tests true`) và test set CSV hiện có, vì dữ liệu hiện tại đã bị nhiễm từ trước.
2. **(CRITICAL, vận hành)** `docker start qdrant_nap` trước khi chạy `main.py`/`eval.py`/`live_demo_qwen.py`.
3. **(HIGH)** Ghi lại chính xác (vd. `pip freeze`) phiên bản thực tế của `transformers`, `langchain`, `sentence-transformers`, `torch`, `bitsandbytes` đang dùng vào một file riêng (vd. `requirements-lock.txt`) để đảm bảo khả năng tái lập, thay vì chỉ ghi mức tối thiểu không khớp thực tế.
4. **(HIGH)** Bổ sung mục trong `README.md` hướng dẫn cài `torch`/`torchvision`/`bitsandbytes` đúng kênh cho CUDA 12.6 trên Windows, giải thích lý do tách `requirements-core.txt`.
5. **(MEDIUM)** Đổi `max_memory` trong `pipeline.py` sang dùng `torch.cuda.mem_get_info()` (bộ nhớ trống thực tế) thay vì `get_device_properties().total_memory`.
6. **(MEDIUM)** Đối chiếu với mục Methodology của bài báo gốc (`Enhance-next-activity-prediction.pdf`) để xác nhận việc test-set/vector-store dùng chung nguồn traces có phải chủ đích hay không; nếu không, cân nhắc tách riêng tập traces cho vector store và test set.
7. **(MEDIUM)** Cập nhật `.gitignore` để loại trừ `tests/outputs/` và `tests/validation/` (theo đúng bản gốc), tránh commit artifact thực nghiệm vào git; cân nhắc `git rm --cached` các file đã lỡ commit (cần xác nhận với nhóm trước khi thực hiện vì đây là thao tác có thể đổi lịch sử làm việc).
8. **(LOW)** Gắn `--evaluation_modality` trong `main.py` vào logic thực tế hoặc ghi chú rõ đây là tham số dành riêng cho `eval.py`.
9. **(LOW)** Ghi rõ trong README/báo cáo nghiên cứu rằng kết quả dùng Qwen2.5-3B/1.5B-Instruct không tương đương trực tiếp với con số 7B trong bài báo gốc.

---

## 10. Overall Assessment

### 10.1 Bảng mức độ hoàn thiện

| Hạng mục | Trạng thái | Vấn đề | Mức độ ưu tiên |
|---|---|---|---|
| Source code | Chạy được (cú pháp/import OK) | Lỗi logic nghiêm trọng ở tiền xử lý prefix | **CRITICAL** |
| Pipeline | Chạy hết luồng về mặt kỹ thuật | Input bị nhiễm nhãn ngay từ bước tiền xử lý | **CRITICAL** |
| RAG retrieval | Hoạt động về mặt kỹ thuật | Corpus bị nhiễm nhãn; self-retrieval đã quan sát thấy thật trong demo | **CRITICAL** |
| LLM integration | Hoạt động (từng infer thành công theo log cũ) | `max_memory` theo tổng VRAM thay vì VRAM trống; version model khác 7B gốc | **MEDIUM–HIGH** |
| Qdrant | Container tồn tại | Đang dừng (Exited 255), không kết nối được lúc kiểm tra | **CRITICAL (vận hành, dễ sửa)** |
| Evaluation | Chưa từng chạy chính thức (`tests/validation/` rỗng) | Code tính metric đúng, nhưng sẽ vô nghĩa cho đến khi fix lỗi #1 | **CRITICAL** |
| Dataset | Đầy đủ 9 log XES | Không có vấn đề về tính đầy đủ | **LOW** |
| Dependencies | Cài đặt được, import OK | Lệch version lớn so với yêu cầu tối thiểu, chưa kiểm chứng hành vi | **HIGH** |
| Reproducibility | Một phần | Thiếu doc cài GPU; artifact thực nghiệm lẫn vào git; version drift | **HIGH** |

### 10.2 Kết luận

- **Những gì đã xác minh (✅):** Phần lớn code (8/10 module cốt lõi) là bản sao trung thực từ tác giả gốc; các thay đổi phần cứng (model list, encoding, max_memory) là hợp lý về mục đích; **một lỗi logic nghiêm trọng, có bằng chứng thực nghiệm cụ thể**, đã được tìm thấy trong bước tạo prefix.
- **Những gì suy luận từ code (🔎):** Rủi ro OOM do tính VRAM sai cách; khả năng self-retrieval trong `eval.py` chính thức (chưa chạy thử để đo tỷ lệ).
- **Những gì chưa thể xác minh (❓):** Liệu thiết kế test-set/vector-store dùng chung nguồn traces có phải là hạn chế đã biết của chính bài báo gốc hay không — cần đọc kỹ phần Methodology/Limitations của `Enhance-next-activity-prediction.pdf`.
- **Framework có sẵn sàng cho thực nghiệm chính thức không?** **Chưa.** Cần sửa lỗi CRITICAL #1 (và khởi động lại Qdrant) trước khi chạy `eval.py` lần đầu. Không có kết quả thực nghiệm chính thức nào tồn tại cho đến nay, nên việc sửa lỗi bây giờ **không làm mất bất kỳ dữ liệu nghiên cứu nào đã có** — đây là thời điểm thuận lợi để khắc phục trước khi bắt đầu thu thập kết quả chính thức.
