# Lỗi Data Leakage trong `src/log_preprocessing.py`

**Mức độ:** 🔴 CRITICAL
**Trạng thái:** ✅ Đã phát hiện, đã fix, đã xác minh lại bằng chạy thực tế
**File liên quan:** `src/log_preprocessing.py` (hàm `process_prefixes()`)
**Lan truyền tới:** `src/vector_store.py`, `src/pipeline.py`, `src/eval.py`, `src/live_demo_qwen.py`

---

## 1. Tóm tắt

Trong quá trình đồng bộ/chỉnh sửa code từ repository gốc của tác giả (`angelo-casciani/rag_next_activity`) sang repository nghiên cứu, hàm `process_prefixes()` trong `src/log_preprocessing.py` đã vô tình **giữ lại chính activity mục tiêu (ground truth) bên trong chuỗi "prefix"** thay vì loại trừ nó như thiết kế gốc.

Hệ quả: khi hệ thống hỏi LLM "activity tiếp theo sau chuỗi này là gì?", **câu hỏi đã chứa sẵn đáp án** ở vị trí cuối cùng của chuỗi. Bài toán "dự đoán next activity" bị biến tướng thành "đọc lại activity cuối cùng vừa được cho sẵn".

---

## 2. Vị trí lỗi và nguyên nhân kỹ thuật

### 2.1 Thiết kế đúng (bản gốc của tác giả)

Bản gốc định nghĩa rõ ràng qua comment trong code:

> *"The last activity is the target: the prefix is everything before it, so it must not be part of the prefix nor contribute its attribute values"*

Để làm được điều này, bản gốc theo dõi chỉ số (`cl_event_idx`) của từng activity trong danh sách sự kiện gốc, rồi dùng nó để **cắt bỏ phần tử cuối** ở hai chỗ:

```python
cl_event_idx = []  # index in events của từng entry trong cl_list
...
if len(cl_list) < 2:
    continue
last_evt = cl_list[-1]
cl_list_string = ','.join(cl_list[:-1])          # ⬅ loại activity mục tiêu khỏi prefix
...
for evt in events[:cl_event_idx[-2] + 1]:         # ⬅ loại event của activity mục tiêu khỏi attribute extraction
    ...
```

### 2.2 Code bị lỗi (repo nghiên cứu, trước khi fix)

Khi refactor, biến `cl_event_idx` bị coi là "không cần thiết" và bị loại bỏ — nhưng vai trò thực sự của nó (cắt bỏ phần tử cuối) lại không được thay thế bằng cơ chế tương đương:

```python
# không còn cl_event_idx
if not cl_list:
    continue
cl_list_string = ','.join(cl_list)                # ⬅ GIỮ LẠI activity mục tiêu trong prefix!
...
for evt in events:                                 # ⬅ KHÔNG loại trừ event của activity mục tiêu
    ...
```

Hai thay đổi này làm cho:
- `cl_list_string` (chuỗi "prefix" được embed vào Qdrant và hiển thị cho LLM) **chứa luôn activity cuối cùng (target)**.
- `attr_vals` (bảng thuộc tính đi kèm) **chứa luôn giá trị thuộc tính của chính event mục tiêu**.

---

## 3. Cơ chế lan truyền lỗi qua toàn bộ pipeline

```
log_preprocessing.process_prefixes()
        │  cl_list_string chứa cả target
        ▼
┌───────────────────────────────────────────────────────────┐
│  main.py / eval.py                                         │
│  prefixes, _, _ = lp.process_prefixes(...)                 │
│  vs.store_traces(prefixes, ...)  ──► embed cl_list_string   │
│                                        vào Qdrant (bị nhiễm) │
│  lp.generate_csv_from_test_set(...) ──► cột "prefix" trong  │
│                                        CSV test set bị nhiễm │
└───────────────────────────────────────────────────────────┘
        │
        ▼
eval.py: el = (prefix, expected_prediction)  # prefix = cl_list_string (có target)
         pipeline_instance.produce_answer(prefix, ...)
        │
        ▼
pipeline.py: produce_answer(question, ...)
    question = prefix  (KHÔNG bị cắt bỏ token cuối ở bước này)
    prompt = f'Here is the prefix to predict: {question}\nAnswer: '
        │
        ▼
   LLM nhận được câu hỏi đã có sẵn đáp án ở cuối chuỗi
```

Ghi chú: `vector_store.retrieve_context()` có `query.split('-')[0].strip()`, nhưng dòng này **chỉ** dùng để tạo câu truy vấn nội bộ cho việc tìm kiếm tương đồng trong Qdrant — **không** áp dụng cho chuỗi `question` thực sự được đưa vào prompt gửi cho LLM.

---

## 4. Bằng chứng thực nghiệm (không suy đoán)

### 4.1 So sánh CSV test set giữa hai repo

File `tests/test_sets/test_set_sepsis_evaluation-attributes.csv`:

| Nguồn | Dòng ví dụ | Activity cuối trong cột `prefix` | Cột `prediction` | Trùng? |
|---|---|---|---|---|
| **Bản gốc** | dòng 2 | `...,LacticAcid` | `Leucocytes` | ❌ Khác nhau (đúng) |
| **Repo nghiên cứu (trước fix)** | dòng 2 | `...,Release A` | `Release A` | ✅ Trùng 100% (lỗi) |

Kiểm tra toàn bộ 4 dòng đầu của file lỗi: **100% các dòng đều có activity cuối trùng khớp chính xác với ground truth.**

### 4.2 Bằng chứng từ log chạy thật (trước khi fix)

File `tests/outputs/live_demo_results_2026-11-12_21-42-33.json`:

```json
{
  "prefix": "...,Admission NC,Admission IC,Leucocytes,LacticAcid,CRP - Values: {...}",
  "expected_next_activity": "CRP"
}
```

Model thậm chí tự nhận diện trong phần suy luận rằng "Past Trace 1" trong ngữ cảnh retrieval **chính là cùng một case với câu hỏi** (cùng giá trị thuộc tính `'La': '0.6'`) — self-retrieval đã xảy ra thật, không chỉ là rủi ro lý thuyết.

---

## 5. Cách fix

Khôi phục đúng 4 điểm trong `process_prefixes()` về logic của bản gốc:

1. Thêm lại `cl_event_idx = []` và `enumerate(events)` để theo dõi chỉ số event của từng activity.
2. Append `cl_event_idx.append(event_idx)` tại cả 3 nhánh append `cl_list`.
3. Đổi điều kiện `if not cl_list:` → `if len(cl_list) < 2:`.
4. Đổi `cl_list_string = ','.join(cl_list)` → `cl_list_string = ','.join(cl_list[:-1])`, và `for evt in events:` → `for evt in events[:cl_event_idx[-2] + 1]:`.

Sau fix, `diff` giữa `src/log_preprocessing.py` của repo nghiên cứu và bản gốc trên GitHub cho kết quả **rỗng tuyệt đối** (giống hệt 100%, exit code 0).

---

## 6. Xác minh sau khi fix

Chạy lại `live_demo_qwen.py` (5 câu hỏi mẫu, Qwen2.5-3B-Instruct, log `sepsis.xes`) sau khi fix và rebuild lại vector store:

| Query | Activity cuối trong `prefix` | Expected next activity | Model dự đoán | Đúng? |
|---|---|---|---|---|
| 1 | Release A | Return ER | Return ER | ✅ |
| 2 | Leucocytes | CRP | CRP | ✅ |
| 3 | CRP | Leucocytes | Leucocytes | ✅ |
| 4 | IV Liquid | CRP | IV Liquid | ❌ |
| 5 | IV Liquid | Admission NC | Admission NC | ✅ |

**Kết quả quan trọng nhất:** activity cuối trong `prefix` giờ **luôn khác** với `expected_next_activity` ở cả 5/5 câu — đúng như kỳ vọng. Model đạt 4/5 (80%) — một kết quả hợp lý cho một bài toán dự đoán thật sự, không còn là "đọc đáp án có sẵn".

---

## 7. Tác động

- **Trước fix:** mọi accuracy/precision/recall/F1-score tính ra từ `eval.py` đều **không có giá trị khoa học**, vì mô hình chỉ cần đọc token cuối của câu hỏi để trả lời đúng gần như tuyệt đối.
- **May mắn:** tại thời điểm phát hiện, `tests/validation/` (nơi lưu kết quả evaluation chính thức của `oracle.py`) hoàn toàn trống — chưa có kết quả nghiên cứu chính thức nào bị lỗi này làm sai lệch. Chỉ có các file demo/smoke-test (`tests/outputs/`) là đã bị ảnh hưởng, và các file này không được dùng để tính metric.
- **Sau fix:** cần chạy lại `--rebuild_db_and_tests true` (giá trị mặc định của `main.py`/`eval.py`) trước khi thực hiện bất kỳ evaluation chính thức nào, để sinh lại vector store và CSV test set bằng code đã sửa.
