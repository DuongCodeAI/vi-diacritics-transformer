# vi-diacritics-transformer

Thêm dấu cho tiếng Việt gõ không dấu bằng **Transformer tự viết từ đầu** (PyTorch thuần, không dùng
`nn.Transformer` / `nn.MultiheadAttention`), chạy ONNX int8 trên CPU và **ngay trên trình duyệt**.

```
"xe may vuot den do phat bao nhieu"  ->  "xe máy vượt đèn đỏ phạt bao nhiêu"
```

Demo (chạy trong trình duyệt, không gửi gì lên server): https://duongcodeai.github.io/vi-diacritics-transformer/

Dùng trong [vn-traffic-law-rag](https://github.com/DuongCodeAI/vn-traffic-law-rag) để xử lý câu hỏi gõ không dấu:
khi chỉ bỏ dấu để so khớp, "o to vuot den do" khớp nhầm "đỏ" với "nồng **độ** cồn".

## Hai cách nhìn bài toán

**1. Gán nhãn từng ký tự (encoder-only)** - cách chính.
Bỏ dấu không đổi độ dài chuỗi, nên mỗi ký tự đầu vào chỉ cần một nhãn = (dấu mũ/móc, dấu thanh):
5 x 6 = 30 lớp. Mỗi chữ gốc chỉ một số lớp hợp lệ ('b' chỉ có lớp 0, 'i' không có mũ...) → mask lúc giải mã,
nên model **không thể** sinh ra chữ không tồn tại hay đổi chữ.

**2. Seq2seq + BPE (encoder-decoder)** - để so sánh.
Coi như dịch máy: câu không dấu → câu có dấu, token BPE tự train. Linh hoạt hơn nhưng chậm hơn và có thể
"bịa" chữ (thêm/bớt/đổi từ).

## Kết quả

> Baseline, tagger, đánh giá tập test đã chạy thật trên Colab (03-04/10/2026). Dòng seq2seq điền sau notebook 03.

**Tagger trên tập val** (3.000 câu Wikipedia, cùng phân phối với test, đo lúc train bằng PyTorch fp32):

| step | epoch | phút T4 | char acc | word acc | sent acc | bịa chữ |
|---|---|---|---|---|---|---|
| 2.000 | 0.13 | 4 | 0.817 | 0.731 | 0.036 | 0% |
| 8.000 | 0.51 | 15 | 0.928 | 0.894 | 0.248 | 0% |
| 16.000 | 1.02 | 31 | 0.949 | 0.925 | 0.346 | 0% |
| 32.000 | 2.05 | 62 | **0.963** | **0.945** | **0.454** | 0% |

Model 4,82M tham số, batch 128. Dừng ở epoch 2.05 (kế hoạch 3 epoch) cho đỡ giờ GPU: val vẫn tăng nhưng chậm,
~0.002 word acc mỗi 4.000 step.
ONNX: fp32 21,5 MB, int8 7,2 MB. So với baseline bigram (word acc 0.855 trên test): giảm số âm tiết sai từ 14,5%
xuống 5,5%, tức **ít lỗi hơn ~2,6 lần**.

**Tập test** (notebook 04, Colab CPU, 04/10/2026). Wikipedia: 3.000 câu không trùng train. nrl: bộ ngoài
`nrl-ai/vn-diacritic-eval` (1.227 câu, 4 thể loại). Tin nhắn: 60 câu kiểu chat tự soạn (`data/real_typing.txt`).
ms/câu đo bằng onnxruntime, CPU 1 luồng.

| tập | hệ thống | char acc | word acc | sent acc | bịa chữ | ms/câu |
|---|---|---|---|---|---|---|
| wiki | bigram baseline | 0.899 | 0.854 | 0.168 | 0% | 0.2 |
| wiki | tagger fp32 | 0.965 | 0.948 | 0.476 | 0% | 17.7 |
| wiki | **tagger int8** | **0.965** | **0.948** | **0.474** | 0% | **12.0** |
| nrl (cả 4) | bigram baseline | 0.836 | 0.764 | 0.117 | 0% | 0.1 |
| nrl (cả 4) | tagger int8 | 0.914 | 0.875 | 0.278 | 0% | 7.0 |
| nrl trang trọng | baseline → tagger int8 | | 0.885 → **0.977** | | 0% | |
| nrl kinh doanh | baseline → tagger int8 | | 0.819 → **0.902** | | 0% | |
| nrl hội thoại | baseline → tagger int8 | | 0.788 → **0.885** | | 0% | |
| nrl văn học | baseline → tagger int8 | | 0.733 → **0.853** | | 0% | |
| tin nhắn | bigram baseline | 0.780 | 0.664 | 0.100 | 0% | 0.1 |
| tin nhắn | tagger int8 | 0.823 | 0.731 | 0.050 | 0% | 3.6 |
| wiki | seq2seq | | | | | |

Đọc bảng:
- int8 gần như không mất gì (word acc 0.948 → 0.948), nhỏ hơn 3 lần (21,5 → 7,2 MB), nhanh hơn ~1,5 lần.
- Văn viết rất tốt (trang trọng 0.977), văn học / hội thoại kém hơn: Wikipedia ít câu kiểu đó.
- **Tin nhắn là điểm yếu thật**: word acc chỉ 0.731. Lỗi điển hình: đại từ thân mật bị đoán theo văn viết
  ("tao" → "tạo", "mày" → "may", "anh" → "ảnh"), từ ngắn đứng cuối câu ("ạ", "nha", "nhé"), tiếng lóng
  ("đen thật sự"). Cần thêm dữ liệu chat thật để train, không phải model to hơn.
- Bịa chữ 0% ở mọi tập: tagger chỉ gắn dấu, không thể đổi chữ.

- char acc chỉ tính trên ký tự "có lựa chọn" (nguyên âm, d); tính cả phụ âm/dấu cách thì số đẹp giả.
- word acc là số người dùng cảm nhận được.

## Chi tiết cài đặt

- `model.py`: scaled dot-product attention, multi-head (tự chia/gộp head), sinusoidal positional encoding,
  pre-LayerNorm, encoder/decoder layer, causal mask. ~5M tham số (6 layer, d_model 256).
- `labels.py`: tách ký tự thành chữ gốc + shape + tone qua Unicode NFD; ghép lại phải đặt dấu mũ/móc **trước**
  dấu thanh, không thì NFC không gộp được thành một ký tự.
- `data.py`: lọc câu Wikipedia (câu ít âm tiết có dấu → bỏ, tránh dạy model "không thêm dấu"); chuẩn hoá
  "hoà/hòa", "thuỷ/thủy" về một kiểu trước khi tạo nhãn; chia train/test theo hash câu.
- `baseline.py`: bigram âm tiết (chỉ đếm), để biết Transformer hơn bao nhiêu.
- `restorer.py`: API suy luận bằng onnxruntime; **giữ dấu người dùng đã gõ**; từ nào model kém tự tin thì
  để nguyên không dấu thay vì đoán bừa.
- `viz.py`: attention map, xem ký tự 'a' trong "ma" nhìn vào đâu để chọn ma / mà / má / mã / ma túy.
- `docs/`: demo web dùng onnxruntime-web, model int8 vài MB.

## Chạy

```bash
pip install -e ".[infer]"
```
```python
from vi_diacritics import Restorer
r = Restorer.load("export/", min_conf=0.6)
r.restore("toi dang di hoc")
r.restore_detail("ma toi khong biet")   # kèm độ tự tin từng từ
```

Train: `notebooks/01_data_baseline` (CPU) → `02_train_tagger` (T4) → `03_train_seq2seq` (T4) → `04_eval_export`.
Chạy trên Colab: mở notebook từ GitHub (File → Open notebook → GitHub → DuongCodeAI/vi-diacritics-transformer), chọn T4 GPU,
thêm Secret `HF_TOKEN` (cho notebook 04). Output và checkpoint lưu trên Google Drive (`MyDrive/ai-portfolio/vi-diacritics-transformer`);
mỗi lần train tối đa 150 phút, bị ngắt thì chạy lại notebook là train tiếp từ `last.pt`.
Chạy thử nhanh trên CPU: `python -m vi_diacritics.train --task tagger --data <thư mục có train.txt, val.txt> --layers 2 --d-model 64`.

## Hạn chế

- Train trên Wikipedia (văn viết) → văn nói, teencode, tên riêng nước ngoài yếu hơn; đo riêng bằng tập tin nhắn.
- Gõ sai chính tả (không phải chỉ thiếu dấu) không sửa được: tagger chỉ thêm dấu, không đổi chữ.
