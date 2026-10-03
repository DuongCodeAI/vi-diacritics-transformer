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

> Chưa chạy. Điền sau khi train trên Kaggle (notebook 01-04).

Tập test: 3.000 câu Wikipedia (không trùng train), `nrl-ai/vn-diacritic-eval` (4 thể loại),
và ~60 câu kiểu tin nhắn tự soạn (`data/real_typing.txt`). ms/câu đo trên CPU 1 luồng.

| tập | hệ thống | char acc | word acc | sent acc | bịa chữ | ms/câu |
|---|---|---|---|---|---|---|
| wiki | bigram baseline | | | | 0% | |
| wiki | tagger fp32 | | | | 0% | |
| wiki | tagger int8 | | | | 0% | |
| wiki | seq2seq | | | | | |
| tin nhắn | tagger int8 | | | | 0% | |
| nrl hội thoại / văn học | tagger int8 | | | | 0% | |

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
