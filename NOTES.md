# Ghi chú quyết định

## Vì sao gán nhãn ký tự thay vì seq2seq
- Bỏ dấu giữ nguyên độ dài → bài toán thực chất là phân loại từng vị trí. Seq2seq phải học thêm việc
  "chép lại đúng chữ" mà không được gì.
- Mask nhãn hợp lệ theo chữ gốc → bịa chữ = 0% theo thiết kế, đáng tin khi đưa vào pipeline RAG.
- Nhanh: một lần forward, không decode từng token. Chạy được trên trình duyệt.
- Seq2seq vẫn train để có số so sánh, không phải để deploy.

## Nhãn 30 lớp
- Shape: không / mũ (â ê ô) / trăng (ă) / móc (ơ ư) / gạch (đ). Tone: 6 thanh.
- 'a' có 3 shape x 6 tone = 18 lựa chọn, 'd' có 2, phụ âm khác có 1.
- Bẫy Unicode: compose phải theo thứ tự chữ + shape + tone. Đặt tone trước shape thì NFC không gộp được
  (cùng combining class 230 nên không tự sắp lại) → ra 2 code point, so chuỗi sai.

## Dữ liệu
- Wikipedia vi (20231101), stream từ HF, lấy tối đa 2 triệu câu train.
- Lọc câu có < 30% âm tiết mang dấu: tên riêng nước ngoài, tiếng Anh, code. Không lọc thì model học thiên về
  "không thêm dấu".
- Hai kiểu bỏ dấu "hoà" / "hòa" cùng tồn tại → nhãn nhiễu. Chuẩn hoá oa/oe/uy (âm tiết mở) về kiểu dấu ở
  nguyên âm đầu, trừ "qu" + y.
- Chia theo hash câu: câu trùng luôn vào cùng một tập.
- Không dùng corpus báo chí binhvq: tác giả đã ngừng phân phối (08/2026) vì vấn đề bản quyền.

## Model
- Pre-LN thay vì post-LN (paper gốc): ổn định hơn, ít phụ thuộc warmup dài, hợp với ngân sách GPU Kaggle.
- Sinusoidal PE: không giới hạn cứng độ dài lúc suy luận, câu ký tự-level dễ dài 200+.
- Chia sqrt(d) trong attention: tránh tích vô hướng lớn làm softmax bão hoà.
- Mask lớp không hợp lệ bằng finfo.min (không phải -inf) để fp16 không ra NaN.

## Smoke test trên CPU (trước khi tốn quota GPU)
- Train model tí hon trên 60 câu tự soạn: accuracy đứng yên ~0.48 (= đoán "không dấu" hết).
  Tưởng bug, kiểm tra thì: (1) cho học thuộc 1 batch thì loss 1.04 → 0.04, model đúng; (2) dữ liệu smoke
  lặp 8-30 lần + sampler gom câu cùng độ dài → mỗi batch toàn bản sao của 1 câu; (3) model 70K tham số cần
  ~100+ lượt qua dữ liệu mới thoát pha "đoán lớp đa số". Không phải bug, chỉ là smoke test quá ngắn.
- torch 2.14 export ONNX mặc định dùng dynamo (cần onnxscript) → dùng `dynamo=False`.

## Lần train thật đầu tiên bị kẹt (Colab T4, 03/10)
- Từ step 2.000 đến 8.000: word acc đứng yên 0.219-0.225, char acc ~0.45 = đúng bằng "để nguyên không dấu".
  Loss cũng chỉ 0.61 → 0.55. Lần này không phải smoke test ngắn: 8.000 step x 128 câu là 1 triệu câu.
- Nguyên nhân: `nn.Embedding` khởi tạo N(0, 1), rồi nhân sqrt(d_model) = 16 như paper → embedding cỡ 16,
  trong khi positional encoding chỉ cỡ 1. Thông tin vị trí bị lấn át, mà pre-LN còn chuẩn hoá luôn đầu vào
  mỗi layer → model gần như là "túi ký tự", không biết ký tự nào đứng cạnh ký tự nào → không phân biệt được
  "ma" trong "con ma" với "ma" trong "mà thôi".
- Paper gốc nhân sqrt(d) vì embedding khởi tạo nhỏ (std 1/sqrt(d)) và dùng chung với lớp output. Sửa: khởi tạo
  std = 1/sqrt(d) (`_init_embedding` trong `model.py`).
- Sau khi sửa: step 2.000 word acc 0.731 (trước 0.219), step 32.000 lên 0.945.
- Bài học: nhân sqrt(d) chỉ đúng khi đi cùng cách khởi tạo của nó; nên in tỉ lệ norm(embedding)/norm(PE) ngay
  khi dựng model.

## Restorer
- Giữ nguyên ký tự người dùng đã gõ có dấu (gõ thiếu dấu vài chữ là chuyện thường).
- `min_conf`: từ kém tự tin để nguyên không dấu. Với RAG, chữ không dấu vẫn khớp được index bỏ dấu;
  thêm sai dấu thì hỏng hẳn.

## Việc cần làm
- Chạy notebook 03 (seq2seq) và 04 (test + int8 + tốc độ), điền nốt bảng.
- Thử tăng d_model hoặc thêm dữ liệu hội thoại nếu tập tin nhắn yếu.
- Bổ sung câu nhắn tin thật của mình vào `data/real_typing.txt` (hiện là câu tự soạn).
