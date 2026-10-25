# Ghi chú học forecasting radar–Himawari

## 1. Bài toán và trạng thái

Mục tiêu đã chỉ định: dự báo CAPPI reflectivity sau đúng 10 phút bằng image-to-image
regression (hồi quy từ ảnh sang ảnh). Đơn vị là dBZ, không phải lượng mưa, sét hay bão.
Chỉ dùng signature A. Preprocessing và 8.774 cặp NPZ/JSON gốc giữ nguyên.

Đã triển khai: tạo forecast pairs đúng timestamp, chia tập theo ngày UTC, kiểm tra
đường dẫn và chống dùng chung sample/time/leakage group giữa các split. Day 11 bổ sung
nguồn vệ tinh chu kỳ trước và audit đủ 10 header/cặp; xem mục 8 cho trạng thái mới nhất.
Day 12 đã tính train-only normalization statistics (mục 9), chưa tạo tensor chuẩn hóa.
Chưa triển khai: Dataset/DataLoader, baseline, U-Net, training, inference/API.
Không cần cài PyTorch cho bước tạo pair.

## 2. Tôi cần hiểu — forecast pairs

1. **Khái niệm:** một NPZ chỉ mô tả quan trắc tại một thời điểm, chưa chứa mục tiêu tương lai.
   Mẫu dự báo là liên kết từ input NPZ tại t tới radar trong target NPZ tại t+600 giây.
2. **Code:** `src/nhabe_forecasting/pairs.py`, `scripts/09_build_forecast_pairs.py`;
   quyết định thí nghiệm nằm trong `configs/forecasting.yaml`.
3. **Input/output:** đọc index và kiểm tra NPZ tồn tại, ghi CSV mới và summary JSON.
   Không sao chép hoặc sửa NPZ. Summary giữ SHA256 index/config để tái lập.
4. **Shape:** giai đoạn này xử lý hàng metadata, không tạo tensor. Hai NPZ tham chiếu
   vẫn chứa radar `(256,256)`, Himawari `(5,256,256)` và các mask.
5. **Kiểm chứng:** lookup theo radar midpoint UTC; target phải đúng `t + timedelta(minutes=10)`.
   CSV được mở lại và đối chiếu từng trường. Số cặp được tính, không ép về dự kiến.
6. **Lỗi thường gặp:** dùng tên file/thời gian danh nghĩa thay midpoint, làm tròn giây,
   dùng nearest-neighbour temporal matching, lấy B làm target, trùng pair, đường dẫn sai.
7. **Ba câu hỏi:** Vì sao một NPZ chưa là mẫu dự báo? Cặp chênh 599,5 giây có được nhận
   không? CSV tham chiếu file khác gì sao chép toàn bộ tensor?

Kiểm tra index hiện tại: 4.387 mẫu A, tất cả NPZ tồn tại; có 842 cặp đúng 600 giây
cùng ngày. Có 4.284 cặp theo chu kỳ vệ tinh danh nghĩa liên tiếp nhưng không phải tất cả
đều có radar midpoint chênh 600 giây. Số dự kiến 4.284 không phải tiêu chí ép kết quả.
Việc lọc exact timestamp làm giảm số mẫu và có thể tạo selection bias; cần báo cáo giới hạn này.

## 3. Tôi cần hiểu — chia tập và temporal leakage

1. **Khái niệm:** data leakage là việc thông tin không hợp lệ cho đánh giá lọt vào quy trình
   học/đánh giá. Ảnh gần nhau có tương quan; random từng ảnh dễ tạo kết quả quá lạc quan.
2. **Code:** `src/nhabe_forecasting/splits.py`, `tests/test_forecast_splits.py`.
3. **Input/output:** split gắn trong manifest: train 01–21, validation 22–26, test 27–31/08/2025.
   Đây là ngày UTC. Input và target phải cùng ngày, cùng split.
4. **Shape:** không thay shape mảng; chỉ phân chia danh sách cặp.
5. **Kiểm chứng:** không dùng chung sample ID, thời điểm hoặc leakage group giữa split;
   không trùng input-target edge. Một sample có thể là target của pair trước và input của
   pair sau **trong cùng split**, điều này không tự nó là cross-split leakage.
6. **Lỗi thường gặp:** chia input trước rồi để target sang split kế tiếp; tính normalization
   trước khi chia tập; dùng test để chọn checkpoint. Train trước validation/test mô phỏng
   tốt hơn dự báo thời gian tương lai, nhưng không bảo đảm tổng quát hóa sang tháng khác.
7. **Ba câu hỏi:** Vì sao không random từng ảnh? Pair 21/08 23:55 → 22/08 00:05 bị loại
   vì những quy tắc nào? Một sample xuất hiện trong hai pair cùng train có luôn là leakage không?

Kết quả exact pairs theo cấu hình: train 519, validation 144, test 179.
`qc_status=PASS` của pair chỉ là kiểm tra cấu trúc, không phải radar QC hoặc chứng minh
mọi feature đã có sẵn tại thời điểm phát hành dự báo.

## 4. Vấn đề nhân quả phát hiện trước Day 11

Trong 4.387 mẫu A, local Himawari line time muộn hơn radar midpoint từ 147,625976 đến
163,983053 giây. Ví dụ radar đầu ngày có midpoint 00:01:31,5Z nhưng vệ tinh là
00:04:05,140228Z. Nếu dự báo được phát hành lúc radar midpoint thì ảnh này chưa được
quan trắc. Thời điểm quan trắc còn khác thời điểm sản phẩm có sẵn để sử dụng.
Radar volume cũng kết thúc sau midpoint. Cặp exact đầu tiên có radar midpoint
00:21:31,5Z, radar end 00:22:54Z và satellite line time 00:24:05,121761Z.
Không được mặc định toàn bộ radar volume có sẵn tại midpoint, kể cả mô hình radar-only.

Manifest giữ thêm thời gian vệ tinh và độ lệch. Không được coi pipeline multimodal là
dự báo nhân quả từ t cho đến khi chốt quy tắc issue time và feature availability.
Chưa thay satellite cycle, chưa đổi định nghĩa lead time, chưa sửa NPZ gốc.

## 5. Lộ trình sau khi chốt thời gian input

- **Normalization:** chỉ pixel hợp lệ của train, mean/std riêng radar và năm band;
  ghi JSON có provenance. Không dùng validation/test để tính thống kê. Cần chốt cách
  đếm sample dùng chung giữa các pair để tránh vô tình tăng trọng số một quan trắc.
- **NaN và mask:** NPZ gốc vẫn giữ NaN. Sau chuẩn hóa mới đặt phần thiếu bằng 0 trong
  tensor bộ nhớ; 0 lúc đó là giá trị đệm, không phải 0 dBZ hoặc 0 K. Mask phải được giữ.
- **Dataset/DataLoader:** radar-only dự kiến `(B,2,256,256)`; multimodal `(B,8,256,256)`;
  target và target mask `(B,1,256,256)`. B ở đây là batch size, không phải signature B.
  Nếu một band thiếu nhưng all-band mask là False, phải xác định cách đệm các band
  nhất quán; không được vô tình xem chúng là đủ năm band hợp lệ.
- **Persistence baseline:** dự báo radar(t+10)=radar(t), không train. Trước tính MAE/RMSE
  cần chốt evaluation mask chung cho cả ba phương pháp, kiểm tra số pixel bị loại.
- **CNN/U-Net:** convolution khai thác cấu trúc lân cận; encoder giảm kích thước không
  gian, bottleneck biểu diễn nén, decoder khôi phục, skip connections truyền chi tiết.
  Small U-Net base channels 16, output một lớp reflectivity, không output xác suất mưa.
- **Masked loss:** chỉ tính lỗi trên pixel target hợp lệ, không chia cho mask rỗng.
  Training mask và evaluation mask có thể khác mục đích nhưng phải công bố rõ.
- **Training/validation/test:** train tối ưu trọng số, validation chọn checkpoint/early
  stopping, test đánh giá cuối. Không chọn mô hình dựa vào kết quả test.
- **Overfitting:** trước full training phải forward/backward, gradient hữu hạn và học
  được 16–32 pair. Loss giảm trên tập nhỏ kiểm tra luồng dữ liệu, không chứng minh tổng quát hóa.
- **Metrics:** masked MAE đo sai số tuyệt đối, masked RMSE nhạy với sai số lớn, đều báo
  ở dBZ. Cần định nghĩa tích lũy theo pixel hay theo pair và dùng cùng protocol.
- **Inference:** đọc NPZ/checkpoint/stats, kiểm tra contract, chuẩn hóa như train,
  dự báo và đổi về dBZ; không ghi đè input. CLI trước API, Docker inference riêng.
- **Giới hạn:** chỉ một tháng, exact pairing giảm mẫu, chưa radar QC chuyên sâu,
  chưa chứng minh model tốt hơn persistence hoặc Himawari mang lại cải thiện.

Những mục trên là kế hoạch, không phải chức năng đã hoàn thành. Mỗi giai đoạn sẽ được
bổ sung code, test, output thật và phần “Tôi cần hiểu” trước khi chuyển bước.

## 6. Chạy bước pair từ bất kỳ thư mục nào

Truyền đường dẫn tuyệt đối vào script và `--data-root` của workspace chứa dataset.
Code nằm trong repository portfolio; dữ liệu vẫn ở workspace nghiên cứu bên ngoài.

```bash
python /absolute/path/to/portfolio/scripts/09_build_forecast_pairs.py \
  --data-root /absolute/path/to/research-workspace \
  --output /absolute/path/to/research-workspace/output/forecasting/observation_timing
```

Output phải là thư mục trống để không ghi đè một thí nghiệm đã có. Không gọi preprocessing
hay đọc raw. Generated manifest có đường dẫn riêng trên máy và được giữ ngoài Git.

## 7. Day 10 — kiểm tra header vệ tinh đại diện (20/10/2026)

### Phạm vi và cách kiểm tra

Chỉ đọc 20 tệp HSD có sẵn: chu kỳ 0010 ngày 01 và 02/08/2025, mỗi chu kỳ gồm
B08/B13/B14/B15/B16 × S04/S05. Ngày 01 lấy từ raw, ngày 02 từ staging. Không tải mới,
không ghi vào nguồn, không hiệu chuẩn/resample ảnh, không sửa NPZ/JSON preprocessing.

Đọc tối đa 1.000.000 byte nén đầu mỗi tệp và giới hạn đầu ra giải nén 65.536 byte;
phân tích Block 1 và Block 9 bằng dtype của SatPy 0.60.0. Kiểm tra số/thứ tự/độ dài
block trước khi sử dụng. MJD chuyển sang UTC từ mốc 1858-11-17. Các tệp nguồn có mẫu tên
`HS_H09_YYYYMMDD_0010_Bbb_FLDK_R20_Sss10.DAT.bz2`; bb và ss là band/segment ở trên.

Theo [JMA HSD User's Guide](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/fig/HS_D_users_guide_en_v13.pdf),
Block 1 có observation start/end và file creation time, Block 9 lưu line number/time.
File creation time không phải timestamp nhận file trên máy hoặc thời điểm tensor sẵn sàng.

### Kết quả thực tế (UTC)

| Ngày / chu kỳ | Tệp | Observation start | Observation end | File creation |
| --- | --- | --- | --- | --- |
| 01/08 0010 | B13 S04 | 00:12:38.862031 | 00:14:05.130994 | 00:18:10 |
| 01/08 0010 | B13 S05 | 00:14:05.130994 | 00:15:05.838700 | 00:18:10 |
| 02/08 0010 | B13 S04 | 00:12:38.826253 | 00:14:05.095220 | 00:18:11 |
| 02/08 0010 | B13 S05 | 00:14:05.095220 | 00:15:05.802927 | 00:18:12 |

Ở cả hai chu kỳ, line-number records của S04 trải từ 1651–2200, S05 từ 2201–2750.
Trong 10 tệp mỗi chu kỳ, observation end muộn nhất lần lượt là 00:15:05.838700 và
00:15:05.802927; file creation muộn nhất là 00:18:18 và 00:18:20 (làm tròn tới giây).
Radar A chu kỳ 0020 kết thúc lúc 00:22:54 ở cả hai ngày. Những thời gian này hỗ trợ
candidate dùng vệ tinh chu kỳ trước cho radar hiện tại, chưa chứng minh mọi cặp trong tháng.

JSON ngày 01 ghi observation interval 00:10:21.545791–00:19:40.952263, rộng hơn interval
B13/S04 đọc trực tiếp. JSON ngày 02 ghi đúng interval B13/S04, nhưng không bao gồm S05.
Do đó không dùng một observation_end trong JSON để chứng minh cả 10 tệp đã hoàn tất.
Đây là vấn đề phạm vi metadata; không suy ra mảng brightness temperature sai và không
sửa metadata lịch sử. Array audit PASS không thay thế audit thời gian phục vụ forecasting.

### Quy tắc đề xuất, chưa áp dụng vào config/manifest

1. Cutoff nghiên cứu là radar input observation end, không phải midpoint.
2. Radar target A có midpoint đúng input midpoint +600 giây, cùng ngày/split.
3. Chọn đúng chu kỳ danh nghĩa trước đó 10 phút cho satellite; không tự lùi thêm khi thiếu.
4. Kiểm tra header cả 10 nguồn: observation end và file creation không sau cutoff.
   File thiếu/thời gian không hợp lệ thì loại và ghi lý do; vẫn phải kiểm tra nguồn cùng split.
5. Tách đường dẫn radar input, satellite input và target trong manifest mới; giữ nguyên NPZ.
6. Để so sánh công bằng, ba phương pháp dùng cùng tập pair cuối và evaluation protocol.

Đây là mô phỏng theo timestamp quan trắc/tạo tệp, không có bằng chứng về download latency,
radar processing latency hoặc tensor availability. Không gọi cutoff là thời điểm phát hành
vận hành đã được đo. Target midpoint cách cutoff khoảng 8 phút 34,5 giây đến 8 phút 47 giây
trong 822 ứng viên kiểm tra JSON trước đó, không phải horizon vận hành đúng 10 phút.
822 (508/141/173) vẫn chỉ là số ứng viên trước bước kiểm tra đủ header từng pair.

### Tôi cần hiểu

- **Khái niệm:** giới hạn thời gian phải áp dụng cho mọi nguồn góp vào input, không chỉ một band.
- **Code:** hôm nay đọc bằng Python kiểm tra tạm; chưa thêm hoặc sửa module chọn nguồn.
- **Input/output:** header HSD + JSON → bảng bằng chứng; không tạo tensor hoặc manifest mới.
- **Shape:** radar `(256,256)`, Himawari `(5,256,256)` giữ nguyên.
- **Kiểm chứng:** đọc Block 1/9, đối chiếu từng tệp với cutoff; 20 tệp không đại diện toàn tháng.
- **Lỗi thường gặp:** coi B13/S04 đại diện mọi segment, hoặc coi creation time là thời gian tải xong.
- **Ba câu hỏi:** Vì sao phải kiểm tra S05? File tạo xong đã chắc có trên máy chưa?
  Vì sao chưa thể công bố tất cả 822 cặp đã vượt qua audit thời gian?

## 8. Day 11 — tạo manifest có kiểm tra đủ nguồn (20/10/2026)

Sau khi người dùng đồng ý quy tắc Day 10, đã cập nhật config forecasting riêng và script 09.
`pipeline.yaml` và toàn bộ NPZ/JSON preprocessing không đổi. Manifest 842 cặp ban đầu vẫn
giữ nguyên như kết quả diagnostic, không dùng nó làm tập multimodal đã kiểm tra timing.

### Tôi cần hiểu

1. **Khái niệm:** một forecast pair hiện tham chiếu ba nguồn: radar input, satellite từ NPZ
   chu kỳ trước, và radar target. Không đọc nhầm vệ tinh từ NPZ radar hiện tại.
2. **Code:** `pairs.py` tìm target exact; `timeline.py` chọn source và kiểm tra cutoff;
   `satellite_timing.py` đọc header có giới hạn; `splits.py` kiểm tra cả satellite role;
   script `09_build_forecast_pairs.py` ghi/mở lại CSV và summary.
3. **Input/output:** index + JSON + header HSD → manifest, header audit, exclusions và summary
   ở `output/forecasting/observation_timing/`. Không sao chép tensor hay chạy preprocessing.
4. **Shape:** chưa tạo batch; nguồn radar `(256,256)`, vệ tinh `(5,256,256)` giữ nguyên.
5. **Kiểm chứng:** preview 2 cặp/20 headers trước, rồi kiểm tra đủ tập ứng viên. 52 local tests
   PASS (21 DE + 17 pair/split + 14 header/timeline/CLI). Mở lại toàn bộ manifest/audit,
   xác nhận source paths, JSON SHA256, index/config/artifact SHA256, đúng 10 headers/cặp,
   cutoff và split isolation; mở NPZ ba cặp đầu kiểm tra shape/dtype/mask PASS.
6. **Lỗi thường gặp:** dùng sai satellite_npz_path, đánh đồng header audit với image audit,
   xem file creation là download completion, hoặc gọi khoảng 8,6 phút là 10 phút từ cutoff.
7. **Ba câu hỏi:** Vệ tinh phải đọc từ trường đường dẫn nào? Vì sao 20 cặp bị loại?
   Kết quả timing PASS có chứng minh dữ liệu đã tải và xử lý xong trước cutoff không?

### Kết quả thực tế

| Nội dung | Số lượng |
| --- | ---: |
| Exact candidates | 842 |
| Accepted pairs | 822 |
| Train | 508 |
| Validation | 141 |
| Test | 173 |
| Headers đã kiểm tra | 8.220 |
| Loại vì chu kỳ trước không có source A trong index | 15 |
| Loại vì vệ tinh qua ngày | 5 |

Tất cả 822 cặp vượt qua quy tắc thời gian quan trắc/tạo tệp đã chọn. Khoảng từ cutoff đến
target midpoint là 514,5–527 giây. Không suy ra vận hành thời gian thực, meteorological QC,
hoặc chất lượng dự báo. Chưa normalization, baseline, train hoặc inference.

Schema mới: `nhabe-forecasting-observation-timing-v1`. Tham khảo
[forecast manifest dictionary](forecasting_manifest.md). Docker chỉ được bổ sung config cần
cho tests; chưa rebuild/verify image hoặc hosted CI của phần mới. Chưa commit/push.

## 9. Day 12 — train-only normalization statistics (20/10/2026)

### Tôi cần hiểu

1. **Khái niệm:** fit mean/std từ pixel hợp lệ của input train. Mỗi trường có một bộ
   thống kê; không phải tính lại theo từng ảnh hoặc theo từng split.
2. **Code:** `normalization.py`, `scripts/10_compute_train_statistics.py`, policy riêng
   `configs/normalization.yaml`; không sửa forecasting config đã được fingerprint ở Day 11.
3. **Input/output:** manifest đã audit → normalization.json và normalization_sources.csv
   trong output mới. Ghi policy, count, mean/std/min/max và provenance/checksums.
4. **Shape:** không thay shape NPZ; mỗi trường radar hoặc band nhận hai số mean/std.
   Chưa có model tensor/batch. Source float32 và mask bool giữ nguyên.
5. **Kiểm chứng:** 67 local tests PASS, trong đó 15 tests normalization. Dùng sentinel
   paths trong tests để chứng minh không mở validation/test/target-only arrays. Reopen
   JSON/CSV; rehash toàn bộ nguồn đã dùng; kiểm tra độc lập normalized train mean≈0/std≈1.
6. **Lỗi thường gặp:** thay NaN bằng 0 trước fit, dùng all-band mask thay mask từng band,
   đếm một nguồn nhiều lần trong cùng trường, dùng mean/std test hoặc bỏ qua std=0.
7. **Ba câu hỏi:** Dùng mean/std nào khi đọc test? Tại sao không gom target-only images
   vào thống kê đã chọn? File NPZ chứa NaN có bị ghi đè thành ảnh chuẩn hóa không?

### Quy tắc và kết quả thật

508 train pairs → 508 radar input sources + 508 satellite sources, dedupe theo từng role.
Một NPZ có thể cung cấp radar cho pair này và vệ tinh cho pair khác. Có 1.016 source-role
records trên 909 NPZ khác nhau; không phải 1.016 NPZ độc lập. Mẫu từng là target ở pair
khác vẫn được dùng nếu nó là input train, nhưng không gom thêm target-only sources.

Chọn pixel-weighted moments, float64 accumulator, population std (`ddof=0`). Mỗi pixel
hợp lệ có trọng số bằng nhau; không lấy trung bình các mean của ảnh nếu số pixel khác nhau.
Radar target sau này dùng cùng mean/std radar input; validation/test không fit lại.

| Trường | Pixel hợp lệ | Mean | Std | Đơn vị |
| --- | ---: | ---: | ---: | --- |
| Radar | 5.607.919 | 8,382429 | 11,441322 | dBZ |
| B08 | 33.273.492 | 229,124144 | 8,139035 | K |
| B13 | 33.273.492 | 267,501815 | 25,988496 | K |
| B14 | 33.273.492 | 265,261867 | 26,045386 | K |
| B15 | 33.273.492 | 261,517266 | 24,909850 | K |
| B16 | 33.273.492 | 250,584941 | 19,192389 | K |

Đây là giá trị làm tròn để đọc, code phải dùng JSON đủ precision. Mean/std không nói
vùng nào mưa mạnh; mean dBZ chỉ phục vụ feature scaling. Hàng triệu pixel không phải
hàng triệu mẫu khí tượng độc lập. Thống kê radar giữ cả dBZ âm và 0, không threshold.

Kiểm tra độc lập: tất cả source NPZ/JSON SHA256 không đổi; nguồn thống kê đúng tập train
đã chọn; cả sáu trường sau chuẩn hóa trên pixel train hợp lệ có |mean|<1e-10 và
|std−1|<1e-10. Chưa áp dụng transform lên validation/test, chưa đệm NaN thành 0 trong model.
Tham khảo [protocol/schema normalization](normalization.md) và
[NumPy std](https://numpy.org/doc/stable/reference/generated/numpy.std.html).

Chưa cài PyTorch, train, baseline, rebuild Docker/hosted CI hoặc commit/push. Docker
allowlist/COPY chỉ bổ sung normalization.yaml để suite có đủ config khi kiểm tra sau này.

## 10. Day 13 — môi trường PyTorch riêng (24/10/2026)

PyTorch 2.14.1 cài trực tiếp vào môi trường khoa học đã xung đột OpenMP: wheel PyTorch
nạp `torch/lib/libomp.dylib`, trong khi NumPy/OpenBLAS của Conda nạp `env/lib/libomp.dylib`.
Không dùng `KMP_DUPLICATE_LIB_OK=TRUE` vì đó chỉ bỏ qua cảnh báo và có thể làm phép tính
không đáng tin cậy. Môi trường preprocessing vẫn dùng `doan_khdl`; AI dùng `.venv` riêng.

Trong `.venv`: Python 3.12, PyTorch 2.14.1, NumPy 2.5.2; tensor thử nghiệm
`(2,8,256,256)` float32 tạo thành công và MPS available=True. Đây chỉ chứng minh backend
được nhận diện, chưa chứng minh một lần train dài sẽ ổn định hoặc nhanh hơn CPU.

## 11. Day 14 — Dataset và DataLoader (24/10/2026)

### Tôi cần hiểu

1. **Khái niệm:** Dataset biến một row manifest thành một sample tensor; DataLoader gom
   nhiều sample thành batch. Dataset không phải mô hình và không thực hiện training.
2. **Ba nguồn:** radar từ `input_npz_path`, Himawari từ `satellite_npz_path`, radar tương
   lai từ `target_npz_path`. Không lấy Himawari trong NPZ radar hiện tại.
3. **Chuẩn hóa:** chỉ pixel hợp lệ dùng `(x-mean)/std`. NaN được thay bằng 0 sau chuẩn hóa
   trong RAM; file nguồn không đổi và mask vẫn chỉ ra vị trí hợp lệ.
4. **Shape:** radar-only `(B,2,256,256)`; multimodal `(B,8,256,256)`; target và target mask
   `(B,1,256,256)`. Target mask là bool, các tensor số là float32.
5. **Loss sau này:** phải dùng target mask để pixel thiếu không đóng góp vào sai số.
6. **Lỗi thường gặp:** dùng sai satellite path, chuẩn hóa NaN như số thật, bỏ mask, fit lại
   mean/std theo validation/test, hoặc hiểu zero padding là reflectivity 0 dBZ.

### Kết quả thực tế

Năm test PyTorch riêng PASS: shape/dtype, normalization, padding và mask, provenance
manifest, exact lead time, dữ liệu lỗi và DataLoader. 67 test preprocessing/timing cũ vẫn
PASS. Smoke run mở hai cặp train thật: multimodal `(2,8,256,256)`, radar-only
`(2,2,256,256)`, target/mask `(2,1,256,256)`; tất cả tensor số finite. Train split có
508 cặp. `training_performed=false`: chưa chạy U-Net, optimizer, loss hoặc cập nhật weight.

Tại pixel chỉ thiếu một số band, các band hợp lệ vẫn được giữ; band thiếu được zero-pad và
all-band mask bằng 0. Contract 8 kênh chưa thêm năm per-band mask riêng. Thay đổi số kênh
sau này phải là thí nghiệm có kiểm soát. Xem [dataset contract](forecasting_dataset.md).

## 12. Day 15 — masked metrics và persistence baseline (24/10/2026)

### Tôi cần hiểu

1. **Persistence:** dự báo radar tương lai bằng radar hiện tại. Đây là baseline không học
   tham số, dùng để biết U-Net có thực sự cải thiện so với “giữ nguyên ảnh gần nhất” không.
2. **Mask đánh giá:** dùng `input_radar_mask AND target_mask`. Persistence không có dự báo
   quan trắc tại nơi radar đầu vào thiếu; target thiếu cũng không có ground truth để chấm.
3. **MAE/RMSE:** MAE là trung bình trị tuyệt đối sai số; RMSE lấy căn của trung bình bình
   phương sai số nên nhạy hơn với lỗi lớn. Cả hai báo bằng dBZ.
4. **Aggregation:** cộng sai số trên mọi pixel hợp lệ rồi chia một lần. Không lấy trung bình
   metric từng ảnh, vì số pixel hợp lệ giữa các ảnh khác nhau.
5. **Split:** chỉ mở 141 cặp validation. Test 173 cặp vẫn khóa cho đánh giá cuối sau khi
   chọn mô hình. Không threshold reflectivity và không diễn giải thành chất lượng mưa/sét.
6. **Loss:** `masked_mse` giữ gradient và có thể dùng khi train; metric validation được
   tính độc lập bằng accumulator float64. Hôm nay chưa có optimizer hay weight update.

### Kết quả thực tế

| Đại lượng | Kết quả |
| --- | ---: |
| Validation pairs | 141 |
| Tổng grid pixels | 9.240.576 |
| Input valid | 1.627.007 (17,607%) |
| Target valid | 1.595.558 (17,267%) |
| Giao hợp lệ để chấm | 1.139.645 (12,333%) |
| Persistence MAE | 5,539317 dBZ |
| Persistence RMSE | 8,036566 dBZ |

Per-pair MAE có min/median/max 2,378 / 4,881 / 9,519 dBZ; RMSE là
3,185 / 6,910 / 12,734 dBZ. Primary result vẫn là global pixel-weighted metric.

Chín test AI PASS, gồm 5 Dataset và 4 masked metric. 67 test preprocessing/timing cũ vẫn
PASS. Output JSON/CSV được mở lại và có fingerprint. `training_performed=false` và
`test_split_opened=false`. Xem [persistence baseline](persistence_baseline.md).

## 13. Day 16 — Small U-Net forward/backward smoke (24/10/2026)

### Tôi cần hiểu

1. **Encoder:** giảm dần kích thước ảnh để học ngữ cảnh rộng hơn; số feature channels tăng
   16→32→64→128. **Decoder:** tăng kích thước trở lại 256×256.
2. **Skip connection:** nối feature cùng độ phân giải từ encoder sang decoder để giữ thông
   tin vị trí chi tiết đã bị mất khi pooling.
3. **GroupNorm:** chuẩn hóa theo nhóm channel trong từng sample, không phụ thuộc thống kê
   của cả batch như BatchNorm; phù hợp batch nhỏ.
4. **Output:** một channel tuyến tính, không sigmoid/softmax. Reflectivity chuẩn hóa là biến
   liên tục có thể âm hoặc dương.
5. **Backward:** tính gradient của masked MSE để xác nhận đường truyền đạo hàm hoạt động.
   Không có optimizer nghĩa là trọng số chưa được cập nhật và chưa gọi là training.
6. **Mask:** loss dùng target mask. Primary evaluation sau này dùng giao input-radar và
   target mask để so sánh công bằng với persistence.

### Kết quả thực tế

| Mode | Input | Output | Trainable parameters | CPU | MPS |
| --- | --- | --- | ---: | --- | --- |
| Radar-only | `(1,2,256,256)` | `(1,1,256,256)` | 482.593 | PASS | chưa cần chạy |
| Multimodal | `(1,8,256,256)` | `(1,1,256,256)` | 483.457 | PASS | PASS |

Hai model chỉ khác 864 tham số ở convolution đầu: 6 channel bổ sung × 16 filter × 3×3.
Các tầng còn lại giống nhau. Cả 50 parameter tensors có gradient hữu hạn. Loss từ trọng số
ngẫu nhiên không phải metric dự báo và không so với persistence. Không tạo optimizer,
checkpoint hoặc chạy epoch. 14 AI tests và 67 preprocessing/timing tests PASS.

Xem [model contract](model.md) và cấu hình khóa tại `configs/model.yaml`.

## 14. Day 17 — overfit gate trên 16 cặp (25/10/2026)

Mục tiêu của overfit gate không phải tạo mô hình tốt, mà kiểm tra toàn bộ chuỗi Dataset →
U-Net → masked loss → gradient → optimizer có thực sự học được. Dùng 16 cặp train đầu,
multimodal, batch 2, AdamW learning rate 0,001, 25 epoch và seed 2025. Không dùng
validation/test và không lưu checkpoint từ tập nhỏ.

Lần đầu phát hiện phép chuyển tensor kết hợp từ MPS sang CPU đồng thời đổi float32→float64
làm metric NaN, dù prediction và error gốc hữu hạn. Đã sửa thành chuyển thiết bị trước, đổi
dtype sau. Lần hai loss giảm đúng nhưng code có điều kiện cuối epoch phải bằng đúng epoch tốt
nhất, trong khi điều kiện này không có trong policy. Hai lần FAIL được giữ ngoài Git làm dấu
vết; không xóa hoặc trình bày như kết quả thành công.

Lần canonical v3 PASS: evaluation MSE giảm từ 69,030397 xuống 0,166506; tốt nhất 0,165023
ở epoch 24; tỷ lệ cuối/đầu 0,002412, thấp hơn ngưỡng 0,5. Runtime 21,84 giây trên MPS.
Kết luận duy nhất: pipeline có khả năng cập nhật weight và giảm loss trên tập nhỏ.

## 15. Day 18 — huấn luyện chính thức (25/10/2026)

Hai thí nghiệm dùng cùng seed, 508 train pairs, 141 validation pairs, kiến trúc base 16,
AdamW, learning rate 0,001, weight decay 0,0001, batch 4 và early stopping patience 7.
Metric chọn checkpoint là global RMSE dBZ trên giao mask radar input/target, cùng miền với
persistence. Chỉ biến input mode thay đổi. Cả 173 test pairs vẫn khóa.

| Phương pháp | Best epoch | MAE validation | RMSE validation |
| --- | ---: | ---: | ---: |
| Persistence | – | 5,539317 dBZ | 8,036566 dBZ |
| Radar-only Small U-Net | 23 | 4,494850 dBZ | 6,211665 dBZ |
| Radar + Himawari Small U-Net | 23 | 4,421153 dBZ | 6,039912 dBZ |

Radar-only giảm MAE 18,86% và RMSE 22,71% so với persistence. Multimodal giảm MAE
20,19% và RMSE 24,84%. So với radar-only, multimodal giảm MAE 1,64% và RMSE 2,77% trên
validation. Hai run đều dừng ở epoch 30; runtime lần lượt 646,50 và 823,12 giây trên MPS.

Checkpoint canonical đã mở lại bằng `weights_only=True`, tái tạo đúng metric và có SHA256.
Một radar v1 đã train xong nhưng bị safe loader từ chối vì metadata TorchVersion; output này
được giữ là failed attempt. v2 lưu version thành chuỗi và PASS hoàn chỉnh.

Không kết luận multimodal tốt hơn một cách tổng quát vì mới có một seed, một tháng và một
trạm. Đây là kết quả validation phục vụ model development; chưa phải test result hay bằng
chứng vận hành thực tế. Xem [training journal](training_journal.md) và
[official training results](training_results.md).

## 16. Day 19 — đánh giá locked test (25/10/2026)

Trước khi mở test, toàn bộ source/config/docs được commit tại `b5b8399`; repository sạch.
Final evaluator kiểm tra lại SHA256 của manifest, normalization, config, history và hai
checkpoint. Hai checkpoint được mở bằng `weights_only=True`. Sau đó mới tạo Dataset test.

173 test pairs tạo 11.337.728 grid cells; giao mask hợp lệ dùng để chấm gồm 1.746.163 pixel
(15,401%). Cả ba phương pháp dùng đúng cùng pair order và common mask.

| Phương pháp | Test MAE | Test RMSE |
| --- | ---: | ---: |
| Persistence | 5,011121 dBZ | 7,407763 dBZ |
| Radar-only U-Net | 4,234776 dBZ | 5,956063 dBZ |
| Radar + Himawari U-Net | 4,215341 dBZ | 5,903871 dBZ |

Radar-only giảm test RMSE 19,60% so với persistence. Multimodal giảm 20,30% so với
persistence và 0,88% so với radar-only. Thứ tự giống validation nhưng lợi thế multimodal nhỏ
hơn. Không điều chỉnh model/hyperparameter sau khi xem test.

Kết quả ủng hộ chọn multimodal làm candidate cho portfolio, không chứng minh satellite luôn
tốt hơn vì chỉ có một seed, một tháng và một trạm. Xem [locked-test audit](final_evaluation.md).

## 17. Day 20–21 — hình báo cáo và inference một mẫu (25/10/2026)

Hai hình báo cáo được tạo hoàn toàn từ history/summary đã khóa: đường validation RMSE và bảng
so sánh validation với locked test. Việc vẽ không nạp dữ liệu huấn luyện và không sửa model.

Inference dùng checkpoint multimodal epoch 23. Lệnh nhận một `pair_id` đã audit, đọc radar
hiện tại và Himawari chu kỳ trước, nhưng không mở radar target tương lai. Dự báo chuẩn hóa được
đổi ngược về dBZ bằng mean/std radar tính chỉ từ train. NPZ đầu ra giữ `float32 [256,256]` và
mask boolean; pixel ngoài mask radar input là NaN chứ không phải 0.

Mẫu validation `NHB250822004007_RAWNDW4__to__NHB250822005007_RAWNDWB` chạy CPU PASS:
7.292 pixel hợp lệ (11,1267%), forward khoảng 0,079 giây; NPZ/JSON mở lại đúng hash, shape,
dtype và NaN–mask. Đây là offline inference từ manifest đã chuẩn bị, chưa phải API hay hệ thống
nhận dữ liệu thời gian thực. Xem [inference contract](inference.md).

## 18. Day 22 — xem định tính mà không cherry-pick (25/10/2026)

Chọn cố định validation index 0, 70 và 140, tương ứng đầu, giữa và cuối theo thời gian. Quy
tắc được chốt trước khi xem hình nên không chọn thủ công trường hợp đẹp. Prediction được tạo
trước mà không đọc target; bước visualization sau đó mới đọc target để tính sai số và vẽ.

Ba RMSE lần lượt là 10,214; 5,095 và 3,622 dBZ trên common-valid mask. Sự khác nhau cho thấy
độ khó từng tình huống không đồng nhất. Chỉ ba mẫu không đại diện cho phân phối toàn validation,
vì vậy kết luận định lượng vẫn phải dựa trên metric global của 141 validation và 173 test pairs.
Dải màu -10–60 dBZ và error 0–30 dBZ chỉ là display range, không threshold dữ liệu.

## 19. Day 23 — FastAPI và Docker AI tách riêng (25/10/2026)

API chỉ nhận `pair_id` có trong audited manifest. Client không được gửi path, array hay thay
config. Ba endpoint gồm `/health`, `/model-info` và `/predict`. Model nạp một lần khi startup;
một lock tuần tự hóa inference. Output đã tồn tại trả 409 để không ghi đè, pair không có trả 404.

Docker AI tách khỏi preprocessing image. Dữ liệu, manifest, normalization và checkpoint đều là
read-only mounts; output là mount ghi. Port Compose chỉ bind localhost. Lần build đầu được dừng
vì PyTorch Linux mặc định kéo CUDA; bản sửa dùng official CPU index, giữ Torch 2.14.1.

Kết quả: image 278.781.609 byte (~266 MiB), `torch 2.14.1+cpu`, CUDA `None`, dependency check
PASS. Compose healthy; health/model-info/predict qua HTTP thật PASS; output `[256,256]` float32
và mask mở lại PASS. Image không chứa checkpoint, NPZ hay RAW. Đây vẫn là offline research API,
không có auth/TLS/rate limit/real-time ingestion và không được gọi là production.

## 20. Day 24 — chuẩn bị CI AI không dùng dữ liệu thật (25/10/2026)

Workflow có job thứ ba riêng cho AI/API. CI tự tạo một radar và Himawari synthetic, manifest
một pair, normalization giả và checkpoint random chưa train. File target tương lai cố ý không
tồn tại. Fixture tự ghi `meteorological_observation=false` và `trained_checkpoint=false` để
không bị hiểu nhầm là kết quả nghiên cứu.

Job cài Torch CPU-only, chạy 30 tests, build Docker AI, khởi động API, gọi health/model/predict,
mở lại NPZ/JSON, kiểm tra image không có data/checkpoint và luôn dọn container. Mô phỏng local
ARM64 PASS với 34.621 valid pixels; future target không được đọc. Workflow YAML/structure PASS.
Chưa push nên chưa được gọi là GitHub AMD64 CI PASS.

## 21. Day 25 — xác nhận CI AI trên Linux AMD64 (25/10/2026)

Sáu commit AI được push cùng nhau sau khi local audit PASS. GitHub Actions run 40157818399
hoàn thành xanh cả ba job: Python 1 phút 7 giây, AI/API 1 phút 54 giây và preprocessing Docker
1 phút 53 giây. Job AI chạy lại 30 tests, tạo fixture synthetic không có future target, build
CPU-only image, gọi health/model-info/predict, mở lại prediction, quét image và dọn container.

Kết quả này xác nhận workflow synthetic chạy được trên Linux AMD64 ngoài máy phát triển ARM64.
Nó không phải bằng chứng model chạy production, không thay cho đánh giá locked test và không
xác nhận preprocessing RAW/HSD thật bên trong Docker.
