## PHẦN 1 — TÓM TẮT BÀI TOÁN TỪ README

Bài cuối khóa là một **binary classification challenge** trên dữ liệu bảng đã bị ẩn danh hoàn toàn: không có tên cột, không có ngữ cảnh, tất cả feature đều là số, nhãn là `0/1`. Bạn chỉ được dùng `X_train` và `y_train` để huấn luyện; `X_test`, `y_test`, `X_challenge`, `y_challenge` chỉ dùng để đánh giá. Mục tiêu không phải chỉ train model, mà là xây một pipeline ML hoàn chỉnh và **vượt toàn bộ benchmark** đã cho. 

Bộ dữ liệu gồm 6 file: `X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`, `X_challenge.csv`, `y_challenge.csv`. README nhấn mạnh benchmark được tạo bằng các model chạy **cấu hình mặc định, không có xử lý dữ liệu**, nên dư địa cải thiện nằm ở EDA, preprocessing, feature engineering, tuning và ensemble. 

Mốc benchmark cao nhất trên **test** là LightGBM với `F1 = 0.9801`, `AUC = 0.9981`; còn trên **challenge** mốc F1 cao nhất là `0.9700` với SGD(log). Bài nộp bắt buộc có đủ 3 phần: **code**, **slide tối thiểu 12 trang**, và **thuyết trình 10–15 phút + 5 phút hỏi đáp**. Code phải có đúng chuỗi bước: load dữ liệu → EDA → tiền xử lý → feature engineering → huấn luyện → tuning → đánh giá trên cả test và challenge; đồng thời phải cố định `random_state=42`, tái hiện được kết quả, và không để notebook lỗi hay rác output. 

## PHẦN 2 — NHỮNG KIẾN THỨC CẦN HỌC TRƯỚC

### 1) EDA cho dữ liệu bảng và binary classification

* **Học để làm gì trong bài này:** Vì dữ liệu bị ẩn danh, bạn không thể dựa vào ý nghĩa nghiệp vụ của cột; bạn phải đọc dữ liệu bằng thống kê, phân phối, tương quan, missing, outlier, phân phối nhãn. README cũng yêu cầu rõ phần EDA phải có phân phối nhãn, phân phối từng feature, ma trận tương quan và phát hiện outlier. 
* **Học đến mức nào là đủ:** Biết kiểm tra shape, dtype, missing, duplicate; xem label balance; vẽ histogram/boxplot; kiểm tra tương quan feature-feature và feature-target; phát hiện cột gần hằng, cột lệch mạnh, cột có outlier bất thường.
* **Bắt buộc hay nâng cao:**

  * Bắt buộc: missing, outlier, phân phối nhãn, phân phối feature, correlation.
  * Nâng cao: kiểm tra drift sơ bộ giữa train/test/challenge, PCA/UMAP để nhìn cụm dữ liệu.

### 2) Missing values, duplicates, outliers

* **Học để làm gì:** README nói rõ benchmark chưa xử lý dữ liệu, nên đây là chỗ ăn điểm rất lớn. Nếu bỏ qua missing/outlier, các model tuyến tính, KNN, MLP thường hụt hiệu năng đáng kể. 
* **Học đến mức nào là đủ:** Biết khi nào dùng median imputation, khi nào thêm cờ missing; biết outlier có nên clip/winsorize hay giữ nguyên; biết duplicate có thể gây leakage hoặc làm CV ảo.
* **Bắt buộc hay nâng cao:**

  * Bắt buộc: missing + outlier.
  * Nâng cao: isolation-based outlier filtering, robust transform.

### 3) Class imbalance

* **Học để làm gì:** README yêu cầu kiểm tra và xử lý imbalance nếu có. Với binary classification, imbalance ảnh hưởng trực tiếp đến precision/recall/F1 của nhãn 1 — mà README cũng ghi rõ precision/recall/F1 được tính trên nhãn 1 ở benchmark challenge. 
* **Học đến mức nào là đủ:** Biết đọc phân phối nhãn; biết dùng `class_weight`, threshold tuning, hoặc SMOTE/oversampling trong CV đúng cách.
* **Bắt buộc hay nâng cao:**

  * Bắt buộc: kiểm tra imbalance và thử `class_weight`.
  * Nâng cao: resampling trong pipeline CV.

### 4) Preprocessing: scaling, imputation, robust transforms

* **Học để làm gì:** README yêu cầu rõ phần tiền xử lý phải có scaling, imputation và giải thích lý do chọn phương pháp. Các model như Logistic Regression, SGD, LinearSVC, MLP, KNN đặc biệt cần scaling tốt. 
* **Học đến mức nào là đủ:** Biết khác nhau giữa StandardScaler, RobustScaler, MinMaxScaler; biết chỉ fit trên train fold; biết ghép bằng `Pipeline`.
* **Bắt buộc hay nâng cao:**

  * Bắt buộc: imputation + ít nhất 1 scaling pipeline chuẩn.
  * Nâng cao: power transform / quantile transform.

### 5) Feature engineering cho dữ liệu bảng số

* **Học để làm gì:** README nói thẳng đây là một hướng cải thiện chính: tạo feature mới, loại bỏ feature, chọn feature theo căn cứ. Vì dữ liệu vô danh, feature engineering ở đây nên mang tính thống kê hơn là nghiệp vụ. 
* **Học đến mức nào là đủ:** Biết thử feature selection theo variance, correlation, mutual information, model-based importance; biết tạo feature tổng hợp đơn giản như ratio, interaction, row-wise stats nếu hợp lý.
* **Bắt buộc hay nâng cao:**

  * Bắt buộc: loại cột kém hữu ích, thử một nhánh selection.
  * Nâng cao: interaction features, nonlinear transforms, stacking features.

### 6) Cross-validation và chống leakage

* **Học để làm gì:** README bắt buộc có cross-validation trong huấn luyện và tuning. Nếu làm preprocessing ngoài CV, kết quả sẽ đẹp giả và dễ fail ở test/challenge. 
* **Học đến mức nào là đủ:** Biết dùng `StratifiedKFold`; biết mọi bước imputation/scaling/feature selection phải nằm trong `Pipeline`; biết tách rõ train/test/challenge.
* **Bắt buộc hay nâng cao:**

  * Bắt buộc: CV đúng chuẩn.
  * Nâng cao: repeated CV, nested CV.

### 7) Metrics cho binary classification

* **Học để làm gì:** README yêu cầu in đầy đủ Accuracy, Precision, Recall, F1, AUC trên cả test và challenge; benchmark test có AUC, còn challenge nhấn mạnh F1 trên nhãn 1. 
* **Học đến mức nào là đủ:** Biết ý nghĩa từng metric; biết vì sao không chỉ nhìn accuracy; biết khi nào ưu tiên F1, khi nào ưu tiên AUC; biết threshold 0.5 chưa chắc là tối ưu.
* **Bắt buộc hay nâng cao:**

  * Bắt buộc: đọc 5 metric cơ bản.
  * Nâng cao: threshold tuning theo F1, precision-recall tradeoff.

### 8) Hyperparameter tuning

* **Học để làm gì:** README bắt buộc có bước tối ưu bằng grid search, random search hoặc Bayesian optimization. Đây là phần cần có sau khi baseline ổn, không nên làm quá sớm. 
* **Học đến mức nào là đủ:** Biết chọn search space vừa phải; biết tune theo metric chính; biết ưu tiên model hứa hẹn nhất trước.
* **Bắt buộc hay nâng cao:**

  * Bắt buộc: Grid/Random Search cho 1–2 model.
  * Nâng cao: Bayesian optimization, Optuna.

### 9) Ensemble / stacking / blending

* **Học để làm gì:** README gợi ý ensemble là một hướng cải thiện thêm sau EDA, preprocessing và tuning. 
* **Học đến mức nào là đủ:** Hiểu voting mềm, averaging xác suất, stacking cơ bản.
* **Bắt buộc hay nâng cao:**

  * Bắt buộc: không bắt buộc ngay tuần 1.
  * Nâng cao: dành cho cuối tuần khi đã có 2–3 model mạnh.

### 10) Trình bày kết quả vào slide và thuyết trình

* **Học để làm gì:** README yêu cầu tối thiểu 12 slide với cấu trúc rất cụ thể, và phần trình bày bị chấm ở cả hiểu bài toán, lý luận pipeline, kết quả tái hiện được, phân tích thất bại và trả lời câu hỏi. 
* **Học đến mức nào là đủ:** Biết biến notebook thành câu chuyện: dữ liệu → vấn đề → quyết định → kết quả → bài học.
* **Bắt buộc hay nâng cao:**

  * Bắt buộc: dựng dàn ý slide ngay trong tuần 1.
  * Nâng cao: demo live, dashboard đẹp.

## PHẦN 3 — KẾ HOẠCH 1 TUẦN CHI TIẾT

### Day 1

* **Mục tiêu:** Hiểu đề thật chắc, dựng khung project, load dữ liệu sạch sẽ, kiểm tra toàn cảnh.
* **Học gì:**

  * Cấu trúc một pipeline tabular classification hoàn chỉnh.
  * Cách đọc các metric Accuracy / Precision / Recall / F1 / AUC.
  * Nguyên tắc không leakage: chỉ train bằng `X_train`, `y_train`.
* **Làm gì:**

  * Đọc lại README và chép ra 1 trang note: mục tiêu, benchmark, yêu cầu nộp.
  * Tạo cấu trúc thư mục: `data/`, `notebooks/`, `src/`, `reports/`, `slides/`, `results/`.
  * Viết notebook/script đầu tiên để load đủ 6 file, check shape, dtype, missing, duplicates, phân phối nhãn.
  * In benchmark ra thành một bảng riêng để sau này so sánh.
  * Chốt metric chính dùng để ra quyết định nội bộ: ưu tiên **F1** và **AUC** trên validation; nhưng vẫn lưu đủ 5 metric vì README bắt buộc.
* **Đầu ra cuối ngày:**

  * 1 notebook load dữ liệu chạy được.
  * 1 file note “yêu cầu đề bài”.
  * 1 bảng benchmark tự tổng hợp.
* **Tiêu chí hoàn thành:**

  * Bạn nói lại được: dữ liệu có những file nào, được train trên gì, benchmark cần vượt cái gì, nộp những gì.
  * Chạy notebook không lỗi và biết sơ bộ dataset có missing/imbalance hay không.

### Day 2

* **Mục tiêu:** Hoàn thành EDA sơ bộ nhưng đủ dùng để ra quyết định preprocessing.
* **Học gì:**

  * EDA cho dữ liệu bảng số: histogram, boxplot, correlation, missing pattern, label distribution.
  * Cách phát hiện cột vô dụng: near-constant, duplicated columns, lệch mạnh, outlier cực đoan.
* **Làm gì:**

  * Phân tích phân phối nhãn.
  * Vẽ thống kê mô tả cho toàn bộ feature.
  * Kiểm tra missing theo cột và theo hàng.
  * Kiểm tra outlier bằng IQR/z-score/boxplot cho top cột bất thường.
  * Tính correlation matrix hoặc top correlated features.
  * So sánh phân phối train vs test vs challenge cho một số feature quan trọng để cảm nhận drift.
  * Viết 5–10 phát hiện quan trọng nhất.
* **Đầu ra cuối ngày:**

  * 1 section EDA khá rõ trong notebook.
  * 3–5 biểu đồ có thể bê thẳng vào slide.
  * 1 danh sách quyết định ban đầu: có cần imputation không, có cần scaling không, có dấu hiệu imbalance không, có cần xử lý outlier không.
* **Tiêu chí hoàn thành:**

  * Bạn không chỉ “vẽ biểu đồ”, mà đã rút ra được quyết định kỹ thuật cụ thể từ EDA.
  * Có ít nhất vài phát hiện đủ mạnh để giải thích trong slide 3–4 sau này.

### Day 3

* **Mục tiêu:** Dựng preprocessing pipeline chuẩn và có baseline đầu tiên.
* **Học gì:**

  * `Pipeline` trong scikit-learn.
  * Imputation, scaling, `StratifiedKFold`, `cross_validate`.
  * Vì sao preprocessing phải nằm trong pipeline.
* **Làm gì:**

  * Tạo 2–3 pipeline preprocessing khác nhau, ví dụ:

    1. MedianImputer + StandardScaler
    2. MedianImputer + RobustScaler
    3. MedianImputer + no scaling cho tree-based
  * Huấn luyện baseline nhanh với các model đơn giản và dễ đọc: Logistic Regression, SGD(log), RandomForest, ExtraTrees.
  * Chạy CV trên train.
  * Chọn cách log kết quả chuẩn: mỗi run lưu metric, seed, pipeline, model, tham số.
* **Đầu ra cuối ngày:**

  * 1 baseline bảng kết quả CV.
  * 1 preprocessing pipeline rõ ràng, tái dùng được.
  * 1 format lưu thí nghiệm.
* **Tiêu chí hoàn thành:**

  * Có ít nhất 3–4 baseline chạy xong.
  * Đã biết mô hình nào hứa hẹn nhất ở giai đoạn đầu.

### Day 4

* **Mục tiêu:** Làm feature engineering/selection có căn cứ và thử nhóm model mạnh hơn.
* **Học gì:**

  * Variance threshold, mutual information, model-based feature importance.
  * Tư duy feature engineering cho tabular numerical data vô danh.
* **Làm gì:**

  * Loại cột gần hằng hoặc rõ ràng vô ích.
  * Thử nhánh feature selection dựa trên mutual information hoặc tree importance.
  * Nếu hợp lý, thử thêm row-wise features như mean/std/min/max của từng hàng hoặc interaction đơn giản giữa top feature quan trọng.
  * Thử nhóm model mạnh hơn theo benchmark README: HistGradientBoosting, LightGBM, XGBoost nếu môi trường cho phép; nếu chưa, ưu tiên HistGradBoost + ExtraTrees + Logistic/SGD tốt.
  * So sánh nhánh “raw features” và “selected/engineered features”.
* **Đầu ra cuối ngày:**

  * 1 bảng so sánh “trước/sau feature engineering”.
  * 1 shortlist 2–3 pipeline/model đáng đầu tư.
* **Tiêu chí hoàn thành:**

  * Mọi bước loại/tạo feature đều có lý do, không làm cảm tính.
  * Có ít nhất 1 nhánh cho kết quả tốt hơn baseline Day 3.

### Day 5

* **Mục tiêu:** Tuning có trọng tâm để tiến sát hoặc vượt benchmark từng phần.
* **Học gì:**

  * Grid Search / Random Search.
  * Chọn không gian siêu tham số vừa đủ.
  * Threshold tuning theo F1 nếu cần.
* **Làm gì:**

  * Chọn 2 model tốt nhất từ Day 4 để tune.
  * Tune riêng cho từng họ model:

    * Logistic/SGD: `C`, penalty, alpha, class_weight
    * ExtraTrees/RandomForest: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
    * HistGradBoost/LightGBM/XGBoost: learning rate, depth/leaves, regularization, subsampling
  * Đánh giá bằng CV trước, sau đó test trên `X_test`, `y_test`.
  * Nếu F1 tốt nhưng precision/recall lệch, thử threshold tuning.
* **Đầu ra cuối ngày:**

  * 1 model tuned tốt nhất trên validation/test.
  * 1 bảng “baseline vs tuned”.
* **Tiêu chí hoàn thành:**

  * Bạn có ít nhất 1 model vượt baseline rõ rệt.
  * Có bằng chứng tuning giúp gì, không chỉ “thử đại”.

### Day 6

* **Mục tiêu:** Đánh giá trên cả test và challenge, thử ensemble nhẹ, bắt đầu đóng gói câu chuyện trình bày.
* **Học gì:**

  * Voting/blending/stacking cơ bản.
  * Cách phân tích vì sao model thắng hoặc thua benchmark.
* **Làm gì:**

  * Chạy model tốt nhất trên cả `test` và `challenge`, in đủ Accuracy, Precision, Recall, F1, AUC theo yêu cầu README.
  * So benchmark: đã vượt được những model nào, còn thiếu gì.
  * Nếu có 2–3 model bổ sung nhau, thử soft voting hoặc blend xác suất.
  * Viết phần “what worked / what failed”.
  * Chọn biểu đồ và bảng tốt nhất cho slide.
* **Đầu ra cuối ngày:**

  * 1 bảng kết quả cuối tạm thời trên test và challenge.
  * 1 kết luận sơ bộ: pipeline nào là ứng viên nộp bài.
* **Tiêu chí hoàn thành:**

  * Đã có kết quả đầy đủ theo đúng format đề yêu cầu.
  * Có thể nói rõ pipeline nào đang là “main pipeline” và vì sao.

### Day 7

* **Mục tiêu:** Ổn định bài nộp tuần 1: code sạch, story rõ, slide khung xong.
* **Học gì:**

  * Cách chuyển notebook thành thuyết trình 10–15 phút.
  * Cách giải thích quyết định kỹ thuật ngắn gọn nhưng có lý.
* **Làm gì:**

  * Dọn notebook/script: bỏ cell lỗi, thêm comment quan trọng, cố định seed, kiểm tra rerun.
  * Chốt file kết quả và bảng benchmark comparison.
  * Tạo dàn ý tối thiểu 12 slide theo đúng cấu trúc README.
  * Viết bullet cho phần thuyết trình:

    * dữ liệu hiểu ra sao
    * pipeline làm gì và vì sao
    * kết quả so benchmark
    * bài học và hướng tiếp theo
  * Tự rehearsal 1 lượt 10 phút.
* **Đầu ra cuối ngày:**

  * 1 notebook/script sạch, chạy được.
  * 1 bản slide khung.
  * 1 danh sách việc cần làm cho tuần 2 để tối ưu tiếp.
* **Tiêu chí hoàn thành:**

  * Bạn có thể demo notebook, chỉ ra pipeline chính, chỉ ra kết quả, và trình bày được câu chuyện làm bài liền mạch.
  * Tài liệu đã đủ nền để tuần sau tập trung tối ưu sâu hơn thay vì quay lại vá nền tảng.

## PHẦN 4 — CHIẾN LƯỢC ĐỂ ĐẠT KẾT QUẢ TỐT NHẤT

### 1) Thứ tự thử model hợp lý

Dựa trên benchmark trong README, mình khuyên đi theo thứ tự này:

1. **Logistic Regression / SGD(log)** để có baseline nhanh, dễ đọc, dễ kiểm soát scaling và threshold.
2. **ExtraTrees / RandomForest / HistGradientBoosting** để khai thác dữ liệu số và độ phi tuyến.
3. **LightGBM / XGBoost** nếu môi trường cho phép, vì benchmark test đang rất mạnh ở nhóm boosting.
4. **Blending/Voting** chỉ sau khi đã có 2 model mạnh thật sự. 

Lý do: bạn cần một baseline nhanh để hiểu dữ liệu trước, rồi mới lao vào boosting. Nhảy ngay vào model phức tạp dễ tốn thời gian nhưng không biết mình đang được lợi từ preprocessing, feature engineering hay chỉ ăn may từ tuning.

### 2) Hướng có khả năng vượt benchmark cao hơn

README đã gợi ý khá rõ: benchmark mặc định không xử lý dữ liệu, nên cách thắng dễ nhất không phải “mô hình kỳ dị”, mà là:

* preprocessing đúng,
* feature selection/engineering có căn cứ,
* CV chuẩn,
* tuning có trọng tâm,
* có thể thêm ensemble ở cuối. 

Nói thẳng: **quick win** thường đến từ `imputation + scaling đúng + loại feature rác + tuning nhẹ` trước khi đến các thứ phức tạp như stacking.

### 3) Quick wins nên làm trước

* Kiểm tra missing và impute đúng cách.
* So sánh StandardScaler với RobustScaler.
* Loại cột near-constant hoặc cực nhiễu.
* Dùng `StratifiedKFold`.
* Tune threshold theo F1 nếu mô hình trả xác suất.
* So sánh raw features với 1 nhánh feature selection nhẹ.

Đây là các bước có chi phí thấp nhưng khả năng nâng điểm khá tốt.

### 4) Phần có thể đào sâu nếu còn thời gian

* Bayesian optimization thay cho grid/random search.
* Stacking giữa model tuyến tính và tree-based.
* Phân tích drift giữa train với challenge.
* Error analysis trên các mẫu khó.
* So sánh nhiều chiến lược feature engineering thống kê.

### 5) Bẫy cần tránh

* **Leakage:** fit scaler/imputer trên toàn bộ train trước khi CV.
* **Tuning quá sớm:** chưa có baseline đã lao vào search khổng lồ.
* **Chỉ nhìn accuracy:** trong bài này phải bám F1/AUC và nhìn nhãn 1.
* **EDA quá dài:** vẽ rất nhiều nhưng không ra quyết định.
* **Notebook lộn xộn:** README chấm cả tính tái hiện và tính trình bày. 

### 6) Chiến lược thời gian hợp lý

* Nửa đầu tuần: hiểu dữ liệu + xây pipeline đúng.
* Giữa tuần: baseline + feature engineering + shortlist model.
* Cuối tuần: tuning có chọn lọc + đánh giá test/challenge + chuẩn bị slide.

Cách này giúp bạn đến khoảng Day 5 đã có một baseline đủ tốt, thay vì đến Day 7 vẫn còn loay hoay load data.

## PHẦN 5 — CHECKLIST SAU 1 TUẦN

Sau 1 tuần, bạn nên có trong tay:

* [ ] 1 notebook hoặc script **chạy được từ đầu đến cuối** theo đúng pipeline yêu cầu của đề. 
* [ ] Phần load dữ liệu đủ 6 file, có check shape, dtype, missing. 
* [ ] 1 EDA sơ bộ nhưng có giá trị ra quyết định: label distribution, feature distribution, correlation, outlier. 
* [ ] 1 preprocessing pipeline rõ ràng, giải thích được vì sao chọn scaling/imputation đó. 
* [ ] 1 nhánh feature engineering hoặc feature selection có căn cứ.
* [ ] 1–3 model baseline đã chạy bằng cross-validation. 
* [ ] Ít nhất 1 model tuned sơ bộ.
* [ ] 1 bảng so sánh metric giữa các model.
* [ ] Kết quả đánh giá đầy đủ trên cả `test` và `challenge`: Accuracy, Precision, Recall, F1, AUC. 
* [ ] 1 danh sách rõ “cái gì hiệu quả / cái gì không hiệu quả”.
* [ ] 1 dàn ý slide 12 trang bám đúng cấu trúc README. 
* [ ] 1 khung nói 10–15 phút cho phần thuyết trình. 

Vị trí bạn nên đạt sau tuần 1 là: **đã hiểu đề rất chắc, có pipeline đầu tiên tử tế, có baseline đủ mạnh để so benchmark, có hướng tối ưu tiếp theo rõ ràng, và đã bắt đầu đóng gói kết quả cho code + slide + thuyết trình**.

Muốn, mình có thể làm tiếp cho bạn một bản **“roadmap triển khai notebook/script theo từng section”** để bạn mở máy lên là làm ngay từng phần.
