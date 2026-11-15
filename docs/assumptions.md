# 已確認的補充設定

狀態：使用者於 2026-10-26 明確同意本表補充設定，並授權白天資料完成後開始三模型正式訓練。
以下仍是補充設定，**不是論文已公開的完整設定**；確認不代表取得了作者原始超參數。

| 項目 | 建議值 | 依據／限制 |
|---|---|---|
| learning rate | 0.001 | Keras Adam 常見預設；論文只寫 Adam |
| batch size | 32 | 論文未明確給值；若本機記憶體不足需另調整並記錄 |
| 隨機種子 | 42 | 論文說固定種子但未給數值；PyTorch/Python 打亂亦不同於 TensorFlow |
| 像素處理 | RGB、除以 255，無額外標準化 | 未公開，選擇透明且簡單的處理 |
| resize | PIL bilinear、直接縮放至 244×244 | 插值與 EXIF 處理未公開；不額外偵測或增強 |
| MaxPool padding | valid | 3×3、stride 2 明述；padding 未明述，採 Keras 常見預設 |
| BatchNorm | epsilon 0.001、moving retention 0.99 | Keras 常見預設；自訂 PyTorch 層使用 population variance 更新移動統計 |
| BatchNorm scale | false；保留 beta | **由表 3 推得**：三模型分別少 2016、3011、2112 個參數，恰為各 BN 通道總和；關閉 scale 可精確對上三模型參數數目。但僅靠數目不能證明作者一定關閉的是 scale 而非 center |
| 初始化 | Glorot uniform；Dense bias=0 | 採 Keras 常見預設；DW fan-in/out 依 Keras kernel 形狀計算 |
| Adam | betas=(0.9,0.999)、epsilon=1e-7、weight_decay=0 | Keras 常見預設；PyTorch Adam 與 TensorFlow Adam 對 epsilon／bias correction 的實作仍有數值差異，不宣稱逐步等價 |
| 計時 | GPU 暖機 3 組，再測 8×25；同步 GPU | 論文明述 8×25，未交代暖機及同步；另記每組與每張，不猜測表 3 單位 |
| CompCars 抽樣 | 每品牌擴充至 600 張候選；亮度初篩＋逐圖白天外觀審查後選 200 張 | 使用者已指定只用白天；門檻會排除部分偏暗的白天圖片，仍不同於作者資料 |

正式訓練不使用驗證集、最佳 checkpoint、early stopping、增強、排程或混合精度。
只評估第 200 epoch；所有三模型共用同一 manifest。

確認已記錄於 `configs/paper.json` 的 `assumptions_confirmed=true` 與 `assumption_note`。日後若更動設定，應另存實驗並保留差異紀錄。

補充設定參考官方文件（用於解釋預設值，不能證明作者實際採用）：
- https://keras.io/api/layers/normalization_layers/batch_normalization/
- https://keras.io/api/optimizers/adam/
- https://keras.io/api/layers/pooling_layers/max_pooling2d/

## 論文內部不一致的處理

- 第 5 頁正文六層通道漏列 256；偽程式碼及 Figure 9 列出 32,64,128,256,512,1024，採後者。
- Figure 8 的 DR 標為 X/2，正文卻寫固定 16；固定 16 可對上 Table 3，採固定 16。
- Figure 9 的 DR 區塊列中有 DW 名稱，依正文與圖標題判定為標記不一致。
- 表 3 總參數含 BN 非訓練統計；報告分開列 PyTorch 可訓練參數及 BN 統計，避免錯誤比較。
- 論文環境為 Python 3.9.18、TensorFlow 2.10、RTX 3060 Ti；本機為 Python 3.11、PyTorch、RTX 4060 Laptop，因此不能直接比較絕對耗時。
