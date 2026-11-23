本機完整版已附正式 1,000 張原圖、20 組最終權重與續訓 checkpoint。請重建 .venv 後從任務 02 開始檢查。

# 未裁切版交付

本資料夾只使用 CompCars 原圖直接縮放至 244×244，包含三核心、五基準、十二消融及五種影像干擾。YOLO11m-cls 是品牌分類基準。

GitHub 版提供程式、最終權重、三核心評估所需 checkpoint 與實測數據，圖片需依官方條件取得後還原。

## 助教操作順序

1. VS Code 開啟資料夾，執行「01 建立與檢查環境」。
2. GitHub 下載者先執行 git lfs pull，再依下列指令還原圖片；完整版已附固定 1,000 張。
3. 執行「02 檢查圖片與原始切分」。
4. 執行「03 使用原權重驗證結果」或「期末報告：單張辨識」。
5. 要從頭訓練核心模型用 04，接續用 05，開啟報告用 06。
6. 延伸模型先安裝 requirements.extended.txt。extend.py all 會略過已完成項目。從頭重跑延伸模型請在獨立副本備份後移出 extended_results 既有模型輸出，保留原基準紀錄。

```powershell
git lfs pull
.\.venv\Scripts\python.exe restore_compcars.py 'D:/CompCars/sv_data'
.\.venv\Scripts\python.exe verify_split.py
.\.venv\Scripts\python.exe restore_compcars.py --demo
```

[CompCars 官方來源與條款](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)


> GitHub 版提供程式、固定資料清單與實測報告。CompCars 圖片請依官方條件取得，按 data/manifest.csv 放置原圖。模型權重可重訓產生。

# 車頭品牌辨識：VS Code 助教繳交版

用 VS Code 開啟本資料夾，選「終端機 → 執行工作」，依下表操作。也可按 Ctrl+Shift+P 搜尋「Tasks: Run Task」。需 Windows、Python 3.11 與支援 CUDA 12.8 的 NVIDIA GPU 及驅動。第一次操作建議依序執行 01 → 02 → 03，再用 04 完整重訓。

| 任務 | 何時使用 |
|---|---|
| 01 建立與檢查環境 | 第一次在本機執行。安裝時間取決於網路。若收到搬移的 .venv，先把它移出專案，再執行此任務。 |
| 02 檢查圖片與原始切分 | 驗證 1,000 張原圖、品牌、像素雜湊及原始 800／200 切分。 |
| 03 使用原權重驗證結果 | 重新評估三模型，適合快速核對逐圖預測與分類指標。本機實測約 2 分鐘。 |
| 04 從頭重現三模型 | 從隨機初始化開始，各模型訓練 200 輪、評估並比較。本機重訓實測三模型合計約 44 分鐘（訓練計時），另需資料讀取、寫入 checkpoint 與評估。 |
| 05 繼續中斷的重跑 | 接續最近一次未完成的重跑；每個模型從最後完成的 epoch 開始。 |
| 06 開啟最近一次完整重跑報告 | 在 VS Code 開啟最後完成的完整重跑報告。 |

原始結果在 outputs。每次任務 03 或 04 會在 reproduction 建立獨立資料夾。完成後開啟其中的 reproduction-report.md，比對原始數據與本次結果。prediction-differences.csv 列出預測不同的圖片。各模型子資料夾包含逐圖預測、逐品牌數據與圖表。完整重跑另外保存逐輪訓練紀錄、權重與續訓 checkpoint。

不同硬體、驅動及數值運算可能影響預測與時間。比對會檢查正式清單、測試圖片、標籤與參數量；報告如實呈現分數差異。驗證原權重時，訓練時間沿用原 checkpoint；推論時間每次實測。

固定資料來自 CompCars surveillance，Toyota、Honda、Mazda、Nissan、Volkswagen 各 200 張。訓練與評估使用 data/manifest.csv。獨立切分程式使用打亂前的固定順序及亂數狀態還原本次切分。

論文 DOI 為 10.1109/ACCESS.2025.3574187。設定與差異見 docs/experiment-comparison.md，資料查核與白天篩選紀錄見 docs/source-audit.md。歷史候選清單含已移除圖片的路徑。圖片來自 [CompCars 官方資料集](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)，供非商業研究使用，繳交及存取請依資料集條款及課程授權辦理，請勿公開散布原圖。



## 電腦端延伸實驗

首次安裝延伸套件（先完成任務 01）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.extended.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

使用 VS Code「終端機 → 執行工作」。07 驗證全部新增模型；08 依序訓練 17 個模型並執行影像干擾測試；09 至 13 可執行各組；14 開啟報告。重新執行同一任務會接續最近的 checkpoint。輸出位於 extended_results，各模型完成評估後會寫入 complete.json。

依 Figure 11，五區塊通道為 64／128／256／512／1024。Stride 版把後兩區塊的 3×3 卷積改為步長 2，1×1 維持步長 1；No Padding 版移除卷積 padding，池化沿用原設定。BN 消融以 Identity 取代 BN，卷積 bias 設定照舊。基準採 torchvision 的 VGG16、InceptionV3（關閉輔助分類頭）、ResNet50，另實作 MobileNet V1，YOLO11m-cls 使用官方架構設定及隨機初始化。PyTorch 的分類頭、初始化與 BN 計數可能與 TensorFlow 不同。評估仍計時八組各 25 張，記憶體不足的版本以已驗證的訓練 batch 大小切成子批次。

Figure 13 干擾規則：原圖上做 1–10 像素方框模糊、標準差 0–100 的 Gaussian noise、0–0.018 機率的椒鹽噪聲、每通道扣除 0–100 的亮度，以及 2–11 個水滴。採固定種子；椒鹽最後一點約 0.018，水滴直徑按原圖短邊相對 600 像素縮放，使用模糊及半透明白色圓形近似。作者未公布精確合成程式，這些規則屬本次實作。