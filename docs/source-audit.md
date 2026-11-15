# 來源查核紀錄

查核日期：2026-10-25。未找到不等於確定不存在；下列是此次查核範圍。

## 原論文與作者

- 本機原論文：`../Minimizing_Model_Size_of_CNN-Based_Vehicle_Make_Recognition_for_Frontal_Vehicle_Images.pdf`（相對專案根目錄）。
- DOI：https://doi.org/10.1109/ACCESS.2025.3574187
- IEEE：https://ieeexplore.ieee.org/abstract/document/11015954
- 作者 publications：https://rathachai.creatier.pro/publications
- 搜尋完整論文標題、作者名與 github/dataset，以及 CTv1 vehicle github；未找到可確認屬於本論文的原始程式庫或 1000 張資料下載。
- 本機 PDF 內容未找到原始資料／程式公開連結。IEEE 完整 document 頁未能由網頁工具讀取；不宣稱已核查無法讀取的所有補充附件。
- 已渲染並核對 PDF 第 5、6、8 頁的 Figure 8/9、演算法、Table 1/3；繳交版保留此查核紀錄。
- 未寄信聯絡作者。

## CompCars 官方備援

- 主頁：https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html
- 官方下載指引：https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt
- 指引提供的公開資料夾：https://drive.google.com/drive/folders/18EunmjOJsbE5Lh9zA0cZ4wKV6Um46dkg
- 透過 Google Drive 連接工具確認官方資料夾的 sv_data.zip、sv_data.z01-z03；大型 raw fetch 發生 Transport closed，改採相同官方公開檔案的 HTTP Range。
- 官方 README（web 子集）已保存為 docs/compcars-official-readme.txt；surveillance 自己的 README 與 .mat 必須以該壓縮檔內實際內容為準。
- 2026-10-26 已讀取 surveillance 原始 README 及 `sv_make_model_name.mat`：矩陣為 281×3，第一欄為品牌，第二欄車型，第三欄對應 web 子集 ID。依 README 使用 1-based 列號對應圖片父資料夾的 surveillance_model_id。
- 官方標籤中沒有 Isuzu；Toyota 3051、Honda 2035、Mazda 1626、Nissan 1950 張；依既定備援規則以 Volkswagen 4480 張補第五品牌。
- 實際抽看發現 surveillance 包含夜間圖片，與論文全白天不同；使用者已要求另篩白天，每品牌仍 200 張。採亮度候選篩選後逐圖視覺審查，保留完整紀錄；沒有拍攝時間標籤，須揭露白天外觀判斷限制。
- 官方發布的 ZIP 密碼用於解壓研究資料，不是帳號密碼。
- 原候選下載流程驗證 Range 回應、原檔 ZIP CRC；品牌採官方 sv_make_model_name.mat，不從照片或車型名稱猜測。
- 僅供非商業研究，勿公開重散布原始圖片。資料引用：Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang, “A Large-Scale Car Dataset for Fine-Grained Categorization and Verification”, CVPR 2015。

## 狀態分類

程式可執行、合成資料測試成功、資料已準備、正式訓練完成是四種不同狀態。outputs/smoke.json 只證明軟體試跑，不是車廠辨識結果。實際完成狀態見 outputs/status.md。

