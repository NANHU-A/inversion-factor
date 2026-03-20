# 极坐标价量融合反转因子项目

> 本项目基于中国A股日频行情数据，构建 **价量融合反转因子** 并进行实证检验。

## 1. 项目目标
- 通过将 **收盘价** 与 **成交额** 联合建模，提取价量状态的强度（极径）与方向（极角）。
- 结合象限偏好与角度权重，得到单周期因子，并在【20、60、120、240】日四个周期下合成复合因子。 
- 使用 **未来收益率** 作为标签，评估因子性能（IC、RankIC、分层回测）。
- 在 baseline 基础上实现两种优化：
  1. 因子后处理（去极值、标准化、中性化）。
  2. 依据历史 RankIC 做动态加权。 

## 2. 基本思路
1. **数据预处理**：读取 parquet，统一字段、清洗缺失、去重、时间排序。 
2. **构造未来收益率**：`fwd_ret = future_close / close - 1`（或用复权价）。 
3. **价量差分**：以过去 N 日为参考，计算收盘价与成交额的变化率。 
4. **极径**：使用 **马氏距离** 以体现尺度与相关性。 
5. **极角**：`theta = arctan2(delta_amount, delta_close)`，精准判定四象限。 
6. **角度权重**：以 45° 为中心的衰减函数 `f(theta)`。 
7. **象限偏好**：四象限 alpha 系数对信号取正负。 
8. **单周期因子**：`factor = alpha * rho * f(theta)`。 
9. **复合因子**：等权或动态加权合成。 
10. **性能评估**：IC、RankIC、分层回测。 

## 3. 文件结构
```
├─ exam_QR_周文熙.ipynb          # 主 Notebook
├─ quant_polar_reversal.py        # 函数实现
├─ generate_notebook.py           # Notebook 生成脚本
├─ README.md                      # 说明文件
├─ readme_new.md                  # 本文件
├─ exam_data.parquet              # 数据集
├─ requirements.txt               # 依赖
```

## 4. 运行方式
```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Notebook
jupyter notebook exam_QR_周文熙.ipynb
```

> 如果想直接调用 Python API，请参见 `quant_polar_reversal.py`。

## 5. 需要改进的地方
- 使用复权价替代收盘价做收益率与因子计算，进一步验证稳健性。 
- 加入行业、市值、中性化后再做因子检验。 
- 通过多周期加权探索不同时间尺度的权重动态。 

## 6. 结语
本项目旨在展示从原始行情到因子构造、检验、优化的完整工作流，适合作为量化因子学习与实战的案例。
