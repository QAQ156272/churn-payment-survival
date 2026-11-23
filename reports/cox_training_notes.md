# Cox 训练记录：打分后到下一笔购买的时间

## 数据

- 表：data/processed/cox_model_table.csv
- 目标：从首购后第 14 天打分日起，到下一笔有效购买的时间
- 打分日当天成交记 T=1；E=1 观察到下一单，E=0 右删失

| split | 人数 | 事件 | 删失 |
|---|---|---|---|
| train | 2097 | 925 | 1172 |
| val | 382 | 128 | 254 |

测试集未参与任何拟合或指标计算。

## 配置

- 模型 A 特征：days_since_last_buy
- 模型 B 特征（白名单全量，顺序）：n_invoices, first_order_amount, days_since_last_buy, n_products, top1_product_share
- 惩罚：无惩罚（penalizer=0）
- 变换：无对数变换、无标准化，HR 直接按业务增量换算
- 依赖：lifelines=0.30.3, pandas=2.3.3, numpy=2.5.5, scipy=1.18.1

## 收敛

- penalizer=0：无收敛或数值警告，通过校验

## 验证集 C-index（Harrell）

| 模型 | train C-index | val C-index |
|---|---|---|
| A（单变量） | 0.5345 | 0.5620 |
| B（全特征） | 0.5857 | 0.6416 |

**B − A = 0.0796**

## HR 表

| 模型 | 特征 | 增量 | HR | 95% CI | p |
|---|---|---|---|---|---|
| A: days_since_last_buy only | days_since_last_buy | +1 天 | 0.932 | [0.911, 0.954] | 1.767e-09 |
| B: all whitelisted features | n_invoices | +1 张发票 | 1.066 | [0.889, 1.278] | 0.4885 |
| B: all whitelisted features | first_order_amount | +100 GBP | 1.006 | [1.001, 1.011] | 0.03016 |
| B: all whitelisted features | days_since_last_buy | +1 天 | 0.946 | [0.918, 0.974] | 0.0002493 |
| B: all whitelisted features | n_products | +1 个商品 | 1.003 | [1.001, 1.006] | 0.01297 |
| B: all whitelisted features | top1_product_share | +10 个百分点 | 0.921 | [0.886, 0.957] | 3e-05 |

## 业务解读（基于本次拟合）

1. 本次验证集排序：A=0.5620，B=0.6416，差值 +0.0796。这是验证集点估计，尚不是测试集结论。

2. 全特征模型中，其他变量相同的条件下，days_since_last_buy +1 天：HR=0.946（95% CI [0.918, 0.974]），下一单的瞬时发生率低约 5.4%。

3. 全特征模型中，其他变量相同的条件下，n_invoices +1 张发票：HR=1.066（95% CI [0.889, 1.278]），区间包含 1，不据此判断方向。

## 读法（务必按此解释）

- HR > 1：在其他入模变量相同时，该变量每增加一个单位增量，**发生下一单的瞬时率更高**
- HR < 1：瞬时率更低
- **不是**购买概率的变化，**不是**流失风险，**不是**因果效果
- 95% CI 跨 1 时，不认定方向

## 未做

- 比例风险假设检验（Schoenfeld 残差）留到第 3 步
- 测试集评估、校准、XGBoost 对比不在本步