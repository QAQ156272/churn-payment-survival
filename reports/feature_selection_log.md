# 特征筛选记录

只在 train 上拟合筛选规则，原样应用到 val / test。

## 初始候选（7 列）

| 列 | train 均值 | train 标准差 | 缺失 |
|---|---|---|---|
| n_invoices | 1.1426 | 0.4282 | 0 |
| n_active_days | 1.0801 | 0.2902 | 0 |
| first_order_amount | 426.5182 | 996.1598 | 0 |
| window_amount | 477.3064 | 1224.1355 | 0 |
| days_since_last_buy | 13.4111 | 2.3004 | 0 |
| n_products | 26.6452 | 26.4171 | 0 |
| top1_product_share | 0.2123 | 0.2251 | 0 |

## 逐步删除

| 步骤 | 删除列 | 原因 |
|---|---|---|
| high_correlation | window_amount | |r|=0.969 with first_order_amount |
| vif | n_active_days | VIF=6.11 > 5 |

## 保留特征与 VIF

| 列 | VIF |
|---|---|
| n_invoices | 1.774 |
| first_order_amount | 1.069 |
| days_since_last_buy | 1.707 |
| n_products | 1.372 |
| top1_product_share | 1.262 |

## 保留特征相关矩阵

| | n_invoices | first_order_amount | days_since_last_buy | n_products | top1_product_share |
|---|---|---|---|---|---|
| n_invoices | 1.00 | 0.13 | -0.64 | 0.23 | -0.02 |
| first_order_amount | 0.13 | 1.00 | -0.01 | 0.22 | -0.10 |
| days_since_last_buy | -0.64 | -0.01 | 1.00 | -0.17 | 0.05 |
| n_products | 0.23 | 0.22 | -0.17 | 1.00 | -0.45 |
| top1_product_share | -0.02 | -0.10 | 0.05 | -0.45 | 1.00 |