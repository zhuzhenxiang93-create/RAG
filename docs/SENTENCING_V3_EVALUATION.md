# Sentencing v3 evaluation

罪名报告 micro/macro F1、precision、recall；刑罚类型报告 accuracy、macro F1 和混淆矩阵；
刑期报告 MAE、Median AE、区间准确率；罚金是否判处报告 precision/recall/F1。罚金金额
只在真实 positive 样本计算，并按罪名、来源和可用年份分解。

v2.1.1/v3 公平比较必须使用相同模型、种子和任务定义，并另报共同测试子集。不同原生测试
集上的数字只能描述，不能证明 v3 更好。LeCaRDv2 qrels 是独立专家标签；proxy qrels 必须
明确称为 proxy。
