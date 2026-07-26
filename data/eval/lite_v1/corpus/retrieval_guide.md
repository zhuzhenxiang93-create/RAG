# 检索策略

BM25 根据词频和逆文档频率计算相关性，适合产品编号、错误码、专有名词及罕见关键词等精确查询。

Dense Retrieval 将文本映射为向量，适合查询和文档用词不同但语义相近的场景。

Reciprocal Rank Fusion（RRF）只使用候选在各检索通道中的排名，能够避免直接相加不同量纲的原始分数。

## Parent-Child

Child Chunk 用于精确召回，命中后回溯 Parent Chunk，为生成阶段提供较完整的上下文。Child Chunk 应携带所属章节标题。
