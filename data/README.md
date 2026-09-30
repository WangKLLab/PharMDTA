# 数据文件

代码仓库包含 `component_sequences.csv`，列为 `protein_component_id` 和 `sequence`，共 6812 个蛋白组分。

训练划分 CSV 保留在本地工作目录，不随代码提交。克隆仓库后，请按下面的路径准备数据：

```text
data/
├── component_sequences.csv
├── bindingdb/splits/{train,val,test}.csv
└── kiba/splits/{train,val,test}.csv
```

训练还需要与数据划分配套的 `manifest.json`、口袋配置和图文件，以及 ESM-C-6B 残基特征缓存。`--data-dir` 指向同时包含 `manifest.json` 和 `splits/` 的数据集目录。
