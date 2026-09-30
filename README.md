# PharMDTA

## 目录

```text
PharMDTA/
├── environment.yml         
├── configs/
│   ├── bindingdb.json       
│   └── kiba.json           
├── data/
│   ├── component_sequences.csv 
│   ├── bindingdb/splits/   
│   └── kiba/splits/        
├── src/model/             
└── scripts/                
```
<!--  -->

## 环境

在本目录执行以下命令：

```bash
conda env create -f environment.yml
conda activate pharmdta
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## 训练

BindingDB：

```bash
python scripts/train.py \
  --config configs/bindingdb.json \
  --data-dir /path/to/bindingdb \
  --pocket-contract /path/to/bindingdb_pocket_contract.json \
  --component-sequences data/component_sequences.csv \
  --esmc6b-embeddings /path/to/esmc6b_embeddings \
  --output-dir runs \
  --device cuda \
  --micro-batch-size 16 \
  --num-workers 4
```

训练 KIBA 时，将配置改为 `configs/kiba.json`，并使用 KIBA 对应的数据、口袋和特征缓存路径。默认运行目录分别为 `runs/bindingdb_seed42/` 和 `runs/kiba_seed42/`；可通过 `--run-name` 指定名称。

也可以使用 Shell 入口：

```bash
DATASET=kiba \
DATA_DIR=/path/to/kiba \
POCKET_CONTRACT=/path/to/kiba_pocket_contract.json \
COMPONENT_SEQUENCES=data/component_sequences.csv \
ESMC6B_EMBEDDINGS=/path/to/esmc6b_embeddings \
bash scripts/train.sh
```

## 评估

在上述环境中，使用训练生成的检查点：

```bash
python -m model.evaluate \
  --checkpoint runs/bindingdb_seed42/best.pt \
  --split test \
  --output-dir results/bindingdb_test \
  --device cuda \
  --micro-batch-size 16
```

评估会读取检查点同目录的 `config.json`，请将两者一并保留。输入文件移动后，可通过与训练相同的四个输入路径参数重新指定位置；输出目录应为空或尚不存在。

评估生成 `predictions.csv` 和 `metrics.json`，提供原始标签尺度上的 MSE、RMSE、MAE、Pearson、Spearman、CI、R² 和残差标准差。添加 `--attention-limit 10` 可导出前 10 个可用样本的原子–残基注意力及重要性结果。

## 训练输出

每次训练的主要输出位于对应的 `runs/<运行名称>/`：

- `best.pt`：验证集表现最佳的检查点。
- `last.pt`：包含优化器等状态的最新检查点，用于续训。
- `config.json`：本次运行配置及输入路径。
- `history.csv`：逐 epoch 的训练损失和验证指标。
- `final_metrics.json`：最佳模型的验证集与测试集指标。
- `predictions/val_best.csv`、`predictions/test_best.csv`：最佳模型的预测结果。

## 查看配置和模型规模

以下命令无需读取训练数据或特征缓存：

```bash
python scripts/train.py --config configs/bindingdb.json --dry-run
python scripts/model_summary.py --config configs/bindingdb.json
```


