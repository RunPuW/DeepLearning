# Walkthrough: 数据格式适配

## 修改内容

### [preprocess.py](file:///c:/Users/Administrator/Desktop/整合+预处理数据(1)/整合+预处理数据/preprocess.py)

| 修改项 | 原始 | 修改后 |
|--------|------|--------|
| FPB 路径 | `all-data.csv` | [FinancialPhraseBank/Sentences_50Agree.txt](file:///c:/Users/Administrator/Desktop/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%281%29/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE/FinancialPhraseBank/Sentences_50Agree.txt) |
| SEntFiN 路径 | `SEntFiN-v1.1.csv` | [SEntFiN/SEntFiN.csv](file:///c:/Users/Administrator/Desktop/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%281%29/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE/SEntFiN/SEntFiN.csv) |
| FinEntity 路径 | `FinEntity.json` | `FinEntity/train/data-*.arrow` |
| FinMarBa 路径 | `FinMarBa.csv` | `FinMarBa/train/data-*.arrow` |
| [load_fpb()](file:///c:/Users/Administrator/Desktop/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%281%29/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE/preprocess.py#135-187) | `pd.read_csv()` 读 CSV | 逐行读 [.txt](file:///c:/Users/Administrator/Desktop/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%281%29/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE/FinancialPhraseBank/README.txt)，按最后一个 `@` 分割 text 和 label |
| [load_finentity()](file:///c:/Users/Administrator/Desktop/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%281%29/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE/preprocess.py#269-322) | `json.load()` | `HFDataset.from_file()` + `.to_list()` |
| [load_finmarba()](file:///c:/Users/Administrator/Desktop/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%281%29/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE/preprocess.py#368-431) | `pd.read_csv()` | `HFDataset.from_file()` + `.to_pandas()` |
| label 列检测 | 不含 `Global Sentiment` | 优先检测 `Global Sentiment` 列 |

### [eda.py](file:///c:/Users/Administrator/Desktop/整合+预处理数据(1)/整合+预处理数据/eda.py)

- `TRAIN_PATH`: `final_train.jsonl` → `final_integrated.jsonl`

## 验证结果

[preprocess.py](file:///c:/Users/Administrator/Desktop/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%281%29/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE/preprocess.py) 输出:
```
FPB:       4,846 rows loaded
SEntFiN:  14,371 expanded rows (13 parse errors)
FinEntity: 2,131 expanded rows
FinMarBa:  8,142 rows (100% valid market_label)
Total:    29,490 → 29,489 (legality) → 29,259 (dedup)
```

[eda.py](file:///c:/Users/Administrator/Desktop/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%281%29/%E6%95%B4%E5%90%88+%E9%A2%84%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE/eda.py) 输出: 7 张图 + `eda_summary.json` 全部正常生成。
