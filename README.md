## 数据预处理
### 1. 获取价格数据
将 stocknet-dataset 的原始 CSV 价格数据转换为 FinMem 框架要求的输入格式。
```shell
python D:\codeFiles\python_file\毕设\FinMem-lhy-master\FinMem-lhy-master\data-pipeline\price_data_format_transform.py
```
### 2. 获取新闻数据
基于 stocknet-dataset 的 tweet 数据，为 FinMem 生成 news.pkl。
```shell
python D:\codeFiles\python_file\毕设\FinMem-lhy-master\FinMem-lhy-master\data-pipeline\build_news_from_tweets.py
```
### 3. 获取财报数据
从 SEC EDGAR 下载 10-K 和 10-Q 财报数据，并转换为 FinMem 输入格式。
得到 filing_data.parquet
```shell
python D:\codeFiles\python_file\毕设\FinMem-lhy-master\FinMem-lhy-master\data-pipeline\01_SEC_EDGAR_10k10q_download.py
```
### 4. 合并数据
合并:
  - data/03_model_input/aapl_env_data_full.pkl   (已有 price + news, filing 为空)
  - data/03_primary/filing_data.parquet          (8 份 AAPL 10-K/10-Q)

输出:
  - data/03_model_input/env_data.pkl             (FinMem run.py 直接消费)

```shell
python D:\codeFiles\python_file\毕设\FinMem-lhy-master\FinMem-lhy-master\data-pipeline\build_env_data.py
```
最终得到 env_data.pkl 作为环境数据输入。
## 训练模型 和 测试模型
- 输入：env_data.pkl、config.toml
- 输出：模型对应的 .csv 动作文件
### 1.1. 训练和测试完整 FULL 模型
```shell
python D:\codeFiles\python_file\毕设\FinMem-lhy-master\FinMem-lhy-master\scripts\run_finmem_full.py
```
### 基线和消融对比
#### 1. 训练和测试 单独开启 CFR 模型
```shell
python D:\codeFiles\python_file\毕设\FinMem-lhy-master\FinMem-lhy-master\scripts\run_only_CFR_baseline.py
```
#### 2. 训练和测试 单独开启 ETR 模型
```shell
python D:\codeFiles\python_file\毕设\FinMem-lhy-master\FinMem-lhy-master\scripts\run_only_ETR_baseline.py
```
#### 3. 训练和测试 单独开启 RGR 模型
```shell
python D:\codeFiles\python_file\毕设\FinMem-lhy-master\FinMem-lhy-master\scripts\run_only_RGR_baseline.py
```
#### 4. 训练和测试 对比原始 FinMem 模型
```shell
python D:\codeFiles\python_file\毕设\FinMem-lhy-master\FinMem-lhy-master\scripts\run_finmem_orig_baseline.py
```
#### 5. 训练和测试 FinGPT 模型
```shell
python scripts\run_fingpt_baseline.py
```
#### 6. 训练和测试 FinRL 基线: DQN / PPO / A2C
```shell
python scripts/run_finrl_baselines.py
```
## 评估
一次性跑出所有分组评估结果，包含累计收益曲线、收益率、夏普比率、最大回撤等指标，以及与基线模型的对比分析。
```shell
python D:\codeFiles\python_file\毕设\FinMem-lhy-master\FinMem-lhy-master\eval\run_batch_eval.py
```