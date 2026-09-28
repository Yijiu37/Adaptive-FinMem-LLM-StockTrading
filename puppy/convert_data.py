import pickle
from datetime import date


def convert_env_data(input_path: str, output_path: str, tickers: list = None):
    """
    转换错误格式(tuple)的数据到正确格式(dict)
    """
    print(f"Loading data from {input_path}...")
    with open(input_path, 'rb') as f:
        old_data = pickle.load(f)

    print(f"Original data type: {type(old_data)}")
    print(f"Number of dates: {len(old_data)}")

    # 检查第一个记录
    first_date = list(old_data.keys())[0]
    first_record = old_data[first_date]
    print(f"First date: {first_date} (type: {type(first_date)})")
    print(f"First record type: {type(first_record)}")

    new_data = {}

    if isinstance(first_record, tuple):
        # 处理 tuple 格式: (price_dict, news_dict, filing_q_dict, filing_k_dict)
        print("Converting from tuple format...")
        for cur_date, record in old_data.items():
            price_item, news_item, filing_q_item, filing_k_item = record

            # 提取正确的数据
            if isinstance(price_item, dict) and 'price' in price_item:
                price_data = price_item['price']
            else:
                price_data = price_item if isinstance(price_item, dict) else {}

            if isinstance(news_item, dict) and 'news' in news_item:
                news_data = news_item['news']
            else:
                news_data = news_item if isinstance(news_item, dict) else {}

            if isinstance(filing_q_item, dict) and 'filing_q' in filing_q_item:
                filing_q_data = filing_q_item['filing_q']
            else:
                filing_q_data = filing_q_item if isinstance(filing_q_item, dict) else {}

            if isinstance(filing_k_item, dict) and 'filing_k' in filing_k_item:
                filing_k_data = filing_k_item['filing_k']
            else:
                filing_k_data = filing_k_item if isinstance(filing_k_item, dict) else {}

            new_data[cur_date] = {
                'price': price_data,
                'news': news_data,
                'filing_q': filing_q_data,
                'filing_k': filing_k_data
            }

    elif isinstance(first_record, dict):
        # 检查是否已经是正确格式
        required_keys = {'price', 'news', 'filing_q', 'filing_k'}
        if required_keys.issubset(first_record.keys()):
            print("Data is already in correct format!")
            new_data = old_data
        else:
            print("Converting from dict format...")
            # 尝试其他转换逻辑
            for cur_date, record in old_data.items():
                new_data[cur_date] = {
                    'price': record.get('price', {}),
                    'news': record.get('news', {}),
                    'filing_q': record.get('filing_q', {}),
                    'filing_k': record.get('filing_k', {})
                }
    else:
        raise ValueError(f"Unknown data format: {type(first_record)}")

    # 填充缺失的ticker数据
    if tickers:
        print(f"Filling data for tickers: {tickers}")
        for cur_date in new_data:
            for ticker in tickers:
                if ticker not in new_data[cur_date]['news']:
                    new_data[cur_date]['news'][ticker] = []
                if ticker not in new_data[cur_date]['price']:
                    new_data[cur_date]['price'][ticker] = None

    # 验证转换结果
    print("\nValidating converted data...")
    from pydantic import BaseModel
    from typing import Dict, List

    class OneDateRecord(BaseModel):
        price: Dict[str, float]
        filing_k: Dict[str, str]
        filing_q: Dict[str, str]
        news: Dict[str, List[str]]

    # 检查几个样本
    for i, (cur_date, record) in enumerate(new_data.items()):
        if i < 3:
            try:
                OneDateRecord.model_validate(record)
                print(f"✓ {cur_date}: Valid")
            except Exception as e:
                print(f"✗ {cur_date}: Invalid - {e}")
                print(f"  Record: {record}")

    # 保存转换后的数据
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(new_data, f)

    print(f"Done! Converted {len(new_data)} records.")


if __name__ == '__main__':
    # 使用示例
    convert_env_data(
        input_path='/Users/liuhanyu.666/FinMem/data/03_model_input/env_data.pkl',
        output_path='/Users/liuhanyu.666/FinMem/data/03_model_input/env_data_fixed.pkl',
        tickers=['TSLA']  # 你的股票代码
    )
