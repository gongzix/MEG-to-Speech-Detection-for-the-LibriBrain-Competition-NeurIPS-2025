import csv
from collections import defaultdict
from config import *
def read_csv_files(file_paths):
    """读取CSV文件路径列表中的所有文件并返回数据"""
    if not file_paths:
        raise ValueError("No CSV files provided in the list")
    
    all_data = []
    for file_path in file_paths:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]
            all_data.append(data)
    return all_data

def voting_mechanism(all_data, method='average'):
    """
    对多个CSV文件的数据进行投票
    支持的方法:
    - 'average': 平均值
    - 'max': 最大值
    - 'min': 最小值
    - 'majority': 多数表决(四舍五入)
    """
    segment_dict = defaultdict(list)
    
    for data in all_data:
        for row in data:
            segment_idx = row['segment_idx']
            speech_prob = float(row['speech_prob'])
            segment_dict[segment_idx].append(speech_prob)
    
    voted_data = []
    for segment_idx in sorted(segment_dict.keys(), key=lambda x: int(x)):
        probs = segment_dict[segment_idx]
        
        if method == 'average':
            result = sum(probs) / len(probs)
        elif method == 'max':
            result = max(probs)
        elif method == 'min':
            result = min(probs)
        elif method == 'majority':
            # 将概率转换为0或1后进行多数表决
            votes = [round(p) for p in probs]
            result = round(sum(votes) / len(votes))
        else:
            raise ValueError(f"Unknown voting method: {method}")
        
        voted_data.append({
            'segment_idx': segment_idx,
            'speech_prob': result
        })
    
    return voted_data

def write_output_csv(data, output_path):
    """将投票结果写入指定的输出路径"""
    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['segment_idx', 'speech_prob'])
        writer.writeheader()
        writer.writerows(data)

def get_voting_method():
    """获取用户选择的投票方法"""
    print("\nAvailable voting methods:")
    print("1. Average (default)")
    print("2. Maximum")
    print("3. Minimum")
    print("4. Majority vote")
    
    choice = input("Enter your choice (1-4, default 1): ").strip()
    
    methods = {
        '1': 'average',
        '2': 'max',
        '3': 'min',
        '4': 'majority'
    }
    
    return methods.get(choice, 'average')

def main():
    print("CSV Voting Tool")
    print("=" * 40)
    
    # 获取CSV文件路径列表
    file_paths = [
                    "./best_vote_files/tcn_s30_375.csv",
                    "./best_vote_files/tcn_s30_500.csv",
                    "./best_vote_files/tcn_s30_1000.csv", #0.901
                    "./best_vote_files/tcn_s150_1000.csv",
                    "./best_vote_files/trans_s150_1000.csv", #0.892

                    "./best_vote_files/tcn-L-step=10W_s1_1000.csv", #0.907

                    "./best_vote_files/tcn-L-20-s30_s30_1000_threshold=0.58.csv", #0.9118
                    "./best_vote_files/newtcn_s1_1000_threshold=0.6.csv", #0.9100
                    "./best_vote_files/SEANetBackbone_s30_1000_07301340_threshold=0.58.csv", #0.9000
                    "./best_vote_files/crisscross-w8-epoch=35-val_loss=0.34.csv", #0.894

                    ]
    
    
    # 获取投票方法
    method = get_voting_method()
    
    import os
    from datetime import datetime

    # 获取当前日期和时间，格式为 YYYYMMDD_HHMM
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M")

    # 在文件名中加入日期和时间
    file_path = os.path.join(
        RESULTS_DIR, 
        f"voted_holdout_speech_predictions_{current_datetime}.csv"
    )    
    output_path = file_path


    # 读取所有CSV文件
    print("\nReading CSV files...")
    all_data = read_csv_files(file_paths)
    print(f"Successfully read {len(file_paths)} CSV files")
    
    # 进行投票
    print(f"Applying {method} voting mechanism...")
    voted_data = voting_mechanism(all_data, method)
    voted_data = voting_mechanism(all_data, method)  # 假设返回的是字典列表

    for item in voted_data:
        item['speech_prob'] = 1 if item['speech_prob'] >= 0.5 else 0
    # 写入输出文件
    print("Writing output file...")
    write_output_csv(voted_data, output_path)
    print(f"Successfully wrote voted results to {output_path}")


if __name__ == "__main__":
    main()