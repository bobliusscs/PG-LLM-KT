#!/usr/bin/env python3
"""
数据集拆分脚本

功能：
1. 将 dataSet/processed 文件夹下所有数据集的 user_group_sequences.txt 文件拆分为训练集、验证集和测试集
2. 按照 70%:10%:20% 的比例进行拆分
3. 对于 junyi 数据集，先随机抽取 2500 个学生，然后进行拆分
4. 保存到 dataSet/data_split 目录下对应的数据集文件夹中
5. 对超过200的序列进行截断

用法：
    python split_dataset.py
    
可选参数：
    --train_ratio: 训练集比例，默认 0.70
    --val_ratio: 验证集比例，默认 0.10
    --test_ratio: 测试集比例，默认 0.20
    --sample_size: 数据集抽样大小，默认 2500
    --max_seq_length: 最大序列长度，默认 200
"""

import os
import random
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional


def convert_float_to_int(correct_sequence: str) -> str:
    """
    将正确性序列中的浮点数转换为整数
    
    Args:
        correct_sequence: 正确性序列字符串，格式如 "1.0,0.0,1.0"
        
    Returns:
        转换后的整数序列字符串，格式如 "1,0,1"
    """
    if not correct_sequence:
        return correct_sequence
    
    try:
        # 分割序列
        values = correct_sequence.split(',')
        # 转换每个值
        converted_values = []
        for value in values:
            value = value.strip()
            if value in ['1.0', '1']:
                converted_values.append('1')
            elif value in ['0.0', '0']:
                converted_values.append('0')
            else:
                # 如果是其他浮点数，尝试转换
                try:
                    float_val = float(value)
                    int_val = int(float_val)
                    converted_values.append(str(int_val))
                except ValueError:
                    # 如果转换失败，保持原值
                    converted_values.append(value)
        
        return ','.join(converted_values)
    except Exception:
        # 如果处理失败，返回原值
        return correct_sequence


def truncate_sequences(sequences: List[Tuple[str, str, str, str]], max_length: int) -> List[Tuple[str, str, str, str]]:
    """
    截断超过最大长度的序列，保留序列的开头部分
    
    Args:
        sequences: 用户序列列表
        max_length: 最大序列长度
        
    Returns:
        截断后的用户序列列表
    """
    truncated_sequences = []
    
    for user_id, problem_seq, skill_seq, correct_seq in sequences:
        # 分割序列
        problem_list = problem_seq.split(',')
        skill_list = skill_seq.split(';')
        correct_list = correct_seq.split(',')
        
        # 如果序列长度超过最大长度，则截断
        if len(problem_list) > max_length:
            problem_list = problem_list[:max_length]
            skill_list = skill_list[:max_length]
            correct_list = correct_list[:max_length]
            
            # 重新组合序列
            problem_seq = ','.join(problem_list)
            skill_seq = ';'.join(skill_list)
            correct_seq = ','.join(correct_list)
        
        truncated_sequences.append((user_id, problem_seq, skill_seq, correct_seq))
    
    return truncated_sequences


def read_user_sequences(file_path: str, dataset_name: Optional[str] = None) -> List[Tuple[str, str, str, str]]:
    """
    读取用户序列文件，返回4元组列表
    
    Args:
        file_path: 文件路径
        dataset_name: 数据集名称，用于特殊处理
        
    Returns:
        List of tuples: (user_id, problem_sequence, skill_sequence, correct_sequence)
    """
    sequences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 每4行为一个用户的数据
    for i in range(0, len(lines), 4):
        if i + 3 < len(lines):
            user_id = lines[i].strip()
            problem_seq = lines[i + 1].strip()
            skill_seq = lines[i + 2].strip()
            correct_seq = lines[i + 3].strip()
            
            # 对于 assistments2012 数据集，将浮点数转换为整数
            if dataset_name == 'assistments2012':
                correct_seq = convert_float_to_int(correct_seq)
            
            sequences.append((user_id, problem_seq, skill_seq, correct_seq))
    
    return sequences


def write_user_sequences(sequences: List[Tuple[str, str, str, str]], file_path: str):
    """
    写入用户序列到文件
    
    Args:
        sequences: 用户序列列表
        file_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for user_id, problem_seq, skill_seq, correct_seq in sequences:
            f.write(f"{user_id}\n")
            f.write(f"{problem_seq}\n")
            f.write(f"{skill_seq}\n")
            f.write(f"{correct_seq}\n")


def split_dataset(dataset_name: str, source_dir: str, target_dir: str, 
                 train_ratio: float = 0.75, val_ratio: float = 0.05, 
                 test_ratio: float = 0.20, sample_size: Optional[int] = None,
                 max_seq_length: int = 250):
    """
    拆分数据集
    
    Args:
        dataset_name: 数据集名称
        source_dir: 源数据目录
        target_dir: 目标目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        sample_size: 采样大小（仅对特定数据集有效）
        max_seq_length: 最大序列长度
    """
    input_file = os.path.join(source_dir, dataset_name, 'user_group_sequences.txt')
    
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return
    
    print(f"处理数据集: {dataset_name}")
    
    # 读取所有用户序列
    sequences = read_user_sequences(input_file, dataset_name)
    total_users = len(sequences)
    print(f"  原始用户数: {total_users}")
    
    # 如果指定了采样大小且数据集用户数超过采样大小，则进行随机采样
    if sample_size and total_users > sample_size:
        print(f"  随机抽取 {sample_size} 个用户")
        sequences = random.sample(sequences, sample_size)
    
    # 截断过长的序列
    if max_seq_length > 0:
        sequences = truncate_sequences(sequences, max_seq_length)
        print(f"  序列已截断至最大长度: {max_seq_length}")
    
    # 随机打乱序列
    random.shuffle(sequences)
    
    # 计算各集合大小
    total_size = len(sequences)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # 确保总数不变
    
    # 拆分训练集、验证集和测试集
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    test_sequences = sequences[train_size + val_size:]
    
    print(f"  训练集用户数: {len(train_sequences)}")
    print(f"  验证集用户数: {len(val_sequences)}")
    print(f"  测试集用户数: {len(test_sequences)}")
    
    # 创建输出目录
    output_dir = os.path.join(target_dir, dataset_name)
    
    # 保存训练集
    train_file = os.path.join(output_dir, 'train_sequences.txt')
    write_user_sequences(train_sequences, train_file)
    print(f"  训练集保存到: {train_file}")
    
    # 保存验证集
    val_file = os.path.join(output_dir, 'val_sequences.txt')
    write_user_sequences(val_sequences, val_file)
    print(f"  验证集保存到: {val_file}")
    
    # 保存测试集
    test_file = os.path.join(output_dir, 'test_sequences.txt')
    write_user_sequences(test_sequences, test_file)
    print(f"  测试集保存到: {test_file}")
    
    # 保存拆分统计信息
    stats = {
        'dataset_name': dataset_name,
        'original_users': total_users,
        'sampled_users': len(sequences),
        'train_users': len(train_sequences),
        'val_users': len(val_sequences),
        'test_users': len(test_sequences),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'sample_size': sample_size,
        'max_seq_length': max_seq_length
    }
    
    stats_file = os.path.join(output_dir, 'split_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  统计信息保存到: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='数据集拆分脚本')
    parser.add_argument('--train_ratio', type=float, default=0.70, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.10, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.20, help='测试集比例')
    parser.add_argument('--sample_size', type=int, default=3000, help='数据集抽样大小')
    parser.add_argument('--max_seq_length', type=int, default=250, help='最大序列长度')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(42)
    
    # 设置路径 - 修正路径计算方式
    # 获取当前脚本的绝对路径，然后向上三级找到项目根目录
    script_dir = Path(__file__).resolve().parent
    process_data_dir = script_dir.parent
    src_dir = process_data_dir.parent
    workspace_root = src_dir.parent
    
    # 正确设置source_dir和target_dir路径
    source_dir = workspace_root / 'dataSet' / 'processed'
    target_dir = workspace_root / 'dataSet' / 'data_split'
    
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"验证集比例: {args.val_ratio}")
    print(f"测试集比例: {args.test_ratio}")
    print(f"数据集抽样大小: {args.sample_size}")
    print(f"最大序列长度: {args.max_seq_length}")
    print()
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 查找所有包含 user_group_sequences.txt 的数据集
    datasets = []
    for item in os.listdir(source_dir):
        dataset_path = os.path.join(source_dir, item)
        if os.path.isdir(dataset_path):
            sequence_file = os.path.join(dataset_path, 'user_group_sequences.txt')
            if os.path.exists(sequence_file):
                datasets.append(item)
    
    print(f"发现数据集: {datasets}")
    print()
    
    # 处理每个数据集
    for dataset_name in sorted(datasets):
        sample_size = None
        if dataset_name == 'junyi' or dataset_name == 'ednet_kt1' or dataset_name == 'assistments2012' or dataset_name == 'assistments2009':
            sample_size = args.sample_size
        
        split_dataset(
            dataset_name=dataset_name,
            source_dir=str(source_dir),
            target_dir=str(target_dir),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            sample_size=sample_size,
            max_seq_length=args.max_seq_length
        )
        print()
    
    print("所有数据集处理完成！")


if __name__ == '__main__':
    main()