#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练脚本 - 支持指定数据源进行训练
"""

import os
import sys
import argparse
import json
import time
import random
from typing import Optional, Tuple

# 添加项目根目录下的src路径到系统路径
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.dirname(os.path.dirname(script_dir))  # 直接指向src目录
project_root = os.path.dirname(src_path)  # 项目根目录
sys.path.append(src_path)

from config_generator import generate_single_dataset_config, generate_merged_config


class TrainingConfig:
    """训练配置类，用于统一管理训练参数"""
    
    def __init__(self, **kwargs):
        # 基础参数
        self.dataset_name = kwargs.get('dataset_name')
        self.data_source = kwargs.get('data_source', 'data_text_graph')
        self.prediction_head = kwargs.get('prediction_head', 'adaptive')
        self.loss_function = kwargs.get('loss_function', 'focal')
        self.pos_weight = kwargs.get('pos_weight', 1.0)
        self.train_only_head = kwargs.get('train_only_head', False)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.train_mode = kwargs.get('train_mode', 'lora')
        self.head_learning_rate = kwargs.get('head_learning_rate', None)  # 添加分类头学习率参数
        
        # 数据处理参数
        self.no_undersample = kwargs.get('no_undersample', False)
        self.undersample_ratio = kwargs.get('undersample_ratio', None)  # 默认为None，表示使用数据集特定配置
        
        # 配置对象
        self.config = None
    
    def generate_config(self, is_merged=False):
        """生成配置对象"""
        if is_merged:
            self.config = generate_merged_config(
                data_source=self.data_source,
                prediction_head=self.prediction_head,
                loss_function=self.loss_function,
                pos_weight=self.pos_weight
            )
        else:
            self.config = generate_single_dataset_config(
                dataset_name=self.dataset_name,
                data_source=self.data_source,
                prediction_head=self.prediction_head,
                loss_function=self.loss_function,
                pos_weight=self.pos_weight
            )
        
        # 更新配置中的数据处理参数
        if self.config and "data_processing" in self.config:
            # 只有当命令行明确指定禁用欠采样时才覆盖配置文件中的设置
            if self.no_undersample:
                self.config["data_processing"]["undersample"] = False
            # 只有当命令行参数指定了undersample_ratio时才覆盖配置中的值
            if self.undersample_ratio is not None:
                self.config["data_processing"]["undersample_ratio"] = self.undersample_ratio
        
        return self.config


def merge_all_datasets(data_source='data_text_graph'):
    """合并所有数据集的训练集和验证集"""
    print(f"🔍 开始合并所有数据集（数据源: {data_source}）...")
    
    # 定义所有数据集
    datasets = [
        'assistments2009',
        'assistments2012', 
        'ednet_kt1',
        'hnu_sys2023',
        'junyi',
        'kdd2010'
    ]
    
    # 根据数据源决定文件名和合并后的文件名
    if data_source == 'data_text_graph':
        train_file_name = "train.json"  # 使用 train_select.json
        merged_train_file = os.path.join(project_root, "dataSet", data_source, "merged_train_select.json")
        filter_correct = False  # 只使用 is_correct=true 的样本
    else:
        train_file_name = "train.json"  # 使用传统的 train.json
        merged_train_file = os.path.join(project_root, "dataSet", data_source, "merged_train.json")
        filter_correct = False
        
    merged_val_file = os.path.join(project_root, "dataSet", data_source, "merged_val.json")
    
    # 合并训练集
    merged_train_data = []
    total_samples = 0
    filtered_samples = 0
    
    for dataset in datasets:
        train_file = os.path.join(project_root, "dataSet", data_source, dataset, train_file_name)
        if os.path.exists(train_file):
            print(f"  合并 {dataset} 训练集...")
            try:
                with open(train_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_samples += len(data)
                    
                    if filter_correct:
                        # 只保留 is_correct=true 的样本
                        filtered_data = [item for item in data if item.get('is_correct') == True]
                        filtered_samples += len(filtered_data)
                        merged_train_data.extend(filtered_data)
                        print(f"    原始数据: {len(data)} 条，筛选后: {len(filtered_data)} 条 (is_correct=true)")
                    else:
                        # 使用所有数据
                        merged_train_data.extend(data)
                        print(f"    添加了 {len(data)} 条记录")
            except Exception as e:
                print(f"    跳过 {dataset} 训练集 (错误: {e})")
        else:
            print(f"    跳过 {dataset} 训练集 (文件不存在)")
    
    if filter_correct:
        print(f"  筛选统计: 总样本 {total_samples} 条 → 正确预测样本 {filtered_samples} 条 (筛选率: {filtered_samples/total_samples*100:.1f}%)")
    
    # 打乱合并后的训练数据
    random.shuffle(merged_train_data)
    print(f"  训练数据打乱完成，共 {len(merged_train_data)} 条记录")
    
    # 保存合并后的训练集
    os.makedirs(os.path.dirname(merged_train_file), exist_ok=True)
    try:
        with open(merged_train_file, 'w', encoding='utf-8') as f:
            json.dump(merged_train_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 合并训练集保存完成，共 {len(merged_train_data)} 条记录")
    except Exception as e:
        print(f"❌ 保存合并训练集失败: {e}")
        return False, None, None
    
    # 合并验证集（直接使用完整验证集）
    merged_val_data = []
    for dataset in datasets:
        val_file = os.path.join(project_root, "dataSet", data_source, dataset, "val.json")
        if os.path.exists(val_file):
            print(f"  处理 {dataset} 验证集...")
            try:
                # 直接使用完整验证集数据
                with open(val_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                merged_val_data.extend(data)
                print(f"    添加了 {len(data)} 条记录")
            except Exception as e:
                print(f"    跳过 {dataset} 验证集 (错误: {e})")
        else:
            print(f"    跳过 {dataset} 验证集 (文件不存在)")
    
    # 打乱合并后的验证数据
    random.shuffle(merged_val_data)
    print(f"  验证数据打乱完成，共 {len(merged_val_data)} 条记录")
    
    # 保存合并后的验证集
    try:
        with open(merged_val_file, 'w', encoding='utf-8') as f:
            json.dump(merged_val_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 合并验证集保存完成，共 {len(merged_val_data)} 条记录")
    except Exception as e:
        print(f"❌ 保存合并验证集失败: {e}")
        return False, merged_train_file, None
    
    return True, merged_train_file, merged_val_file


def analyze_data_balance(data_file_path: str, description="数据"):
    """分析数据的类别平衡情况"""
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计不同标签的数量
        label_counts = {}
        total_count = len(data)
        
        for item in data:
            output = item.get('output', '').strip().lower()
            # 标准化标签
            if output in ['正确', 'right', '1']:
                label = 'positive'
            elif output in ['错误', 'wrong', '0']:
                label = 'negative'
            else:
                label = 'unknown'
            
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"📊 {description}类别分析:")
        print(f"   总样本数: {total_count}")
        
        for label, count in label_counts.items():
            ratio = count / total_count * 100 if total_count > 0 else 0
            print(f"   {label}: {count} ({ratio:.2f}%)")
        
        # 计算不平衡比例
        pos_count = label_counts.get('positive', 0)
        neg_count = label_counts.get('negative', 0)
        
        if pos_count > 0 and neg_count > 0:
            imbalance_ratio = max(pos_count, neg_count) / min(pos_count, neg_count)
            print(f"   类别不平衡比例: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 10:
                print(f"   ⚠️ 严重类别不平衡! 建议调整Focal Loss参数")
                print(f"   💡 建议: alpha=0.1-0.3, gamma=3-5")
            elif imbalance_ratio > 3:
                print(f"   ⚠️ 中等类别不平衡，Focal Loss应该有效")
                print(f"   💡 建议: alpha=0.2-0.5, gamma=2-3")
            else:
                print(f"   ✅ 类别相对平衡")
        
        return label_counts
        
    except Exception as e:
        print(f"   ❌ 数据分析失败: {e}")
        return {}


def calculate_pos_weight(label_counts):
    """根据标签计数计算正样本权重"""
    pos_count = label_counts.get('positive', 0)
    neg_count = label_counts.get('negative', 0)
    
    # 如果任一类样本数为0，返回默认权重1.0
    if pos_count == 0 or neg_count == 0:
        return 1.0
    
    # 计算正样本权重: 负样本数 / 正样本数
    pos_weight = neg_count / pos_count
    return pos_weight


def preprocess_train_data(train_file_path: str, dataset_name: str, data_source: str) -> Optional[str]:
    """预处理训练数据，筛选 is_correct=true 的样本"""
    print(f"  正在筛选 {dataset_name} 中 is_correct=true 的样本...")
    
    try:
        with open(train_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 筛选 is_correct=true 的样本
        filtered_data = [item for item in data if item.get('is_correct') == True]
        
        print(f"    原始数据: {len(data)} 条")
        print(f"    筛选后: {len(filtered_data)} 条 (is_correct=true)")
        print(f"    筛选率: {len(filtered_data)/len(data)*100:.1f}%")
        
        if len(filtered_data) == 0:
            print(f"    ⚠️ 警告: 没有找到 is_correct=true 的样本")
            return None
        
        # 保存筛选后的数据到临时文件
        temp_file = train_file_path.replace('.json', '_filtered.json')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"    ✅ 筛选后的数据保存到: {temp_file}")
        return temp_file
        
    except Exception as e:
        print(f"    ❌ 数据筛选失败: {e}")
        return None


def prepare_training_data(config, dataset_name: str, data_source: str) -> Tuple[Optional[str], Optional[str]]:
    """准备训练数据，包括预处理、欠采样等"""
    # 检查数据文件是否存在
    train_file_path = os.path.join(project_root, config["data"]["train_file"])
    val_file_path = os.path.join(project_root, config["data"]["val_file"])
    
    if not os.path.exists(train_file_path):
        print(f"❌ 训练文件不存在: {train_file_path}")
        return None, None
    
    if not os.path.exists(val_file_path):
        print(f"❌ 验证文件不存在: {val_file_path}")
        return None, None
    
    # 如果需要筛选，先预处理数据
    processed_train_file = train_file_path
    if config["data"].get("filter_correct_only", False):
        print(f"🔍 检测到 data_text_graph 数据源，将筛选 is_correct=true 的样本...")
        processed_train_file = preprocess_train_data(train_file_path, dataset_name, data_source)
        if not processed_train_file:
            return None, None
    
    # 分析数据平衡性
    print(f"📊 分析数据平衡性...")
    label_counts = analyze_data_balance(processed_train_file, f"{dataset_name} 训练")
    analyze_data_balance(val_file_path, f"{dataset_name} 验证")
    
    # 如果使用BCE损失函数，自动计算正样本权重
    if config.get("loss", {}).get("function") == "bce":
        pos_weight = calculate_pos_weight(label_counts)
        config["loss"]["pos_weight"] = pos_weight
        print(f"⚖️ 自动计算的正样本权重: {pos_weight:.2f}")
    
    # 检查是否需要进行欠采样
    undersample_enabled = config.get("data_processing", {}).get("undersample", False)
    undersample_ratio = config.get("data_processing", {}).get("undersample_ratio", "3:1")
    if undersample_enabled and should_undersample(label_counts):
        print(f"⚠️ 检测到类别不平衡，将对训练数据进行欠采样...")
        processed_train_file = undersample_data(processed_train_file, dataset_name, undersample_ratio)
        if not processed_train_file:
            print(f"❌ 欠采样失败，使用原始数据继续训练")
        else:
            print(f"✅ 欠采样完成，使用平衡数据进行训练")
            # 重新分析欠采样后的数据平衡性，并更新BCE损失函数的权重
            undersampled_label_counts = analyze_data_balance(processed_train_file, f"{dataset_name} 训练(欠采样后)")
            # 如果使用BCE损失函数，重新计算正样本权重（基于欠采样后的数据）
            if config.get("loss", {}).get("function") == "bce":
                pos_weight = calculate_pos_weight(undersampled_label_counts)
                config["loss"]["pos_weight"] = pos_weight
                print(f"⚖️ 基于欠采样后数据重新计算的正样本权重: {pos_weight:.2f}")
    
    # 直接使用完整验证集
    processed_val_file = val_file_path
    
    return processed_train_file, processed_val_file


def build_training_args(config, train_file: str, val_file: str, prediction_head: str, max_grad_norm: float, train_mode: str, train_only_head: bool, head_learning_rate: Optional[float] = None, use_precomputed_embeddings: bool = False):
    """构建训练参数"""
    # 确保模型路径是绝对路径
    model_path = config["model"]["model_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    
    # 如果head_learning_rate为None，设置为默认值（基础学习率的10倍）
    if head_learning_rate is None:
        head_learning_rate = config["training"]["learning_rate"] * 10
    
    # 构造命令行参数
    cmd_args = [
        "--model_path", model_path,
        "--data_path", train_file,
        "--val_path", val_file,
        "--output_dir", os.path.join(project_root, config["paths"]["output_dir"]),
        "--max_seq_length", str(config["model"]["max_seq_length"]),
        "--batch_size", str(config["training"]["batch_size"]),
        "--gradient_accumulation_steps", str(config["training"]["gradient_accumulation_steps"]),
        "--learning_rate", str(config["training"]["learning_rate"]),
        "--num_epochs", str(config["training"]["epochs"]),
        "--warmup_ratio", str(config["training"]["warmup_ratio"]),
        "--weight_decay", str(config["training"]["weight_decay"]),
        "--save_steps", str(config["training"]["save_steps"]),
        "--eval_steps", str(config["training"]["eval_steps"]),
        "--logging_steps", str(config["training"]["logging_steps"]),
        "--eval_split_ratio", "0.03",
        "--early_stopping_patience", str(config["training"]["early_stopping_patience"]),
        "--seed", "3407",
        "--prediction_head", prediction_head,
        "--max_grad_norm", str(max_grad_norm),
        "--train_mode", train_mode
    ]
    
    # 添加分类头学习率参数（如果指定且不等于基础学习率）
    if head_learning_rate is not None and head_learning_rate != config["training"]["learning_rate"]:
        cmd_args.extend(["--head_learning_rate", str(head_learning_rate)])
    
    # 只有在不是只训练分类头且使用LoRA模式时才添加LoRA参数
    if not train_only_head and train_mode == "lora":
        cmd_args.extend([
            "--lora_r", str(config["model"]["lora_r"]),
            "--lora_alpha", str(config["model"]["lora_alpha"]),
            "--lora_dropout", str(config["model"]["lora_dropout"]),
        ])
    
    # 添加只训练分类头的参数
    if train_only_head:
        cmd_args.append("--train_only_head")
    
    # 添加预计算嵌入参数（仅在冻结模式下有效）
    if use_precomputed_embeddings and train_mode == "freeze":
        cmd_args.append("--use_precomputed_embeddings")
    
    # 添加损失函数参数
    if "loss" in config:
        cmd_args.extend([
            "--loss_function", config["loss"]["function"],
        ])
        # 根据损失函数类型添加相应的参数
        if config["loss"]["function"] == "focal":
            cmd_args.extend([
                "--focal_alpha", str(config["loss"]["alpha"]),
                "--focal_gamma", str(config["loss"]["gamma"])
            ])
        elif config["loss"]["function"] == "weighted_bce" and "pos_weight" in config["loss"]:
            cmd_args.extend(["--pos_weight", str(config["loss"]["pos_weight"])])
        elif config["loss"]["function"] == "bce" and "pos_weight" in config["loss"]:
            cmd_args.extend(["--pos_weight", str(config["loss"]["pos_weight"])])
    
    # 添加欠采样参数（在预处理阶段已经处理了欠采样，所以这里禁用训练脚本中的欠采样）
    cmd_args.extend(["--no_undersample"])
    
    return cmd_args


def run_training(cmd_args, dataset_name: str) -> bool:
    """执行训练"""
    # 保存原始参数
    original_argv = sys.argv.copy()
    
    try:
        # 设置新的命令行参数
        sys.argv = ["loraAndPredictor.py"] + cmd_args
        
        # 导入并调用训练函数
        from model.loraAndPredictor import main as lora_main
        lora_main()
        
        print(f"✅ {dataset_name} 训练完成")
        return True
        
    except Exception as e:
        print(f"❌ {dataset_name} 训练失败: {e}")
        return False
    finally:
        # 恢复原始参数
        sys.argv = original_argv


def cleanup_temp_files(train_file: str, val_file: str, original_train_file: str, original_val_file: str):
    """清理临时文件"""
    temp_files = []
    if train_file != original_train_file and os.path.exists(train_file):
        temp_files.append(train_file)
    if val_file != original_val_file and os.path.exists(val_file):
        temp_files.append(val_file)
    
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"  🧹 已清理临时文件: {temp_file}")
        except:
            pass


def run_single_dataset_training(training_config: TrainingConfig) -> bool:
    """运行单个数据集训练"""
    dataset_name = training_config.dataset_name
    data_source = training_config.data_source
    prediction_head = training_config.prediction_head
    loss_function = training_config.loss_function
    pos_weight = training_config.pos_weight
    train_only_head = training_config.train_only_head
    max_grad_norm = training_config.max_grad_norm
    train_mode = training_config.train_mode
    
    # 确保 dataset_name 不为 None
    if not dataset_name:
        print("❌ 数据集名称不能为空")
        return False
    
    print(f"\n{'='*60}")
    print(f"开始训练数据集: {dataset_name}")
    print(f"数据源: {data_source}")
    print(f"{'='*60}")
    
    # 生成配置
    config = training_config.generate_config(is_merged=False)
    if not config:
        print("❌ 配置生成失败")
        return False
    
    # 准备训练数据
    processed_train_file, processed_val_file = prepare_training_data(config, dataset_name, data_source)
    if not processed_train_file or not processed_val_file:
        return False
    
    print(f"✅ 训练文件: {config['data']['train_file']}")
    print(f"✅ 验证文件: {config['data']['val_file']}")
    print(f"📁 输出目录: {config['paths']['output_dir']}")
    
    # 获取分类头学习率（如果指定）
    head_learning_rate = getattr(training_config, 'head_learning_rate', None)
    
    # 构建训练参数（在freeze模式下自动启用预计算嵌入）
    use_precomputed = (train_mode == "freeze")
    if use_precomputed:
        print(f"🚀 将使用预计算嵌入加速训练（freeze模式自动启用）")
    cmd_args = build_training_args(
        config, processed_train_file, processed_val_file,
        prediction_head, max_grad_norm, train_mode, train_only_head, head_learning_rate,
        use_precomputed_embeddings=use_precomputed
    )
    
    # 添加调试信息
    if "loss" in config:
        print(f"🔥 损失函数参数:")
        print(f"   损失函数: {config['loss']['function']}")
        if config["loss"]["function"] == "focal":
            print(f"   Alpha: {config['loss']['alpha']}")
            print(f"   Gamma: {config['loss']['gamma']}")
        elif config["loss"]["function"] == "weighted_bce" and "pos_weight" in config["loss"]:
            print(f"   正样本权重: {config['loss']['pos_weight']}")
        elif config["loss"]["function"] == "bce" and "pos_weight" in config["loss"]:
            print(f"   自动计算的正样本权重: {config['loss']['pos_weight']}")
    
    # 不添加欠采样参数，因为我们已经在预处理阶段完成了欠采样
    undersample_enabled = config.get("data_processing", {}).get("undersample", True)
    if undersample_enabled:
        print(f"📈 欠采样: 已在预处理阶段完成")
    else:
        print(f"📈 欠采样: 禁用")
    
    # 确保梯度裁剪值在0.1到5之间
    print(f"⚖️ 梯度裁剪: {max(0.1, min(5.0, max_grad_norm))} (限制在0.1-5.0范围内)")
    
    # 执行训练
    success = run_training(cmd_args, dataset_name)
    
    # 清理临时文件
    original_train_file = os.path.join(project_root, config["data"]["train_file"])
    original_val_file = os.path.join(project_root, config["data"]["val_file"])
    cleanup_temp_files(processed_train_file, processed_val_file, original_train_file, original_val_file)
    
    return success


def run_merged_training(training_config: TrainingConfig) -> bool:
    """运行合并数据集训练"""
    data_source = training_config.data_source
    prediction_head = training_config.prediction_head
    loss_function = training_config.loss_function
    pos_weight = training_config.pos_weight
    train_only_head = training_config.train_only_head
    max_grad_norm = training_config.max_grad_norm
    train_mode = training_config.train_mode
    
    print(f"\n{'='*60}")
    print(f"开始合并数据集训练")
    print(f"数据源: {data_source}")
    print(f"{'='*60}")
    
    # 先合并数据集
    success, merged_train_file, merged_val_file = merge_all_datasets(data_source)
    if not success or not merged_train_file or not merged_val_file:
        print("❌ 数据集合并失败，无法继续训练")
        return False
    
    # 生成配置
    config = training_config.generate_config(is_merged=True)
    if not config:
        print("❌ 配置生成失败")
        return False
    
    # 分析合并后的数据平衡性
    print(f"📊 分析合并后数据平衡性...")
    label_counts = analyze_data_balance(merged_train_file, "合并训练数据")
    analyze_data_balance(merged_val_file, "合并验证数据")
    
    # 如果使用BCE损失函数，使用用户指定的正样本权重
    if config.get("loss", {}).get("function") == "bce":
        config["loss"]["pos_weight"] = pos_weight
        print(f"⚖️ 使用用户指定的正样本权重: {pos_weight:.2f}")
    
    # 检查是否需要进行欠采样
    undersample_enabled = config.get("data_processing", {}).get("undersample", True)
    undersample_ratio = config.get("data_processing", {}).get("undersample_ratio", "3:1")
    if undersample_enabled and should_undersample(label_counts):
        print(f"⚠️ 检测到类别不平衡，将对合并后的训练数据进行欠采样...")
        undersampled_train_file = undersample_data(merged_train_file, "merged", undersample_ratio)
        if undersampled_train_file:
            merged_train_file = undersampled_train_file
            print(f"✅ 欠采样完成，使用平衡数据进行训练")
            # 重新分析欠采样后的数据平衡性，并更新BCE损失函数的权重
            undersampled_label_counts = analyze_data_balance(merged_train_file, "合并训练数据(欠采样后)")
            # 如果使用BCE损失函数，重新计算正样本权重（基于欠采样后的数据）
            if config.get("loss", {}).get("function") == "bce":
                pos_weight = calculate_pos_weight(undersampled_label_counts)
                config["loss"]["pos_weight"] = pos_weight
                print(f"⚖️ 基于欠采样后数据重新计算的正样本权重: {pos_weight:.2f}")
        else:
            print(f"❌ 欠采样失败，使用原始数据继续训练")
    
    # 直接使用完整验证集
    processed_val_file = merged_val_file
    
    print(f"✅ 训练文件: {config['data']['train_file']}")
    print(f"✅ 验证文件: {config['data']['val_file']}")
    print(f"📁 输出目录: {config['paths']['output_dir']}")
    
    # 构建训练参数（启用预计算嵌入功能，仅在freeze模式下有效）
    use_precomputed = (train_mode == "freeze" and train_only_head)
    if use_precomputed:
        print(f"🚀 将使用预计算嵌入加速训练（适用于冻结模式）")
    cmd_args = build_training_args(
        config, merged_train_file, processed_val_file,
        prediction_head, max_grad_norm, train_mode, train_only_head,
        use_precomputed_embeddings=use_precomputed
    )
    
    # 添加调试信息
    if "loss" in config:
        print(f"🔥 损失函数参数:")
        print(f"   损失函数: {config['loss']['function']}")
        if config["loss"]["function"] == "focal":
            print(f"   Alpha: {config['loss']['alpha']}")
            print(f"   Gamma: {config['loss']['gamma']}")
        elif config["loss"]["function"] == "weighted_bce" and "pos_weight" in config["loss"]:
            print(f"   正样本权重: {config['loss']['pos_weight']}")
        elif config["loss"]["function"] == "bce" and "pos_weight" in config["loss"]:
            print(f"   用户指定的正样本权重: {config['loss']['pos_weight']}")
    
    # 不添加欠采样参数，因为我们已经在预处理阶段完成了欠采样
    if undersample_enabled:
        print(f"📈 欠采样: 已在预处理阶段完成")
    else:
        print(f"📈 欠采样: 禁用")
    
    # 执行训练
    success = run_training(cmd_args, "合并数据集")
    
    # 不再清理任何合并后的数据文件（训练集和验证集），因为后续还需要使用
    print("  📝 保留合并后的数据文件供后续使用")
    
    return success


def run_all_datasets_training(training_config: TrainingConfig) -> dict:
    """运行所有数据集的合并训练（将所有数据集合并成一个大数据集）"""
    data_source = training_config.data_source
    prediction_head = training_config.prediction_head
    loss_function = training_config.loss_function
    pos_weight = training_config.pos_weight
    train_only_head = training_config.train_only_head
    max_grad_norm = training_config.max_grad_norm
    train_mode = training_config.train_mode 
    
    print(f"\n{'='*80}")
    print(f"开始合并所有数据集进行统一训练")
    print(f"数据源: {data_source}")
    print(f"数据集: assistments2009, assistments2012, ednet_kt1, hnu_sys2023, junyi, kdd2010")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # 检查是否启用欠采样
    config = training_config.generate_config(is_merged=True)
    undersample_enabled = config.get("data_processing", {}).get("undersample", False) if config else False
    
    if undersample_enabled:
        # 先对每个数据集进行欠采样，然后再合并
        print(f"🔍 先对每个数据集进行欠采样，然后再合并...")
        success, merged_train_file, merged_val_file = merge_all_datasets_with_undersampling(
            data_source, prediction_head, loss_function, pos_weight
        )
    else:
        # 直接合并所有数据集
        print(f"🔍 直接合并所有数据集（欠采样已禁用）...")
        success, merged_train_file, merged_val_file = merge_all_datasets(data_source)
    
    if not success or not merged_train_file or not merged_val_file:
        print("❌ 数据集合并失败，无法继续训练")
        return {"success": False, "reason": "data_merge_failed"}
    
    # 如果没有生成配置，重新生成
    if not config:
        config = training_config.generate_config(is_merged=True)
        if not config:
            print("❌ 配置生成失败")
            return {"success": False, "reason": "config_generation_failed"}
    
    # 分析合并后的数据平衡性
    print(f"📊 分析合并后数据平衡性...")
    label_counts = analyze_data_balance(merged_train_file, "统一训练数据")
    analyze_data_balance(merged_val_file, "统一验证数据")
    
    # 如果使用BCE损失函数，使用用户指定的正样本权重
    if config.get("loss", {}).get("function") == "bce":
        # 注意：对于统一训练，我们使用用户指定的权重而不是自动计算的权重
        config["loss"]["pos_weight"] = pos_weight
        print(f"⚖️ 使用用户指定的正样本权重: {pos_weight:.2f}")
    
    # 直接使用完整验证集
    processed_val_file = merged_val_file
    
    print(f"\n🚀 开始统一训练合并数据集...")
    print(f"✅ 训练文件: {config['data']['train_file']}")
    print(f"✅ 验证文件: {config['data']['val_file']}")
    print(f"📁 输出目录: {config['paths']['output_dir']}")
    
    # 构建训练参数（在freeze模式下自动启用预计算嵌入）
    use_precomputed = (train_mode == "freeze")
    if use_precomputed:
        print(f"🚀 将使用预计算嵌入加速训练（freeze模式自动启用）")
    cmd_args = build_training_args(
        config, merged_train_file, processed_val_file,
        prediction_head, max_grad_norm, train_mode, train_only_head,
        use_precomputed_embeddings=use_precomputed
    )
    
    # 添加调试信息
    if "loss" in config:
        print(f"🔥 损失函数参数:")
        print(f"   损失函数: {config['loss']['function']}")
        if config["loss"]["function"] == "focal":
            print(f"   Alpha: {config['loss']['alpha']}")
            print(f"   Gamma: {config['loss']['gamma']}")
        elif config["loss"]["function"] == "weighted_bce" and "pos_weight" in config["loss"]:
            print(f"   正样本权重: {config['loss']['pos_weight']}")
        elif config["loss"]["function"] == "bce" and "pos_weight" in config["loss"]:
            print(f"   用户指定的正样本权重: {config['loss']['pos_weight']}")
    
    # 显示欠采样状态
    if undersample_enabled:
        print(f"📈 欠采样: 已在预处理阶段完成（每个数据集单独处理）")
    else:
        print(f"📈 欠采样: 禁用")
    
    # 执行训练
    success = run_training(cmd_args, "所有数据集")
    
    # 不再清理任何合并后的数据文件（训练集和验证集），因为后续还需要使用
    print("  📝 保留合并后的数据文件供后续使用")
    
    total_time = time.time() - start_time
    print(f"✅ 所有数据集合并训练完成（耗时: {total_time:.1f}秒）")
    return {"success": True, "duration": total_time}


def should_undersample(label_counts: dict) -> bool:
    """判断是否需要进行欠采样"""
    pos_count = label_counts.get('positive', 0)
    neg_count = label_counts.get('negative', 0)
    
    # 如果任一类样本数为0，不需要欠采样
    if pos_count == 0 or neg_count == 0:
        return False
    
    # 计算不平衡比例
    imbalance_ratio = max(pos_count, neg_count) / min(pos_count, neg_count)
    
    # 如果不平衡比例超过3，建议进行欠采样
    return imbalance_ratio > 3


def undersample_data(data_file_path: str, dataset_name: str, ratio="1:1") -> Optional[str]:
    """对数据进行欠采样以平衡类别"""
    try:
        print(f"  正在对 {dataset_name} 数据进行欠采样 (比例: {ratio})...")
        
        with open(data_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 分离正负样本
        positive_samples = []
        negative_samples = []
        
        for item in data:
            output = item.get('output', '').strip().lower()
            # 标准化标签
            if output in ['正确', 'right', '1']:
                positive_samples.append(item)
            elif output in ['错误', 'wrong', '0']:
                negative_samples.append(item)
        
        print(f"    正样本: {len(positive_samples)} 条")
        print(f"    负样本: {len(negative_samples)} 条")
        
        if len(positive_samples) == 0 or len(negative_samples) == 0:
            print(f"    ⚠️ 一类样本为空，无法进行欠采样")
            return data_file_path
        
        # 解析比例字符串
        try:
            pos_ratio, neg_ratio = map(int, ratio.split(':'))
            if pos_ratio <= 0 or neg_ratio <= 0:
                raise ValueError("比例必须为正整数")
        except:
            print(f"    ⚠️ 无效的比例格式 '{ratio}'，使用默认 1:1")
            pos_ratio, neg_ratio = 1, 1
        
        # 计算目标样本数量
        if pos_ratio == neg_ratio:
            # 1:1 比例，采用原来的平衡策略
            target_count = min(len(positive_samples), len(negative_samples))
            if len(positive_samples) > len(negative_samples):
                positive_samples = random.sample(positive_samples, target_count)
            else:
                negative_samples = random.sample(negative_samples, target_count)
        else:
            # 自定义比例
            # 计算可以保持比例的最大样本数
            max_pos_from_ratio = len(negative_samples) * pos_ratio // neg_ratio
            max_neg_from_ratio = len(positive_samples) * neg_ratio // pos_ratio
            
            if max_pos_from_ratio <= len(positive_samples):
                # 正样本是限制因素
                target_pos_count = max_pos_from_ratio
                target_neg_count = target_pos_count * neg_ratio // pos_ratio
            else:
                # 负样本是限制因素
                target_neg_count = max_neg_from_ratio
                target_pos_count = target_neg_count * pos_ratio // neg_ratio
            
            # 随机采样
            if target_pos_count < len(positive_samples):
                positive_samples = random.sample(positive_samples, target_pos_count)
            if target_neg_count < len(negative_samples):
                negative_samples = random.sample(negative_samples, target_neg_count)
        
        # 合并采样后的数据
        balanced_data = positive_samples + negative_samples
        
        # 打乱数据
        random.shuffle(balanced_data)
        
        print(f"    欠采样后数据: {len(balanced_data)} 条")
        actual_pos = len([s for s in balanced_data if s.get('output', '').strip().lower() in ['正确', 'right', '1']])
        actual_neg = len(balanced_data) - actual_pos
        print(f"    正负样本比例: {actual_pos}:{actual_neg}")
        
        # 保存欠采样后的数据到临时文件，确保使用ASCII编码避免全角字符问题
        temp_file = data_file_path.replace('.json', '_undersampled.json')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(balanced_data, f, ensure_ascii=True, indent=2)
        
        print(f"    ✅ 欠采样后的数据保存到: {temp_file}")
        return temp_file
        
    except Exception as e:
        print(f"    ❌ 数据欠采样失败: {e}")
        return None


def merge_all_datasets_with_undersampling(data_source='data_text_graph', prediction_head='adaptive', loss_function='focal', pos_weight=1.0) -> Tuple[bool, Optional[str], Optional[str]]:
    """先对每个数据集进行欠采样，然后再合并所有数据集的训练集和验证集"""
    print(f"🔍 开始对每个数据集进行欠采样并合并（数据源: {data_source}）...")
    
    # 定义所有数据集
    datasets = [
        'assistments2009',
        'assistments2012', 
        'ednet_kt1',
        'hnu_sys2023',
        'junyi',
        'kdd2010'
    ]
    
    # 根据数据源决定文件名和合并后的文件名
    if data_source == 'data_text_graph':
        train_file_name = "train.json"  # 使用 train_select.json
        merged_train_file = os.path.join(project_root, "dataSet", data_source, "merged_train_select_balanced.json")
        filter_correct = False  # 只使用 is_correct=true 的样本
    else:
        train_file_name = "train.json"  # 使用传统的 train.json
        merged_train_file = os.path.join(project_root, "dataSet", data_source, "merged_train_balanced.json")
        filter_correct = False
        
    merged_val_file = os.path.join(project_root, "dataSet", data_source, "merged_val.json")
    
    # 合并训练集（先欠采样再合并）
    merged_train_data = []
    total_samples = 0
    balanced_samples = 0
    
    for dataset in datasets:
        train_file = os.path.join(project_root, "dataSet", data_source, dataset, train_file_name)
        if os.path.exists(train_file):
            print(f"  处理 {dataset} 训练集...")
            try:
                with open(train_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_samples += len(data)
                    
                    if filter_correct:
                        # 只保留 is_correct=true 的样本
                        filtered_data = [item for item in data if item.get('is_correct') == True]
                        print(f"    原始数据: {len(data)} 条，筛选后: {len(filtered_data)} 条 (is_correct=true)")
                        data = filtered_data
                    
                    # 分析当前数据集的平衡性
                    label_counts = {}
                    for item in data:
                        output = item.get('output', '').strip().lower()
                        # 标准化标签
                        if output in ['正确', 'right', '1']:
                            label = 'positive'
                        elif output in ['错误', 'wrong', '0']:
                            label = 'negative'
                        else:
                            label = 'unknown'
                        label_counts[label] = label_counts.get(label, 0) + 1
                    
                    print(f"    数据集 {dataset} 类别分析: 正样本 {label_counts.get('positive', 0)} 条, 负样本 {label_counts.get('negative', 0)} 条")
                    
                    # 如果需要欠采样，则进行欠采样
                    if should_undersample(label_counts):
                        print(f"    ⚠️ 检测到 {dataset} 类别不平衡，将进行欠采样...")
                        # 生成临时配置以获取欠采样参数
                        temp_config = generate_single_dataset_config(
                            dataset_name=dataset, 
                            data_source=data_source, 
                            prediction_head=prediction_head, 
                            loss_function=loss_function, 
                            pos_weight=pos_weight
                        )
                        # 从配置中获取数据集特定的欠采样比例
                        undersample_ratio = "3:1"  # 默认值
                        if temp_config and "data_processing" in temp_config:
                            undersample_ratio = temp_config["data_processing"].get("undersample_ratio", "3:1")
                        
                        # 创建临时文件进行欠采样
                        temp_file = train_file.replace('.json', '_temp.json')
                        with open(temp_file, 'w', encoding='utf-8') as temp_f:
                            json.dump(data, temp_f, ensure_ascii=False, indent=2)
                        
                        # 进行欠采样
                        balanced_file = undersample_data(temp_file, dataset, undersample_ratio)
                        if balanced_file and os.path.exists(balanced_file):
                            with open(balanced_file, 'r', encoding='utf-8') as bf:
                                balanced_data = json.load(bf)
                            balanced_samples += len(balanced_data)
                            merged_train_data.extend(balanced_data)
                            print(f"    欠采样后: {len(balanced_data)} 条记录")
                            # 清理临时文件
                            os.remove(temp_file)
                            os.remove(balanced_file)
                        else:
                            # 欠采样失败，使用原始数据
                            merged_train_data.extend(data)
                            print(f"    欠采样失败，使用原始数据: {len(data)} 条记录")
                    else:
                        # 不需要欠采样，直接使用原始数据
                        merged_train_data.extend(data)
                        print(f"    数据已平衡，直接使用: {len(data)} 条记录")
                        
            except Exception as e:
                print(f"    跳过 {dataset} 训练集 (错误: {e})")
        else:
            print(f"    跳过 {dataset} 训练集 (文件不存在)")
    
    print(f"  总样本统计: 原始 {total_samples} 条 → 平衡后 {balanced_samples} 条")
    
    # 打乱合并后的训练数据
    random.shuffle(merged_train_data)
    print(f"  训练数据打乱完成，共 {len(merged_train_data)} 条记录")
    
    # 保存合并后的训练集
    os.makedirs(os.path.dirname(merged_train_file), exist_ok=True)
    try:
        with open(merged_train_file, 'w', encoding='utf-8') as f:
            json.dump(merged_train_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 合并训练集保存完成，共 {len(merged_train_data)} 条记录")
    except Exception as e:
        print(f"❌ 保存合并训练集失败: {e}")
        return False, None, None
    
    # 合并验证集（直接使用完整验证集）
    merged_val_data = []
    for dataset in datasets:
        val_file = os.path.join(project_root, "dataSet", data_source, dataset, "val.json")
        if os.path.exists(val_file):
            print(f"  处理 {dataset} 验证集...")
            try:
                # 直接使用完整验证集数据
                with open(val_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                merged_val_data.extend(data)
                print(f"    添加了 {len(data)} 条记录")
            except Exception as e:
                print(f"    跳过 {dataset} 验证集 (错误: {e})")
        else:
            print(f"    跳过 {dataset} 验证集 (文件不存在)")
    
    # 打乱合并后的验证数据
    random.shuffle(merged_val_data)
    print(f"  验证数据打乱完成，共 {len(merged_val_data)} 条记录")
    
    # 保存合并后的验证集
    try:
        with open(merged_val_file, 'w', encoding='utf-8') as f:
            json.dump(merged_val_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 合并验证集保存完成，共 {len(merged_val_data)} 条记录")
    except Exception as e:
        print(f"❌ 保存合并验证集失败: {e}")
        return False, merged_train_file, None
    
    return True, merged_train_file, merged_val_file


def main():
    parser = argparse.ArgumentParser(description='统一训练脚本 - 支持指定数据源进行训练')
    parser.add_argument('--dataset', type=str,
                       choices=['assistments2009', 'assistments2012', 'ednet_kt1', 'hnu_sys2023', 'junyi', 'kdd2010'],
                       help='训练单个数据集')
    parser.add_argument('--data_source', type=str, choices=['data_text', 'data_text_graph'],
                       default='data_text_graph',
                       help='数据源类型, 默认: data_text_graph')
    parser.add_argument('--merged', action='store_true',
                       help='合并并训练所有数据集 与 --all 相同, 保留兼容性')
    parser.add_argument('--all', action='store_true',
                       help='合并所有数据集并进行统一训练 推荐使用')
    parser.add_argument('--no_undersample', action='store_true',
                       help='禁用欠采样功能')
    parser.add_argument('--undersample_ratio', type=str, default=None,
                       help='欠采样时的正负样本比例, 格式为 "正样本:负样本", 例如 "3:1" 表示正样本:负样本 = 3:1')
    parser.add_argument('--prediction_head', type=str, default='adaptive',
                       help='预测头类型: adaptive(自适应多层特征), linear(简单线性), dynamic(动态门控+置信度), enhanced_dynamic(增强动态门控,适合多数据集), mlp(多层感知机)',
                       choices=['adaptive', 'linear', 'dynamic', 'enhanced_dynamic', 'mlp'])
    parser.add_argument('--loss_function', type=str, default='bce',
                       help='损失函数类型: focal, bce, weighted_bce',
                       choices=['focal', 'bce', 'weighted_bce'])
    parser.add_argument('--pos_weight', type=float, default=1.0,
                       help='正样本权重，仅在使用weighted_bce时有效')
    parser.add_argument('--train_only_head', action='store_true',
                       help='是否只训练分类头，冻结模型其他参数')
    parser.add_argument('--train_mode', type=str, default='lora',
                       help='训练模式: freeze(固定权重), lora(LoRA微调), full(全参数微调)',
                       choices=['freeze', 'lora', 'full'])
    # 添加分类头学习率参数
    parser.add_argument('--head_learning_rate', type=float, default=None,
                       help='分类头学习率，默认为基础模型学习率的10倍')
    # 添加梯度裁剪参数
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                       help='梯度裁剪的最大范数，默认为1.0，范围应在0.1到5之间')
    
    args = parser.parse_args()
    
    if not any([args.dataset, args.merged, args.all]):
        print("错误: 必须指定 --dataset, --merged 或 --all 参数")
        parser.print_help()
        return
    
    # 设置随机种子以确保结果可重现
    random.seed(3407)
    
    # 创建训练配置对象
    training_config = TrainingConfig(
        dataset_name=args.dataset,
        data_source=args.data_source,
        prediction_head=args.prediction_head,
        loss_function=args.loss_function,
        pos_weight=args.pos_weight,
        train_only_head=args.train_only_head,
        max_grad_norm=max(0.1, min(5.0, args.max_grad_norm)),  # 限制在0.1到5之间
        train_mode=args.train_mode,
        head_learning_rate=args.head_learning_rate,
        no_undersample=args.no_undersample,
        undersample_ratio=args.undersample_ratio
    )
    
    # 简化日志输出，只显示关键训练参数
    print("\n" + "="*50)
    print("训练配置信息:")
    print("="*50)
    print(f"数据源: {training_config.data_source}")
    print(f"欠采样: {'禁用' if training_config.no_undersample else '启用'}")
    if not training_config.no_undersample:
        # 显示实际使用的欠采样比例
        if training_config.undersample_ratio is not None:
            print(f"欠采样比例: {training_config.undersample_ratio} (来自命令行参数)")
        else:
            # 从生成的配置中获取数据集特定的欠采样比例
            config_undersample_ratio = "3:1"  # 默认值
            if training_config.config and "data_processing" in training_config.config:
                config_undersample_ratio = training_config.config["data_processing"].get("undersample_ratio", "3:1")
            print(f"欠采样比例: {config_undersample_ratio} (来自数据集配置)")
    print(f"损失函数: {training_config.loss_function}")
    if training_config.loss_function == 'weighted_bce':
        print(f"正样本权重: {training_config.pos_weight}")
    print(f"预测头类型: {training_config.prediction_head}")
    print(f"只训练分类头: {training_config.train_only_head}")
    print(f"训练模式: {training_config.train_mode}")
    # 添加学习率信息
    config = training_config.generate_config(is_merged=False) if training_config.dataset_name else training_config.generate_config(is_merged=True)
    if config and "training" in config:
        base_lr = config["training"]["learning_rate"]
        head_lr = getattr(training_config, 'head_learning_rate', None)
        if head_lr is not None and head_lr != base_lr:
            print(f"基础模型学习率: {base_lr}")
            print(f"分类头学习率: {head_lr}")
        else:
            print(f"学习率: {base_lr}")
    print("="*50)
    
    if args.dataset:
        # 训练单个数据集
        success = run_single_dataset_training(training_config)
        if success:
            print(f"\n🎉 {args.dataset} 训练成功！")
        else:
            print(f"\n💥 {args.dataset} 训练失败！")
            
    elif args.merged:
        # 合并并训练所有数据集
        success = run_merged_training(training_config)
        if success:
            print(f"\n🎉 合并数据集训练成功！")
        else:
            print(f"\n💥 合并数据集训练失败！")
            
    elif args.all:
        # 合并所有数据集进行统一训练
        result = run_all_datasets_training(training_config)
        if result["success"]:
            print(f"\n🎉 所有数据集合并训练成功！")
        else:
            print(f"\n💥 所有数据集合并训练失败！原因: {result.get('reason', '未知错误')}")


if __name__ == "__main__":
    main()