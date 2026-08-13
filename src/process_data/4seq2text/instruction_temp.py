#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将数据集转换为LLM训练的指令格式

功能：
- 读取 dataSet/data_split 下的数据集
- 转换为包含系统提示、指令、输出的JSON格式
- 支持长序列重叠拆分、图谱上下文、中英文双语
- 输出到 dataSet/data_text 目录（启用图上下文时输出到 dataSet/data_text_graph）

详细使用说明请参考：docs/instruction_template_guide.md
"""

import os
import sys
import json
import glob
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable


def iter_dataset_sequence_groups(file_path: str) -> Iterable[Tuple[str, List[str], List[str], List[int]]]:
    """
    从数据集分割文件流式迭代读取每个用户的 (question_ids, concept_ids, corrects)
    输入格式（每个用户一段，4行一组）：
    1) 用户ID行：如 1、2、12 等
    2) 题目ID行：逗号分隔，例如: 12243,12267,...
    3) 知识点ID行：分号分隔，例如: 19;19;... 或者元素可能为 [3,4]、[12,7,29]
    4) 正确性行：逗号分隔的 0/1 或 0.0/1.0
    """
    pending_line: str | None = None
    state = 'start'  # start(uid) -> q -> c -> r -> start(uid)
    question_line: str = ''
    concept_line: str = ''
    correct_line: str = ''
    current_uid: str = ''

    def parse_and_yield(uid: str, q_line: str, c_line: str, r_line: str):
        questions = [t.strip() for t in q_line.split(',') if t.strip()]
        concepts = [t.strip() for t in c_line.split(';') if t.strip()]
        # 正确性：将任意非零视为 True
        corrects_inner: List[int] = []
        for token in r_line.split(','):
            token = token.strip()
            if token == '':
                continue
            try:
                val = float(token)
            except ValueError:
                val = 0.0
            corrects_inner.append(1 if val != 0.0 else 0)

        min_len = min(len(questions), len(concepts), len(corrects_inner))
        if min_len > 0:
            yield uid, questions[:min_len], concepts[:min_len], corrects_inner[:min_len]

    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            line = pending_line if pending_line is not None else f.readline()
            pending_line = None
            if not line:
                break
            line = line.strip()
            if line == '':
                continue

            if state == 'start':
                # 将起始行视为该组用户UID（数值或字符串均可）
                current_uid = line
                state = 'q'
                continue

            if state == 'q':
                question_line = line
                state = 'c'
                continue

            if state == 'c':
                concept_line = line
                state = 'r'
                continue

            if state == 'r':
                correct_line = line
                # 完整一组，尝试解析
                for item in parse_and_yield(current_uid, question_line, concept_line, correct_line):
                    yield item
                # 下一行即为下一组的 UID
                state = 'start'
                continue


def get_result_text(is_correct: int, lang: str) -> str:
    """
    根据语言固定返回：
    - zh: 正确 / 错误
    - en: right / wrong
    """
    if lang == 'en':
        return 'right' if is_correct else 'wrong'
    return '正确' if is_correct else '错误'


def get_text_config_for_dataset(
    dataset_name: str,
    text_cfg: Dict[str, Any],
    desc_overrides: Dict[str, Any]
) -> Tuple[str, str, Dict[str, List[str]], str, str]:
    """
    根据数据集名称返回对应的文本模板与结果词典：
    - assistments2009/assistments2012/ednet/ednet_kt1/kdd2010 使用英文模板
    - hnu_sys2023/junyi 使用中文模板（来自配置）
    其他默认使用中文模板（来自配置）
    """
    english_sets = {"assistments2009", "assistments2012", "ednet", "ednet_kt1", "kdd2010"}
    chinese_sets = {"hnu_sys2023", "junyi"}

    dataset_desc_map_en = {
        "assistments2009": (
            "ASSISTments 2009-2010 (K-12 math): 3.8K students, 16.9K problems, 101 skills, 274K interactions. "
            "Short sequences (avg=72 steps), skill-centric tutoring system with immediate feedback."
        ),
        "assistments2012": (
            "ASSISTments 2012-2013 (K-12 math): 27.1K students, 51K problems, 198 skills, 2.6M interactions. "
            "Medium sequences (avg=97 steps), enhanced skill tagging, multi-skill problems common."
        ),
        "ednet": (
            "EdNet (TOEIC English): 784K students, 12.3K bundles, 188 parts, 95M interactions. "
            "Long sequences (avg=130 steps), language learning platform with part-based skills."
        ),
        "ednet_kt1": (
            "EdNet KT1 (TOEIC English): 784K students, 12.3K bundles, 188 parts, 95M interactions. "
            "Long sequences (avg=130 steps), standardized split for knowledge tracing research."
        ),
        "kdd2010": (
            "KDD Cup 2010 'Bridge to Algebra': 1.1K students, 19K problems, 564 skills, 1.8M interactions. "
            "Very long sequences (avg=1587 steps), step-level cognitive tutoring with multi-skill problems."
        )
    }
    dataset_desc_map_zh = {
        "hnu_sys2023": (
            "湖南大学系统（2023）：708名学生，220道题目，43个知识点，10.2万次交互。"
            "中等序列（平均=145步），高校在线学习平台，知识点标注精细，序列密集分布。"
        ),
        "junyi": (
            "均一教育平台（K-12数学）：6.99万学生，2.58万练习，1326个主题，1621万次交互。"
            "长序列（平均=232步），中文数学学习平台，主题映射清晰，练习多样化。"
        )
    }

    if dataset_name in english_sets:
        template = (
            "Dataset: {dataset}. {dataset_desc} "
            "The student with UID={student_id} has previously, in chronological order, answered {interactions}.\n"
        )
        interaction_template = (
            "t={step}: Q{question_id} {concept_label} {result}"
        )
        result_mapping = {
            'correct': ['right'],
            'incorrect': ['wrong']
        }
        desc_override = (desc_overrides.get('en', {}) if isinstance(desc_overrides, dict) else {}).get(dataset_name)
        return template, interaction_template, result_mapping, (desc_override or dataset_desc_map_en.get(dataset_name, "")), 'en'

    # 中文模板
    template = (
        '数据集：{dataset}。{dataset_desc} 学生 UID={student_id} 按时间顺序曾回答：{interactions}。'
    )
    interaction_template = 't={step}：Q{question_id} {concept_label} {result}'
    result_mapping = text_cfg.get('result_mapping', {'correct': ['正确'], 'incorrect': ['错误']})
    desc_override = (desc_overrides.get('zh', {}) if isinstance(desc_overrides, dict) else {}).get(dataset_name)
    return template, interaction_template, result_mapping, (desc_override or dataset_desc_map_zh.get(dataset_name, '')), 'zh'


def format_instruction(
    dataset_name: str,
    student_id: str,
    questions: List[str],
    concepts: List[str],
    corrects: List[int],
    template: str,
    interaction_template: str,
    result_mapping: Dict[str, List[str]],
    dataset_desc: str = '',
    lang: str = 'zh',
    start_step: int = 1,
    target_question: str = '',
    target_concepts: str = '',
    prereq_concepts: str = '',
    use_graph_context: bool = False,
    simple_mode: bool = False
) -> str:
    interactions: List[str] = []
    
    def build_concept_set(concept_token: str) -> str:
        """构建概念集合格式 {id1,id2,...}"""
        token = concept_token.strip()
        if token.startswith('[') and token.endswith(']'):
            inner = token[1:-1]
            parts = [p.strip() for p in inner.split(',') if p.strip()]
            return '{' + ','.join(parts) + '}'
        else:
            return '{' + token + '}'

    # 构建学习历史记录
    if simple_mode:
        # 简化模式：不包含时间戳
        for q, c, r in zip(questions, concepts, corrects):
            result_text = get_result_text(r, lang)
            concept_set = build_concept_set(c)
            interaction_text = f"q={q}, c={concept_set}, {result_text}"
            interactions.append(interaction_text)
        interactions_text = '; '.join(interactions)
        
        # 构建目标概念集合
        target_concept_set = build_concept_set(target_concepts)
        
        # 简化格式：学生学习记录+要预测的问题
        if lang == 'en':
            instruction_parts = []
            instruction_parts.append(f"Student learning history: {interactions_text}. Predict whether the student will answer correctly: q={target_question}, c={target_concept_set}.")
            
            # 在简洁模式下也可以使用图上下文
            if use_graph_context and prereq_concepts:
                prereq_set = '{' + prereq_concepts + '}'
                instruction_parts.append(f"Graph Context: The prerequisite knowledge points for the target concepts are {prereq_set}.")
            
            instruction = ' '.join(instruction_parts)
        else:
            instruction_parts = []
            instruction_parts.append(f"学生学习记录：{interactions_text}。预测学生是否能正确回答：q={target_question}, c={target_concept_set}。")
            
            # 在简洁模式下也可以使用图上下文
            if use_graph_context and prereq_concepts:
                prereq_set = '{' + prereq_concepts + '}'
                instruction_parts.append(f"图谱上下文：目标概念的前置知识点是{prereq_set}。")
            
            instruction = ''.join(instruction_parts)
    else:
        # 原有逻辑
        for idx, (q, c, r) in enumerate(zip(questions, concepts, corrects), start=start_step):
            result_text = get_result_text(r, lang)
            concept_set = build_concept_set(c)
            # 新格式：t = 351: q = 6597, c = {128}, wrong
            interaction_text = f"t = {idx}: q = {q}, c = {concept_set}, {result_text}"
            interactions.append(interaction_text)

        interactions_text = '; '.join(interactions)
        
        # 构建目标概念集合
        target_concept_set = build_concept_set(target_concepts)
        target_step = len(questions) + start_step
        
        # 构建完整的instruction
        if lang == 'en':
            # 英文格式
            instruction_parts = []
            instruction_parts.append(f"Student History: The student with ID={student_id} answered in order: {interactions_text}")
            instruction_parts.append(f"Prediction Target: Predict whether the student will answer correctly at t = {target_step}: q = {target_question}, c = {target_concept_set}.")
            
            if use_graph_context and prereq_concepts:
                prereq_set = '{' + prereq_concepts + '}'
                instruction_parts.append(f"Graph Context: The prerequisite knowledge points for the target concepts are {prereq_set}.")
            
            instruction = '\n'.join(instruction_parts)

            instruction = instruction + "\n\nPrediction:\n\n"
        else:
            # 中文格式
            instruction_parts = []
            instruction_parts.append(f"学生历史：学生ID={student_id}按时间顺序回答：{interactions_text}")
            instruction_parts.append(f"预测目标：预测学生在t = {target_step}: q = {target_question}, c = {target_concept_set}时是否能正确回答。")
            
            if use_graph_context and prereq_concepts:
                prereq_set = '{' + prereq_concepts + '}'
                instruction_parts.append(f"图谱上下文：目标概念的前置知识点是{prereq_set}。")
            
            instruction = '\n'.join(instruction_parts)
        
            instruction = instruction + "\n\n预测结果：\n\n"
    
    return instruction


def get_system_prompt(lang: str, simple_mode: bool = False) -> str:
    """获取系统提示词"""
    # 简洁模式和标准模式使用相同的系统提示词
    if lang == 'en':
        return ("This is a knowledge tracing task. You are given a student's learning history consisting of question IDs, concept IDs, and correctness outcomes. Use temporal progression, mastery transitions, performance trends and graph context to infer whether the student will answer the next question correctly. Answer only 'right' or 'wrong'.")
    else:
        return ("这是一个知识追踪任务。您需要根据学生的学习历史（包括题目ID、概念ID和正确性结果），利用时序进展、掌握转换、表现趋势和图谱上下文来推断学生是否能正确回答下一个问题。只回答'正确'或'错误'。")


def split_sequence(questions: List[str], concepts: List[str], corrects: List[int], max_length: int = 50, step_size: int = 20, sequence_mode: str = 'hybrid') -> List[Tuple[List[str], List[str], List[int], str, str, int, int]]:
    """将长序列拆分为多个子序列，使用滑动窗口策略
    
    Args:
        questions: 题目ID列表
        concepts: 概念ID列表
        corrects: 正确性列表
        max_length: 窗口大小（最大长度限制）
        step_size: 步长（滑动距离）
        sequence_mode: 序列生成模式
            - 'short': 只生成短序列数据（2到max_length，包含max_length）
            - 'long': 只生成长序列数据（max_length及以上）
            - 'hybrid': 融合模式，生成所有数据（默认，不重复生成max_length）
        
    Returns:
        List of (history_questions, history_concepts, history_corrects, target_question, target_concept, target_correct, start_step)
        
    示例：
        对于长度为100的序列，max_length=30, step_size=10时：
        
        short模式：
          - t=1（历史），预测t=2
          - t=1到t=2（历史），预测t=3
          - ...
          - t=1到t=29（历史），预测t=30
        
        long模式：
          - t=1到t=29（历史），预测t=30
          - t=11到t=39（历史），预测t=40
          - t=21到t=49（历史），预测t=50
          - ...
          - t=71到t=99（历史），预测t=100
        
        hybrid模式：
          - 短序列：t=2到t=29 (29-1=28个样本)
          - 长序列：t=30到t=100 (8个样本)
          - 总计：36个样本，无重复
    """
    if len(questions) < 2:
        return []
    
    segments = []
    total_length = len(questions)
    
    # 第一阶段：生成短序列样本（步长为1）
    # 仅在 'short' 或 'hybrid' 模式下生成
    if sequence_mode in ['short', 'hybrid']:
        # 从序列开头开始，逐步增加历史长度
        # short模式：预测目标为t=2, t=3, ..., t=max_length
        # hybrid模式：预测目标为t=2, t=3, ..., t=max_length-1（避免与long模式重复）
        if sequence_mode == 'short':
            end_idx = min(max_length, total_length)
        else:  # hybrid模式
            end_idx = min(max_length - 1, total_length)
        
        for target_idx in range(1, end_idx):
            # target_idx 是要预测的位置（0-based索引）
            # 历史记录是 [0, target_idx)
            history_questions = questions[:target_idx]
            history_concepts = concepts[:target_idx]
            history_corrects = corrects[:target_idx]
            
            target_question = questions[target_idx]
            target_concept = concepts[target_idx]
            target_correct = corrects[target_idx]
            
            segments.append((
                history_questions,
                history_concepts,
                history_corrects,
                target_question,
                target_concept,
                target_correct,
                1  # start_step（短序列都从第1步开始）
            ))
    
    # 如果是纯短序列模式，或序列长度不超过窗口大小，直接返回
    if sequence_mode == 'short' or total_length <= max_length:
        return segments
    
    # 第二阶段：生成长序列样本，使用滑动窗口策略
    # 仅在 'long' 或 'hybrid' 模式下生成
    if sequence_mode not in ['long', 'hybrid']:
        return segments
    
    window_start = 0
    
    while window_start + max_length <= total_length:
        # 当前窗口的结束位置
        window_end = window_start + max_length
        
        # 提取历史记录（窗口内的前max_length-1个记录）
        history_questions = questions[window_start:window_end-1]
        history_concepts = concepts[window_start:window_end-1]
        history_corrects = corrects[window_start:window_end-1]
        
        # 目标记录是窗口的最后一个记录
        target_question = questions[window_end-1]
        target_concept = concepts[window_end-1]
        target_correct = corrects[window_end-1]
        
        segments.append((
            history_questions,
            history_concepts,
            history_corrects,
            target_question,
            target_concept,
            target_correct,
            window_start + 1  # start_step（从1开始计数）
        ))
        
        # 移动到下一个窗口
        window_start += step_size
    
    # 处理最后一个窗口（如果还有剩余数据）
    if window_start < total_length - 1:
        # 使用序列末尾的max_length个记录作为最后一个窗口
        window_start = total_length - max_length
        history_questions = questions[window_start:total_length-1]
        history_concepts = concepts[window_start:total_length-1]
        history_corrects = corrects[window_start:total_length-1]
        
        target_question = questions[total_length-1]
        target_concept = concepts[total_length-1]
        target_correct = corrects[total_length-1]
        
        segments.append((
            history_questions,
            history_concepts,
            history_corrects,
            target_question,
            target_concept,
            target_correct,
            window_start + 1  # start_step（从1开始计数）
        ))
    
    return segments


def save_json_array(data: List[Dict[str, Any]], out_path: str):
    """保存为JSON数组格式"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_prereq_graph(dataset_name: str, knowledge_graph_root: str = 'dataSet/knowledge_graph', graph_config: str = 'w55') -> Dict[str, List[str]]:
    """
    加载前置知识图谱，返回概念ID到前置概念列表的映射
    
    Args:
        dataset_name: 数据集名称
        knowledge_graph_root: 知识图谱根目录
        graph_config: 图谱配置，支持以下格式：
                     1. 权重配置：w19, w37, w55, w73, w91
                        - w19: 时序权重0.1, 条件权重0.9
                        - w37: 时序权重0.3, 条件权重0.7
                        - w55: 时序权重0.5, 条件权重0.5
                        - w73: 时序权重0.7, 条件权重0.3
                        - w91: 时序权重0.9, 条件权重0.1
                     2. 密度筛选配置：w55_d10, w37_d15 等
                        - 格式为 {权重}_{密度}，例如 w55_d10 表示权重5:5、密度10%
    """
    # 直接指定图谱文件路径
    kg_file = Path(knowledge_graph_root) / dataset_name / graph_config / 'prereq_graph.json'
    
    prereq_map = {}
    if kg_file.exists():
        # 使用指定的图谱配置文件
        try:
            with open(kg_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
                
                # 处理边列表格式的图谱数据
                if isinstance(graph_data, list):
                    # 边列表格式: [{"from_id": x, "to_id": y, "from_name": "x", "to_name": "y"}, ...]
                    for edge in graph_data:
                        if 'to_id' in edge and 'from_id' in edge:
                            to_concept = str(edge['to_id'])  # 转换为字符串
                            from_concept = str(edge['from_id'])  # 转换为字符串
                            if to_concept not in prereq_map:
                                prereq_map[to_concept] = []
                            prereq_map[to_concept].append(from_concept)
                elif isinstance(graph_data, dict):
                    # 字典格式: {"concept_id": ["prereq1", "prereq2", ...]}
                    prereq_map = graph_data
            
            print(f"    已加载知识图谱: {kg_file} (配置: {graph_config}, 包含 {len(prereq_map)} 个概念)")
        except Exception as e:
            print(f"    警告：加载知识图谱失败 {kg_file}: {e}")
    else:
        print(f"    未找到知识图谱文件: {kg_file}")
        # 尝试查找所有可用的配置
        available_configs = []
        kg_pattern = str(Path(knowledge_graph_root) / dataset_name / '*' / 'prereq_graph.json')
        kg_files = glob.glob(kg_pattern)
        for kg_file_path in kg_files:
            config_name = Path(kg_file_path).parent.name
            available_configs.append(config_name)
        if available_configs:
            print(f"    可用的图谱配置: {', '.join(sorted(available_configs))}")
    
    return prereq_map


def get_prereq_concepts(concept_str: str, prereq_map: Dict[str, List[str]]) -> str:
    """根据目标概念获取前置概念"""
    concepts = []
    if concept_str.startswith('[') and concept_str.endswith(']'):
        # 多概念格式 [3,4]
        inner = concept_str[1:-1]
        concepts = [c.strip() for c in inner.split(',') if c.strip()]
    else:
        # 单概念
        concepts = [concept_str.strip()]
    
    # 收集所有前置概念
    all_prereqs = set()
    for concept in concepts:
        if concept in prereq_map:
            all_prereqs.update(prereq_map[concept])
    
    return ','.join(sorted(all_prereqs)) if all_prereqs else ''


def list_available_graph_configs(dataset_name: str, knowledge_graph_root: str = 'dataSet/knowledge_graph') -> List[str]:
    """列出指定数据集可用的图谱配置"""
    dataset_kg_root = Path(knowledge_graph_root) / dataset_name
    configs = []
    
    if dataset_kg_root.exists():
        for config_dir in dataset_kg_root.iterdir():
            if config_dir.is_dir():
                prereq_file = config_dir / 'prereq_graph.json'
                if prereq_file.exists():
                    configs.append(config_dir.name)
    
    return sorted(configs)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='将 dataSet/data_split 目录下的数据集转换为语言模型训练的文本数据')
    parser.add_argument('--input-root', type=str, default='dataSet/data_split', help='数据分割根目录')
    parser.add_argument('--output-root', type=str, default='dataSet/data_text', help='输出根目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--preview', type=int, default=0, help='仅预览前N条样本，不写出文件')
    parser.add_argument('--include-datasets', type=str, default='', help='仅处理这些数据集，逗号分隔，例如: assistments2009,junyi')
    parser.add_argument('--use-graph-context', action='store_true', help='是否在指令中包含图谱上下文信息')
    parser.add_argument('--knowledge-graph-root', type=str, default='dataSet/knowledge_graph', help='知识图谱根目录')
    parser.add_argument('--graph-config', type=str, default='', help='指定使用的图谱配置，如 w37、w55、w55_d10、w37_d15 等。不指定则使用默认策略（按字母顺序第一个）')
    parser.add_argument('--list-graph-configs', type=str, default='', help='列出指定数据集的所有可用图谱配置，例如: --list-graph-configs hnu_sys2023')
    parser.add_argument('--max-length', type=int, default=30, help='学习记录的窗口大小（最大长度限制）')
    parser.add_argument('--step-size', type=int, default=10, help='滑动窗口的默认步长')
    parser.add_argument('--dataset-step-sizes', type=str, default='assistments2009:10,ednet_kt1:32,assistments2012:16,hnu_sys2023:1,junyi:20,kdd2010:15', help='为特定数据集指定步长，格式: dataset1:step1,dataset2:step2 例如: assistments2009:15,hnu_sys2023:25')
    parser.add_argument('--simple-mode', action='store_true', help='启用简化模式，生成不包含时间戳的简单指令数据')
    parser.add_argument('--sequence-mode', type=str, default='hybrid', choices=['short', 'long', 'hybrid'], 
                        help='序列生成模式: short(只生成2到max_length的短序列), long(只生成max_length及以上的长序列), hybrid(融合模式，生成所有数据且无重复，默认)')
    args = parser.parse_args()

    random.seed(args.seed)

    # 如果指定了 --list-graph-configs，则列出可用配置并退出
    if args.list_graph_configs:
        dataset_name = args.list_graph_configs
        configs = list_available_graph_configs(dataset_name, args.knowledge_graph_root)
        if configs:
            print(f'\n数据集 {dataset_name} 的可用图谱配置:')
            for i, config in enumerate(configs, 1):
                # 解析配置名称
                parts = config.split('_')
                if len(parts) == 1 and parts[0].startswith('w'):
                    # 新格式：wXY (如 w19, w37, w55)
                    weight_part = parts[0]
                    weight_map = {
                        'w19': '权重 1:9 (时序:0.1, 条件:0.9)',
                        'w37': '权重 3:7 (时序:0.3, 条件:0.7)',
                        'w55': '权重 5:5 (时序:0.5, 条件:0.5)',
                        'w73': '权重 7:3 (时序:0.7, 条件:0.3)',
                        'w91': '权重 9:1 (时序:0.9, 条件:0.1)'
                    }
                    weight_desc = weight_map.get(weight_part, weight_part)
                    print(f'  {i}. {config} - {weight_desc}')
                elif len(parts) == 2 and parts[0].startswith('w') and parts[1].startswith('d'):
                    # 密度筛选格式：wXY_dZZ (如 w55_d10)
                    weight_part = parts[0]
                    density_part = parts[1]
                    weight_map = {
                        'w19': '权重 1:9',
                        'w37': '权重 3:7',
                        'w55': '权重 5:5',
                        'w73': '权重 7:3',
                        'w91': '权重 9:1'
                    }
                    weight_desc = weight_map.get(weight_part, weight_part)
                    density_value = density_part[1:]  # 去掉 'd' 前缀
                    print(f'  {i}. {config} - {weight_desc}, 密度 {density_value}%')
                elif len(parts) == 2 and parts[0].startswith('t') and parts[1].startswith('w'):
                    # 旧格式：tXXX_wYY (如 t060_w37，兼容旧版)
                    threshold_part = parts[0]
                    weight_part = parts[1]
                    weight_desc = {
                        'w37': '权重 3:7 (时序:0.3, 条件:0.7)',
                        'w55': '权重 5:5 (时序:0.5, 条件:0.5)',
                        'w73': '权重 7:3 (时序:0.7, 条件:0.3)'
                    }.get(weight_part, weight_part)
                    threshold_desc = {
                        't055': '阈值 0.55',
                        't060': '阈值 0.60', 
                        't065': '阈值 0.65'
                    }.get(threshold_part, threshold_part)
                    print(f'  {i}. {config} - {threshold_desc}, {weight_desc}')
                else:
                    print(f'  {i}. {config}')
        else:
            print(f'数据集 {dataset_name} 没有可用的图谱配置')
        sys.exit(0)

    # 修正路径计算方式，确保使用正确的项目根目录
    # 获取当前脚本的绝对路径，然后向上三级找到项目根目录
    script_dir = Path(__file__).resolve().parent
    process_data_dir = script_dir.parent
    src_dir = process_data_dir.parent
    workspace_root = src_dir.parent
    
    # 正确设置input_root和output_root路径
    input_root = workspace_root / args.input_root
    # 如果启用了图上下文，输出到 data_text_graph 目录
    if args.use_graph_context:
        output_root = workspace_root / args.output_root.replace('data_text', 'data_text_graph')
    else:
        output_root = workspace_root / args.output_root
    
    print(f"输入根目录: {input_root}")
    print(f"输出根目录: {output_root}")
    
    # 查找所有数据集目录
    if not input_root.exists():
        print(f'错误：输入根目录 {input_root} 不存在')
        sys.exit(1)
        
    dataset_dirs = [d for d in input_root.iterdir() if d.is_dir()]
    include_set = set([x.strip() for x in args.include_datasets.split(',') if x.strip()])
    if include_set:
        dataset_dirs = [d for d in dataset_dirs if d.name in include_set]
    
    if not dataset_dirs:
        print(f'未在 {input_root} 下找到任何数据集目录')
        sys.exit(1)

    # 解析数据集特定的步长配置
    dataset_step_sizes = {}
    if args.dataset_step_sizes:
        for item in args.dataset_step_sizes.split(','):
            if ':' in item:
                dataset_name, step_size = item.split(':')
                try:
                    dataset_step_sizes[dataset_name.strip()] = int(step_size.strip())
                except ValueError:
                    print(f"警告：无法解析数据集步长配置 '{item}'，将使用默认步长")
    
    # 处理每个数据集
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f'\n处理数据集: {dataset_name}')
        
        # 确定该数据集使用的步长
        step_size = dataset_step_sizes.get(dataset_name, args.step_size)
        print(f'  使用步长: {step_size} (窗口大小: {args.max_length})')
        
        # 加载前置知识图谱（如果需要的话）
        prereq_map = {}
        if args.use_graph_context:
            # 如果没有指定图谱配置，自动选择第一个可用的配置
            graph_config = args.graph_config
            if not graph_config:
                available_configs = list_available_graph_configs(dataset_name, str(workspace_root / args.knowledge_graph_root))
                if available_configs:
                    graph_config = available_configs[0]  # 使用字母顺序第一个
                    print(f'    自动选择图谱配置: {graph_config}')
                else:
                    print(f'    警告：数据集 {dataset_name} 没有可用的图谱配置')
                    graph_config = 'w55'  # 使用默认配置作为后备
            
            prereq_map = load_prereq_graph(dataset_name, str(workspace_root / args.knowledge_graph_root), graph_config)
        
        # 为该数据集选择模板
        ds_template, ds_interaction_template, ds_result_mapping, ds_desc, ds_lang = get_text_config_for_dataset(dataset_name, {}, {})
        
        # 处理训练、验证、测试集
        for split_name in ['train_sequences.txt', 'val_sequences.txt', 'test_sequences.txt']:
            split_file = dataset_dir / split_name
            if not split_file.exists():
                print(f'  跳过不存在的文件: {split_file}')
                continue
                
            output_name = split_name.replace('_sequences.txt', '.json')
            output_file = output_root / dataset_name / output_name
            
            print(f'  处理 {split_name} -> {output_name}')
            
            instructions = []
            
            for uid, questions, concepts, corrects in iter_dataset_sequence_groups(str(split_file)):
                if len(questions) < 5:  # 至少需要5个交互才能做预测
                    continue
                
                # 使用拆分函数处理序列，使用特定于数据集的步长和序列模式
                segments = split_sequence(questions, concepts, corrects, args.max_length, step_size, args.sequence_mode)
                
                for seg_idx, (history_questions, history_concepts, history_corrects, target_question, target_concept, target_correct, start_step) in enumerate(segments):
                    # 获取前置知识点（如果启用图谱上下文）
                    prereq_concepts = ''
                    if args.use_graph_context:
                        prereq_concepts = get_prereq_concepts(target_concept, prereq_map)
                    
                    # 生成instruction
                    instruction = format_instruction(
                        dataset_name=dataset_name,
                        student_id=uid,
                        questions=history_questions,
                        concepts=history_concepts,
                        corrects=history_corrects,
                        template=ds_template,
                        interaction_template=ds_interaction_template,
                        result_mapping=ds_result_mapping,
                        dataset_desc=ds_desc,
                        lang=ds_lang,
                        start_step=start_step,
                        target_question=target_question,
                        target_concepts=target_concept,
                        prereq_concepts=prereq_concepts,
                        use_graph_context=args.use_graph_context,
                        simple_mode=args.simple_mode
                    )
                    
                    # 获取目标答案
                    target_answer = get_result_text(target_correct, ds_lang)
                    
                    # 获取系统提示词
                    system_prompt = get_system_prompt(ds_lang, args.simple_mode)
                    
                    instructions.append({
                        'system': system_prompt,
                        'instruction': instruction,
                        'output': target_answer,
                        'dataset_name': dataset_name
                    })
            
            # 仅预览模式
            if args.preview and args.preview > 0:
                preview_n = min(args.preview, len(instructions))
                for item in instructions[:preview_n]:
                    print(json.dumps(item, ensure_ascii=False))
                print(f"预览完毕，共展示 {preview_n}/{len(instructions)} 条。")
                continue
            
            # 保存文件
            if instructions:
                save_json_array(instructions, str(output_file))
                print(f'    保存 {len(instructions)} 条记录到 {output_file}')
            else:
                print(f'    没有生成任何记录')
    
    print('\n处理完成！')
    
    if args.use_graph_context:
        print(f'\n注意：本次运行启用了图谱上下文功能，输出已保存到 {output_root} 目录。')
    else:
        print(f'\n注意：本次运行未启用图谱上下文功能，输出已保存到 {output_root} 目录。如需包含前置知识信息，请使用 --use-graph-context 参数。')


if __name__ == '__main__':
    main()


