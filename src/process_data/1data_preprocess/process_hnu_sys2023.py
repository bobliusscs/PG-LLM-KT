"""
将 hnu_sys2023 数据集处理为统一格式，并进行重新编号：

输入：
- 学生答题记录：data/raw/hnu_sys2023/HNU_SYS_2023.txt
  每行：学生基本信息\t"qid ans,qid ans,..."（题目与回答之间空格，题与题之间逗号）
- 题目-知识点映射：data/raw/hnu_sys2023/question2skill.csv
  第一列为题目编号，其余列为知识点编号（列名为数字），单元格为1表示该题包含该知识点

输出：
- data/processed/hnu_sys2023/user_group_sequences.txt
  第1行：新学生编号（从1开始）
  第2行：新题目编号序列（逗号分隔）
  第3行：新知识点序列（单知识点为数字；多知识点为列表如 [1,2,3]；题与题之间分号分隔）
  第4行：正确标记（0/1，逗号分隔）
- data/processed/hnu_sys2023/map/{user_id_map.csv, problem_id_map.csv, skill_name_map.csv}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json

import pandas as pd


RAW_DIR = Path('dataSet/raw/hnu_sys2023')
RESP_FILE = RAW_DIR / 'HNU_SYS_2023.txt'
Q2S_FILE = RAW_DIR / 'question2skill.csv'

OUT_DIR = Path('dataSet/processed/hnu_sys2023')
OUT_FILE = OUT_DIR / 'user_group_sequences.txt'


def parse_question2skill_map() -> Dict[int, List[int]]:
    """读取 question2skill.csv，返回 {question_id: [skill_id,...]}。
    - 假定第一列是题目编号，其余列列名为知识点编号
    - 单元格为 '1' 表示关联
    """
    if not Q2S_FILE.exists():
        raise FileNotFoundError(f'未找到题目-知识点映射文件：{Q2S_FILE}')

    df = pd.read_csv(Q2S_FILE, dtype=str)
    if df.empty or df.shape[1] < 2:
        raise ValueError('question2skill.csv 格式异常，至少需要题目列和一个知识点列')

    # 识别题目列（第一列）
    question_col = df.columns[0]
    skill_cols = list(df.columns[1:])

    q2s: Dict[int, List[int]] = {}
    for _, row in df.iterrows():
        q_raw = str(row[question_col]).strip()
        if q_raw == '' or q_raw.lower() == 'nan':
            continue
        if not re.match(r'^\d+$', q_raw):
            # 尝试从形如 "题X" 中提取数字
            m = re.search(r'(\d+)', q_raw)
            if not m:
                continue
            q_id = int(m.group(1))
        else:
            q_id = int(q_raw)

        skills: List[int] = []
        for col in skill_cols:
            val = str(row[col]).strip() if col in row else ''
            if val in {'1', '1.0'}:
                # 列名作为知识点编号
                col_name = str(col).strip()
                m2 = re.search(r'(\d+)', col_name)
                if m2:
                    skills.append(int(m2.group(1)))
        q2s[q_id] = sorted(set(skills))

    return q2s


def format_tags_item(tag_ids: List[int]) -> str:
    if not tag_ids:
        return '[]'
    if len(tag_ids) == 1:
        return str(tag_ids[0])
    return '[' + ','.join(str(x) for x in tag_ids) + ']'


def parse_user_id(info_part: str) -> int | None:
    """从学生基本信息部分提取原始学生编号，优先匹配 "<num>:<num>"，返回左侧数字。"""
    m = re.search(r'(\d+)\s*:\s*\d+', info_part)
    if m:
        return int(m.group(1))
    # 其次尝试直接提取第一个数字序列
    m2 = re.search(r'(\d+)', info_part)
    if m2:
        return int(m2.group(1))
    return None


def read_responses() -> List[Tuple[int, List[int], List[int]]]:
    """读取 HNU_SYS_2023.txt，返回列表：(raw_user_id, questions[], answers[])
    - questions 为原始题目编号（整数）
    - answers 为 0/1
    """
    if not RESP_FILE.exists():
        raise FileNotFoundError(f'未找到答题记录文件：{RESP_FILE}')

    records: List[Tuple[int, List[int], List[int]]] = []
    with open(RESP_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' not in line:
                # 非标准行，跳过
                continue
            info_part, seq_part = line.split('\t', 1)
            uid = parse_user_id(info_part)
            if uid is None:
                continue

            q_ids: List[int] = []
            ans: List[int] = []
            for item in seq_part.split(','):
                item = item.strip()
                if not item:
                    continue
                parts = item.split()
                if len(parts) < 2:
                    continue
                q_raw, a_raw = parts[0], parts[1]
                # 题目编号：提取数字
                mq = re.search(r'(\d+)', q_raw)
                if not mq:
                    continue
                qid = int(mq.group(1))
                a_val = 1 if str(a_raw).strip() == '1' else 0
                q_ids.append(qid)
                ans.append(a_val)

            if q_ids:
                records.append((uid, q_ids, ans))

    return records


def main():
    q2s_map = parse_question2skill_map()
    records = read_responses()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 扫描并构建重新编号映射
    user_set: Set[int] = set()
    problem_set: Set[int] = set()
    skill_set: Set[int] = set()

    for uid, qids, _ in records:
        user_set.add(uid)
        for q in qids:
            problem_set.add(q)
            # 按用户建议：知识点映射使用 原始题号+1
            for s in q2s_map.get(q + 1, []):
                skill_set.add(s)

    user_sorted = sorted(user_set)
    problem_sorted = sorted(problem_set)
    skill_sorted = sorted(skill_set)

    user_id_to_new: Dict[int, int] = {u: i + 1 for i, u in enumerate(user_sorted)}
    problem_id_to_new: Dict[int, int] = {q: i + 1 for i, q in enumerate(problem_sorted)}
    skill_id_to_new: Dict[int, int] = {s: i + 1 for i, s in enumerate(skill_sorted)}

    # 写出映射表
    maps_dir = OUT_DIR / 'map'
    maps_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        'old_user_id': user_sorted,
        'new_user_id': [user_id_to_new[u] for u in user_sorted]
    }).to_csv(maps_dir / 'user_id_map.csv', index=False, encoding='utf-8')

    pd.DataFrame({
        'old_problem_id': problem_sorted,
        'new_problem_id': [problem_id_to_new[q] for q in problem_sorted]
    }).to_csv(maps_dir / 'problem_id_map.csv', index=False, encoding='utf-8')

    pd.DataFrame({
        'skill_name': skill_sorted,
        'skill_id': [skill_id_to_new[s] for s in skill_sorted]
    }).to_csv(maps_dir / 'skill_name_map.csv', index=False, encoding='utf-8')

    # 写出序列文件
    with open(OUT_FILE, 'w', encoding='utf-8') as out:
        for uid, qids, ans in records:
            new_uid = user_id_to_new.get(uid)
            if new_uid is None:
                continue
            # 问题新编号
            new_qids = [str(problem_id_to_new[q]) for q in qids if q in problem_id_to_new]
            if not new_qids:
                continue
            # 知识点新编号（按题序）
            tag_items: List[str] = []
            for q in qids:
                # 按用户建议：知识点映射使用 原始题号+1
                skills = q2s_map.get(q + 1, [])
                mapped = [skill_id_to_new[s] for s in skills if s in skill_id_to_new]
                tag_items.append(format_tags_item(mapped))
            # 正误序列
            ans_seq = [str(int(x)) for x in ans[:len(new_qids)]]

            out.write(f"{new_uid}\n")
            out.write(','.join(new_qids) + '\n')
            out.write(';'.join(tag_items[:len(new_qids)]) + '\n')
            out.write(','.join(ans_seq) + '\n')

    print(f'映射表输出目录：{maps_dir}')
    print(f'序列文件已写出：{OUT_FILE}')

    # 统计信息
    num_users = len(user_id_to_new)
    num_problems = len(problem_id_to_new)
    num_skills = len(skill_id_to_new)
    
    # 计算序列长度统计
    seq_lens = []
    total_records = 0
    for uid, qids, _ in records:
        if uid in user_id_to_new:
            seq_len = len(qids)
            seq_lens.append(seq_len)
            total_records += seq_len
    
    min_seq_len = min(seq_lens) if seq_lens else 0
    max_seq_len = max(seq_lens) if seq_lens else 0
    mean_seq_len = sum(seq_lens) / len(seq_lens) if seq_lens else 0
    
    print(f"学生总数（user_id）：{num_users}")
    print(f"每个学生序列长度：最小值={min_seq_len}，最大值={max_seq_len}，平均值={mean_seq_len:.2f}")
    print(f"不同问题总数（problem_id）：{num_problems}")
    print(f"不同知识点总数（skill）：{num_skills}")
    print(f"总答题数：{total_records}")

    # 保存统计信息到JSON文件
    dataset_stats = {
        "dataset_name": "hnu_sys2023",
        "num_users": int(num_users),
        "sequence_length": {
            "min": int(min_seq_len),
            "max": int(max_seq_len),
            "mean": float(mean_seq_len)
        },
        "num_problems": int(num_problems),
        "num_skills": int(num_skills),
        "total_records": int(total_records)
    }
    
    stats_file = OUT_DIR / 'dataset_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)

    print(f"数据集统计信息已保存到 {stats_file}")


if __name__ == '__main__':
    main()


