"""
构建知识前驱关系图（重构版本 - 移除LLM，优化结构）

输入：
- user_group_sequences.txt：用户学习序列
- map/skill_name_map.csv：技能映射

输出：
- prereq_graph.json：边列表，包含分数和解释
- prereq_graph.csv：简化的边表
- cycle_removals.json：去环记录

说明：
- 基于时序优先和条件依赖两个统计指标构建前驱关系
- 支持断点续传和并行处理
- 自动去环以确保DAG结构
- 性能优化：
  * 当学生数超过 MAX_STUDENTS_LIMIT (默认10000) 时进行随机抽样
  * 一次性计算所有统计量，减少重复计算
  * 支持通过环境变量调整参数，如: PREREQ_MAX_STUDENTS=5000
"""

from __future__ import annotations

import json
import math
import os
import time
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import pandas as pd

# 确保控制台输出实时刷新
try:
    if hasattr(sys.stdout, 'reconfigure') and callable(getattr(sys.stdout, 'reconfigure', None)):
        getattr(sys.stdout, 'reconfigure')(line_buffering=True)
except Exception:
    pass


# ======================== 配置 ========================
class Config:
    """配置类，集中管理所有参数"""
    
    # 基础路径
    WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    PROCESSED_DIR = os.path.join(WORKSPACE_ROOT, "data", "processed")
    KNOWLEDGE_GRAPH_DIR = os.path.join(WORKSPACE_ROOT, "data", "knowledge_graph")
    
    # 动态设置的路径（由set_dataset方法设置）
    DATA_DIR: str = ""
    SEQUENCE_FILE: str = ""
    MAP_DIR: str = ""
    SKILL_MAP_FILE: str = ""
    GRAPH_DIR: str = ""
    OUTPUT_GRAPH_JSON: str = ""
    OUTPUT_GRAPH_CSV: str = ""
    OUTPUT_CYCLE_JSON: str = ""
    OUTPUT_NEO4J_NODES: str = ""
    OUTPUT_NEO4J_RELS: str = ""
    OUTPUT_NEO4J_CYPHER: str = ""
    CHECKPOINT_PKL: str = ""
    PROCESSED_PAIRS_PKL: str = ""
    _base_graph_dir: str = ""  # 基础图谱目录
    
    @classmethod
    def set_dataset(cls, dataset_name: str):
        """设置当前处理的数据集路径"""
        cls.DATA_DIR = os.path.join(cls.PROCESSED_DIR, dataset_name)
        cls.SEQUENCE_FILE = os.path.join(cls.DATA_DIR, "user_group_sequences.txt")
        cls.MAP_DIR = os.path.join(cls.DATA_DIR, "map")
        cls.SKILL_MAP_FILE = os.path.join(cls.MAP_DIR, "skill_name_map.csv")
        
        # 输出路径（不包含配置名称，在set_config中设置）
        cls._base_graph_dir = os.path.join(cls.KNOWLEDGE_GRAPH_DIR, dataset_name)
    
    @classmethod
    def set_config(cls, config: dict):
        """设置当前构建配置"""
        cls.W_TEMPORAL = config["w_temporal"]
        cls.W_CONDITIONAL = config["w_conditional"]
        cls.CURRENT_CONFIG_NAME = config["name"]
        
        # 设置输出路径（包含配置名称）
        cls.GRAPH_DIR = os.path.join(cls._base_graph_dir, config["name"])
        cls.OUTPUT_GRAPH_JSON = os.path.join(cls.GRAPH_DIR, "prereq_graph.json")
        cls.OUTPUT_GRAPH_CSV = os.path.join(cls.GRAPH_DIR, "prereq_graph.csv")
        cls.OUTPUT_CYCLE_JSON = os.path.join(cls.GRAPH_DIR, "cycle_removals.json")
        cls.OUTPUT_NEO4J_NODES = os.path.join(cls.GRAPH_DIR, "neo4j_nodes.csv")
        cls.OUTPUT_NEO4J_RELS = os.path.join(cls.GRAPH_DIR, "neo4j_relations.csv")
        cls.OUTPUT_NEO4J_CYPHER = os.path.join(cls.GRAPH_DIR, "neo4j_import.cypher")
        
        # 检查点文件
        cls.CHECKPOINT_PKL = os.path.join(cls.GRAPH_DIR, "edges_checkpoint.pkl")
        cls.PROCESSED_PAIRS_PKL = os.path.join(cls.GRAPH_DIR, "processed_pairs.pkl")
    
    # 阈值参数
    MIN_SUPPORT_STUDENTS = int(os.getenv("PREREQ_MIN_SUPPORT", "60"))
    MAX_STUDENTS_LIMIT = int(os.getenv("PREREQ_MAX_STUDENTS", "30000"))  # 最大统计学生数量
    TEMPORAL_THRESHOLD = float(os.getenv("PREREQ_TEMPORAL_THRESHOLD", "0.5"))
    CONDITIONAL_THRESHOLD = float(os.getenv("PREREQ_CONDITIONAL_THRESHOLD", "0.5"))
    
    # 性能优化说明：
    # - MIN_SUPPORT_STUDENTS: 最小支持学生数，低于此值的边对会被过滤
    # - MAX_STUDENTS_LIMIT: 当数据集学生数超过此值时会进行随机抽样，提高处理效率
    #   可通过环境变量 PREREQ_MAX_STUDENTS 设置，如: PREREQ_MAX_STUDENTS=5000
    
    # 图谱构建配置
    GRAPH_CONFIGS = [
        {"w_temporal": 0.1, "w_conditional": 0.9, "name": "w19"},
        {"w_temporal": 0.3, "w_conditional": 0.7, "name": "w37"},
        {"w_temporal": 0.5, "w_conditional": 0.5, "name": "w55"},
        {"w_temporal": 0.7, "w_conditional": 0.3, "name": "w73"},
        {"w_temporal": 0.9, "w_conditional": 0.1, "name": "w91"},
    ]
    
    # 当前配置（运行时设置）
    W_TEMPORAL = 0.5
    W_CONDITIONAL = 0.5
    CURRENT_CONFIG_NAME = "default"
    
    # 并行处理配置（高度优化）
    MAX_WORKERS = max(6, min(96, (os.cpu_count() or 4) * 3))  # 进一步提升并发度，使用CPU核心数的3倍
    BATCH_SIZE = int(os.getenv("PREREQ_BATCH_SIZE", "5000"))  # 大幅增大批次大小，减少任务调度开销
    SAVE_INTERVAL = int(os.getenv("PREREQ_SAVE_INTERVAL", "2000"))  # 减少保存频率，最小化I/O中断
    CHUNK_SIZE = int(os.getenv("PREREQ_CHUNK_SIZE", "100"))  # 新增：数据块大小，用于内存友好的分块处理
    PREFETCH_FACTOR = int(os.getenv("PREREQ_PREFETCH_FACTOR", "4"))  # 新增：预取因子，提高数据访问效率
    
    # 功能开关
    ENABLE_CHECKPOINT = os.getenv("PREREQ_ENABLE_CHECKPOINT", "1").lower() in ("1", "true", "yes")
    ENABLE_NEO4J = os.getenv("PREREQ_ENABLE_NEO4J", "1").lower() in ("1", "true", "yes")
    ENABLE_PREFILTER = os.getenv("PREREQ_ENABLE_PREFILTER", "1").lower() in ("1", "true", "yes")


# ======================== 数据结构 ========================
@dataclass
class EdgeScore:
    """边分数数据类"""
    from_id: int
    to_id: int
    from_name: str
    to_name: str
    temporal_precedence_score: float
    conditional_dependency_score: float
    final_score: float
    edge_type: str
    explanation: str
    
    @property
    def is_valid(self) -> bool:
        """边是否有效（不是none类型）"""
        return self.edge_type != "none"


# ======================== 工具函数 ========================
def log(msg: str) -> None:
    """日志输出函数"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ======================== 数据加载 ========================
class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def read_skill_map(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
        """读取技能映射"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到技能映射文件：{path}")
            
        df = pd.read_csv(path)
        required_cols = {"skill_name", "skill_id"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"skill_name_map.csv 缺少必要列：{required_cols}")
            
        id_to_name = {int(row["skill_id"]): str(row["skill_name"]) for _, row in df.iterrows()}
        name_to_id = {v: k for k, v in id_to_name.items()}
        return id_to_name, name_to_id
    
    @staticmethod
    def read_user_sequences(path: str) -> Dict[int, Dict[str, List]]:
        """读取用户学习序列，处理多技能问题"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到序列文件：{path}")

        user_data: Dict[int, Dict[str, List]] = {}
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if len(lines) % 4 != 0:
            raise ValueError("user_group_sequences.txt 行数不是4的倍数")

        for i in range(0, len(lines), 4):
            user_id = int(lines[i])
            # 跳过问题ID行
            skill_items = [x for x in lines[i + 2].split(';') if x]
            corrects = [int(float(x)) for x in lines[i + 3].split(',') if x]

            # 解析技能ID（处理多技能问题）
            expanded_skills = []
            expanded_corrects = []
            
            for skill_item, correct in zip(skill_items, corrects):
                if skill_item.startswith('[') and skill_item.endswith(']'):
                    # 处理多技能问题：[30,34,42,179,181]
                    try:
                        skill_list_str = skill_item[1:-1]  # 去掉方括号
                        skill_ids = [int(s.strip()) for s in skill_list_str.split(',') if s.strip()]
                        # 为每个技能创建一个交互记录
                        for skill_id in skill_ids:
                            expanded_skills.append(skill_id)
                            expanded_corrects.append(correct)
                    except ValueError as e:
                        log(f"警告：用户 {user_id} 的技能列表解析失败 '{skill_item}'：{e}，跳过")
                        continue
                else:
                    # 单技能问题
                    try:
                        skill_id = int(skill_item)
                        expanded_skills.append(skill_id)
                        expanded_corrects.append(correct)
                    except ValueError as e:
                        log(f"警告：用户 {user_id} 的技能ID解析失败 '{skill_item}'：{e}，跳过")
                        continue

            if expanded_skills:  # 只保存有有效技能的用户
                user_data[user_id] = {"skills": expanded_skills, "corrects": expanded_corrects}

        log(f"成功读取 {len(user_data)} 个用户的学习序列（后续将根据 MAX_STUDENTS_LIMIT={Config.MAX_STUDENTS_LIMIT} 进行可能的抽样）")
        return user_data


# ======================== 统计计算 ========================
class StatisticsCalculator:
    """统计指标计算器（内存优化版）"""
    
    def __init__(self, user_data: Dict[int, Dict[str, List]]):
        # 如果学生数超过限制，随机抽样
        if len(user_data) > Config.MAX_STUDENTS_LIMIT:
            log(f"原始学生数：{len(user_data)}，超过限制 {Config.MAX_STUDENTS_LIMIT}，将进行随机抽样")
            
            # 设置随机种子以保证可重现性
            random.seed(42)
            selected_users = random.sample(list(user_data.keys()), Config.MAX_STUDENTS_LIMIT)
            self.user_data = {uid: user_data[uid] for uid in selected_users}
            
            log(f"抽样后学生数：{len(self.user_data)}，抽样比例：{len(self.user_data)/len(user_data):.1%}")
        else:
            self.user_data = user_data
            log(f"学生数量：{len(self.user_data)}，未超过限制")
            
        self.user_first_mastery = self._compute_first_mastery_indices()
        # 新增：预计算技能映射和缓存结构
        self._skill_cache = self._build_skill_cache()
        self._mastery_cache = self._build_mastery_cache()
    
    def _build_skill_cache(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """构建技能级别的缓存，提高数据访问效率"""
        skill_cache = defaultdict(list)
        
        for user_id, record in self.user_data.items():
            for idx, (skill_id, correct) in enumerate(zip(record["skills"], record["corrects"])):
                skill_cache[skill_id].append((user_id, idx, correct))
        
        # 对每个技能按用户ID排序，优化内存局部性
        for skill_id in skill_cache:
            skill_cache[skill_id].sort(key=lambda x: x[0])  # 按user_id排序
        
        return dict(skill_cache)
    
    def _build_mastery_cache(self) -> Dict[int, Dict[int, int]]:
        """构建掌握索引缓存，优化查找性能"""
        mastery_cache = defaultdict(dict)
        
        for user_id, first_mastery in self.user_first_mastery.items():
            mastery_cache[user_id] = first_mastery
        
        return dict(mastery_cache)
    
    def _compute_first_mastery_indices(self) -> Dict[int, Dict[int, int]]:
        """计算每个用户对每个技能的首次掌握索引"""
        result: Dict[int, Dict[int, int]] = {}
        
        for user_id, record in self.user_data.items():
            first_idx: Dict[int, int] = {}
            for idx, (skill_id, correct) in enumerate(zip(record["skills"], record["corrects"])):
                if correct == 1 and skill_id not in first_idx:
                    first_idx[skill_id] = idx
            result[user_id] = first_idx
            
        return result
    
    def temporal_precedence_score(self, from_skill: int, to_skill: int) -> Tuple[float, int, int]:
        """
        计算时序优先分数（内存优化版）
        返回: (分数, 满足条件的学生数, 总学生数)
        """
        num_total = 0
        num_yes = 0
        
        # 使用缓存的掌握数据进行快速查找
        for user_id, user_mastery in self._mastery_cache.items():
            if to_skill in user_mastery:  # 必须掌握目标技能
                num_total += 1
                to_idx = user_mastery[to_skill]
                if from_skill in user_mastery and user_mastery[from_skill] < to_idx:
                    num_yes += 1
                    
        score = (num_yes / num_total) if num_total > 0 else 0.0
        return score, num_yes, num_total
    
    def conditional_dependency_score(self, from_skill: int, to_skill: int) -> Tuple[float, float, float, int, int]:
        """
        计算条件依赖分数（内存优化版）
        返回: (分数, 掌握后正确率, 掌握前正确率, 掌握后答题数, 掌握前答题数)
        """
        correct_after = total_after = 0
        correct_before = total_before = 0

        # 使用技能缓存进行快速访问
        if to_skill in self._skill_cache:
            for user_id, idx, correct in self._skill_cache[to_skill]:
                first_idx_from = self._mastery_cache[user_id].get(from_skill, math.inf)
                
                if idx > first_idx_from:  # 掌握from_skill之后
                    total_after += 1
                    if correct == 1:
                        correct_after += 1
                else:  # 掌握from_skill之前
                    total_before += 1
                    if correct == 1:
                        correct_before += 1

        p_after = (correct_after / total_after) if total_after > 0 else 0.0
        p_before = (correct_before / total_before) if total_before > 0 else 0.0
        
        # 归一化到[0,1]区间
        score = (p_after - p_before + 1.0) / 2.0
        return score, p_after, p_before, total_after, total_before


# ======================== 边处理 ========================
class EdgeProcessor:
    """边处理器"""
    
    def __init__(self, stats_calc: StatisticsCalculator, id_to_name: Dict[int, str]):
        self.stats_calc = stats_calc
        self.id_to_name = id_to_name
    
    def process_edge_pair(self, from_id: int, to_id: int) -> Optional[EdgeScore]:
        """处理单个边对"""
        if from_id == to_id:
            return None
            
        # 计算统计指标
        temporal_score, num_yes, num_total = self.stats_calc.temporal_precedence_score(from_id, to_id)
        
        # 支持度过滤
        if num_total < Config.MIN_SUPPORT_STUDENTS:
            return None
            
        cond_score, p_after, p_before, n_after, n_before = self.stats_calc.conditional_dependency_score(from_id, to_id)
        
        # 预过滤检查
        if Config.ENABLE_PREFILTER:
            if temporal_score < Config.TEMPORAL_THRESHOLD or cond_score < Config.CONDITIONAL_THRESHOLD:
                return None
        
        # 计算最终分数
        final_score = Config.W_TEMPORAL * temporal_score + Config.W_CONDITIONAL * cond_score
        edge_type = self._classify_edge(final_score)
        
        if edge_type == "none":
            return None
            
        # 生成解释
        explanation = (
            f"时序优先: {num_yes}/{num_total} 学生在掌握 {self.id_to_name[to_id]} 前已掌握 {self.id_to_name[from_id]}; "
            f"条件依赖: 掌握后正确率={p_after:.2f} (n={n_after}), 掌握前正确率={p_before:.2f} (n={n_before})"
        )
        
        return EdgeScore(
            from_id=from_id,
            to_id=to_id,
            from_name=self.id_to_name[from_id],
            to_name=self.id_to_name[to_id],
            temporal_precedence_score=temporal_score,
            conditional_dependency_score=cond_score,
            final_score=final_score,
            edge_type=edge_type,
            explanation=explanation
        )
    
    @staticmethod
    def _classify_edge(final_score: float) -> str:
        """分类边的强度（无阈值过滤）"""
        return "valid"


# ======================== 检查点管理 ========================
class CheckpointManager:
    """检查点管理器"""
    
    @staticmethod
    def load_checkpoint() -> Tuple[List[EdgeScore], Set[Tuple[int, int]]]:
        """加载检查点"""
        edges = []
        processed_pairs = set()
        
        if not Config.ENABLE_CHECKPOINT:
            return edges, processed_pairs
            
        try:
            if os.path.exists(Config.CHECKPOINT_PKL):
                with open(Config.CHECKPOINT_PKL, 'rb') as f:
                    edges = pickle.load(f)
                log(f"从检查点加载 {len(edges)} 条已计算边")
                    
            if os.path.exists(Config.PROCESSED_PAIRS_PKL):
                with open(Config.PROCESSED_PAIRS_PKL, 'rb') as f:
                    processed_pairs = pickle.load(f)
                log(f"从检查点加载 {len(processed_pairs)} 个已处理边对")
                
        except Exception as e:
            log(f"检查点加载失败：{e}，从头开始")
            edges = []
            processed_pairs = set()
            
        return edges, processed_pairs
    
    @staticmethod
    def save_checkpoint(edges: List[EdgeScore], processed_pairs: Set[Tuple[int, int]]):
        """保存检查点"""
        if not Config.ENABLE_CHECKPOINT:
            return
            
        try:
            ensure_dir(Config.CHECKPOINT_PKL)
            
            with open(Config.CHECKPOINT_PKL, 'wb') as f:
                pickle.dump(edges, f)
            
            with open(Config.PROCESSED_PAIRS_PKL, 'wb') as f:
                pickle.dump(processed_pairs, f)
                
            log(f"已保存检查点：{len(edges)} 条边，{len(processed_pairs)} 个已处理对")
        except Exception as e:
            log(f"检查点保存失败：{e}")


# ======================== 去环算法 ========================
class CycleBreaker:
    """去环处理器"""
    
    @staticmethod
    def detect_cycle(adj: Dict[int, Set[int]], skills: List[int]) -> Optional[List[int]]:
        """检测环，返回环的节点序列或None"""
        visited = {s: 0 for s in skills}  # 0: 未访问, 1: 正在访问, 2: 已完成
        stack = []

        def dfs(node: int) -> Optional[List[int]]:
            visited[node] = 1
            stack.append(node)
            
            for neighbor in adj.get(node, set()):
                if visited[neighbor] == 0:
                    cycle = dfs(neighbor)
                    if cycle is not None:
                        return cycle
                elif visited[neighbor] == 1:
                    # 找到回边，构造环
                    if neighbor in stack:
                        idx = stack.index(neighbor)
                        return stack[idx:] + [neighbor]
            
            stack.pop()
            visited[node] = 2
            return None

        for skill in skills:
            if visited[skill] == 0:
                cycle = dfs(skill)
                if cycle is not None:
                    return cycle
                    
        return None
    
    @staticmethod
    def break_cycles(edges: List[EdgeScore]) -> Tuple[List[EdgeScore], List[Dict]]:
        """去环处理，返回保留的边和删除记录"""
        # 构建邻接表和映射
        skills = sorted({e.from_id for e in edges} | {e.to_id for e in edges})
        adj = defaultdict(set)
        score_map = {}
        edge_map = {}

        for edge in edges:
            adj[edge.from_id].add(edge.to_id)
            score_map[(edge.from_id, edge.to_id)] = edge.final_score
            edge_map[(edge.from_id, edge.to_id)] = edge

        removals = []
        round_idx = 0
        
        # 循环去环
        while True:
            cycle = CycleBreaker.detect_cycle(adj, skills)
            if not cycle:
                break

            # 找环中分数最低的边
            cycle_edges = list(zip(cycle[:-1], cycle[1:]))
            worst_edge = min(cycle_edges, key=lambda x: score_map.get(x, -1.0))
            worst_edge_obj = edge_map[worst_edge]

            # 记录删除信息
            removal = {
                "remove_from": worst_edge_obj.from_id,
                "remove_to": worst_edge_obj.to_id,
                "remove_score": worst_edge_obj.final_score,
                "reason": "cycle breaking (lowest score in cycle)",
                "cycle_path": cycle,
                "round": round_idx,
            }
            removals.append(removal)
            
            log(f"[去环] 第{round_idx}轮 删除边 {worst_edge_obj.from_id}->{worst_edge_obj.to_id} "
                f"(score={worst_edge_obj.final_score:.4f}), 环路径={cycle}")
            
            # 删除边
            adj[worst_edge[0]].discard(worst_edge[1])
            score_map.pop(worst_edge, None)
            edge_map.pop(worst_edge, None)
            round_idx += 1

        kept_edges = list(edge_map.values())
        return kept_edges, removals


# ======================== 输出生成 ========================
class OutputGenerator:
    """输出文件生成器"""
    
    @staticmethod
    def save_outputs(edges: List[EdgeScore], removals: List[Dict]):
        """保存所有输出文件"""
        ensure_dir(Config.OUTPUT_GRAPH_JSON)
        
        OutputGenerator._save_json(edges)
        OutputGenerator._save_csv(edges)
        OutputGenerator._save_cycle_removals(removals)
        
        if Config.ENABLE_NEO4J:
            OutputGenerator._save_neo4j_files(edges)
    
    @staticmethod
    def _save_json(edges: List[EdgeScore]):
        """保存JSON格式的边文件"""
        edges_json = [
            {
                "from_id": e.from_id,
                "to_id": e.to_id,
                "from_name": e.from_name,
                "to_name": e.to_name,
                "type": e.edge_type,
                "final_score": round(e.final_score, 4),
                "config": Config.CURRENT_CONFIG_NAME,
                "weights": {
                    "temporal": Config.W_TEMPORAL,
                    "conditional": Config.W_CONDITIONAL
                },
                "components": {
                    "temporal_precedence_score": round(e.temporal_precedence_score, 4),
                    "conditional_dependency_score": round(e.conditional_dependency_score, 4),
                },
                "explanation": e.explanation,
            }
            for e in sorted(edges, key=lambda x: -x.final_score)
        ]
        
        with open(Config.OUTPUT_GRAPH_JSON, "w", encoding="utf-8") as f:
            json.dump(edges_json, f, ensure_ascii=False, indent=2)
        log(f"已保存JSON边文件：{Config.OUTPUT_GRAPH_JSON}，共 {len(edges_json)} 条边")
    
    @staticmethod
    def _save_csv(edges: List[EdgeScore]):
        """保存CSV格式的边文件"""
        df = pd.DataFrame([
            {
                "from_id": e.from_id,
                "to_id": e.to_id,
                "from_name": e.from_name,
                "to_name": e.to_name,
                "type": e.edge_type,
                "final_score": e.final_score,
            }
            for e in edges
        ])
        
        df.to_csv(Config.OUTPUT_GRAPH_CSV, index=False, encoding="utf-8")
        log(f"已保存CSV边文件：{Config.OUTPUT_GRAPH_CSV}，形状={df.shape}")
    
    @staticmethod
    def _save_cycle_removals(removals: List[Dict]):
        """保存去环记录"""
        with open(Config.OUTPUT_CYCLE_JSON, "w", encoding="utf-8") as f:
            json.dump(removals, f, ensure_ascii=False, indent=2)
        log(f"已保存去环记录：{Config.OUTPUT_CYCLE_JSON}，共 {len(removals)} 条记录")
    
    @staticmethod
    def _save_neo4j_files(edges: List[EdgeScore]):
        """保存Neo4j导入文件"""
        # 节点文件
        nodes = {}
        for e in edges:
            nodes[e.from_id] = e.from_name
            nodes[e.to_id] = e.to_name
            
        nodes_df = pd.DataFrame([
            {"skill_id": sid, "skill_name": sname} 
            for sid, sname in sorted(nodes.items())
        ])
        nodes_df.to_csv(Config.OUTPUT_NEO4J_NODES, index=False, encoding="utf-8")
        log(f"已保存Neo4j节点：{Config.OUTPUT_NEO4J_NODES}，共 {len(nodes_df)} 个")

        # 关系文件
        rels_df = pd.DataFrame([
            {"from_id": e.from_id, "to_id": e.to_id, "relation": e.edge_type}
            for e in edges
        ])
        rels_df.to_csv(Config.OUTPUT_NEO4J_RELS, index=False, encoding="utf-8")
        log(f"已保存Neo4j关系：{Config.OUTPUT_NEO4J_RELS}，共 {len(rels_df)} 条")

        # Cypher导入脚本
        cypher = f"""// 清空数据库（谨慎使用）
// MATCH ()-[r]-() DELETE r; MATCH (n) DELETE n;

// 导入节点
LOAD CSV WITH HEADERS FROM 'file:///{os.path.basename(Config.OUTPUT_NEO4J_NODES)}' AS row
MERGE (s:Skill {{id: toInteger(row.skill_id)}})
SET s.name = row.skill_name;

// 导入关系
LOAD CSV WITH HEADERS FROM 'file:///{os.path.basename(Config.OUTPUT_NEO4J_RELS)}' AS row
MATCH (a:Skill {{id: toInteger(row.from_id)}}), (b:Skill {{id: toInteger(row.to_id)}})
MERGE (a)-[r:PREREQ]->(b)
SET r.relation = row.relation;
"""
        with open(Config.OUTPUT_NEO4J_CYPHER, "w", encoding="utf-8") as f:
            f.write(cypher)
        log(f"已保存Neo4j导入脚本：{Config.OUTPUT_NEO4J_CYPHER}")


# ======================== 数据集管理器 ========================
class DatasetManager:
    """数据集管理器，用于发现和验证数据集"""
    
    @staticmethod
    def get_available_datasets() -> List[str]:
        """获取所有可用的数据集名称"""
        datasets = []
        if not os.path.exists(Config.PROCESSED_DIR):
            return datasets
            
        for item in os.listdir(Config.PROCESSED_DIR):
            dataset_path = os.path.join(Config.PROCESSED_DIR, item)
            if os.path.isdir(dataset_path):
                # 检查必要文件是否存在
                sequence_file = os.path.join(dataset_path, "user_group_sequences.txt")
                skill_map_file = os.path.join(dataset_path, "map", "skill_name_map.csv")
                
                if os.path.exists(sequence_file) and os.path.exists(skill_map_file):
                    datasets.append(item)
                else:
                    log(f"跳过数据集 {item}：缺少必要文件")
        
        return sorted(datasets)
    
    @staticmethod
    def validate_dataset(dataset_name: str) -> bool:
        """验证数据集是否有效"""
        Config.set_dataset(dataset_name)
        
        if not os.path.exists(Config.SEQUENCE_FILE):
            log(f"数据集 {dataset_name} 缺少序列文件：{Config.SEQUENCE_FILE}")
            return False
            
        if not os.path.exists(Config.SKILL_MAP_FILE):
            log(f"数据集 {dataset_name} 缺少技能映射文件：{Config.SKILL_MAP_FILE}")
            return False
            
        return True
    
    @staticmethod
    def is_dataset_completed(dataset_name: str) -> bool:
        """检查数据集是否已经完成所有配置的处理"""
        Config.set_dataset(dataset_name)
        
        # 检查所有配置是否都已完成
        completed_configs = []
        missing_configs = []
        
        for config in Config.GRAPH_CONFIGS:
            # 临时设置配置以获取正确的输出路径
            Config.set_config(config)
            
            # 检查关键输出文件是否存在
            if (os.path.exists(Config.OUTPUT_GRAPH_JSON) and 
                os.path.exists(Config.OUTPUT_GRAPH_CSV) and 
                os.path.exists(Config.OUTPUT_CYCLE_JSON)):
                completed_configs.append(config['name'])
            else:
                missing_configs.append(config['name'])
        
        is_completed = len(missing_configs) == 0
        
        if is_completed:
            log(f"数据集 {dataset_name} 已完成所有 {len(Config.GRAPH_CONFIGS)} 个配置的处理")
        else:
            log(f"数据集 {dataset_name} 缺少 {len(missing_configs)} 个配置：{', '.join(missing_configs[:3])}{'...' if len(missing_configs) > 3 else ''}")
        
        return is_completed
    
    @staticmethod
    def get_missing_configs(dataset_name: str) -> List[Dict]:
        """获取数据集中缺失的配置列表"""
        Config.set_dataset(dataset_name)
        missing_configs = []
        
        for config in Config.GRAPH_CONFIGS:
            Config.set_config(config)
            
            # 检查关键输出文件是否存在
            if not (os.path.exists(Config.OUTPUT_GRAPH_JSON) and 
                   os.path.exists(Config.OUTPUT_GRAPH_CSV) and 
                   os.path.exists(Config.OUTPUT_CYCLE_JSON)):
                missing_configs.append(config)
        
        return missing_configs


# ======================== 主处理类 ========================
class PrereqGraphBuilder:
    """前驱关系图构建器主类"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.checkpoint_manager = CheckpointManager()
        self.output_generator = OutputGenerator()
        self.dataset_manager = DatasetManager()
    
    def compute_base_scores(self, user_data: Dict[int, Dict[str, List]], id_to_name: Dict[int, str]) -> Dict[Tuple[int, int], Dict]:
        """一次性计算所有边对的基础统计量（深度优化版）"""
        stats_calc = StatisticsCalculator(user_data)
        
        # 生成候选边对
        skills = sorted(id_to_name.keys())
        candidate_pairs = [(a, b) for a, b in combinations(skills, 2)] + \
                         [(b, a) for a, b in combinations(skills, 2)]
        
        total_pairs = len(candidate_pairs)
        log(f"总候选边对数：{total_pairs}")
        log(f"并行处理配置：线程数={Config.MAX_WORKERS}, 批次大小={Config.BATCH_SIZE}, 块大小={Config.CHUNK_SIZE}")
        
        base_scores = {}
        processed_count = 0
        start_time = time.perf_counter()
        
        # 优化的分批处理：使用更大的批次和更好的内存管理
        for batch_idx, i in enumerate(range(0, len(candidate_pairs), Config.BATCH_SIZE)):
            batch_pairs = candidate_pairs[i:i + Config.BATCH_SIZE]
            batch_start = time.perf_counter()
            
            log(f"批次 [{batch_idx+1}] 边对 {i+1}-{min(i+Config.BATCH_SIZE, len(candidate_pairs))}/{len(candidate_pairs)}")
            
            # 使用更高效的并行处理策略
            batch_results = self._process_batch_optimized(stats_calc, batch_pairs)
            
            # 合并结果
            for pair, result in batch_results.items():
                if result is not None:
                    base_scores[pair] = result
                    processed_count += 1
            
            batch_time = time.perf_counter() - batch_start
            speed = len(batch_pairs) / batch_time if batch_time > 0 else 0
            log(f"批次完成：用时 {batch_time:.2f}s, 速度 {speed:.0f} 边对/秒, 有效结果 {len(batch_results)} 个")
            
            # 内存管理：定期清理临时变量
            if batch_idx % 10 == 0 and batch_idx > 0:
                import gc
                gc.collect()
        
        total_time = time.perf_counter() - start_time
        avg_speed = processed_count / total_time if total_time > 0 else 0
        log(f"统计量计算完成：总用时 {total_time:.2f}s, 平均速度 {avg_speed:.0f} 边对/秒")
        log(f"有效边对：{len(base_scores)}/{total_pairs} ({len(base_scores)/total_pairs:.1%})")
        
        return base_scores
    
    def _process_batch_optimized(self, stats_calc: StatisticsCalculator, batch_pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Optional[Dict]]:
        """优化的批次处理函数"""
        batch_results = {}
        
        # 使用更高效的线程池配置
        with ThreadPoolExecutor(
            max_workers=Config.MAX_WORKERS,
            thread_name_prefix="StatCalc"
        ) as executor:
            # 分块提交任务，提高调度效率
            chunk_futures = {}
            
            for chunk_start in range(0, len(batch_pairs), Config.CHUNK_SIZE):
                chunk_pairs = batch_pairs[chunk_start:chunk_start + Config.CHUNK_SIZE]
                future = executor.submit(self._process_chunk, stats_calc, chunk_pairs)
                chunk_futures[future] = chunk_pairs
            
            # 收集结果
            for future in as_completed(chunk_futures):
                try:
                    chunk_results = future.result(timeout=30)  # 设置超时
                    batch_results.update(chunk_results)
                except Exception as e:
                    chunk_pairs = chunk_futures[future]
                    log(f"警告：处理块 {len(chunk_pairs)} 个边对失败：{e}")
        
        return batch_results
    
    def _process_chunk(self, stats_calc: StatisticsCalculator, chunk_pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Optional[Dict]]:
        """处理单个数据块"""
        chunk_results = {}
        
        for from_id, to_id in chunk_pairs:
            try:
                result = self._compute_pair_stats(stats_calc, from_id, to_id)
                chunk_results[(from_id, to_id)] = result
            except Exception as e:
                log(f"警告：计算边对 ({from_id}, {to_id}) 失败：{e}")
                chunk_results[(from_id, to_id)] = None
        
        return chunk_results
    
    def _compute_pair_stats(self, stats_calc: StatisticsCalculator, from_id: int, to_id: int) -> Optional[Dict]:
        """计算单个边对的统计量（性能优化版）"""
        if from_id == to_id:
            return None
        
        # 预先检查数据可用性，避免不必要计算
        if (from_id not in stats_calc._skill_cache and 
            to_id not in stats_calc._skill_cache):
            return None
            
        try:
            # 计算统计指标
            temporal_score, num_yes, num_total = stats_calc.temporal_precedence_score(from_id, to_id)
            
            # 支持度过滤（提前过滤，减少后续计算）
            if num_total < Config.MIN_SUPPORT_STUDENTS:
                return None
                
            cond_score, p_after, p_before, n_after, n_before = stats_calc.conditional_dependency_score(from_id, to_id)
            
            # 基础预过滤（使用较低的阈值，避免过度过滤）
            if temporal_score < Config.TEMPORAL_THRESHOLD or cond_score < Config.CONDITIONAL_THRESHOLD:
                return None
            
            return {
                'temporal_score': temporal_score,
                'conditional_score': cond_score,
                'temporal_stats': (num_yes, num_total),
                'conditional_stats': (p_after, p_before, n_after, n_before)
            }
            
        except Exception as e:
            # 记录异常但不中断处理
            return None
    
    def generate_edges_from_scores(self, base_scores: Dict[Tuple[int, int], Dict], config: Dict, id_to_name: Dict[int, str]) -> List[EdgeScore]:
        """基于预计算的统计量生成特定配置的边列表"""
        edges = []
        
        for (from_id, to_id), stats in base_scores.items():
            # 计算加权最终分数
            final_score = config['w_temporal'] * stats['temporal_score'] + config['w_conditional'] * stats['conditional_score']
            
            # 无阈值过滤，直接生成边
            if True:
                # 生成解释
                num_yes, num_total = stats['temporal_stats']
                p_after, p_before, n_after, n_before = stats['conditional_stats']
                
                explanation = (
                    f"时序优先: {num_yes}/{num_total} 学生在掌握 {id_to_name[to_id]} 前已掌握 {id_to_name[from_id]}; "
                    f"条件依赖: 掌握后正确率={p_after:.2f} (n={n_after}), 掌握前正确率={p_before:.2f} (n={n_before})"
                )
                
                edge = EdgeScore(
                    from_id=from_id,
                    to_id=to_id,
                    from_name=id_to_name[from_id],
                    to_name=id_to_name[to_id],
                    temporal_precedence_score=stats['temporal_score'],
                    conditional_dependency_score=stats['conditional_score'],
                    final_score=final_score,
                    edge_type="valid",
                    explanation=explanation
                )
                edges.append(edge)
        
        # 按最终分数从高到低排序
        edges.sort(key=lambda x: x.final_score, reverse=True)
        return edges
    
    def process_single_dataset(self, dataset_name: str, force_reprocess: bool = False) -> bool:
        """处理单个数据集，一次计算统计量，为每个配置生成图谱"""
        try:
            log(f"\n{'='*60}")
            log(f"开始处理数据集：{dataset_name}")
            log(f"{'='*60}")
            
            # 设置当前数据集路径
            Config.set_dataset(dataset_name)
            
            # 验证数据集
            if not self.dataset_manager.validate_dataset(dataset_name):
                log(f"数据集 {dataset_name} 验证失败，跳过")
                return False
            
            # 检查是否已完成处理
            if not force_reprocess and self.dataset_manager.is_dataset_completed(dataset_name):
                log(f"✓ 数据集 {dataset_name} 已完成所有配置的处理，跳过")
                return True
            
            # 获取缺失的配置
            missing_configs = self.dataset_manager.get_missing_configs(dataset_name)
            if not force_reprocess and len(missing_configs) == 0:
                log(f"✓ 数据集 {dataset_name} 所有配置都已存在，跳过")
                return True
            
            # 确定需要处理的配置
            configs_to_process = Config.GRAPH_CONFIGS if force_reprocess else missing_configs
            
            if not force_reprocess and missing_configs:
                log(f"发现 {len(missing_configs)} 个未完成的配置，将继续处理")
            
            log("[1/4] 读取映射与序列...")
            id_to_name, _ = self.data_loader.read_skill_map(Config.SKILL_MAP_FILE)
            user_data = self.data_loader.read_user_sequences(Config.SEQUENCE_FILE)
            
            # 打印原始数据统计
            num_users = len(user_data)
            total_interactions = sum(len(v["skills"]) for v in user_data.values())
            log(f"技能数：{len(id_to_name)}，原始用户数：{num_users}，总作答数：{total_interactions}")

            log("[2/4] 计算所有统计量（一次性计算）...")
            # 一次性计算所有边对的基础统计量（在 StatisticsCalculator 内部会进行抽样）
            base_scores = self.compute_base_scores(user_data, id_to_name)
            log(f"计算完成，共 {len(base_scores)} 个边对的统计量")
            
            log(f"[3/4] 为 {len(configs_to_process)} 个配置生成知识图谱...")
            success_configs = []
            failed_configs = []
            
            for i, config in enumerate(configs_to_process, 1):
                log(f"\n处理配置 [{i}/{len(configs_to_process)}]: {config['name']} "
                    f"(时序权重:{config['w_temporal']:.1f}, 条件权重:{config['w_conditional']:.1f})")
                
                try:
                    # 设置当前配置
                    Config.set_config(config)
                    
                    # 基于预计算的统计量生成边
                    edges = self.generate_edges_from_scores(base_scores, config, id_to_name)
                    log(f"  生成 {len(edges)} 条边")

                    # 去环
                    edges_kept, removals = CycleBreaker.break_cycles(edges)
                    log(f"  去环后边数量：{len(edges_kept)}，删除 {len(removals)} 条")

                    # 保存结果
                    self.output_generator.save_outputs(edges_kept, removals)
                    
                    log(f"  ✓ 配置 {config['name']} 完成")
                    success_configs.append(config['name'])
                    
                except Exception as e:
                    log(f"  ✗ 配置 {config['name']} 失败：{e}")
                    failed_configs.append(config['name'])
                    continue
            
            log("[4/4] 数据集处理完成")
            log(f"成功配置: {len(success_configs)}/{len(configs_to_process)}")
            if success_configs:
                log(f"  ✓ {', '.join(success_configs)}")
            if failed_configs:
                log(f"  ✗ {', '.join(failed_configs)}")
            
            return len(success_configs) > 0
            
        except Exception as e:
            log(f"处理数据集 {dataset_name} 时发生错误：{e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_single_dataset(self, dataset_name: str, force_reprocess: bool = False):
        """处理单个指定的数据集"""
        start_time = time.perf_counter()
        log(f"开始处理指定数据集：{dataset_name}")
        log(f"性能配置：最大线程数={Config.MAX_WORKERS}, 批次大小={Config.BATCH_SIZE}, 块大小={Config.CHUNK_SIZE}")
        
        # 获取所有可用数据集
        datasets = self.dataset_manager.get_available_datasets()
        
        if not datasets:
            log("未找到任何有效的数据集")
            return
        
        # 检查指定的数据集是否存在
        if dataset_name not in datasets:
            log(f"错误：数据集 '{dataset_name}' 不存在或无效")
            log(f"可用的数据集：{', '.join(datasets)}")
            return
        
        log(f"找到指定数据集：{dataset_name}")
        
        # 检查数据集状态
        if self.dataset_manager.is_dataset_completed(dataset_name):
            if not force_reprocess:
                log(f"✓ 数据集 {dataset_name} 已完成所有配置的处理")
                log(f"如需强制重新处理，请使用 --force 参数")
                return
            else:
                log(f"强制重新处理模式：将重新处理数据集 {dataset_name}")
        else:
            missing_configs = self.dataset_manager.get_missing_configs(dataset_name)
            if missing_configs:
                log(f"发现 {len(missing_configs)} 个未完成的配置，将继续处理")
        
        # 处理数据集
        dataset_start = time.perf_counter()
        success = self.process_single_dataset(dataset_name, force_reprocess)
        dataset_time = time.perf_counter() - dataset_start
        total_time = time.perf_counter() - start_time
        
        # 输出结果统计
        log(f"\n{'='*60}")
        if success:
            log(f"✓ 数据集 {dataset_name} 处理成功！")
        else:
            log(f"✗ 数据集 {dataset_name} 处理失败")
        
        log(f"\n处理统计：")
        log(f"  数据集: {dataset_name}")
        log(f"  处理时间: {dataset_time:.1f}s")
        log(f"  总运行时间: {total_time:.1f}s")
        log(f"  配置数量: {len(Config.GRAPH_CONFIGS)} 个")
        
        if success:
            avg_config_time = dataset_time / len(Config.GRAPH_CONFIGS)
            log(f"  平均每配置: {avg_config_time:.1f}s")
        
        log(f"\n结果保存在：{Config.KNOWLEDGE_GRAPH_DIR}/{dataset_name}/")
        log(f"目录结构：[config_name]/prereq_graph.json")
        log(f"可用配置：")
        for config in Config.GRAPH_CONFIGS:
            log(f"  - {config['name']}: 时序权重={config['w_temporal']:.1f}, 条件权重={config['w_conditional']:.1f}")
        log(f"{'='*60}")
    
    def run(self, force_reprocess: bool = False):
        """主运行函数 - 处理所有数据集（性能优化版）"""
        start_time = time.perf_counter()
        log("开始批量构建知识前驱图...")
        log(f"性能配置：最大线程数={Config.MAX_WORKERS}, 批次大小={Config.BATCH_SIZE}, 块大小={Config.CHUNK_SIZE}")
        
        # 获取所有可用数据集
        datasets = self.dataset_manager.get_available_datasets()
        
        if not datasets:
            log("未找到任何有效的数据集")
            return
        
        log(f"发现 {len(datasets)} 个数据集：{', '.join(datasets)}")
        
        # 预检查数据集完成情况
        completed_datasets = []
        partial_datasets = []
        pending_datasets = []
        
        for dataset_name in datasets:
            if self.dataset_manager.is_dataset_completed(dataset_name):
                completed_datasets.append(dataset_name)
            else:
                missing_configs = self.dataset_manager.get_missing_configs(dataset_name)
                if len(missing_configs) == len(Config.GRAPH_CONFIGS):
                    pending_datasets.append(dataset_name)
                else:
                    partial_datasets.append(dataset_name)
        
        # 打印数据集状态统计
        log(f"\n数据集状态统计：")
        log(f"  ✓ 已完成: {len(completed_datasets)} 个 - {', '.join(completed_datasets) if completed_datasets else '无'}")
        log(f"  ▷ 部分完成: {len(partial_datasets)} 个 - {', '.join(partial_datasets) if partial_datasets else '无'}")
        log(f"  ○ 未开始: {len(pending_datasets)} 个 - {', '.join(pending_datasets) if pending_datasets else '无'}")
        
        if not force_reprocess and len(completed_datasets) == len(datasets):
            log(f"\n✓ 所有数据集都已完成处理，无需重新运行")
            log(f"如需强制重新处理，请使用 force_reprocess=True 参数")
            return
        
        # 确定需要处理的数据集
        if force_reprocess:
            datasets_to_process = datasets
            log(f"\n强制重新处理模式：将处理所有 {len(datasets)} 个数据集")
        else:
            datasets_to_process = partial_datasets + pending_datasets
            if completed_datasets:
                log(f"\n跳过已完成的 {len(completed_datasets)} 个数据集")
            log(f"将处理 {len(datasets_to_process)} 个数据集")
        
        if not datasets_to_process:
            log("无需处理的数据集")
            return
        
        # 处理结果统计
        success_count = 0
        failed_datasets = []
        skipped_count = len(completed_datasets) if not force_reprocess else 0
        total_processing_time = 0
        
        # 依次处理每个数据集
        for i, dataset_name in enumerate(datasets_to_process, 1):
            dataset_start = time.perf_counter()
            log(f"\n进度：[{i}/{len(datasets_to_process)}] 处理数据集 {dataset_name}")
            
            if self.process_single_dataset(dataset_name, force_reprocess):
                success_count += 1
                dataset_time = time.perf_counter() - dataset_start
                total_processing_time += dataset_time
                log(f"✓ 数据集 {dataset_name} 处理成功，用时 {dataset_time:.1f}s")
            else:
                failed_datasets.append(dataset_name)
                log(f"✗ 数据集 {dataset_name} 处理失败")
        
        total_time = time.perf_counter() - start_time
        
        # 输出最终统计
        log(f"\n{'='*60}")
        log(f"批量处理完成！")
        log(f"数据集处理统计：")
        log(f"  ✓ 成功: {success_count} 个")
        if skipped_count > 0:
            log(f"  ↻ 跳过（已完成）: {skipped_count} 个")
        if failed_datasets:
            log(f"  ✗ 失败: {len(failed_datasets)} 个 - {', '.join(failed_datasets)}")
        
        total_processed = success_count + len(failed_datasets)
        total_success = success_count + skipped_count
        log(f"  总计: {total_success}/{len(datasets)} 个数据集已完成")
        
        # 性能统计
        log(f"\n性能统计：")
        log(f"  总运行时间: {total_time:.1f}s")
        if total_processed > 0:
            log(f"  平均处理时间: {total_processing_time/total_processed:.1f}s/数据集")
        if success_count > 0:
            avg_configs_per_dataset = len(Config.GRAPH_CONFIGS)
            log(f"  平均配置生成速度: {avg_configs_per_dataset*success_count/total_processing_time:.1f} 配置/秒")
        
        log(f"")
        log(f"每个数据集生成 {len(Config.GRAPH_CONFIGS)} 个不同配置的知识图谱：")
        for config in Config.GRAPH_CONFIGS:
            log(f"  - {config['name']}: 时序权重={config['w_temporal']:.1f}, 条件权重={config['w_conditional']:.1f}")
        log(f"")
        log(f"结果保存在：{Config.KNOWLEDGE_GRAPH_DIR}")
        log(f"目录结构：[dataset_name]/[config_name]/prereq_graph.json")
        log(f"优化说明：")
        log(f"  - 统计量只计算一次，显著减少重复计算")
        log(f"  - 限制最大统计学生数: {Config.MAX_STUDENTS_LIMIT}，对大数据集进行随机抽样")
        log(f"  - 自动跳过已完成的数据集，支持断点续传")
        log(f"  - 高度优化的并行处理和内存管理")
        log(f"{'='*60}")


def main():
    """主入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="构建知识前驱关系图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\n示例用法:
  python build_prereq_graph.py                    # 正常运行，跳过已完成的数据集
  python build_prereq_graph.py --force            # 强制重新处理所有数据集
  python build_prereq_graph.py --dataset assistments2009  # 只处理指定数据集
  python build_prereq_graph.py --dataset assistments2009 --force  # 强制重新处理指定数据集
  
环境变量配置:
  set PREREQ_MAX_STUDENTS=8000                     # 设置最大学生数限制
  set PREREQ_MIN_SUPPORT=200                       # 设置最小支持学生数
"""
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新处理所有数据集，忽略已存在的结果"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="指定要处理的数据集名称（如：assistments2009, hnu_sys2023等）"
    )
    
    parser.add_argument(
        "--max-students",
        type=int,
        help=f"设置最大统计学生数限制（当前: {Config.MAX_STUDENTS_LIMIT}）"
    )
    
    args = parser.parse_args()
    
    # 应用命令行参数
    if args.max_students:
        Config.MAX_STUDENTS_LIMIT = args.max_students
        log(f"设置最大学生数限制为: {Config.MAX_STUDENTS_LIMIT}")
    
    builder = PrereqGraphBuilder()
    
    # 根据参数选择处理方式
    if args.dataset:
        # 处理指定数据集
        builder.run_single_dataset(args.dataset, force_reprocess=args.force)
    else:
        # 批量处理所有数据集
        builder.run(force_reprocess=args.force)


if __name__ == "__main__":
    main()
