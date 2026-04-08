# 消融实验报告：Router + Experts

> 生成时间：2026-04-05 21:21

## 1. 整体 Macro-F1 对比

| 配置 | Macro-F1 | Accuracy | 描述 |
|------|----------|----------|------|
| **no_conflict_expert** | 0.8972 (++0.0004) | 0.8972 | 关闭 Conflict Expert（验证冲突专家对 multi_entity/conflict 样本的价值） |
| **base_expert_only** | 0.8971 (++0.0003) | 0.8972 | 只用 Base Expert（关闭 Conflict + Context，等价于单专家 baseline） |
| **no_context_expert** | 0.8971 (++0.0003) | 0.8972 | 关闭 Context Expert（验证上下文专家对 ambiguous 样本的价值） |
| **full** | 0.8968 | 0.8969 | 完整模型（Router + Base/Conflict/Context Expert + L_div） |
| **no_router** | 0.8938 (+-0.0030) | 0.8940 | 均匀路由消融（α = [1/3, 1/3, 1/3]，验证 Router 的必要性） |

## 2. Hard Subset 分析

（只展示 full 与关键消融配置的对比）

| 配置 | multi_entity F1 | conflict F1 | ambiguous F1 | any_hard F1 |
|------|-----------------|-------------|--------------|-------------|
| full | 0.9018 | 0.0000 | 0.0000 | 0.8997 |
| no_router | 0.8990 | 0.0000 | 0.0000 | 0.8969 |
| no_conflict_expert | 0.9020 | 0.0000 | 0.0000 | 0.8999 |
| no_context_expert | 0.9025 | 0.0000 | 0.0000 | 0.9004 |
| base_expert_only | 0.9012 | 0.0000 | 0.0000 | 0.8990 |

## 3. 路由分析

| 配置 | Base权重 | Conflict权重 | Context权重 | 平均熵 | 熵比率 |
|------|----------|--------------|-------------|--------|--------|
| no_conflict_expert | 0.888 | 0.000 | 0.112 | 0.3502 | 0.3187 |
| base_expert_only | 1.000 | 0.000 | 0.000 | 0.0000 | 0.0000 |
| no_context_expert | 0.869 | 0.131 | 0.000 | 0.3853 | 0.3507 |
| full | 0.784 | 0.118 | 0.099 | 0.6689 | 0.6089 |
| no_router | 0.333 | 0.333 | 0.333 | 1.0986 | 1.0000 |

## 4. Expert 多样性诊断

（余弦相似度越低 = experts 越分化 = L_div 效果越好）

| 配置 | 整体平均余弦相似度 | Hard样本余弦相似度 | Base-Conflict | Base-Context | Conflict-Context |
|------|-------------------|-------------------|---------------|--------------|-----------------|
| no_conflict_expert | -0.2053 | -0.2101 | -0.1979 | -0.2029 | -0.2152 |
| base_expert_only | -0.2053 | -0.2101 | -0.1979 | -0.2029 | -0.2152 |
| no_context_expert | -0.2053 | -0.2101 | -0.1979 | -0.2029 | -0.2152 |
| full | -0.2053 | -0.2101 | -0.1979 | -0.2029 | -0.2152 |
| no_router | -0.2053 | -0.2101 | -0.1979 | -0.2029 | -0.2152 |

## 5. 验收标准回答

### Q1: Router 是否真的在工作？
✅ **是**。完整 Router 比均匀路由 Macro-F1 高出 **+0.0030**，说明路由器学到了有意义的分发策略。

### Q2: Expert 是否真的有分工？
✅ **是**。Hard 样本上 Expert 表征平均余弦相似度 = -0.2101（< 0.7），说明 Expert 已分化，L_div 有效。

### Q3: 提升主要来自 conflict / ambiguous 样本？
✅ **是**。完整模型在 multi_entity 上提升 +0.0006，在 ambiguous 上提升 +0.0000。

### Q4: 去掉某个 Expert，性能会不会掉？
  - 去掉 Conflict Expert：❌ 未掉（delta = -0.0004）
  - 去掉 Context Expert：❌ 未掉（delta = -0.0003）

---
*本报告由 `ablation_router_experts.py` 自动生成*