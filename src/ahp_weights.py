# -*- coding: utf-8 -*-
"""
AHP 层次分析法权重模块（全行业 ToB 客户分级 v2.0）
三层结构：目标层 → 准则层（客户价值/成交可行性/客户需求度）→ 方案层（10维度）
流程：两两比较判断矩阵 → 特征向量求层内权重 → 一致性检验CR<0.1
      → 大类权重×层内权重 = 10维全局权重
被 scoring_model.py 和 app.py 共同调用。自检：python ahp_weights.py
"""
import numpy as np

RI = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}

# ========== 准则层：客户价值 / 成交可行性 / 客户需求度 ==========
# 价值:可行性 = 3:2，价值:需求 = 1:1，可行性:需求 = 2:3
CAT_MATRIX = [
    [1,   3/2, 1  ],
    [2/3, 1,   2/3],
    [1,   3/2, 1  ],
]

# ========== B1 客户价值层：组织规模 / 预算能力 / 增长潜力 / 战略价值 ==========
# 预算能力最重要，规模和增长并列，战略价值最轻
VALUE_MATRIX = [
    [1,   2/3, 1,   3/2],
    [3/2, 1,   3/2, 2  ],
    [1,   2/3, 1,   3/2],
    [2/3, 1/2, 2/3, 1  ],
]

# ========== B2 成交可行性层：决策链复杂度 / 合作成熟度 / 竞争与替换成本 ==========
# 合作成熟度最重要，决策链和竞争替换并列（抢客户场景下替换成本权重提升）
FEAS_MATRIX = [
    [1,   2/3, 1  ],
    [3/2, 1,   3/2],
    [1,   2/3, 1  ],
]

# ========== B3 客户需求度层：显性需求信号 / 降本增效与替换潜力 / 行业需求刚性 ==========
# 降本增效与替换潜力最重要（开拓新客户核心），显性需求信号其次，行业刚性最轻
DEMAND_MATRIX = [
    [1,   2/3, 3/2],
    [3/2, 1,   2  ],
    [2/3, 1/2, 1  ],
]

WEIGHT_ORDER = [
    '组织规模得分', '预算能力得分', '增长潜力得分', '战略价值得分',
    '决策链复杂度得分', '合作成熟度得分', '竞争与替换成本得分',
    '显性需求信号得分', '降本增效与替换潜力得分', '行业需求刚性得分',
]

# 维度中文显示名
DIMENSION_NAMES = {
    '组织规模得分': '组织规模',
    '预算能力得分': '预算能力',
    '增长潜力得分': '增长潜力',
    '战略价值得分': '战略价值',
    '决策链复杂度得分': '决策链复杂度',
    '合作成熟度得分': '合作成熟度',
    '竞争与替换成本得分': '竞争与替换成本',
    '显性需求信号得分': '显性需求信号',
    '降本增效与替换潜力得分': '降本增效与替换潜力',
    '行业需求刚性得分': '行业需求刚性',
}

# 维度所属大类
DIMENSION_CATEGORY = {
    '组织规模得分': '客户价值',
    '预算能力得分': '客户价值',
    '增长潜力得分': '客户价值',
    '战略价值得分': '客户价值',
    '决策链复杂度得分': '成交可行性',
    '合作成熟度得分': '成交可行性',
    '竞争与替换成本得分': '成交可行性',
    '显性需求信号得分': '客户需求度',
    '降本增效与替换潜力得分': '客户需求度',
    '行业需求刚性得分': '客户需求度',
}


def _local_weights(matrix):
    """计算单个判断矩阵的特征向量权重、λmax、CR"""
    M = np.array(matrix, dtype=float)
    n = M.shape[0]
    vals, vecs = np.linalg.eig(M)
    k = np.argmax(vals.real)
    lam = vals[k].real
    w = vecs[:, k].real
    w = w / w.sum()
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    CR = CI / RI[n] if RI[n] > 0 else 0.0
    return w, lam, CR


def get_ahp_weights():
    """返回10维全局权重字典（按WEIGHT_ORDER顺序）"""
    wc, _, _ = _local_weights(CAT_MATRIX)
    wv, _, _ = _local_weights(VALUE_MATRIX)
    wf, _, _ = _local_weights(FEAS_MATRIX)
    wd, _, _ = _local_weights(DEMAND_MATRIX)
    g = {}
    for n, x in zip(['组织规模得分', '预算能力得分', '增长潜力得分', '战略价值得分'], wv):
        g[n] = wc[0] * x
    for n, x in zip(['决策链复杂度得分', '合作成熟度得分', '竞争与替换成本得分'], wf):
        g[n] = wc[1] * x
    for n, x in zip(['显性需求信号得分', '降本增效与替换潜力得分', '行业需求刚性得分'], wd):
        g[n] = wc[2] * x
    return {k: g[k] for k in WEIGHT_ORDER}


def get_cr_report():
    """返回所有矩阵的一致性检验报告"""
    lines = []
    for name, M in [
        ('准则层（价值/可行性/需求）', CAT_MATRIX),
        ('B1 客户价值层', VALUE_MATRIX),
        ('B2 成交可行性层', FEAS_MATRIX),
        ('B3 客户需求度层', DEMAND_MATRIX),
    ]:
        _, lam, cr = _local_weights(M)
        status = '通过' if cr < 0.1 else '不通过!!'
        lines.append(f"  {name}：λmax={lam:.4f}，CR={cr:.4f} -> {status}")
    return "\n".join(lines)


def get_category_weights():
    """返回三大类权重"""
    wc, _, _ = _local_weights(CAT_MATRIX)
    return {'客户价值': wc[0], '成交可行性': wc[1], '客户需求度': wc[2]}


if __name__ == '__main__':
    print("=" * 60)
    print("AHP 层次分析法一致性检验（全行业ToB客户分级 v2.0）")
    print("=" * 60)
    print(get_cr_report())

    print("\n" + "=" * 60)
    print("三大类权重：")
    cat = get_category_weights()
    for k, v in cat.items():
        print(f"  {k}: {v*100:.1f}%")

    print("\n10维全局权重：")
    w = get_ahp_weights()
    for k, v in w.items():
        cat_name = DIMENSION_CATEGORY[k]
        print(f"  [{cat_name}] {DIMENSION_NAMES[k]:<12} {v*100:>5.1f}%")
    print(f"\n权重合计：{sum(w.values())*100:.2f}%（应≈100%）")
