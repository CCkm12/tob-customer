# -*- coding: utf-8 -*-
"""
AHP 层次分析法权重模块
流程：两两比较判断矩阵（1-9标度）→ 特征向量求层内权重 → 一致性检验CR<0.1
      → 大类权重×层内权重 = 12维全局权重（层次总排序）
被 scoring_model.py 和 app.py 共同调用。自检：python ahp_weights.py
"""
import numpy as np

RI = {1:0,2:0,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49,11:1.51,12:1.54}

# 三大类：客户价值/成交可行性/需求赛道
CAT_MATRIX = [[1,2,2],[1/2,1,1],[1/2,1,1]]
# 客户价值层内：规模/增长/盈利/财务健康
VALUE_MATRIX = [[1,1/2,1,2],[2,1,2,2],[1,1/2,1,2],[1/2,1/2,1/2,1]]
# 成交可行性层内：地域/决策链/成熟度/上市年限/企业性质
FEAS_MATRIX = [[1,1,2,2,3],[1,1,2,2,2],[1/2,1/2,1,1,2],
               [1/2,1/2,1,1,2],[1/3,1/2,1/2,1/2,1]]
# 需求赛道层内：行业景气/技术投入/需求匹配
DEMAND_MATRIX = [[1,1,2],[1,1,1],[1/2,1,1]]

WEIGHT_ORDER = ['规模得分','增长得分','盈利得分','财务健康得分','技术投入得分',
                '成熟度得分','上市年限得分','地域得分','行业景气得分',
                '企业性质得分','决策链得分','需求匹配得分']

def _local_weights(matrix):
    M = np.array(matrix, dtype=float); n = M.shape[0]
    vals, vecs = np.linalg.eig(M)
    k = np.argmax(vals.real); lam = vals[k].real
    w = vecs[:, k].real; w = w/w.sum()
    CI = (lam-n)/(n-1); CR = CI/RI[n] if RI[n] > 0 else 0.0
    return w, lam, CR

def get_ahp_weights():
    wc,_,_ = _local_weights(CAT_MATRIX)
    wv,_,_ = _local_weights(VALUE_MATRIX)
    wf,_,_ = _local_weights(FEAS_MATRIX)
    wd,_,_ = _local_weights(DEMAND_MATRIX)
    g = {}
    for n,x in zip(['规模得分','增长得分','盈利得分','财务健康得分'], wv): g[n]=wc[0]*x
    for n,x in zip(['地域得分','决策链得分','成熟度得分','上市年限得分','企业性质得分'], wf): g[n]=wc[1]*x
    for n,x in zip(['行业景气得分','技术投入得分','需求匹配得分'], wd): g[n]=wc[2]*x
    return {k:g[k] for k in WEIGHT_ORDER}

def get_cr_report():
    lines=[]
    for name,M in [('三大类',CAT_MATRIX),('客户价值层',VALUE_MATRIX),
                   ('成交可行性层',FEAS_MATRIX),('需求赛道层',DEMAND_MATRIX)]:
        _,lam,cr=_local_weights(M)
        lines.append(f"  {name}：λmax={lam:.4f}，CR={cr:.4f} -> {'通过' if cr<0.1 else '不通过!!'}")
    return "\n".join(lines)

if __name__ == '__main__':
    print("AHP 一致性检验："); print(get_cr_report())
    w = get_ahp_weights()
    print("\n12维全局权重：")
    for k,v in w.items(): print(f"  {k:<8} {v:.4f}")
    print(f"权重合计：{sum(w.values()):.6f}（应≈1）")
