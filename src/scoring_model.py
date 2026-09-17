# -*- coding: utf-8 -*-
"""
客户评分模型 v2.0（全行业 ToB 客户分级）
功能：数据清洗 → 10维度特征工程 → AHP加权评分 → A/B/C/D分级
适用：政企、医院、学校、企业等全行业客户，同时覆盖增量（没系统）和存量（替换）场景
被 app.py 调用。自检：python scoring_model.py
"""
import pandas as pd
import numpy as np
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ahp_weights import get_ahp_weights, get_cr_report, DIMENSION_NAMES, WEIGHT_ORDER

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 导入模板列定义（全行业客户）
# ============================================================
IMPORT_TEMPLATE_COLUMNS = [
    # 基础信息
    '客户名称', '客户类型', '所属行业', '企业性质', '所在地区',
    # 规模指标（不同行业用不同指标，允许部分缺失）
    '员工人数', '营业收入_万元', '床位数量', '学生人数', '行政级别',
    # 财务与预算
    'IT年度预算_万元', '营收同比增长率', '净利润同比增长率', '毛利率',
    # 组织与决策
    '成立年限', '是否需要招投标', '现有供应商名称', '现有合同到期时间', '现有系统使用年限',
    # 需求信号（可由爬虫自动填充，也可手动标注）
    '有采购公告', '有重新招标', '合同半年内到期', '有新建项目', '有数字化转型规划',
    'IT岗位招聘数', '现有供应商负面新闻', '客户抱怨现有产品',
    # 销售维护字段
    '合作成熟度阶段', '竞争激烈程度', '绑定程度',
    # 战略与行业
    '行业排名', '是否行业标杆',
]

# 列名别名映射（兼容各种导入格式）
COLUMN_ALIASES = {
    '客户名称': ['客户名称', '公司名称', '企业名称', 'name', 'company', '公司'],
    '客户类型': ['客户类型', '类型', 'customer_type'],
    '所属行业': ['所属行业', '行业', 'industry'],
    '企业性质': ['企业性质', '公司性质', '性质', 'enterprise_type'],
    '所在地区': ['所在地区', '地区', '地址', 'region', 'address'],
    '员工人数': ['员工人数', '人数', '员工数', 'employees'],
    '营业收入_万元': ['营业收入_万元', '营业收入', '营收', 'revenue'],
    '床位数量': ['床位数量', '床位数', '床位', 'beds'],
    '学生人数': ['学生人数', '学生数', '在校生人数', 'students'],
    '行政级别': ['行政级别', '级别', 'admin_level'],
    'IT年度预算_万元': ['IT年度预算_万元', 'IT预算', '信息化预算', 'it_budget'],
    '营收同比增长率': ['营收同比增长率', '营收增长率', 'revenue_growth'],
    '净利润同比增长率': ['净利润同比增长率', '净利润增长率', 'profit_growth'],
    '毛利率': ['毛利率', 'gross_margin'],
    '成立年限': ['成立年限', '经营年限', 'years_established'],
    '是否需要招投标': ['是否需要招投标', '需招投标', '招投标', 'need_bidding'],
    '现有供应商名称': ['现有供应商名称', '现有供应商', '供应商', 'current_vendor'],
    '现有合同到期时间': ['现有合同到期时间', '合同到期时间', '合同到期', 'contract_expiry'],
    '现有系统使用年限': ['现有系统使用年限', '系统使用年限', '使用年限', 'system_age'],
    '有采购公告': ['有采购公告', '采购公告', '采购', 'has_procurement'],
    '有重新招标': ['有重新招标', '重新招标', '替换招标', 'has_rebidding'],
    '合同半年内到期': ['合同半年内到期', '合同即将到期', 'contract_expiring_soon'],
    '有新建项目': ['有新建项目', '新建项目', '扩张', 'has_new_project'],
    '有数字化转型规划': ['有数字化转型规划', '数字化规划', '转型规划', 'has_digital_plan'],
    'IT岗位招聘数': ['IT岗位招聘数', 'IT招聘数', '招聘数', 'it_jobs'],
    '现有供应商负面新闻': ['现有供应商负面新闻', '供应商负面', 'vendor_negative'],
    '客户抱怨现有产品': ['客户抱怨现有产品', '客户抱怨', 'customer_complaint'],
    '合作成熟度阶段': ['合作成熟度阶段', '接触阶段', '合作阶段', 'maturity_stage'],
    '竞争激烈程度': ['竞争激烈程度', '竞争程度', 'competition_level'],
    '绑定程度': ['绑定程度', '替换难度', 'switching_cost'],
    '行业排名': ['行业排名', '排名', 'industry_rank'],
    '是否行业标杆': ['是否行业标杆', '行业标杆', 'is_benchmark'],
}


def normalize_import_columns(df):
    """把常见列名映射成评分模型字段"""
    rename = {}
    used = set()
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            used.add(canonical)
            continue
        for alias in aliases:
            key = alias.lower()
            if key in lower_map and lower_map[key] not in used:
                rename[lower_map[key]] = canonical
                used.add(lower_map[key])
                break
    if rename:
        df = df.rename(columns=rename)
    return df


def _is_blank(value):
    return _str(value) == ''


# 新客户导入时，销售未填写则自动补全的默认字段（对应得分见各维度函数注释）
NEW_CUSTOMER_FIELD_DEFAULTS = {
    # 字段: (默认值说明, 对应得分效果)
    '合作成熟度阶段': '仅有线索',          # → 合作成熟度 25 分
    # 现有供应商名称: 保持空 → 视为增量客户，C7 默认 60 分
    # 绑定程度: 保持空 → 不加减分
    # 竞争激烈程度: 保持空 → 不加减分
    '客户抱怨现有产品': '否',              # 空视为否 → 不加分
    # 是否行业标杆: 保持空 → 战略价值默认 35 分
    # 行业排名: 保持空 → 不触发排名分档
}


def apply_new_customer_defaults(df):
    """
    新客户导入缺省字段自动填写。
    仅填充空白，不覆盖用户已填写的值。

    | 字段           | 默认值           | 对应得分              |
    |----------------|------------------|-----------------------|
    | 合作成熟度阶段 | 仅有线索         | 25 分                 |
    | 现有供应商名称 | 空（增量客户）   | C7 默认 60 分         |
    | 绑定程度       | 空               | 不加减分              |
    | 竞争激烈程度   | 空               | 不加减分              |
    | 客户抱怨现有产品 | 空→否          | 不加分                |
    | 是否行业标杆   | 空               | 默认 35 分            |
    | 行业排名       | 空               | 不触发排名分档        |
    """
    out = df.copy()
    fill_cols = (
        '合作成熟度阶段', '现有供应商名称', '绑定程度', '竞争激烈程度',
        '客户抱怨现有产品', '是否行业标杆', '行业排名',
    )
    for col in fill_cols:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype='object')
        else:
            out[col] = out[col].astype('object')

    blank_maturity = out['合作成熟度阶段'].map(_is_blank)
    out.loc[blank_maturity, '合作成熟度阶段'] = NEW_CUSTOMER_FIELD_DEFAULTS['合作成熟度阶段']

    blank_complaint = out['客户抱怨现有产品'].map(_is_blank)
    out.loc[blank_complaint, '客户抱怨现有产品'] = NEW_CUSTOMER_FIELD_DEFAULTS['客户抱怨现有产品']

    # 以下字段刻意保持空：现有供应商名称 / 绑定程度 / 竞争激烈程度 / 是否行业标杆 / 行业排名
    return out


FOLLOWUP_OVERRIDE_FIELDS = [
    '合作成熟度阶段', '现有供应商名称', '绑定程度', '竞争激烈程度',
    '客户抱怨现有产品', '是否行业标杆', '行业排名',
    '跟进日期',  # 用于成熟度时间衰减
]


def _normalize_customer_name(name):
    """弱规范化客户名：去空格/括号后缀/公司后缀，便于跟进表匹配。"""
    s = _str(name)
    if not s:
        return ''
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'[（(][^）)]*[）)]', '', s)
    for suf in ('有限责任公司', '股份有限公司', '有限公司', '集团公司', '集团', '公司', '厂'):
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[: -len(suf)]
            break
    return s.lower()


def load_sales_followup(path=None):
    """读取销售跟进表；不存在则返回空表。"""
    fp = path or os.path.join(DATA_DIR, 'sales_followup.csv')
    if not os.path.exists(fp):
        return pd.DataFrame(columns=['客户名称'] + FOLLOWUP_OVERRIDE_FIELDS)
    df = pd.read_csv(fp)
    if '客户名称' not in df.columns and '公司名称' in df.columns:
        df['客户名称'] = df['公司名称']
    return df


def apply_followup_overrides(df, followup_df=None):
    """
    按客户名称把销售跟进表字段覆盖到评分输入表。
    支持去后缀的弱匹配；允许空值覆盖（表示清空后走默认规则）。
    """
    if followup_df is None:
        followup_df = load_sales_followup()
    if followup_df is None or followup_df.empty or '客户名称' not in followup_df.columns:
        return df

    out = df.copy()
    for col in FOLLOWUP_OVERRIDE_FIELDS:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype='object')
        else:
            out[col] = out[col].astype('object')

    fu = followup_df.copy()
    fu['_key'] = fu['客户名称'].map(_normalize_customer_name)
    fu = fu[fu['_key'] != ''].drop_duplicates(subset=['_key'], keep='last')
    if fu.empty:
        return out

    fu_map = fu.set_index('_key')
    name_a = out['客户名称'].map(_normalize_customer_name) if '客户名称' in out.columns else pd.Series([''] * len(out), index=out.index)
    name_b = out['公司名称'].map(_normalize_customer_name) if '公司名称' in out.columns else pd.Series([''] * len(out), index=out.index)

    for idx in out.index:
        key = name_a.at[idx] or name_b.at[idx]
        if key not in fu_map.index:
            continue
        fr = fu_map.loc[key]
        for col in FOLLOWUP_OVERRIDE_FIELDS:
            if col in fr.index:
                val = fr[col]
                out.at[idx, col] = '' if pd.isna(val) else val
    return out


def _stock_signal_strength(row):
    """
    存量迹象强度 0~4（不再仅靠供应商名二分）。
    供应商名权重最高；系统年限/重招/抱怨/合同到期为辅助信号。
    """
    strength = 0
    has_vendor = pd.notna(row.get('现有供应商名称')) and str(row.get('现有供应商名称', '')).strip() != ''
    if has_vendor:
        strength += 2
    if pd.notna(_parse_number(row.get('现有系统使用年限'))):
        strength += 1
    if _parse_bool(row.get('有重新招标')) == 1:
        strength += 1
    if _parse_bool(row.get('客户抱怨现有产品')) == 1:
        strength += 1
    if _parse_bool(row.get('合同半年内到期')) == 1 or _str(row.get('现有合同到期时间')):
        strength += 1
    return min(4, strength)


def _parse_followup_date(value):
    s = _str(value)
    if not s:
        return None
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%Y%m%d'):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt, errors='raise')).normalize()
        except Exception:
            continue
    try:
        return pd.Timestamp(pd.to_datetime(s, errors='raise')).normalize()
    except Exception:
        return None


def _days_since_followup(row):
    dt = _parse_followup_date(row.get('跟进日期'))
    if dt is None:
        return None
    return max(0, int((pd.Timestamp.today().normalize() - dt).days))


def _has_objective_sales_corroboration(row):
    """高成熟度是否有客观信号背书（防刷分）。"""
    if _str(row.get('现有供应商名称')):
        return True
    if _parse_bool(row.get('有采购公告')) == 1:
        return True
    if _parse_bool(row.get('有重新招标')) == 1:
        return True
    if _parse_bool(row.get('合同半年内到期')) == 1:
        return True
    if _parse_bool(row.get('有新建项目')) == 1:
        return True
    if _parse_bool(row.get('有数字化转型规划')) == 1:
        return True
    jobs = _parse_number(row.get('IT岗位招聘数'))
    if pd.notna(jobs) and jobs >= 1:
        return True
    days = _days_since_followup(row)
    if days is not None and days <= 60:
        return True
    return False


def grade_customer(score):
    """A/B/C/D 分级。门槛按当前样本分位校准：约前 5% 为 A，前 20% 为 B。"""
    if score >= 59:
        return 'A级（高价值优先跟进）'
    elif score >= 55:
        return 'B级（重点跟进）'
    elif score >= 50:
        return 'C级（常规跟进）'
    else:
        return 'D级（低优先级/观察）'


def _parse_number(value):
    """安全解析数值，失败返回NaN"""
    if pd.isna(value) or value == '' or value is None:
        return np.nan
    try:
        return float(str(value).replace(',', '').replace('%', '').strip())
    except Exception:
        return np.nan


def _parse_bool(value):
    """解析布尔值（是/否/1/0/true/false）"""
    if pd.isna(value) or value == '' or value is None:
        return np.nan
    v = str(value).strip().lower()
    if v in ['是', '有', '1', 'true', 'yes', 'y']:
        return 1
    if v in ['否', '无', '没有', '0', 'false', 'no', 'n']:
        return 0
    return np.nan


def _str(value):
    if pd.isna(value) or value is None:
        return ''
    s = str(value).strip()
    return '' if s.lower() in ('nan', 'none') else s


def clean_industry_name(s):
    if pd.isna(s):
        return '未知'
    s = re.split(r'概念', str(s))[0]
    s = re.sub(r'[ⅠⅡⅢⅣⅤIⅡ]', '', s)
    s = s.strip().strip('：:').strip()
    s = re.split(r'\s+', s)[0] if s else s
    return s if s else '未知'


def _extract_number(text, patterns):
    blob = _str(text)
    for pat in patterns:
        m = re.search(pat, blob)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except ValueError:
                continue
    return np.nan


def enrich_legacy_fields(df):
    """兼容旧版上市公司表：回填营收/市值/性质/短行业，并保留公司名称。"""
    if '公司名称' not in df.columns and '客户名称' in df.columns:
        df['公司名称'] = df['客户名称']
    if '客户名称' not in df.columns and '公司名称' in df.columns:
        df['客户名称'] = df['公司名称']

    blob = (
        df.get('所属行业', pd.Series('', index=df.index)).map(_str) + ' ' +
        df.get('公司亮点', pd.Series('', index=df.index)).map(_str) + ' ' +
        df.get('主营业务', pd.Series('', index=df.index)).map(_str)
    )

    if '营业收入_万元' not in df.columns:
        df['营业收入_万元'] = np.nan
    rev_wan = pd.to_numeric(df['营业收入_万元'], errors='coerce')
    rev_yi = blob.map(lambda t: _extract_number(t, [
        r'营业总收入[：:\s]*([0-9.]+)\s*亿元',
        r'营业总收入[：:\s]*([0-9.]+)\s*亿',
    ]))
    df['营业收入_万元'] = rev_wan
    df.loc[df['营业收入_万元'].isna() & rev_yi.notna(), '营业收入_万元'] = rev_yi * 10000

    if '总市值(亿)' not in df.columns:
        df['总市值(亿)'] = np.nan
    mcap = blob.map(lambda t: _extract_number(t, [r'总市值[：:\s]*([0-9.]+)\s*亿']))
    df.loc[df['总市值(亿)'].isna() | (pd.to_numeric(df['总市值(亿)'], errors='coerce') < 1), '总市值(亿)'] = mcap

    df['行业清洗'] = df['所属行业'].apply(clean_industry_name) if '所属行业' in df.columns else '未知'

    nature = []
    regions = []
    ctypes = []
    for i, row in df.iterrows():
        text = _str(row.get('公司亮点')) + _str(row.get('主营业务')) + _str(row.get('客户名称'))
        n = _str(row.get('企业性质'))
        if not n:
            if any(k in text for k in ('央企', '国企', '国资', '国有')):
                n = '国企'
            elif any(k in text for k in ('民营', '民企')):
                n = '民企'
            elif _str(row.get('股票代码')):
                n = '上市'
        nature.append(n)
        r = _str(row.get('所在地区')) or _str(row.get('办公地址'))
        if not r:
            for city in ('深圳', '广州', '东莞', '佛山', '珠海', '北京', '上海', '杭州', '苏州', '成都'):
                if city in text:
                    r = city
                    break
        regions.append(r)
        ct = _str(row.get('客户类型'))
        if not ct:
            ct = '企业'
        ctypes.append(ct)
    df['企业性质'] = nature
    df['所在地区'] = regions
    df['客户类型'] = ctypes
    # 是否行业标杆：空值保持空，由 apply_new_customer_defaults / score_strategic_value 处理（默认 35 分）
    return df


# ============================================================
# C1 组织规模得分（分行业指标）
# ============================================================
def score_organization_scale(row):
    """
    分行业选择规模指标，分档打分。
    企业：员工人数/营收；医院：床位；学校：学生数；政府：行政级别。
    """
    industry = _str(row.get('行业清洗')) or clean_industry_name(row.get('所属行业', ''))
    customer_type = _str(row.get('客户类型'))
    employees = _parse_number(row.get('员工人数'))
    revenue = _parse_number(row.get('营业收入_万元'))
    beds = _parse_number(row.get('床位数量'))
    students = _parse_number(row.get('学生人数'))
    admin_level = _str(row.get('行政级别'))

    # 医院：优先用床位
    if '医院' in industry or '医疗' in customer_type or pd.notna(beds):
        if pd.notna(beds):
            if beds >= 1000: return 90
            if beds >= 500: return 75
            if beds >= 200: return 55
            if beds >= 50: return 35
            return 20

    # 学校：优先用学生数
    if '学校' in industry or '教育' in customer_type or pd.notna(students):
        if pd.notna(students):
            if students >= 20000: return 90
            if students >= 10000: return 75
            if students >= 3000: return 55
            if students >= 500: return 35
            return 20

    # 政府/事业单位：用行政级别
    if '政府' in customer_type or '政务' in industry or admin_level:
        level_map = {'部委': 95, '省厅': 85, '市局': 65, '区县': 45, '街道': 25, '基层': 15}
        for k, v in level_map.items():
            if k in admin_level:
                return v

    # 企业：用员工人数或营收
    if pd.notna(employees):
        if employees >= 5000: return 95
        if employees >= 1000: return 78
        if employees >= 500: return 62
        if employees >= 300: return 50
        if employees >= 100: return 35
        return 20
    if pd.notna(revenue):
        if revenue >= 500000: return 95  # 50亿
        if revenue >= 100000: return 78  # 10亿
        if revenue >= 10000: return 62   # 1亿
        if revenue >= 1000: return 45    # 1000万
        return 25

    return 50  # 缺失中性分


# ============================================================
# C2 预算能力得分
# ============================================================
def score_budget(row):
    """
    已知IT预算直接分档；未知则按营收推算（IT支出通常占营收1-3%）或按规模推断。
    """
    it_budget = _parse_number(row.get('IT年度预算_万元'))
    revenue = _parse_number(row.get('营业收入_万元'))
    scale_score = score_organization_scale(row)  # 复用规模得分做推断

    if pd.notna(it_budget):
        if it_budget >= 1000: return 95
        if it_budget >= 500: return 82
        if it_budget >= 100: return 65
        if it_budget >= 50: return 50
        if it_budget >= 10: return 35
        return 20

    # 按营收推算IT预算（取2%作为中位数）
    if pd.notna(revenue):
        estimated = revenue * 0.02
        if estimated >= 1000: return 85
        if estimated >= 500: return 72
        if estimated >= 100: return 58
        if estimated >= 50: return 45
        return 30

    # 按规模推断
    if scale_score >= 80: return 65
    if scale_score >= 60: return 50
    if scale_score >= 40: return 38
    return 28


# ============================================================
# C3 增长潜力得分
# ============================================================
def score_growth_potential(row):
    """
    营收增长率 + 新建项目信号 + IT招聘趋势。
    规模逼近阈值（如员工从300涨到500）额外加分。
    """
    revenue_growth = _parse_number(row.get('营收同比增长率'))
    profit_growth = _parse_number(row.get('净利润同比增长率'))
    has_new_project = _parse_bool(row.get('有新建项目'))
    it_jobs = _parse_number(row.get('IT岗位招聘数'))
    employees = _parse_number(row.get('员工人数'))

    score = 40  # 基础分

    # 营收增长
    if pd.notna(revenue_growth):
        if revenue_growth > 30: score += 25
        elif revenue_growth > 15: score += 18
        elif revenue_growth > 5: score += 10
        elif revenue_growth > 0: score += 3
        elif revenue_growth > -10: score -= 5
        else: score -= 15

    # 新建项目/扩张
    if has_new_project == 1:
        score += 15

    # IT岗位招聘（表明在加强IT团队，可能有项目）
    if pd.notna(it_jobs):
        if it_jobs >= 10: score += 12
        elif it_jobs >= 5: score += 8
        elif it_jobs >= 1: score += 4

    # 规模逼近阈值信号（员工300-500之间，即将跨过需要系统的阈值）
    if pd.notna(employees) and 300 <= employees < 500:
        score += 8

    return max(0, min(100, score))


# ============================================================
# C4 战略价值得分
# ============================================================
def score_strategic_value(row):
    """
    战略价值：标杆与排名可叠加（不再互斥）。
    空标注 → 基础 35；标杆加分；排名加分；有标杆无排名时不加满（防刷）。
    """
    score = 35
    is_benchmark = _parse_bool(row.get('是否行业标杆'))
    rank = _parse_number(row.get('行业排名'))

    if is_benchmark == 1:
        score += 30 if pd.notna(rank) else 22

    if pd.notna(rank):
        if rank <= 10:
            score += 25
        elif rank <= 30:
            score += 15
        elif rank <= 50:
            score += 8
        else:
            score += 3

    return max(0, min(100, score))


# ============================================================
# C5 决策链复杂度得分（越简单分越高）
# ============================================================
def score_decision_chain(row):
    """
    决策链越简单，成交越快，得分越高。
    基于企业性质和招投标要求推断。
    """
    enterprise_type = _str(row.get('企业性质'))
    customer_type = _str(row.get('客户类型'))
    need_bidding = _parse_bool(row.get('是否需要招投标'))

    # 政府/事业单位：决策链最复杂
    if '政府' in customer_type or '政务' in enterprise_type or '事业' in enterprise_type:
        return 30

    # 国企/央企：决策链复杂
    if '国企' in enterprise_type or '央企' in enterprise_type or '国有' in enterprise_type:
        return 40

    # 上市公司：决策链中等偏复杂
    if '上市' in enterprise_type or '股份' in enterprise_type:
        return 55

    # 明确需要招投标
    if need_bidding == 1:
        return 40

    # 民企/外企：决策链相对简单
    if '民企' in enterprise_type or '私营' in enterprise_type or '外资' in enterprise_type:
        return 70

    # 小微企业/初创：决策链最简单
    if '小微' in enterprise_type or '初创' in enterprise_type:
        return 85

    return 55  # 未知默认中等


# ============================================================
# C6 合作成熟度得分（销售手动维护）
# ============================================================
def score_cooperation_maturity(row):
    """
    与客户的接触深度。
    - 默认「仅有线索」→ 25
    - 跟进过旧：成熟度衰减
    - 高成熟度但无客观背书：扣分（防刷）
    """
    stage = str(row.get('合作成熟度阶段', '') or '')

    stage_map = {
        '已有合作': 95,
        '商务谈判': 80,
        '深度接触': 65,
        '初步接触': 45,
        '仅有线索': 25,
        '完全陌生': 10,
    }
    score = 25
    for k, v in stage_map.items():
        if k in stage:
            score = v
            break

    days = _days_since_followup(row)
    if days is None:
        if score >= 65:
            score -= 10
    elif days > 180:
        score -= 25
    elif days > 90:
        score -= 15
    elif days > 60:
        score -= 8

    if score >= 80 and not _has_objective_sales_corroboration(row):
        score -= 20
    elif score >= 65 and not _has_objective_sales_corroboration(row):
        score -= 10

    return max(10, min(100, score))


# ============================================================
# C7 竞争与替换成本得分（竞争越小、替换越容易，分越高）
# ============================================================
def score_competition_and_switching(row):
    """
    竞争越小、替换越容易，分越高。
    - 存量强度 0：纯增量，基础 60；绑定仍可半幅生效
    - 存量强度 1：疑似存量，基础 50
    - 存量强度 ≥2：存量，基础 40
    """
    has_vendor = pd.notna(row.get('现有供应商名称')) and str(row.get('现有供应商名称', '')).strip() != ''
    system_age = _parse_number(row.get('现有系统使用年限'))
    competition = str(row.get('竞争激烈程度', '') or '').strip()
    binding = str(row.get('绑定程度', '') or '').strip()
    contract_expiry = str(row.get('现有合同到期时间', '') or '')
    expiring_soon = _parse_bool(row.get('合同半年内到期'))
    stock_lv = _stock_signal_strength(row)

    if stock_lv == 0:
        score = 60
    elif stock_lv == 1:
        score = 50
    else:
        score = 40

    if expiring_soon == 1 or ('半年' in contract_expiry) or ('6个月' in contract_expiry):
        score += 25 if stock_lv >= 1 else 10
    elif '1年' in contract_expiry or '一年' in contract_expiry:
        score += 15 if stock_lv >= 1 else 5

    if pd.notna(system_age):
        if system_age >= 8:
            score += 20
        elif system_age >= 5:
            score += 12
        elif system_age >= 3:
            score += 5

    # 绑定：无供应商时半幅生效
    bind_factor = 1.0 if has_vendor else 0.5
    if binding:
        if '深' in binding or '高' in binding:
            score -= int(20 * bind_factor)
        elif '中等' in binding:
            score -= int(5 * bind_factor)
        elif '浅' in binding or '低' in binding:
            score += int(10 * bind_factor)

    if competition:
        if '激烈' in competition:
            score -= 10
        elif '小' in competition or '少' in competition:
            score += 25 if stock_lv == 0 else 10
        elif stock_lv == 0 and ('中等' in competition or competition in ('中',)):
            score = min(score, 65)

    if stock_lv == 0 and not competition and not binding:
        score = 60

    return max(0, min(100, score))


# ============================================================
# C8 显性需求信号得分（增量+存量双场景）
# ============================================================
def score_explicit_demand(row):
    """
    基础分20，信号叠加，上限100。
    客户抱怨：有供应商 +15；无供应商也给 +8。
    """
    score = 20
    has_vendor = pd.notna(row.get('现有供应商名称')) and str(row.get('现有供应商名称', '')).strip() != ''

    if _parse_bool(row.get('有采购公告')) == 1:
        score += 30
    if _parse_bool(row.get('有重新招标')) == 1:
        score += 30
    if _parse_bool(row.get('合同半年内到期')) == 1:
        score += 20
    if _parse_bool(row.get('有新建项目')) == 1:
        score += 20
    if _parse_bool(row.get('有数字化转型规划')) == 1:
        score += 20

    if _parse_bool(row.get('客户抱怨现有产品')) == 1:
        score += 15 if has_vendor else 8

    if _parse_bool(row.get('现有供应商负面新闻')) == 1:
        score += 15

    it_jobs = _parse_number(row.get('IT岗位招聘数'))
    if pd.notna(it_jobs):
        if it_jobs >= 10:
            score += 12
        elif it_jobs >= 5:
            score += 8
        elif it_jobs >= 1:
            score += 4

    return min(100, score)


# ============================================================
# C9 降本增效与替换潜力得分（第一权重，增量+存量双场景）
# ============================================================
def score_cost_reduction_and_replacement(row):
    """
    核心维度。增量 + 存量；客户抱怨在无供应商时也给弱加分。
    """
    score = 20

    employees = _parse_number(row.get('员工人数'))
    revenue = _parse_number(row.get('营业收入_万元'))
    beds = _parse_number(row.get('床位数量'))
    students = _parse_number(row.get('学生人数'))
    industry = _str(row.get('行业清洗')) or clean_industry_name(row.get('所属行业', ''))
    profit_growth = _parse_number(row.get('净利润同比增长率'))
    gross_margin = _parse_number(row.get('毛利率'))
    system_age = _parse_number(row.get('现有系统使用年限'))
    has_vendor = pd.notna(row.get('现有供应商名称')) and str(row.get('现有供应商名称', '')).strip() != ''
    vendor_negative = _parse_bool(row.get('现有供应商负面新闻'))
    customer_complaint = _parse_bool(row.get('客户抱怨现有产品'))
    stock_lv = _stock_signal_strength(row)

    scale_reached = False
    if pd.notna(employees) and employees >= 500:
        scale_reached = True
    if pd.notna(revenue) and revenue >= 10000:
        scale_reached = True
    if pd.notna(beds) and beds >= 300:
        scale_reached = True
    if pd.notna(students) and students >= 5000:
        scale_reached = True
    if scale_reached:
        score += 20

    it_budget = _parse_number(row.get('IT年度预算_万元'))
    has_digital_plan = _parse_bool(row.get('有数字化转型规划'))
    if stock_lv == 0 and pd.isna(it_budget) and has_digital_plan != 1:
        score += 20
    elif stock_lv == 0:
        score += 10
    elif stock_lv == 1 and not has_vendor:
        score += 5

    labor_intensive_industries = ['制造', '物流', '零售', '医疗', '护理', '餐饮', '建筑', '物业', '仓储']
    if any(kw in industry for kw in labor_intensive_industries):
        score += 15

    cost_pressure = False
    if pd.notna(profit_growth) and profit_growth < 0:
        cost_pressure = True
    if pd.notna(gross_margin) and gross_margin < 20:
        cost_pressure = True
    if cost_pressure:
        score += 10

    if pd.notna(employees) and employees >= 1000:
        score += 10

    if has_vendor or stock_lv >= 1:
        if pd.notna(system_age):
            if system_age >= 8:
                score += 15
            elif system_age >= 5:
                score += 10
        if vendor_negative == 1:
            score += 15
        if customer_complaint == 1:
            score += 15 if has_vendor else 8
        if cost_pressure and has_vendor:
            score += 10
    elif customer_complaint == 1:
        score += 8

    return min(100, score)


# ============================================================
# C10 行业需求刚性得分（静态查表）
# ============================================================
def score_industry_rigidity(row):
    """
    按行业分类查表，判断该行业对IT/数字化的普遍需求程度。
    """
    industry = _str(row.get('行业清洗')) or clean_industry_name(row.get('所属行业', ''))
    customer_type = _str(row.get('客户类型'))
    text = industry + customer_type

    # 80-100：强监管+政策驱动，信息化刚需
    high_rigidity = ['医疗', '医院', '金融', '银行', '证券', '保险', '政务', '政府', '军工', '国防']
    for kw in high_rigidity:
        if kw in text:
            return 88

    # 60-80：有明确数字化政策要求 / 科技信息化主业
    mid_high_rigidity = ['教育', '学校', '能源', '电力', '交通', '运输', '制造', '通信',
                         '运营商', '软件', '半导体', '计算机', 'IT', '电子', '自动化']
    for kw in mid_high_rigidity:
        if kw in text:
            return 70

    # 40-60：有需求但非刚性
    mid_rigidity = ['零售', '物流', '房地产', '服务', '酒店', '旅游', '文化', '传媒']
    for kw in mid_rigidity:
        if kw in text:
            return 50

    # 20-40：信息化程度低
    low_rigidity = ['农业', '采矿', '建筑', '餐饮', '渔牧', '林业']
    for kw in low_rigidity:
        if kw in text:
            return 30

    return 45  # 未知行业默认中等偏低


# ============================================================
# 主评分函数
# ============================================================
def score_companies(raw_df, verbose=False):
    """
    对任意客户表做 10 维度 AHP 评分并分级。
    缺字段按规则推断或中性分处理，至少需要「客户名称」。
    """
    df = normalize_import_columns(raw_df.copy())
    if '客户名称' not in df.columns and '公司名称' in df.columns:
        df['客户名称'] = df['公司名称']
    if '客户名称' not in df.columns:
        raise ValueError("导入数据必须包含「客户名称」或「公司名称」列")
    df = df[df['客户名称'].notna() & (df['客户名称'].astype(str).str.strip() != '')]
    if df.empty:
        raise ValueError("没有有效的客户名称，无法评分")

    for col in IMPORT_TEMPLATE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = enrich_legacy_fields(df)
    df = apply_followup_overrides(df)  # 销售跟进表优先覆盖
    df = apply_new_customer_defaults(df)

    # 10维度打分
    df['组织规模得分'] = df.apply(score_organization_scale, axis=1)
    df['预算能力得分'] = df.apply(score_budget, axis=1)
    df['增长潜力得分'] = df.apply(score_growth_potential, axis=1)
    df['战略价值得分'] = df.apply(score_strategic_value, axis=1)
    df['决策链复杂度得分'] = df.apply(score_decision_chain, axis=1)
    df['合作成熟度得分'] = df.apply(score_cooperation_maturity, axis=1)
    df['竞争与替换成本得分'] = df.apply(score_competition_and_switching, axis=1)
    df['显性需求信号得分'] = df.apply(score_explicit_demand, axis=1)
    df['降本增效与替换潜力得分'] = df.apply(score_cost_reduction_and_replacement, axis=1)
    df['行业需求刚性得分'] = df.apply(score_industry_rigidity, axis=1)

    # AHP加权总分
    weights = get_ahp_weights()
    if verbose:
        print("AHP 一致性检验：")
        print(get_cr_report())
        print("\n10维全局权重：")
        for k, v in weights.items():
            print(f"  {DIMENSION_NAMES[k]:<12} {v*100:>5.1f}%")

    df['总分'] = 0.0
    for col, weight in weights.items():
        df['总分'] += df[col] * weight
    df['总分'] = df['总分'].round(2)
    df['客户分级'] = df['总分'].apply(grade_customer)
    if '公司名称' not in df.columns:
        df['公司名称'] = df['客户名称']
    return df


def merge_scored_into_base(base_df, scored_df):
    """按股票代码或客户名称合并：已有客户覆盖更新，新客户追加。"""
    if base_df is None or base_df.empty:
        return scored_df.copy()
    base = base_df.copy()
    scored = scored_df.copy()
    all_cols = list(dict.fromkeys(list(base.columns) + list(scored.columns)))
    for col in all_cols:
        if col not in base.columns:
            base[col] = np.nan
        if col not in scored.columns:
            scored[col] = np.nan
    base = base[all_cols]
    scored = scored[all_cols]

    def _key_row(row):
        code = _str(row.get('股票代码'))
        if code:
            digits = re.sub(r'\D', '', code)
            if digits:
                return 'c:' + (digits.zfill(6) if len(digits) <= 6 else digits[-6:])
        name = _str(row.get('客户名称')) or _str(row.get('公司名称'))
        return 'n:' + name

    base['_merge_key'] = base.apply(_key_row, axis=1)
    scored['_merge_key'] = scored.apply(_key_row, axis=1)
    kept = base[~base['_merge_key'].isin(set(scored['_merge_key']))]
    return pd.concat([kept, scored], ignore_index=True).drop(columns=['_merge_key'])


def main():
    print("=" * 70)
    print("客户评分模型 v2.0（全行业ToB客户分级）")
    print("=" * 70)

    # 尝试读取数据
    input_file = os.path.join(DATA_DIR, 'company_full_data.csv')
    template_file = os.path.join(DATA_DIR, 'customer_import_template.csv')

    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        print(f"读取到 {len(df)} 家公司数据（旧格式，将自动映射列名）")
    elif os.path.exists(template_file):
        df = pd.read_csv(template_file)
        print(f"读取到导入模板 {len(df)} 条数据")
    else:
        print("未找到数据文件，使用示例数据演示...")
        sample_data = [
            {'客户名称': '示例制造企业', '客户类型': '企业', '所属行业': '制造业',
             '企业性质': '民企', '员工人数': 800, '营业收入_万元': 15000,
             '营收同比增长率': 25, '有新建项目': '是'},
            {'客户名称': '示例三甲医院', '客户类型': '事业单位', '所属行业': '医疗',
             '床位数量': 1200, '有采购公告': '是', '现有系统使用年限': 7},
            {'客户名称': '示例高校', '客户类型': '事业单位', '所属行业': '教育',
             '学生人数': 25000, '有数字化转型规划': '是'},
        ]
        df = pd.DataFrame(sample_data)

    print("\n正在进行 10 维度 AHP 评分...")
    df = score_companies(df, verbose=True)

    # 统计分级分布
    grade_counts = df['客户分级'].value_counts()
    print("\n分级分布：")
    for grade in ['A级（高价值优先跟进）', 'B级（重点跟进）', 'C级（常规跟进）', 'D级（低优先级/观察）']:
        count = grade_counts.get(grade, 0)
        print(f"  {grade}：{count} 家（{count/len(df)*100:.1f}%）")

    # 保存结果
    output_file = os.path.join(OUTPUT_DIR, 'customer_scoring_result.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    v2_file = os.path.join(OUTPUT_DIR, 'customer_scoring_result_v2.csv')
    df.to_csv(v2_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存：{output_file}")

    # TOP10
    print("\nTOP 高价值客户：")
    top = df.sort_values('总分', ascending=False).head(min(10, len(df)))
    for _, row in top.iterrows():
        print(f"  {row['总分']:.1f}分 | {row['客户分级'][:2]} | {row['客户名称']}")


if __name__ == '__main__':
    main()
