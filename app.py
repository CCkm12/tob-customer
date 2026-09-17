"""
ToB客户智能评分系统 - 完整闭环版
两层架构：爬虫评分（找客户）+ AI联网尽调（懂客户）
运行：streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import re
import sys

st.set_page_config(page_title="ToB客户智能评分系统", page_icon="📊", layout="wide")

# ===== API Key 安全读取：云端读 st.secrets，本地读 .env =====
# 先把云端 st.secrets 的Key同步到环境变量（供 company_researcher 使用）
try:
    for _k in ["DEEPSEEK_API_KEY", "TAVILY_API_KEY"]:
        if hasattr(st, 'secrets') and _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass
# 加载本地 .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'outputs', 'customer_scoring_result.csv')
EXTRA_FILE = os.path.join(BASE_DIR, 'data', 'extra_data.csv')
PRODUCT_FILE = os.path.join(BASE_DIR, 'data', 'our_products.csv')
FOLLOWUP_FILE = os.path.join(BASE_DIR, 'data', 'sales_followup.csv')
PRODUCT_COLUMNS = ['产品名称', '产品类别', '适配行业', '目标客户规模', '核心能力', '产品亮点']
FOLLOWUP_COLUMNS = [
    '客户名称',
    '合作成熟度阶段', '现有供应商名称', '绑定程度', '竞争激烈程度',
    '客户抱怨现有产品', '是否行业标杆', '行业排名',
    '跟进日期', '跟进人', '备注',
]
FOLLOWUP_SCORE_FIELDS = [
    '合作成熟度阶段', '现有供应商名称', '绑定程度', '竞争激烈程度',
    '客户抱怨现有产品', '是否行业标杆', '行业排名', '跟进日期',
]
FOLLOWUP_DEFAULT_GUIDE = [
    {'字段': '合作成熟度阶段', '默认值': '仅有线索', '对应得分': '25 分；高成熟度需背书，过期跟进会衰减'},
    {'字段': '现有供应商名称', '默认值': '空（视为增量客户）', '对应得分': '影响 C7/C9；空则 C7 默认 60'},
    {'字段': '绑定程度', '默认值': '空', '对应得分': '有供应商全幅；无供应商半幅仍生效'},
    {'字段': '竞争激烈程度', '默认值': '空', '对应得分': '空不加减；有值改 C7'},
    {'字段': '客户抱怨现有产品', '默认值': '空（否）', '对应得分': '是：有供应商+15，无供应商+8'},
    {'字段': '是否行业标杆', '默认值': '空', '对应得分': '基础 35；标杆可与排名叠加'},
    {'字段': '行业排名', '默认值': '空', '对应得分': '空不触发；有值与标杆叠加'},
]

SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

RESEARCH_AVAILABLE = False
RESEARCH_IMPORT_ERROR = ""
research_company = None
generate_deep_report = None
save_research_outputs = None


def _load_company_researcher():
    """可靠加载尽调模块：优先按文件绝对路径 importlib 加载，避免 cwd/sys.path 问题。"""
    global RESEARCH_AVAILABLE, RESEARCH_IMPORT_ERROR
    global research_company, generate_deep_report, save_research_outputs
    RESEARCH_IMPORT_ERROR = ""
    errors = []

    # 方式1：绝对路径加载 src/company_researcher.py
    try:
        import importlib.util
        researcher_path = os.path.join(SRC_DIR, 'company_researcher.py')
        if not os.path.isfile(researcher_path):
            raise FileNotFoundError(f"找不到文件：{researcher_path}")
        spec = importlib.util.spec_from_file_location(
            "customer_scoring_company_researcher", researcher_path
        )
        if spec is None or spec.loader is None:
            raise ImportError("无法创建 module spec")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        research_company = mod.research_company
        generate_deep_report = mod.generate_deep_report
        save_research_outputs = getattr(mod, 'save_research_outputs', None)
        RESEARCH_AVAILABLE = True
        return True
    except Exception as e:
        errors.append(f"绝对路径加载失败：{e}")

    # 方式2：常规 from company_researcher import ...
    try:
        if SRC_DIR not in sys.path:
            sys.path.insert(0, SRC_DIR)
        from company_researcher import (  # noqa: WPS433
            research_company as _rc,
            generate_deep_report as _gd,
            save_research_outputs as _sr,
        )
        research_company = _rc
        generate_deep_report = _gd
        save_research_outputs = _sr
        RESEARCH_AVAILABLE = True
        return True
    except Exception as e:
        errors.append(f"sys.path 导入失败：{e}")

    RESEARCH_AVAILABLE = False
    RESEARCH_IMPORT_ERROR = " | ".join(errors)
    research_company = None
    generate_deep_report = None
    save_research_outputs = None
    return False


_load_company_researcher()

try:
    from scoring_model import score_companies, merge_scored_into_base, IMPORT_TEMPLATE_COLUMNS
    try:
        from scoring_model import _normalize_customer_name as normalize_customer_name
    except Exception:
        def normalize_customer_name(name):
            s = '' if name is None or (isinstance(name, float) and pd.isna(name)) else str(name).strip()
            return s
    SCORING_AVAILABLE = True
except Exception:
    SCORING_AVAILABLE = False
    def normalize_customer_name(name):
        s = '' if name is None or (isinstance(name, float) and pd.isna(name)) else str(name).strip()
        return s
    IMPORT_TEMPLATE_COLUMNS = [
        '客户名称', '客户类型', '所属行业', '企业性质', '所在地区',
        '员工人数', '营业收入_万元', '床位数量', '学生人数', '行政级别',
        'IT年度预算_万元', '营收同比增长率', '净利润同比增长率', '毛利率',
    ]

try:
    from ahp_weights import get_ahp_weights, DIMENSION_NAMES as AHP_DIM_NAMES, WEIGHT_ORDER
    DIMENSION_WEIGHTS = {k: float(v) for k, v in get_ahp_weights().items()}
    DIMENSION_NAMES = dict(AHP_DIM_NAMES)
except Exception:
    WEIGHT_ORDER = [
        '组织规模得分', '预算能力得分', '增长潜力得分', '战略价值得分',
        '决策链复杂度得分', '合作成熟度得分', '竞争与替换成本得分',
        '显性需求信号得分', '降本增效与替换潜力得分', '行业需求刚性得分',
    ]
    DIMENSION_WEIGHTS = {k: 0.1 for k in WEIGHT_ORDER}
    DIMENSION_NAMES = {k: k.replace('得分', '') for k in WEIGHT_ORDER}


def clean_industry(s):
    if pd.isna(s):
        return '未知'
    s = re.split(r'概念', str(s))[0]
    s = re.sub(r'[ⅠⅡⅢⅣⅤIⅡ]', '', s)
    s = s.strip().strip('：:').strip()
    return s if s else '未知'


def pad_stock_code(v):
    if pd.isna(v) or str(v).strip() in ('', 'nan', 'None'):
        return ''
    s = str(v).strip()
    digits = re.sub(r'\D', '', s)
    if digits and len(digits) <= 6:
        return digits.zfill(6)
    return s


def count_grade_prefix(series, letter):
    return int(series.fillna('').astype(str).str.startswith(letter).sum())


def available_score_dims(df):
    return [d for d in WEIGHT_ORDER if d in df.columns]


@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE)
    if '公司名称' not in df.columns and '客户名称' in df.columns:
        df['公司名称'] = df['客户名称']
    if '客户名称' not in df.columns and '公司名称' in df.columns:
        df['客户名称'] = df['公司名称']
    if '行业清洗' in df.columns:
        df['所属行业'] = df['行业清洗'].where(df['行业清洗'].notna() & (df['行业清洗'].astype(str).str.strip() != ''), df['所属行业'])
    df['所属行业'] = df['所属行业'].apply(clean_industry)
    if '股票代码' in df.columns:
        df['股票代码'] = df['股票代码'].map(pad_stock_code)
    if '总分' in df.columns:
        df['总分'] = pd.to_numeric(df['总分'], errors='coerce')
    if '客户分级' in df.columns:
        df['客户分级'] = df['客户分级'].fillna('未分级').astype(str)
    return df


@st.cache_data
def load_extra_data():
    if not os.path.exists(EXTRA_FILE):
        return None
    return pd.read_csv(EXTRA_FILE)


@st.cache_data
def load_products():
    if not os.path.exists(PRODUCT_FILE):
        return None
    df = pd.read_csv(PRODUCT_FILE)
    for col in PRODUCT_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    return df[PRODUCT_COLUMNS]


@st.cache_data
def load_followups():
    if not os.path.exists(FOLLOWUP_FILE):
        return pd.DataFrame(columns=FOLLOWUP_COLUMNS)
    df = pd.read_csv(FOLLOWUP_FILE)
    for col in FOLLOWUP_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    # 丢弃旧版股票代码等多余列
    return df[FOLLOWUP_COLUMNS]


def build_product_template_csv():
    sample = pd.DataFrame([
        {
            '产品名称': '企业级数据中台',
            '产品类别': '数据平台',
            '适配行业': '软件、云计算、人工智能',
            '目标客户规模': '大中型',
            '核心能力': '统一数据采集、治理与服务',
            '产品亮点': '开箱即用指标、权限隔离、可对接信创环境',
        },
        {
            '产品名称': '中小企业数字化套件',
            '产品类别': 'SaaS',
            '适配行业': '全部',
            '目标客户规模': '小型',
            '核心能力': '低成本上云与协同办公',
            '产品亮点': '按年订阅、实施周期短',
        },
    ])
    return sample[PRODUCT_COLUMNS].to_csv(index=False).encode('utf-8-sig')


def persist_products(df):
    os.makedirs(os.path.dirname(PRODUCT_FILE), exist_ok=True)
    out = df.copy()
    for col in PRODUCT_COLUMNS:
        if col not in out.columns:
            out[col] = ''
    out[PRODUCT_COLUMNS].to_csv(PRODUCT_FILE, index=False, encoding='utf-8-sig')


def persist_followups(df):
    os.makedirs(os.path.dirname(FOLLOWUP_FILE), exist_ok=True)
    out = df.copy()
    for col in FOLLOWUP_COLUMNS:
        if col not in out.columns:
            out[col] = ''
    out[FOLLOWUP_COLUMNS].to_csv(FOLLOWUP_FILE, index=False, encoding='utf-8-sig')


def build_followup_template_csv():
    sample = pd.DataFrame([{
        '客户名称': '示例科技有限公司',
        '合作成熟度阶段': '仅有线索',
        '现有供应商名称': '',
        '绑定程度': '',
        '竞争激烈程度': '',
        '客户抱怨现有产品': '否',
        '是否行业标杆': '',
        '行业排名': '',
        '跟进日期': '2026-09-18',
        '跟进人': '销售姓名',
        '备注': '新客户默认字段示例',
    }])
    return sample[FOLLOWUP_COLUMNS].to_csv(index=False).encode('utf-8-sig')


def apply_followup_row_defaults(row: dict) -> dict:
    """单条跟进记录套用新客户默认规则。"""
    out = dict(row)
    if not str(out.get('合作成熟度阶段', '') or '').strip():
        out['合作成熟度阶段'] = '仅有线索'
    if not str(out.get('客户抱怨现有产品', '') or '').strip():
        out['客户抱怨现有产品'] = '否'
    return out


def rescore_customers_with_followup(customers_df, followup_df):
    """
    把销售跟进字段写回客户数据并重新评级。
    按规范化客户名称弱匹配；返回更新后的完整客户库。
    """
    if customers_df is None or customers_df.empty:
        return customers_df, 0
    if followup_df is None or followup_df.empty:
        return customers_df, 0
    if not SCORING_AVAILABLE:
        raise RuntimeError("评分模块未加载")

    base = customers_df.copy()
    if '客户名称' not in base.columns and '公司名称' in base.columns:
        base['客户名称'] = base['公司名称']

    fu = followup_df.copy()
    fu['_key'] = fu['客户名称'].map(normalize_customer_name)
    fu = fu[fu['_key'].astype(str).str.len() > 0].drop_duplicates(subset=['_key'], keep='last')
    if fu.empty:
        return customers_df, 0
    fu_map = fu.set_index('_key')
    name_keys = set(fu_map.index)

    key_a = base['客户名称'].map(normalize_customer_name)
    key_b = base['公司名称'].map(normalize_customer_name) if '公司名称' in base.columns else pd.Series([''] * len(base), index=base.index)
    match_mask = key_a.isin(name_keys) | key_b.isin(name_keys)
    targets = base[match_mask].copy()
    if targets.empty:
        return customers_df, 0

    for idx, row in targets.iterrows():
        key = normalize_customer_name(row.get('客户名称', '')) or normalize_customer_name(row.get('公司名称', ''))
        if key not in fu_map.index:
            continue
        fr = fu_map.loc[key]
        for col in FOLLOWUP_SCORE_FIELDS:
            if col in fr.index:
                val = fr[col]
                targets.at[idx, col] = '' if pd.isna(val) else val

    rescored = score_companies(targets)
    updated = merge_scored_into_base(base, rescored)
    persist_scored_results(updated)
    load_data.clear()
    return updated, len(targets)


def _split_tags(text):
    if pd.isna(text):
        return []
    parts = re.split(r'[,，、/;；|]+', str(text))
    return [p.strip() for p in parts if p.strip()]


# 行业同义词映射：客户行业关键词 -> 产品描述中可能出现的相关词
INDUSTRY_SYNONYMS = {
    '软件': ['软件', '计算机', 'IT', '信息技术', '信息服务', '互联网', '科技', '数字化'],
    '半导体': ['半导体', '芯片', '集成电路', '电子', '微电子'],
    '计算机设备': ['计算机', '服务器', '硬件', '设备', 'IT', '算力'],
    '通信': ['通信', '通讯', '电信', '5G', '网络', '光通信'],
    '金融': ['金融', '银行', '证券', '保险', '基金', '信托'],
    '医疗': ['医疗', '医药', '医院', '健康', '生物'],
    '教育': ['教育', '高校', '学校', '培训', '职教'],
    '制造业': ['制造', '工业', '工厂', '生产', '智能制造'],
    '政府': ['政府', '政务', '部委', '智慧城市', '央国企', '国企'],
    '能源': ['能源', '电力', '石油', '煤炭', '新能源', '储能'],
    '房地产': ['房地产', '地产', '建筑', '基建'],
    '零售': ['零售', '百货', '超市', '电商', '消费'],
    '交通运输': ['交通', '运输', '物流', '航空', '铁路', '航运'],
}

# 表示"适配所有行业"的关键词（出现在产品适配行业描述中即视为全行业适配）
UNIVERSAL_INDUSTRY_KEYWORDS = [
    '全部', '通用', '不限', '全行业', '各行业', '千行百业',
    '企业', '政企', '全领域', '各行各业', '全场景'
]


def _extract_industry_keywords(text):
    """从复杂的行业描述中提取核心行业关键词（去掉括号内容、数字、描述性文字）"""
    if pd.isna(text) or not str(text).strip():
        return []
    text = str(text)
    # 去掉括号及括号内内容
    text = re.sub(r'[（(][^）)]*[）)]', '', text)
    # 去掉数字、百分比、加号、波浪线
    text = re.sub(r'\d+(\.\d+)?[%％+~～]?', '', text)
    # 去掉描述性词汇
    for w in ['累计', '家用户', '家', '用户', '客户', '份额', '超', '以上', '以下',
              '级', '中心', '服务', '解决方案', '领域', '行业']:
        text = text.replace(w, '')
    # 按分隔符拆分
    tokens = re.split(r'[,，、/;；|：:→]+', text)
    return [t.strip() for t in tokens if t.strip() and 1 < len(t.strip()) <= 12]


def infer_customer_scale_label(company):
    """根据规模得分/注册资本推断大中小，不改动原有打分字段。"""
    score = company.get('组织规模得分', None) if hasattr(company, 'get') else None
    if score is None or (isinstance(score, float) and pd.isna(score)):
        score = company.get('规模得分', None) if hasattr(company, 'get') else None
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    if score is None or pd.isna(score):
        cap = company.get('注册资本_万元', None) if hasattr(company, 'get') else None
        try:
            cap = float(cap)
        except (TypeError, ValueError):
            cap = None
        if cap is not None and not pd.isna(cap):
            if cap >= 50000:
                score = 80
            elif cap >= 5000:
                score = 65
            else:
                score = 45
        else:
            score = 50
    if score >= 70:
        return '大型'
    if score >= 55:
        return '中型'
    return '小型'


def _industry_matched(product_industries, customer_industry):
    """增强版行业匹配：支持复杂描述、通用行业检测、行业同义词映射"""
    if pd.isna(product_industries) or not str(product_industries).strip():
        return False, 0
    product_text = str(product_industries)

    # 1. 检测通用行业关键词（产品描述中出现这些词，视为适配所有行业）
    for kw in UNIVERSAL_INDUSTRY_KEYWORDS:
        if kw in product_text:
            return True, 1

    # 2. 从复杂描述中提取产品行业关键词
    product_keywords = _extract_industry_keywords(product_text)
    if not product_keywords:
        return False, 0

    # 3. 处理客户行业
    if pd.isna(customer_industry) or not str(customer_industry).strip():
        return False, 0
    customer_ind = str(customer_industry).strip()

    # 4. 构建客户行业同义词集合
    customer_synonyms = {customer_ind}
    for key, syns in INDUSTRY_SYNONYMS.items():
        # 客户行业匹配到同义词表的key
        if key in customer_ind or customer_ind in key:
            customer_synonyms.add(key)
            customer_synonyms.update(syns)
            break
        # 客户行业匹配到同义词表的值
        if any(customer_ind in s or s in customer_ind for s in syns):
            customer_synonyms.add(key)
            customer_synonyms.update(syns)
            break

    # 5. 关键词匹配
    hits = 0
    for pk in product_keywords:
        for cs in customer_synonyms:
            if cs and (pk in cs or cs in pk):
                hits += 1
                break

    return hits > 0, hits


def _scale_matched(product_scale, customer_scale):
    tokens = _split_tags(product_scale)
    blob = ''.join(tokens) if tokens else ('' if pd.isna(product_scale) else str(product_scale))
    if not blob or any(k in blob for k in ('全部', '不限', '通用', '全规模')):
        return True, 1
    mapping = {
        '大型': ['大型', '大中型', '大客户', '集团', '上市'],
        '中型': ['中型', '大中型', '中小'],
        '小型': ['小型', '中小', '中小型'],
    }
    keys = mapping.get(customer_scale, [customer_scale])
    if any(k in blob for k in keys):
        return True, 2
    return False, 0


def match_products_for_customer(products_df, company):
    """按客户行业与企业规模筛选适配产品。"""
    if products_df is None or products_df.empty:
        return pd.DataFrame(columns=PRODUCT_COLUMNS + ['匹配说明', '匹配分'])
    industry = company.get('所属行业', '') if hasattr(company, 'get') else ''
    scale_label = infer_customer_scale_label(company)
    rows = []
    for _, p in products_df.iterrows():
        ok_ind, ind_score = _industry_matched(p.get('适配行业', ''), industry)
        ok_scale, scale_score = _scale_matched(p.get('目标客户规模', ''), scale_label)
        if not (ok_ind and ok_scale):
            continue
        reasons = [f"行业匹配（客户：{industry or '未知'}）", f"规模匹配（客户：{scale_label}）"]
        rec = {c: p.get(c, '') for c in PRODUCT_COLUMNS}
        rec['匹配说明'] = '；'.join(reasons)
        rec['匹配分'] = int(ind_score * 10 + scale_score * 5)
        rec['推断客户规模'] = scale_label
        rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=PRODUCT_COLUMNS + ['匹配说明', '匹配分'])
    out = pd.DataFrame(rows).sort_values('匹配分', ascending=False)
    return out


def render_recommended_products(products_df, company):
    st.markdown("### 🧩 推荐产品（按行业与规模匹配）")
    if products_df is None or products_df.empty:
        st.info("尚未导入本公司产品。请到「本公司产品」页上传 CSV 产品清单。")
        return None
    matched = match_products_for_customer(products_df, company)
    scale_label = infer_customer_scale_label(company)
    industry = company.get('所属行业', '') if hasattr(company, 'get') else ''
    st.caption(f"当前客户推断规模：**{scale_label}**　所属行业：**{industry or '未知'}**")
    if matched.empty:
        st.warning("暂无同时满足行业与规模条件的产品。以下是产品库中各产品的匹配状态：")
        for _, p in products_df.iterrows():
            ok_ind, ind_hits = _industry_matched(p.get('适配行业', ''), industry)
            ok_scale, _ = _scale_matched(p.get('目标客户规模', ''), scale_label)
            ind_status = "命中" if ok_ind else "未命中"
            scale_status = "命中" if ok_scale else "未命中"
            pname = str(p.get('产品名称', ''))[:40]
            st.markdown(f"- 行业{ind_status} / 规模{scale_status}　**{pname}**")
        st.caption("提示：可在产品清单中把「适配行业」设为「全部」，或补充客户行业对应的关键词。")
        return None
    show_cols = [c for c in PRODUCT_COLUMNS + ['匹配说明'] if c in matched.columns]
    preview = matched[show_cols].copy()
    for c in preview.columns:
        preview[c] = preview[c].astype(str).str.slice(0, 80)
    st.dataframe(preview, use_container_width=True, hide_index=True)
    names = matched['产品名称'].astype(str).tolist()
    picked = st.selectbox("选择推荐产品，用于生成尽调 / 解决方案", names, key='picked_reco_product')
    row = matched[matched['产品名称'].astype(str) == picked].iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**核心能力：** {str(row.get('核心能力', ''))[:180]}")
        st.markdown(f"**产品类别：** {str(row.get('产品类别', ''))[:80]}")
    with c2:
        st.markdown(f"**产品亮点：** {str(row.get('产品亮点', ''))[:180]}")
        st.markdown(f"**目标客户规模：** {str(row.get('目标客户规模', ''))[:80]}")
    solution_text = (
        f"{row.get('产品名称', '')}（{str(row.get('产品类别', ''))[:40]}）。"
        f"核心能力：{str(row.get('核心能力', ''))[:120]}。"
        f"产品亮点：{str(row.get('产品亮点', ''))[:120]}。"
    )
    st.success("可将上方产品作为方案切入点，在下方 AI 分析中生成对应该客户的解决方案。")
    return solution_text


def render_product_import_tab(products_df):
    st.subheader("📦 本公司产品导入")
    st.caption("上传 CSV 产品清单。导入后，系统会按客户所属行业与企业规模自动推荐适配产品。")
    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader(
            "上传产品清单（CSV）",
            type=['csv'],
            key='product_csv_uploader',
            help="字段：产品名称、产品类别、适配行业、目标客户规模、核心能力、产品亮点",
        )
    with c2:
        st.download_button(
            "下载产品导入模板",
            data=build_product_template_csv(),
            file_name='product_import_template.csv',
            mime='text/csv',
            key='dl_product_tpl',
        )
        write_mode = st.radio(
            "写入方式",
            ['覆盖现有产品库', '追加到现有产品库'],
            index=0,
            key='product_write_mode',
        )

    st.markdown("**必填：** 产品名称、适配行业、目标客户规模　　**建议填写：** 产品类别、核心能力、产品亮点")
    st.markdown(
        "- **适配行业**：多个行业用顿号或逗号分隔；填写「全部」表示适用所有行业。\n"
        "- **目标客户规模**：填写大型 / 中型 / 小型 / 大中型 / 中小 / 不限。"
    )

    if products_df is not None and not products_df.empty:
        st.markdown("#### 当前产品库")
        preview_p = products_df.copy()
        for c in preview_p.columns:
            preview_p[c] = preview_p[c].astype(str).str.slice(0, 80)
        st.dataframe(preview_p, use_container_width=True, hide_index=True)
        st.caption(f"共 {len(products_df)} 条产品")

    if uploaded is None:
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        return

    rename = {}
    alias = {
        '产品名称': ['产品名称', '名称', '产品'],
        '产品类别': ['产品类别', '类别', '类型'],
        '适配行业': ['适配行业', '适用行业', '行业'],
        '目标客户规模': ['目标客户规模', '客户规模', '规模'],
        '核心能力': ['核心能力', '能力'],
        '产品亮点': ['产品亮点', '亮点'],
    }
    lower = {str(c).strip().lower(): c for c in raw.columns}
    for canon, names in alias.items():
        if canon in raw.columns:
            continue
        for n in names:
            if n.lower() in lower:
                rename[lower[n.lower()]] = canon
                break
    if rename:
        raw = raw.rename(columns=rename)

    missing = [c for c in ['产品名称', '适配行业', '目标客户规模'] if c not in raw.columns]
    if missing:
        st.error(f"缺少必填列：{'、'.join(missing)}")
        return

    for col in PRODUCT_COLUMNS:
        if col not in raw.columns:
            raw[col] = ''
    raw = raw[PRODUCT_COLUMNS]
    raw = raw[raw['产品名称'].notna() & (raw['产品名称'].astype(str).str.strip() != '')]
    st.markdown("#### 预览待导入数据")
    st.dataframe(raw.head(30), use_container_width=True, hide_index=True)
    st.caption(f"共 {len(raw)} 条有效产品")

    if st.button("保存到产品库", type="primary", key='save_products'):
        if write_mode == '追加到现有产品库' and products_df is not None and not products_df.empty:
            merged = pd.concat([products_df, raw], ignore_index=True)
            merged = merged.drop_duplicates(subset=['产品名称'], keep='last')
        else:
            merged = raw.drop_duplicates(subset=['产品名称'], keep='last')
        persist_products(merged)
        load_products.clear()
        st.success(f"已保存 {len(merged)} 条产品，页面即将刷新。")
        st.rerun()


def build_score_overview_for_research(company, df):
    """构造尽调用的评分概览：含低分维度，供报告开头与深挖。"""
    overview = {
        '总分': company.get('总分', ''),
        '客户分级': company.get('客户分级', ''),
        '维度得分': [],
        '低分维度': [],
    }
    dims = available_score_dims(df) if df is not None else []
    rows = []
    for dim in dims:
        try:
            sc = float(company[dim]) if pd.notna(company.get(dim)) else None
        except Exception:
            sc = None
        if sc is None:
            continue
        w = DIMENSION_WEIGHTS.get(dim, 0)
        name = DIMENSION_NAMES.get(dim, dim)
        low = sc < 45
        rows.append({'维度': name, '得分': round(sc, 1), '权重': f"{w*100:.1f}%", '是否低分': low, '_sc': sc})
    rows.sort(key=lambda x: x['_sc'])
    for r in rows:
        overview['维度得分'].append({k: v for k, v in r.items() if k != '_sc'})
        if r['是否低分']:
            overview['低分维度'].append(r['维度'])
    # 若都没有明显低分，取最低的 3 个作为深挖重点
    if not overview['低分维度'] and rows:
        overview['低分维度'] = [r['维度'] for r in rows[:3]]
        for item in overview['维度得分']:
            if item['维度'] in overview['低分维度']:
                item['是否低分'] = True
    return overview


def get_sales_followup_for_company(company_name, followup_df, company_row=None):
    """
    按客户名称从销售跟进表取出一手情报（可多条，取规范化名匹配）。
    返回 list[dict]；无匹配返回 []。
    """
    if followup_df is None or followup_df.empty or not company_name:
        return []
    keys = {normalize_customer_name(company_name)}
    if company_row is not None:
        for col in ('客户名称', '公司名称'):
            if hasattr(company_row, 'get'):
                v = company_row.get(col)
            else:
                v = None
            nk = normalize_customer_name(v)
            if nk:
                keys.add(nk)
    keys.discard('')
    if not keys:
        return []

    fu = followup_df.copy()
    fu['_key'] = fu['客户名称'].map(normalize_customer_name)
    matched = fu[fu['_key'].isin(keys)]
    if matched.empty:
        return []

    rows = []
    for _, r in matched.iterrows():
        item = {}
        for col in FOLLOWUP_COLUMNS:
            if col not in matched.columns:
                continue
            val = r.get(col)
            if pd.isna(val):
                item[col] = ''
            else:
                item[col] = str(val).strip()
        rows.append(item)
    return rows


def render_followup_tab(followup_df, customers_df=None):
    st.subheader("销售跟进记录")
    st.caption("维护销售跟进字段；保存后可按客户名称回写并重新评级。数据：data/sales_followup.csv。")

    st.markdown("#### 默认字段与得分对照")
    st.dataframe(pd.DataFrame(FOLLOWUP_DEFAULT_GUIDE), use_container_width=True, hide_index=True)

    st.markdown("#### 当前跟进表")
    if followup_df is None or followup_df.empty:
        st.info("暂无跟进记录。可在下方新增，或上传 CSV。")
    else:
        show_fu = followup_df[[c for c in FOLLOWUP_COLUMNS if c in followup_df.columns]].copy()
        st.dataframe(show_fu, use_container_width=True, hide_index=True)
        st.caption(f"共 {len(show_fu)} 条")
        st.download_button(
            "导出跟进表",
            show_fu.to_csv(index=False).encode('utf-8-sig'),
            'sales_followup.csv',
            mime='text/csv',
            key='dl_followup_current',
        )

    if customers_df is not None and not customers_df.empty and followup_df is not None and not followup_df.empty:
        if st.button("用跟进记录重新评级匹配客户", type="secondary", key='fu_rescore_all'):
            try:
                _, n = rescore_customers_with_followup(customers_df, followup_df)
                if n == 0:
                    st.warning("跟进表中的客户名称未在客户库中匹配到记录。")
                else:
                    st.success(f"已按跟进字段重新评级 {n} 家客户，并写入客户库。")
                    st.rerun()
            except Exception as e:
                st.error(f"重新评级失败：{e}")

    st.divider()
    st.markdown("#### 新增 / 更新一条跟进")
    name_options = []
    if customers_df is not None and not customers_df.empty:
        name_col = '公司名称' if '公司名称' in customers_df.columns else '客户名称'
        name_options = sorted(customers_df[name_col].dropna().astype(str).unique().tolist())

    c1, c2 = st.columns(2)
    with c1:
        if name_options:
            pick_mode = st.radio("客户来源", ['从客户库选择', '手动输入'], horizontal=True, key='fu_pick_mode')
        else:
            pick_mode = '手动输入'
        if pick_mode == '从客户库选择':
            cust_name = st.selectbox("客户名称", name_options, key='fu_name_sel')
        else:
            cust_name = st.text_input("客户名称", key='fu_name_manual')
        maturity = st.selectbox(
            "合作成熟度阶段",
            ['仅有线索', '完全陌生', '初步接触', '深度接触', '商务谈判', '已有合作'],
            index=0,
            key='fu_maturity',
        )
        vendor = st.text_input("现有供应商名称（空=增量客户）", key='fu_vendor')
        binding = st.selectbox("绑定程度", ['', '浅', '中等', '深'], index=0, key='fu_binding')
        competition = st.selectbox("竞争激烈程度", ['', '小', '中等', '激烈'], index=0, key='fu_comp')
    with c2:
        complaint = st.selectbox("客户抱怨现有产品", ['否', '是'], index=0, key='fu_complaint')
        benchmark = st.selectbox("是否行业标杆", ['', '是', '否'], index=0, key='fu_bench')
        rank = st.text_input("行业排名（空=不触发分档）", key='fu_rank')
        follow_date = st.text_input("跟进日期", value=pd.Timestamp.today().strftime('%Y-%m-%d'), key='fu_date')
        follower = st.text_input("跟进人", key='fu_person')
        note = st.text_area("备注", key='fu_note', height=80)

    also_rescore = st.checkbox("保存后同步重新评级该客户", value=True, key='fu_also_rescore')

    if st.button("保存本条跟进", type="primary", key='fu_save_one'):
        if not str(cust_name or '').strip():
            st.error("请填写客户名称")
        else:
            row = apply_followup_row_defaults({
                '客户名称': str(cust_name).strip(),
                '合作成熟度阶段': maturity,
                '现有供应商名称': vendor,
                '绑定程度': binding,
                '竞争激烈程度': competition,
                '客户抱怨现有产品': complaint,
                '是否行业标杆': benchmark,
                '行业排名': rank,
                '跟进日期': follow_date,
                '跟进人': follower,
                '备注': note,
            })
            base = followup_df.copy() if followup_df is not None else pd.DataFrame(columns=FOLLOWUP_COLUMNS)
            for col in FOLLOWUP_COLUMNS:
                if col not in base.columns:
                    base[col] = ''
            # 兼容旧文件里残留的股票代码列
            drop_cols = [c for c in base.columns if c not in FOLLOWUP_COLUMNS]
            if drop_cols:
                base = base.drop(columns=drop_cols, errors='ignore')
            new_df = pd.DataFrame([row])[FOLLOWUP_COLUMNS]
            mask = base['客户名称'].astype(str).str.strip() == row['客户名称']
            kept = base[~mask]
            merged = pd.concat([kept, new_df], ignore_index=True)
            persist_followups(merged)
            load_followups.clear()
            msg = f"已保存「{row['客户名称']}」跟进记录"
            if also_rescore and customers_df is not None and not customers_df.empty:
                try:
                    _, n = rescore_customers_with_followup(customers_df, new_df)
                    msg += f"；已重新评级 {n} 家" if n else "；客户库中未匹配到同名客户，未改评级"
                except Exception as e:
                    st.error(f"跟进已保存，但重新评级失败：{e}")
                    st.rerun()
            st.success(msg)
            st.rerun()

    st.divider()
    st.markdown("#### 批量导入 CSV")
    u1, u2 = st.columns([2, 1])
    with u1:
        uploaded = st.file_uploader("上传跟进表（CSV）", type=['csv'], key='fu_csv_uploader')
    with u2:
        st.download_button(
            "下载跟进模板",
            data=build_followup_template_csv(),
            file_name='sales_followup_template.csv',
            mime='text/csv',
            key='dl_fu_tpl',
        )
        write_mode = st.radio(
            "写入方式",
            ['覆盖现有跟进表', '追加/按客户覆盖'],
            index=1,
            key='fu_write_mode',
        )

    if uploaded is None:
        return

    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        return

    rename = {}
    alias = {
        '客户名称': ['客户名称', '公司名称', '企业名称'],
        '合作成熟度阶段': ['合作成熟度阶段', '接触阶段', '合作阶段'],
        '现有供应商名称': ['现有供应商名称', '现有供应商', '供应商'],
        '绑定程度': ['绑定程度', '替换难度'],
        '竞争激烈程度': ['竞争激烈程度', '竞争程度'],
        '客户抱怨现有产品': ['客户抱怨现有产品', '客户抱怨'],
        '是否行业标杆': ['是否行业标杆', '行业标杆'],
        '行业排名': ['行业排名', '排名'],
        '跟进日期': ['跟进日期', '日期'],
        '跟进人': ['跟进人', '销售', '负责人'],
        '备注': ['备注', '说明'],
    }
    lower = {str(c).strip().lower(): c for c in raw.columns}
    for canon, names in alias.items():
        if canon in raw.columns:
            continue
        for n in names:
            if n.lower() in lower:
                rename[lower[n.lower()]] = canon
                break
    if rename:
        raw = raw.rename(columns=rename)

    if '客户名称' not in raw.columns:
        st.error("缺少必填列：客户名称")
        return

    for col in FOLLOWUP_COLUMNS:
        if col not in raw.columns:
            raw[col] = ''
    rows = []
    for _, r in raw.iterrows():
        if not str(r.get('客户名称', '') or '').strip():
            continue
        rows.append(apply_followup_row_defaults({c: r.get(c, '') for c in FOLLOWUP_COLUMNS}))
    if not rows:
        st.error("没有有效的跟进行")
        return
    incoming = pd.DataFrame(rows)[FOLLOWUP_COLUMNS]
    st.markdown("#### 预览待导入")
    st.dataframe(incoming.head(30), use_container_width=True, hide_index=True)
    st.caption(f"共 {len(incoming)} 条有效记录")
    also_rescore_batch = st.checkbox("导入后同步重新评级匹配客户", value=True, key='fu_also_rescore_batch')

    if st.button("保存跟进表", type="primary", key='fu_save_batch'):
        if write_mode == '覆盖现有跟进表':
            merged = incoming.drop_duplicates(subset=['客户名称'], keep='last')
        else:
            base = followup_df.copy() if followup_df is not None else pd.DataFrame(columns=FOLLOWUP_COLUMNS)
            for col in FOLLOWUP_COLUMNS:
                if col not in base.columns:
                    base[col] = ''
            drop_cols = [c for c in base.columns if c not in FOLLOWUP_COLUMNS]
            if drop_cols:
                base = base.drop(columns=drop_cols, errors='ignore')
            names = set(incoming['客户名称'].astype(str).str.strip())
            kept = base[~base['客户名称'].astype(str).str.strip().isin(names)]
            merged = pd.concat([kept, incoming], ignore_index=True)
            merged = merged.drop_duplicates(subset=['客户名称'], keep='last')
        persist_followups(merged)
        load_followups.clear()
        msg = f"已保存 {len(merged)} 条跟进记录"
        if also_rescore_batch and customers_df is not None and not customers_df.empty:
            try:
                _, n = rescore_customers_with_followup(customers_df, merged)
                msg += f"；已重新评级 {n} 家客户"
            except Exception as e:
                st.error(f"跟进已保存，但重新评级失败：{e}")
                st.rerun()
        st.success(msg)
        st.rerun()


def build_import_template_csv():
    # 示例行刻意留空「销售维护字段」，导入后会自动套用新客户默认值
    sample = pd.DataFrame([{
        '客户名称': '示例科技有限公司',
        '客户类型': '企业',
        '所属行业': '软件',
        '企业性质': '民营企业',
        '所在地区': '广东省深圳市',
        '员工人数': '800',
        '营业收入_万元': '120000',
        '床位数量': '',
        '学生人数': '',
        '行政级别': '',
        'IT年度预算_万元': '500',
        '营收同比增长率': '25%',
        '净利润同比增长率': '18%',
        '毛利率': '42%',
        '成立年限': '12',
        '是否需要招投标': '是',
        '现有供应商名称': '',
        '现有合同到期时间': '',
        '现有系统使用年限': '',
        '有采购公告': '是',
        '有重新招标': '否',
        '合同半年内到期': '否',
        '有新建项目': '是',
        '有数字化转型规划': '是',
        'IT岗位招聘数': '5',
        '现有供应商负面新闻': '否',
        '客户抱怨现有产品': '',
        '合作成熟度阶段': '',
        '竞争激烈程度': '',
        '绑定程度': '',
        '行业排名': '',
        '是否行业标杆': '',
    }])
    return sample[IMPORT_TEMPLATE_COLUMNS].to_csv(index=False).encode('utf-8-sig')


def read_uploaded_companies(uploaded):
    name = uploaded.name.lower()
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)


def persist_scored_results(df):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
    top50 = df.sort_values('总分', ascending=False).head(50)
    top50.to_csv(
        os.path.join(BASE_DIR, 'outputs', 'top50_high_value_customers.csv'),
        index=False, encoding='utf-8-sig'
    )


def render_import_tab(existing_df):
    st.subheader("📥 导入公司数据并自动评级")
    st.caption("上传 CSV / Excel，系统按 10 维度 AHP 模型自动打分并给出 A/B/C/D 分级。缺字段按中性分处理。")
    if not SCORING_AVAILABLE:
        st.error("评分模块未加载，请确认 src/scoring_model.py 可正常导入。")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader(
            "上传公司名单（CSV 或 Excel）",
            type=['csv', 'xlsx', 'xls'],
            help="至少包含「客户名称」或「公司名称」。指标越全，评级越准。",
        )
    with c2:
        st.download_button(
            "📥 下载导入模板",
            data=build_import_template_csv(),
            file_name='customer_import_template.csv',
            mime='text/csv',
        )
        merge_mode = st.radio(
            "写入方式",
            ['追加/覆盖到现有客户库', '仅本次预览，不写入库'],
            index=0,
        )

    st.markdown(
        "**必填：** 客户名称　**建议填写：** 所属行业、企业性质、所在地区、员工人数、"
        "营业收入_万元、IT年度预算_万元、营收同比增长率、毛利率"
    )
    st.caption(
        "未填销售维护字段时自动默认：合作成熟度=仅有线索(25分)；"
        "无现有供应商→增量客户 C7=60分；绑定/竞争空不加减分；"
        "客户抱怨空→否；行业标杆空→战略价值35分；行业排名空不触发分档。"
    )

    if uploaded is None:
        return

    try:
        raw = read_uploaded_companies(uploaded)
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        return

    st.markdown("#### 预览原始数据")
    st.dataframe(raw.head(20), use_container_width=True, hide_index=True)
    st.caption(f"共 {len(raw)} 行")

    if st.button("🚀 开始自动评级", type="primary"):
        try:
            scored = score_companies(raw)
        except Exception as e:
            st.error(f"评分失败：{e}")
            return
        st.session_state['last_import_scored'] = scored
        st.session_state['last_import_merge'] = merge_mode
        st.success(f"已完成 {len(scored)} 家公司评级")

    scored = st.session_state.get('last_import_scored')
    if scored is None:
        return

    show_cols = [c for c in ['股票代码', '公司名称', '客户名称', '所属行业', '总分', '客户分级',
                             '组织规模得分', '预算能力得分', '增长潜力得分', '战略价值得分'] if c in scored.columns]
    st.markdown("#### 评级结果")
    st.dataframe(scored[show_cols].sort_values('总分', ascending=False),
                 use_container_width=True, hide_index=True)
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("导入公司数", f"{len(scored)} 家")
    g2.metric("平均分", f"{scored['总分'].mean():.1f}")
    g3.metric("A级", f"{count_grade_prefix(scored['客户分级'], 'A')} 家")
    g4.metric("B级", f"{count_grade_prefix(scored['客户分级'], 'B')} 家")
    st.download_button(
        "📥 导出本次评级结果",
        scored.to_csv(index=False).encode('utf-8-sig'),
        'imported_companies_scored.csv',
        mime='text/csv',
    )

    if st.session_state.get('last_import_merge') == '追加/覆盖到现有客户库':
        if st.button("💾 写入客户库并刷新看板"):
            merged = merge_scored_into_base(existing_df, scored)
            persist_scored_results(merged)
            load_data.clear()
            st.session_state.pop('last_import_scored', None)
            st.success(f"已写入客户库，当前共 {len(merged)} 家。页面即将刷新。")
            st.rerun()


def main():
    st.title("ToB 客户智能分级与销售情报")
    st.caption("10 维 AHP 分级 · 公司/代码检索 · 产品匹配 · 深度分析。缺字段按规则推断，不阻断看板。")

    df = load_data()
    extra_df = load_extra_data()
    products_df = load_products()
    followup_df = load_followups()

    if df is None:
        st.warning("尚未有评分结果库。你可以先在「导入评级」页上传公司数据自动打分。")
        tab_import, tab_prod, tab_fu = st.tabs(["📥 导入评级", "📦 本公司产品", "销售跟进"])
        with tab_import:
            render_import_tab(None)
        with tab_prod:
            render_product_import_tab(products_df)
        with tab_fu:
            render_followup_tab(followup_df, None)
        return

    with st.sidebar:
        st.header("⚙️ 筛选条件")
        search_query = st.text_input(
            "搜索公司 / 股票代码",
            placeholder="例如：达梦数据 或 688692",
            help="按公司名称或股票代码快速定位，可与下方筛选条件叠加使用",
        ).strip()
        grade_options = ['全部', 'A级（高价值优先跟进）', 'B级（重点跟进）', 'C级（常规跟进）', 'D级（低优先级/观察）']
        selected_grade = st.selectbox("客户分级", grade_options)
        selected_exchange = st.selectbox("交易所", ['全部', '深交所', '上交所'])
        industries = ['全部'] + sorted([x for x in df['所属行业'].dropna().astype(str).unique().tolist() if x and x != 'nan'])
        selected_industry = st.selectbox("所属行业", industries)
        min_score = st.slider("最低总分", 0, 100, 0)
        st.divider()
        st.metric("客户总数", f"{len(df)} 家")
        if extra_df is not None:
            st.metric("市值数据覆盖", f"{len(extra_df)} 家")
        if products_df is not None:
            st.metric("本公司产品", f"{len(products_df)} 个")
        st.metric("销售跟进记录", f"{len(followup_df)} 条")
        st.divider()
        st.success("✅ AI联网尽调已启用" if RESEARCH_AVAILABLE else "⚠️ 联网尽调模块未加载")
        if not RESEARCH_AVAILABLE and RESEARCH_IMPORT_ERROR:
            st.caption(f"加载失败原因：{RESEARCH_IMPORT_ERROR[:200]}")
        if DEEPSEEK_API_KEY:
            st.success("✅ DeepSeek API Key 已配置")
        else:
            st.warning("⚠️ 未配置 DeepSeek API Key，尽调报告不可用")
        if os.getenv("TAVILY_API_KEY", ""):
            st.success("✅ Tavily API Key 已配置")
        else:
            st.warning("⚠️ 未配置 Tavily API Key，联网搜索不可用")

    filtered = df.copy()
    if search_query:
        code_str = filtered['股票代码'].map(pad_stock_code) if '股票代码' in filtered.columns else pd.Series('', index=filtered.index)
        name_str = filtered['公司名称'].astype(str)
        q = search_query.lower()
        q_digits = re.sub(r'\D', '', search_query)
        match_name = name_str.str.lower().str.contains(q, na=False, regex=False)
        if q_digits:
            padded = q_digits.zfill(6) if len(q_digits) <= 6 else q_digits[-6:]
            match_code = (code_str == padded) | code_str.str.endswith(q_digits)
        else:
            match_code = False
        filtered = filtered[match_name | match_code]
    if selected_grade != '全部':
        filtered = filtered[filtered['客户分级'] == selected_grade]
    if selected_exchange != '全部' and '交易所' in filtered.columns:
        filtered = filtered[filtered['交易所'] == selected_exchange]
    if selected_industry != '全部':
        filtered = filtered[filtered['所属行业'] == selected_industry]
    if '总分' in filtered.columns:
        filtered = filtered[filtered['总分'] >= min_score]
    with st.sidebar:
        st.metric("筛选后客户数", f"{len(filtered)} 家")
        if search_query and len(filtered) == 0:
            st.caption("未找到匹配公司，试试简称或完整股票代码")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["数据总览", "客户深度分析", "客户列表", "导入评级", "本公司产品", "销售跟进"]
    )

    # ===== Tab1 数据总览 =====
    with tab1:
        st.subheader("📈 核心指标")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("客户总数", f"{len(df)} 家")
        c2.metric("A级高价值客户", f"{count_grade_prefix(df['客户分级'], 'A')} 家")
        c3.metric("平均评分", f"{df['总分'].mean():.1f} 分" if '总分' in df.columns else "-")
        if '总市值(亿)' in df.columns and pd.to_numeric(df['总市值(亿)'], errors='coerce').median() >= 10:
            c4.metric("市值中位数", f"{pd.to_numeric(df['总市值(亿)'], errors='coerce').median():.0f} 亿")
        else:
            c4.metric("最高评分", f"{df['总分'].max():.1f} 分" if '总分' in df.columns else "-")

        st.divider()
        l, r = st.columns(2)
        with l:
            st.markdown("#### 客户分级分布")
            gc = df['客户分级'].value_counts().reset_index()
            gc.columns = ['分级', '数量']
            fig = px.pie(gc, values='数量', names='分级',
                         color_discrete_sequence=['#FF4B4B', '#FFA500', '#4ECDC4', '#95A5A6'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.markdown("#### 总分分布")
            fig = px.histogram(df, x='总分', nbins=24, color_discrete_sequence=['#3498DB'])
            fig.update_layout(bargap=0.08)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("TOP20 高价值客户")
        if filtered.empty:
            st.info("当前筛选条件下没有客户，TOP20 暂无可展示数据。")
        else:
            name_col = '公司名称' if '公司名称' in filtered.columns else '客户名称'
            top20 = filtered.sort_values('总分', ascending=False).head(20)
            fig = px.bar(top20, x='总分', y=name_col, orientation='h', color='客户分级',
                          color_discrete_map={'A级（高价值优先跟进）': '#FF4B4B', 'B级（重点跟进）': '#FFA500',
                                              'C级（常规跟进）': '#4ECDC4', 'D级（低优先级/观察）': '#95A5A6'},
                          hover_data=['所属行业'] if '所属行业' in top20.columns else None)
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=560)
            st.plotly_chart(fig, use_container_width=True)

    # ===== Tab2 客户深度分析 =====
    with tab2:
        st.subheader("客户深度分析")
        name_col = '公司名称' if '公司名称' in filtered.columns else '客户名称'
        if filtered.empty:
            st.info("当前筛选条件下没有客户。请调整左侧筛选，或用公司名称/股票代码搜索。")
            selected_company = None
            our_product = ""
        else:
            s1, s2 = st.columns([2, 3])
            with s1:
                selected_company = st.selectbox(
                    "选择目标公司",
                    sorted(filtered[name_col].dropna().astype(str).unique().tolist()),
                )
            with s2:
                our_product = st.text_input("我方产品/服务（可手填，或使用下方推荐产品）", placeholder="例如：企业级网络安全解决方案")

        if selected_company:
            company = filtered[filtered[name_col].astype(str) == selected_company].iloc[0]
            company_extra = None
            if extra_df is not None and '股票代码' in filtered.columns and '股票代码' in extra_df.columns:
                em = extra_df[extra_df['股票代码'].map(pad_stock_code) == pad_stock_code(company.get('股票代码'))]
                if len(em) > 0:
                    company_extra = em.iloc[0]
            company_full = company.to_dict()
            if company_extra is not None:
                company_full.update(company_extra.to_dict())

            ci, cr = st.columns([1, 2])
            with ci:
                st.markdown(f"### {company.get(name_col, '')} ({company.get('股票代码', '')})")
                st.markdown(f"**客户分级：** {company.get('客户分级', '-')}")
                try:
                    st.markdown(f"**总分：** {float(company['总分']):.1f} 分")
                except Exception:
                    st.markdown("**总分：** -")
                st.markdown(f"**所属行业：** {company.get('所属行业', '-')}")
                mcap_val = company.get('总市值(亿)')
                if pd.notna(mcap_val):
                    try:
                        st.markdown(f"**总市值：** {float(mcap_val):.1f} 亿")
                    except Exception:
                        pass
                if pd.notna(company.get('毛利率')):
                    st.markdown(f"**毛利率：** {company.get('毛利率')}")
                if pd.notna(company.get('营收同比增长率')):
                    st.markdown(f"**营收增长率：** {company.get('营收同比增长率')}")
                st.markdown("**主营业务：**")
                st.info(str(company.get('主营业务', ''))[:200])
            with cr:
                dims = available_score_dims(df)
                if dims:
                    dn = [DIMENSION_NAMES.get(d, d) for d in dims]
                    vals = []
                    for d in dims:
                        try:
                            vals.append(float(company[d]) if pd.notna(company[d]) else 0)
                        except Exception:
                            vals.append(0)
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=dn+[dn[0]], fill='toself',
                                                   line_color='#FF4B4B', name='该公司'))
                    peers = df[df['所属行业'] == company.get('所属行业')]
                    if len(peers) >= 2:
                        ia = peers[dims].mean(numeric_only=True)
                        fig.add_trace(go.Scatterpolar(
                            r=ia.tolist()+[ia.iloc[0]], theta=dn+[dn[0]], fill='toself',
                            line_color='#3498DB', opacity=0.5, name='行业平均'))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                       title=f"{company.get(name_col, '')} 能力雷达",
                                       legend=dict(orientation="h", y=-0.15))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("当前结果文件还没有 10 维得分子段，请重新运行评分或在「导入评级」写入。")

            st.divider()
            st.markdown("### 评分可解释性")
            dims = available_score_dims(df)
            if dims:
                explain = []
                peers = df[df['所属行业'] == company.get('所属行业')]
                for dim in dims:
                    w = DIMENSION_WEIGHTS.get(dim, 0)
                    try:
                        sc = float(company[dim])
                    except Exception:
                        sc = 0
                    av = float(peers[dim].mean()) if dim in peers.columns and len(peers) else sc
                    explain.append({
                        '维度': DIMENSION_NAMES.get(dim, dim),
                        '原始得分': round(sc, 1),
                        '权重': f"{w*100:.1f}%",
                        '贡献分': round(sc * w, 2),
                        '行业平均': round(av, 1),
                        '与行业差值': round(sc - av, 1),
                    })
                edf = pd.DataFrame(explain).sort_values('贡献分', ascending=False)
                e1, e2 = st.columns(2)
                with e1:
                    fig = px.bar(edf, x='贡献分', y='维度', orientation='h', color='贡献分',
                                 color_continuous_scale='RdYlGn', title='各维度加权贡献')
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                with e2:
                    fig = px.bar(edf.sort_values('与行业差值'), x='与行业差值', y='维度', orientation='h',
                                 color='与行业差值', color_continuous_scale='RdBu', title='相对行业平均')
                    fig.add_vline(x=0, line_color='black')
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(edf, use_container_width=True, hide_index=True)
            else:
                st.info("无法展示维度贡献：结果里缺少得分列。")

            st.divider()
            reco_product_text = render_recommended_products(products_df, company)
            if reco_product_text and not our_product:
                our_product = reco_product_text

            st.divider()
            st.markdown("### 联网深度尽调")
            st.caption(
                "销售跟进一手情报（优先）+ 12 维 advanced 搜索 → "
                "9 段尽调报告；开头含评分概览，围绕低分维度深挖。"
            )
            if not RESEARCH_AVAILABLE:
                st.error("联网尽调模块未加载")
                if RESEARCH_IMPORT_ERROR:
                    st.code(RESEARCH_IMPORT_ERROR)
                if st.button("重试加载尽调模块", key='retry_load_research'):
                    if _load_company_researcher():
                        st.success("加载成功，请再次点击尽调按钮")
                        st.rerun()
                    else:
                        st.error(RESEARCH_IMPORT_ERROR or "仍加载失败")
            elif not our_product:
                st.warning("请先输入我方产品，或在上方选择推荐产品")
            elif not os.getenv("TAVILY_API_KEY", ""):
                st.warning("未配置 TAVILY_API_KEY，请在 .env 或 Streamlit secrets 中填写")
            elif not DEEPSEEK_API_KEY:
                st.warning("未配置 DEEPSEEK_API_KEY，请在 .env 或 Streamlit secrets 中填写")
            elif st.button("开始联网深度尽调", type="primary", key='btn_deep_research'):
                try:
                    score_overview = build_score_overview_for_research(company, df)
                    low_dims = score_overview.get('低分维度') or []
                    sales_fu_rows = get_sales_followup_for_company(
                        company.get(name_col), followup_df, company
                    )
                    prog = st.progress(0, text="准备搜索…")

                    def _on_progress(done, total, msg):
                        pct = int(done / max(total, 1) * 70)
                        prog.progress(min(pct, 70), text=f"{msg}（{done}/{total}）")

                    intel = research_company(
                        company.get(name_col),
                        company.get('所属行业', ''),
                        our_product,
                        verbose=False,
                        low_score_dims=low_dims,
                        progress_callback=_on_progress,
                    )
                    if not intel and not sales_fu_rows:
                        st.error("未搜索到联网情报，且无销售跟进记录，无法生成报告")
                    else:
                        if sales_fu_rows:
                            with st.expander("销售内部一手情报（优先于联网结果）", expanded=True):
                                st.warning("以下为销售内部一手情报，优先级高于联网搜索结果")
                                st.dataframe(
                                    pd.DataFrame(sales_fu_rows),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                        prog.progress(75, text=f"已收集联网情报 {len(intel or [])} 条，AI 撰写报告中…")
                        # 结构化数据不含销售跟进字段，避免与一手情报混在一起
                        basic = {
                            k: str(v) for k, v in company_full.items()
                            if k in [
                                '总市值(亿)', '毛利率', '净利率', '营收同比增长率',
                                '净利润同比增长率', '资产负债率', '总分', '客户分级',
                                '所属行业', '主营业务',
                            ]
                        }
                        report = generate_deep_report(
                            company.get(name_col),
                            intel or [],
                            our_product,
                            company_basic=basic,
                            score_overview=score_overview,
                            sales_followup=sales_fu_rows,
                        )
                        if save_research_outputs:
                            try:
                                save_research_outputs(company.get(name_col), our_product, report, intel or [])
                            except Exception:
                                pass
                        prog.progress(100, text="尽调完成")
                        st.success(
                            f"报告已生成：销售跟进 {len(sales_fu_rows)} 条（优先）+ "
                            f"联网情报 {len(intel or [])} 条；"
                            f"低分深挖：{'、'.join(low_dims) if low_dims else '无'}"
                        )
                        if intel:
                            with st.expander(f"查看原始联网情报（{len(intel)} 条）"):
                                for i, it in enumerate(intel, 1):
                                    st.markdown(f"**[{i}] {it.get('标题', '')}**")
                                    st.caption(
                                        f"维度：{it.get('搜索维度', '')} | "
                                        f"来源：{it.get('来源', '')}"
                                    )
                                    st.text(str(it.get('内容', ''))[:400])
                        st.markdown("---")
                        st.markdown(report)
                        st.download_button(
                            "导出尽调报告",
                            f"{company.get(name_col)} 深度尽调\n\n{report}",
                            f"{company.get(name_col)}_深度尽调.md",
                            mime='text/markdown',
                            key='dl_research',
                        )
                except Exception as e:
                    st.error(f"尽调失败：{e}")

    # ===== Tab3 客户列表 =====
    with tab3:
        st.subheader("客户列表")
        if filtered.empty:
            st.info("当前筛选条件下没有客户。请调整左侧筛选，或用公司名称/股票代码搜索。")
        else:
            ldf = filtered.copy()
            cols = [c for c in ['股票代码', '公司名称', '客户名称', '交易所', '所属行业',
                                '总市值(亿)', '总分', '客户分级'] if c in ldf.columns]
            dims = available_score_dims(ldf)[:4]
            show = cols + [d for d in dims if d not in cols]
            st.dataframe(ldf[show].sort_values('总分', ascending=False) if '总分' in ldf.columns else ldf[show],
                         use_container_width=True, hide_index=True)
            st.download_button("导出筛选结果", filtered.to_csv(index=False).encode('utf-8-sig'),
                               'customer_scoring_filtered.csv', mime='text/csv', key='dl_filtered')

    with tab4:
        render_import_tab(df)

    with tab5:
        render_product_import_tab(products_df)

    with tab6:
        render_followup_tab(followup_df, df)

    st.divider()
    with st.expander("📚 数据来源与架构说明"):
        st.markdown("""
**数据架构：**
- **客户基础数据（公开渠道自动采集）**
- **销售跟进数据（手动维护）**：每次跟进追加一行，记录合作成熟度、现有供应商、竞争态势、客户抱怨等
- **两路数据合并 → 10 维度 AHP 评分（客户价值 / 成交可行性 / 客户需求度）→ A/B/C/D 分级**

**AI 尽调**：12 维 Tavily advanced（新闻/采购限近 6 个月、URL 去重）→ DeepSeek 9 段报告（评分概览 + 30/60/90 行动计划）

**技术栈：** Python · Requests · BeautifulSoup · Pandas · Streamlit · Plotly · Tavily · DeepSeek · numpy

**免责声明：** 数据来自公开渠道，仅供学习研究。
        """)


if __name__ == '__main__':
    main()
