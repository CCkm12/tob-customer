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

SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
try:
    from company_researcher import research_company, generate_deep_report
    RESEARCH_AVAILABLE = True
except Exception:
    RESEARCH_AVAILABLE = False

try:
    from ahp_weights import get_ahp_weights
    DIMENSION_WEIGHTS = get_ahp_weights()
except Exception:
    DIMENSION_WEIGHTS = {
        '规模得分': 0.12, '增长得分': 0.15, '盈利得分': 0.12, '财务健康得分': 0.08,
        '技术投入得分': 0.08, '成熟度得分': 0.05, '上市年限得分': 0.05, '地域得分': 0.08,
        '行业景气得分': 0.10, '企业性质得分': 0.03, '决策链得分': 0.07, '需求匹配得分': 0.07
    }

DIMENSION_NAMES = {
    '规模得分': '企业规模', '增长得分': '增长能力', '盈利得分': '盈利能力', '财务健康得分': '财务健康',
    '技术投入得分': '技术投入', '成熟度得分': '企业成熟', '上市年限得分': '上市年限', '地域得分': '地域匹配',
    '行业景气得分': '行业景气', '企业性质得分': '企业性质', '决策链得分': '决策链效率', '需求匹配得分': '需求匹配'
}


def clean_industry(s):
    if pd.isna(s):
        return '未知'
    s = re.split(r'概念', str(s))[0]
    s = re.sub(r'[ⅠⅡⅢⅣⅤIⅡ]', '', s)
    s = s.strip().strip('：:').strip()
    return s if s else '未知'


@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE)
    df['所属行业'] = df['所属行业'].apply(clean_industry)
    return df


@st.cache_data
def load_extra_data():
    if not os.path.exists(EXTRA_FILE):
        return None
    return pd.read_csv(EXTRA_FILE)


def generate_ai_sales_intel(company_data, our_product):
    if not DEEPSEEK_API_KEY:
        return "❌ 未配置API Key"
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    company_info = f"""
【目标公司信息】
公司名称：{company_data.get('公司名称', '未知')}
股票代码：{company_data.get('股票代码', '未知')}
所属行业：{company_data.get('所属行业', '未知')}
主营业务：{company_data.get('主营业务', '未知')}
公司亮点：{company_data.get('公司亮点', '未知')}
总市值：{company_data.get('总市值(亿)', '未知')}亿
毛利率：{company_data.get('毛利率', '未知')}
净利率：{company_data.get('净利率', '未知')}
营收同比增长率：{company_data.get('营收同比增长率', '未知')}
净利润同比增长率：{company_data.get('净利润同比增长率', '未知')}
资产负债率：{company_data.get('资产负债率', '未知')}
客户评分：{company_data.get('总分', '未知')}分
客户分级：{company_data.get('客户分级', '未知')}
需求类型：{company_data.get('需求类型', '未知')}
需求强度：{company_data.get('需求强度分', '未知')}分（{company_data.get('需求强度等级', '未知')}）
需求信号：{company_data.get('需求信号详情', '未知')}

【我方产品】
{our_product}
"""
    prompt = f"""# 角色
你是资深ToB销售总监。基于以下目标公司真实数据，生成可直接实战的销售情报。

{company_info}

# 要求
1. 基于真实数据分析，引用具体数字
2. 严格按结构输出，内容可执行，不要套话
3. 区分事实与推断
4. 重点结合需求信号展开

# 输出结构
## 🎯 一、需求深度解读
### 1.1 需求信号逐条解读（背后业务逻辑、为什么是采购机会）
### 1.2 需求强度与紧迫性判断
### 1.3 需求与我方产品匹配点（表格：客户需求/产品能力/匹配度/切入话术）

## 👤 二、关键决策人分析
### 2.1 决策链结构（推断）
### 2.2 核心决策人画像（职位/KPI/痛点/沟通风格/怎么接触）

## 💡 三、独特解决方案
### 3.1 定制化方案（卖点/差异化/实施路径/ROI）
### 3.2 竞品切入角度

## 📋 四、个性化互动建议（破冰/需求挖掘/方案呈现/成交 四阶段，含具体话术）

## ✉️ 五、个性化邮件草稿（破冰/价值/方案 三封，含主题和正文）

---
⚠️ 基于公开信息推断，实际请结合沟通调整。"""
    data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "max_tokens": 4500}
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ AI生成失败：{str(e)}"


def main():
    st.title("📊 ToB科技客户智能分级与销售情报系统")
    st.caption("两层架构：①爬虫+评分从990家筛客户 ②AI联网深度尽调懂客户 | 需求识别 · 行业对比 · 评分可解释")

    df = load_data()
    extra_df = load_extra_data()
    demand_file = os.path.join(BASE_DIR, 'outputs', 'demand_analysis_result.csv')
    demand_df = pd.read_csv(demand_file) if os.path.exists(demand_file) else None

    if df is None:
        st.error("❌ 未找到评分结果文件，请先运行 scoring_model.py")
        return

    with st.sidebar:
        st.header("⚙️ 筛选条件")
        grade_options = ['全部', 'A级（高价值优先跟进）', 'B级（重点跟进）', 'C级（常规跟进）', 'D级（低优先级/观察）']
        selected_grade = st.selectbox("客户分级", grade_options)
        selected_exchange = st.selectbox("交易所", ['全部', '深交所', '上交所'])
        industries = ['全部'] + sorted(df['所属行业'].dropna().unique().tolist())
        selected_industry = st.selectbox("所属行业", industries)
        min_score = st.slider("最低总分", 0, 100, 0)
        st.divider()
        st.metric("客户总数", f"{len(df)} 家")
        if extra_df is not None:
            st.metric("市值数据覆盖", f"{len(extra_df)} 家")
        if demand_df is not None:
            st.metric("需求分析覆盖", f"{len(demand_df)} 家")
        st.divider()
        st.success("✅ AI联网尽调已启用" if RESEARCH_AVAILABLE else "⚠️ 联网尽调模块未加载")
        st.success("✅ AI销售情报已启用")

    filtered = df.copy()
    if selected_grade != '全部':
        filtered = filtered[filtered['客户分级'] == selected_grade]
    if selected_exchange != '全部':
        filtered = filtered[filtered['交易所'] == selected_exchange]
    if selected_industry != '全部':
        filtered = filtered[filtered['所属行业'] == selected_industry]
    filtered = filtered[filtered['总分'] >= min_score]
    with st.sidebar:
        st.metric("筛选后客户数", f"{len(filtered)} 家")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 数据总览", "🏭 行业深度对比", "🔍 客户深度分析", "📋 客户列表"])

    # ===== Tab1 =====
    with tab1:
        st.subheader("📈 核心指标")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("客户总数", f"{len(df)} 家")
        c2.metric("A级高价值客户", f"{len(df[df['客户分级'].str.startswith('A')])} 家")
        c3.metric("平均评分", f"{df['总分'].mean():.1f} 分")
        if extra_df is not None and '总市值(亿)' in extra_df.columns:
            c4.metric("总市值覆盖", f"{extra_df['总市值(亿)'].sum()/10000:.1f} 万亿")
        else:
            c4.metric("最高评分", f"{df['总分'].max():.1f} 分")

        if demand_df is not None and '需求强度等级' in demand_df.columns:
            st.divider()
            st.subheader("🎯 客户需求强度总览")
            d1, d2 = st.columns(2)
            with d1:
                lo = ['强需求', '中需求', '弱需求', '需求不明显']
                lc = demand_df['需求强度等级'].value_counts().reindex(lo).dropna()
                fig = px.pie(values=lc.values, names=lc.index, title='需求强度分布',
                             color_discrete_sequence=['#FF4B4B', '#FFA500', '#4ECDC4', '#95A5A6'])
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            with d2:
                at = []
                for t in demand_df['需求类型'].dropna():
                    at.extend(str(t).split('、'))
                tc = pd.Series(at).value_counts().reset_index()
                tc.columns = ['需求类型', '公司数量']
                fig = px.bar(tc, x='公司数量', y='需求类型', orientation='h', color='公司数量',
                             color_continuous_scale='Reds', title='各需求类型客户数')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        l, r = st.columns(2)
        with l:
            st.markdown("#### 🏷️ 客户分级分布")
            gc = df['客户分级'].value_counts().reset_index()
            gc.columns = ['分级', '数量']
            fig = px.pie(gc, values='数量', names='分级',
                         color_discrete_sequence=['#FF4B4B', '#FFA500', '#4ECDC4', '#95A5A6'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        with r:
            st.markdown("#### 📊 评分分布")
            fig = px.histogram(df, x='总分', nbins=30, color_discrete_sequence=['#3498DB'])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

        if extra_df is not None and '总市值(亿)' in extra_df.columns:
            st.divider()
            st.subheader("💰 市值深度分析")
            t = df.copy(); t['sc'] = t['股票代码'].astype(str).str.zfill(6)
            e = extra_df.copy(); e['sc'] = e['股票代码'].astype(str).str.zfill(6)
            mv = t.merge(e[['sc', '总市值(亿)', '市盈率']], on='sc', how='left')
            m1, m2, m3 = st.columns(3)
            m1.metric("平均市值", f"{mv['总市值(亿)'].mean():.1f} 亿")
            m2.metric("市值中位数", f"{mv['总市值(亿)'].median():.1f} 亿")
            m3.metric("最大市值", f"{mv['总市值(亿)'].max():.1f} 亿")
            m4, m5 = st.columns(2)
            with m4:
                fig = px.histogram(mv, x='总市值(亿)', nbins=40, title='总市值分布（对数）',
                                   color_discrete_sequence=['#9B59B6'])
                fig.update_layout(bargap=0.1, xaxis_type='log')
                st.plotly_chart(fig, use_container_width=True)
            with m5:
                fig = px.scatter(mv, x='总市值(亿)', y='总分', color='客户分级', hover_data=['公司名称'],
                                 title='市值 vs 评分（对数）', log_x=True)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🏆 TOP20高价值客户")
        top20 = filtered.sort_values('总分', ascending=False).head(20)
        fig = px.bar(top20, x='总分', y='公司名称', orientation='h', color='客户分级',
                      color_discrete_map={'A级（高价值优先跟进）': '#FF4B4B', 'B级（重点跟进）': '#FFA500',
                                          'C级（常规跟进）': '#4ECDC4', 'D级（低优先级/观察）': '#95A5A6'},
                      hover_data=['所属行业'])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
        st.plotly_chart(fig, use_container_width=True)

    # ===== Tab2 =====
    with tab2:
        st.subheader("🏭 行业深度对比分析")
        ist = df.groupby('所属行业').agg(公司数量=('公司名称', 'count'), 平均评分=('总分', 'mean'),
                                          最高评分=('总分', 'max'), 最低评分=('总分', 'min'),
                                          评分标准差=('总分', 'std')).reset_index()
        ab = df[df['客户分级'].str.startswith(('A', 'B'))].groupby('所属行业').size().reset_index(name='AB级数量')
        ist = ist.merge(ab, on='所属行业', how='left')
        ist['AB级数量'] = ist['AB级数量'].fillna(0).astype(int)
        ist['AB级占比'] = (ist['AB级数量'] / ist['公司数量'] * 100).round(1)
        ist = ist[ist['公司数量'] >= 3].sort_values('平均评分', ascending=False)

        i1, i2 = st.columns(2)
        with i1:
            st.markdown("#### 📊 各行业平均评分排名")
            fig = px.bar(ist.head(20), x='平均评分', y='所属行业', orientation='h', color='公司数量',
                         color_continuous_scale='Reds', title='行业平均评分TOP20')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)
        with i2:
            st.markdown("#### 🎯 各行业A/B级客户占比")
            fig = px.bar(ist.head(20), x='AB级占比', y='所属行业', orientation='h', color='AB级数量',
                         color_continuous_scale='Oranges', title='A/B级占比TOP20(%)')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### 📦 TOP15行业评分分布箱线图")
        ti = ist.head(15)['所属行业'].tolist()
        fig = px.box(df[df['所属行业'].isin(ti)], x='所属行业', y='总分', color='所属行业')
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### 📋 行业详细数据")
        st.dataframe(ist.round(2), use_container_width=True, hide_index=True)
        st.download_button("📥 导出行业分析", ist.round(2).to_csv(index=False).encode('utf-8-sig'),
                           'industry_analysis.csv', mime='text/csv')

        st.divider()
        st.markdown("#### 🔬 选定行业内公司对比")
        sel_ind = st.selectbox("选择行业", sorted(df['所属行业'].dropna().unique().tolist()))
        if sel_ind:
            icos = df[df['所属行业'] == sel_ind].sort_values('总分', ascending=False)
            st.metric(f"{sel_ind} 公司数量", f"{len(icos)} 家")
            q1, q2 = st.columns(2)
            with q1:
                st.plotly_chart(px.histogram(icos, x='总分', nbins=20, title=f'{sel_ind}评分分布',
                                              color_discrete_sequence=['#2ECC71']), use_container_width=True)
            with q2:
                st.plotly_chart(px.pie(icos['客户分级'].value_counts().reset_index(), values='count',
                                        names='客户分级', title=f'{sel_ind}分级分布',
                                        color_discrete_sequence=['#FF4B4B', '#FFA500', '#4ECDC4', '#95A5A6']),
                                use_container_width=True)
            st.dataframe(icos[['股票代码', '公司名称', '总分', '客户分级', '增长得分', '盈利得分', '规模得分']],
                         use_container_width=True, hide_index=True)

    # ===== Tab3 =====
    with tab3:
        st.subheader("🔍 客户深度分析")
        s1, s2 = st.columns([2, 3])
        with s1:
            selected_company = st.selectbox("选择目标公司", sorted(filtered['公司名称'].tolist()))
        with s2:
            our_product = st.text_input("我方产品/服务", placeholder="例如：企业级网络安全解决方案")

        if selected_company:
            company = filtered[filtered['公司名称'] == selected_company].iloc[0]
            company_extra = None
            if extra_df is not None:
                em = extra_df[extra_df['股票代码'].astype(str).str.zfill(6) == str(company['股票代码']).zfill(6)]
                if len(em) > 0:
                    company_extra = em.iloc[0]
            company_demand = None
            if demand_df is not None:
                dm = demand_df[demand_df['股票代码'].astype(str).str.zfill(6) == str(company['股票代码']).zfill(6)]
                if len(dm) > 0:
                    company_demand = dm.iloc[0]
            company_full = company.to_dict()
            if company_extra is not None:
                company_full.update(company_extra.to_dict())
            if company_demand is not None:
                company_full.update(company_demand.to_dict())

            if company_demand is not None:
                st.markdown("### 🎯 需求信号识别")
                dl = company_demand.get('需求强度等级', '未知')
                lc = {'强需求': '🔴', '中需求': '🟠', '弱需求': '🟡', '需求不明显': '⚪'}
                x1, x2, x3 = st.columns(3)
                x1.metric("需求强度", f"{lc.get(dl,'')} {dl}")
                x2.metric("需求强度分", f"{company_demand.get('需求强度分',0)} 分")
                x3.metric("需求类型", str(company_demand.get('需求类型','未知'))[:20])
                st.markdown("**需求信号：**")
                for sig in str(company_demand.get('需求信号详情','')).split(' || '):
                    if sig and sig != 'nan':
                        st.markdown(f"- {sig}")
                st.markdown("**销售切入建议：**")
                for sug in str(company_demand.get('销售切入建议','')).split(' || '):
                    if sug and sug != 'nan':
                        st.success(sug)
                st.divider()

            ci, cr = st.columns([1, 2])
            with ci:
                st.markdown(f"### {company['公司名称']} ({company['股票代码']})")
                st.markdown(f"**客户分级：** {company['客户分级']}")
                st.markdown(f"**总分：** {company['总分']:.1f} 分")
                st.markdown(f"**所属行业：** {company['所属行业']}")
                if company_extra is not None and pd.notna(company_extra.get('总市值(亿)')):
                    st.markdown(f"**总市值：** {company_extra['总市值(亿)']:.1f} 亿")
                if pd.notna(company['毛利率']):
                    st.markdown(f"**毛利率：** {company['毛利率']}")
                if pd.notna(company['营收同比增长率']):
                    st.markdown(f"**营收增长率：** {company['营收同比增长率']}")
                st.markdown("**主营业务：**")
                st.info(str(company['主营业务'])[:200])
            with cr:
                dims = list(DIMENSION_WEIGHTS.keys())
                dn = [DIMENSION_NAMES[d] for d in dims]
                vals = [company[d] for d in dims]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=dn+[dn[0]], fill='toself',
                                               line_color='#FF4B4B', name='该公司'))
                ia = df[df['所属行业'] == company['所属行业']][dims].mean()
                fig.add_trace(go.Scatterpolar(r=ia.tolist()+[ia.iloc[0]], theta=dn+[dn[0]], fill='toself',
                                               line_color='#3498DB', opacity=0.5, name='行业平均'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                   title=f"{company['公司名称']} vs 行业平均",
                                   legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("### 📐 评分可解释性分析")
            explain = []
            for dim, w in DIMENSION_WEIGHTS.items():
                sc = company[dim]
                av = df[df['所属行业'] == company['所属行业']][dim].mean()
                explain.append({'维度': DIMENSION_NAMES[dim], '原始得分': round(sc, 1), '权重': f"{w*100:.0f}%",
                                '权重值': w, '贡献分': round(sc*w, 2), '行业平均': round(av, 1),
                                '与行业差值': round(sc-av, 1)})
            edf = pd.DataFrame(explain).sort_values('贡献分', ascending=False)
            e1, e2 = st.columns(2)
            with e1:
                fig = px.bar(edf, x='贡献分', y='维度', orientation='h', color='贡献分',
                             color_continuous_scale='RdYlGn', title='12维度加权贡献')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            with e2:
                fig = px.bar(edf.sort_values('与行业差值'), x='与行业差值', y='维度', orientation='h',
                             color='与行业差值', color_continuous_scale='RdBu', title='vs行业平均')
                fig.add_vline(x=0, line_color='black')
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(edf[['维度','原始得分','权重','贡献分','行业平均','与行业差值']],
                         use_container_width=True, hide_index=True)
            strengths = edf[edf['与行业差值'] > 5]
            weaknesses = edf[edf['与行业差值'] < -5]
            z1, z2 = st.columns(2)
            with z1:
                st.markdown("**✅ 优势维度：**")
                for _, r in strengths.iterrows():
                    st.markdown(f"- {r['维度']}：{r['原始得分']}分（高{r['与行业差值']:.1f}）")
                if len(strengths) == 0:
                    st.markdown("- 无明显优势")
            with z2:
                st.markdown("**⚠️ 短板维度：**")
                for _, r in weaknesses.iterrows():
                    st.markdown(f"- {r['维度']}：{r['原始得分']}分（低{abs(r['与行业差值']):.1f}）")
                if len(weaknesses) == 0:
                    st.markdown("- 无明显短板")

            st.divider()
            st.markdown("### 🤖 AI分析（两层）")
            at1, at2 = st.tabs(["⚡ 快速销售情报（本地数据）", "🌐 联网深度尽调（实时搜索）"])
            with at1:
                st.caption("基于本地结构化数据快速生成，约10秒")
                if not our_product:
                    st.warning("⚠️ 请先输入我方产品")
                elif st.button("⚡ 生成快速销售情报"):
                    with st.spinner("生成中..."):
                        res = generate_ai_sales_intel(company_full, our_product)
                    st.markdown(res)
                    st.download_button("📥 导出", f"{company['公司名称']}\n\n{res}",
                                       f"{company['公司名称']}_快速情报.txt", mime='text/plain')
            with at2:
                st.caption("AI自动联网搜索新闻/高管/融资/采购，约40-60秒")
                if not RESEARCH_AVAILABLE:
                    st.error("❌ 联网尽调模块未加载，确认 src/company_researcher.py 存在且Tavily Key已配置")
                elif not our_product:
                    st.warning("⚠️ 请先输入我方产品")
                elif st.button("🌐 开始联网深度尽调", type="primary"):
                    try:
                        prog = st.progress(0, text="正在联网搜索情报...")
                        intel = research_company(company['公司名称'], company['所属行业'], verbose=False)
                        prog.progress(50, text=f"搜索到{len(intel)}条情报，AI综合分析中...")
                        if not intel:
                            st.error("❌ 未搜索到情报，检查Tavily Key和网络")
                        else:
                            basic = {k: str(v) for k, v in company_full.items()
                                     if k in ['总市值(亿)','毛利率','净利率','营收同比增长率','总分','客户分级','需求类型']}
                            report = generate_deep_report(company['公司名称'], intel, our_product, basic)
                            prog.progress(100, text="尽调完成")
                            st.success(f"✅ 基于{len(intel)}条真实联网情报生成")
                            with st.expander(f"📎 查看原始搜索情报（{len(intel)}条）"):
                                for i, it in enumerate(intel, 1):
                                    st.markdown(f"**[{i}] {it['标题']}**")
                                    st.caption(f"维度：{it['搜索维度']} | 来源：{it['来源']}")
                                    st.text(it['内容'][:300])
                            st.markdown("---")
                            st.markdown(report)
                            st.download_button("📥 导出尽调报告",
                                               f"{company['公司名称']} 深度尽调\n\n{report}",
                                               f"{company['公司名称']}_深度尽调.txt", mime='text/plain')
                    except Exception as e:
                        st.error(f"❌ 尽调失败：{str(e)}")

    # ===== Tab4 =====
    with tab4:
        st.subheader("📋 客户列表")
        cols = ['股票代码', '公司名称', '交易所', '所属行业', '总分', '客户分级']
        ldf = filtered.copy()
        if extra_df is not None:
            ldf['sc'] = ldf['股票代码'].astype(str).str.zfill(6)
            e = extra_df.copy(); e['sc'] = e['股票代码'].astype(str).str.zfill(6)
            ldf = ldf.merge(e[['sc','总市值(亿)','市盈率']], on='sc', how='left').drop(columns=['sc'])
            cols = ['股票代码','公司名称','交易所','所属行业','总市值(亿)','市盈率','总分','客户分级']
        if demand_df is not None:
            ldf['sc'] = ldf['股票代码'].astype(str).str.zfill(6)
            dd = demand_df.copy(); dd['sc'] = dd['股票代码'].astype(str).str.zfill(6)
            ldf = ldf.merge(dd[['sc','需求类型','需求强度分','需求强度等级']], on='sc', how='left').drop(columns=['sc'])
            cols = cols[:-2] + ['需求强度等级','需求强度分'] + cols[-2:]
        st.dataframe(ldf[cols].sort_values('总分', ascending=False), use_container_width=True)
        st.download_button("📥 导出筛选结果", filtered.to_csv(index=False).encode('utf-8-sig'),
                           'customer_scoring_filtered.csv', mime='text/csv')

    st.divider()
    with st.expander("📚 数据来源与架构说明"):
        st.markdown("""
**两层架构：**
- **第一层 找客户**：巨潮资讯网 + 同花顺F10 + 腾讯财经 → 多线程爬虫 → 12维度AHP评分 → 从990家筛出高价值客户
- **第二层 懂客户**：Tavily联网搜索（新闻/高管/融资/采购）→ DeepSeek综合 → 深度尽调报告

**技术栈：** Python · Requests · BeautifulSoup · Pandas · Streamlit · Plotly · Tavily · DeepSeek

**免责声明：** 数据来自公开渠道，仅供学习研究。
        """)


if __name__ == '__main__':
    main()
