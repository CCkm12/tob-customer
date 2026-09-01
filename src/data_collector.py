"""
数据采集模块
数据源1：巨潮资讯网- 获取A股上市公司列表
数据源2：同花顺F10 - 获取公司详细信息（主营业务、行业、亮点等）
数据全部公开可查：http://www.cninfo.com.cn  http://basic.10jqka.com.cn
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# 数据保存路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 线程数（3个比较安全，太多容易被封）
MAX_WORKERS = 3

# 请求头（模拟浏览器，避免被反爬）
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}


def get_stock_list_from_cninfo():
    """
    从巨潮资讯网获取A股上市公司列表（深交所+上交所）
    数据来源：http://www.cninfo.com.cn（证监会指定信息披露平台）
    """
    print("🔍 正在从巨潮资讯网获取A股上市公司列表...")
    
    all_stocks = []
    
    try:
        # 用all_stock.json接口，包含深交所和上交所所有证券
        url = 'http://www.cninfo.com.cn/new/data/all_stock.json'
        r = requests.get(url, headers=HEADERS, timeout=30)
        data = r.json()
        stock_list = data.get('stockList', [])
        print(f"  巨潮全部证券：{len(stock_list)} 条")
        
        for item in stock_list:
            code = item.get('code', '')
            name = item.get('zwjc', '')
            category = item.get('category', '')
            delisted = item.get('delisted', 'false')
            
            # 只保留A股，且未退市
            if category != 'A股' or delisted == 'true':
                continue
            
            # 用代码前缀判断交易所
            if code[:3] in ['000', '001', '002', '003', '300', '301']:
                all_stocks.append({'股票代码': code, '公司名称': name, '交易所': '深交所'})
            elif code[:3] in ['600', '601', '603', '605', '688']:
                all_stocks.append({'股票代码': code, '公司名称': name, '交易所': '上交所'})
        
        sh_count = len([s for s in all_stocks if s['交易所'] == '上交所'])
        sz_count = len([s for s in all_stocks if s['交易所'] == '深交所'])
        print(f"  ✅ 深交所A股：{sz_count} 家")
        print(f"  ✅ 上交所A股：{sh_count} 家")
        
    except Exception as e:
        print(f"  ⚠️ 获取股票列表失败: {e}")
    
    df = pd.DataFrame(all_stocks)
    print(f"✅ 共获取 {len(df)} 家A股上市公司")
    return df


def filter_tech_companies(df):
    """
    筛选计算机/通信/电子行业的科技公司（通过公司名称关键词）
    """
    print("\n🔄 正在筛选科技行业公司...")
    
    keywords = ['科技', '信息', '软件', '通信', '电子', '网络', '数据', '智能', 
                '数字', '技术', '互联', '云', '芯片', '半导体', '计算机', '安防',
                '光电', '激光', '电路', '系统', '集成', '视讯', '声学', '精密']
    
    mask = df['公司名称'].apply(lambda x: any(kw in x for kw in keywords))
    df_tech = df[mask].copy().reset_index(drop=True)
    
    print(f"✅ 筛选出 {len(df_tech)} 家科技行业公司")
    return df_tech


def get_company_detail_from_10jqka(stock_code, stock_name, exchange):
    """
    从同花顺F10获取公司详细信息
    """
    try:
        url = f'http://basic.10jqka.com.cn/{stock_code}/'
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = 'gbk'
        
        if r.status_code != 200:
            return None
        
        soup = BeautifulSoup(r.text, 'html.parser')
        text = ' '.join(soup.get_text().split())
        
        result = {
            '股票代码': stock_code,
            '公司名称': stock_name,
            '交易所': exchange,
            '主营业务': '',
            '所属行业': '',
            '公司亮点': '',
            '成立日期': '',
            '注册资本': '',
            '上市日期': '',
            '办公地址': '',
        }
        
        # 提取各项信息
        patterns = {
            '主营业务': r'主营业务[：:]\s*(.*?)(?:所属|公司亮点|$)',
            '所属行业': r'所属申万行业[：:]\s*(.*?)(?:公司亮点|主营业务|$)',
            '公司亮点': r'公司亮点[：:]\s*(.*?)(?:主营业务|所属|市场人气|$)',
            '成立日期': r'成立日期[：:]\s*(\d{4}-\d{2}-\d{2})',
            '注册资本': r'注册资本[：:]\s*([\d.]+万?元)',
            '上市日期': r'上市日期[：:]\s*(\d{4}-\d{2}-\d{2})',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                if key in ['主营业务', '公司亮点']:
                    result[key] = match.group(1).strip()[:200]
                else:
                    result[key] = match.group(1).strip()
        
        # 每个请求后小延时
        time.sleep(0.3)
        return result
        
    except Exception as e:
        return None


def main():
    print("=" * 70)
    print("📊 ToB客户智能评分系统 - 数据采集模块（多线程优化版）")
    print(f"⚡ 线程数：{MAX_WORKERS}，预计速度提升{MAX_WORKERS}倍")
    print("📋 数据来源：巨潮资讯网（证监会指定）+ 同花顺F10（公开可查）")
    print("=" * 70)
    
    start_time = time.time()
    
    # 1. 获取A股上市公司列表（单接口，不需要多线程）
    df_stocks = get_stock_list_from_cninfo()
    
    if len(df_stocks) == 0:
        print("\n❌ 未获取到任何股票数据，请检查网络连接")
        return
    
    df_stocks.to_csv(os.path.join(DATA_DIR, 'stock_list_all.csv'), index=False, encoding='utf-8-sig')
    
    # 2. 筛选科技行业公司
    df_tech = filter_tech_companies(df_stocks)
    df_tech.to_csv(os.path.join(DATA_DIR, 'stock_list_tech.csv'), index=False, encoding='utf-8-sig')
    
    # 3. 多线程批量获取公司详细信息
    print(f"\n🔄 正在多线程获取公司详细信息...")
    print(f"   每家间隔0.3秒，{MAX_WORKERS}线程并发\n")
    
    detail_list = []
    total = len(df_tech)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                get_company_detail_from_10jqka, 
                row['股票代码'], 
                row['公司名称'], 
                row['交易所']
            ): idx
            for idx, row in df_tech.iterrows()
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    detail_list.append(result)
                if len(detail_list) % 50 == 0 or len(detail_list) == total:
                    elapsed = time.time() - start_time
                    print(f"  进度：已成功 {len(detail_list)}/{total} 家，用时 {elapsed:.1f} 秒")
            except:
                pass
    
    # 4. 保存详细数据
    df_detail = pd.DataFrame(detail_list)
    df_detail.to_csv(os.path.join(DATA_DIR, 'company_detail.csv'), index=False, encoding='utf-8-sig')
    
    # 5. 合并数据
    df_merged = pd.merge(df_tech, df_detail, on=['股票代码', '公司名称', '交易所'], how='inner')
    df_merged.to_csv(os.path.join(DATA_DIR, 'company_merged_data.csv'), index=False, encoding='utf-8-sig')
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"✅ 数据采集完成！总用时 {elapsed:.1f} 秒")
    print(f"   - A股上市公司总数：{len(df_stocks)} 家")
    print(f"   - 筛选科技行业公司：{len(df_tech)} 家")
    print(f"   - 成功获取详情：{len(df_detail)} 家（{len(df_detail)/len(df_tech)*100:.1f}%）")
    print(f"   - 合并后完整数据：{len(df_merged)} 家")
    print(f"   - 平均每秒处理：{len(df_tech)/elapsed:.1f} 家")
    print(f"   - 数据保存路径：{DATA_DIR}")
    print("=" * 70)
    print("\n📁 生成的文件：")
    print(f"   1. stock_list_all.csv - 全部A股上市公司列表")
    print(f"   2. stock_list_tech.csv - 筛选后的科技行业公司列表")
    print(f"   3. company_detail.csv - 公司详细信息")
    print(f"   4. company_merged_data.csv - 合并后的完整数据（财务采集用这个）")


if __name__ == '__main__':
    main()
