"""
额外数据采集器 - 市值数据 + 近期新闻
数据源：
1. 腾讯财经API - 市值、市盈率、市净率、涨跌幅
2. 搜狗新闻搜索 - 公司近期新闻标题
运行：python src/extra_data_collector.py
"""
import requests
import pandas as pd
import time
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_FILE = os.path.join(BASE_DIR, 'outputs', 'customer_scoring_result.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'extra_data.csv')

# 请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://finance.qq.com/'
}

NEWS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://news.sogou.com/'
}


def get_stock_prefix(code):
    """根据股票代码判断交易所前缀"""
    code = str(code).zfill(6)
    if code.startswith(('600', '601', '603', '605', '688')):
        return f'sh{code}'
    elif code.startswith(('000', '001', '002', '003', '300', '301')):
        return f'sz{code}'
    return None


def fetch_market_data(stock_code, company_name):
    """
    从腾讯财经API获取市值、市盈率、市净率、涨跌幅
    """
    result = {
        '股票代码': stock_code,
        '公司名称': company_name,
        '总市值(亿)': None,
        '流通市值(亿)': None,
        '市盈率': None,
        '市净率': None,
        '涨跌幅(%)': None,
        '当前价': None
    }
    
    prefix = get_stock_prefix(stock_code)
    if not prefix:
        return result
    
    try:
        url = f'https://qt.gtimg.cn/q={prefix}'
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = 'gbk'
        text = resp.text
        
        match = re.search(r'="([^"]+)"', text)
        if not match:
            return result
        
        fields = match.group(1).split('~')
        if len(fields) < 50:
            return result
        
        result['当前价'] = float(fields[3]) if fields[3] else None
        result['涨跌幅(%)'] = float(fields[32]) if fields[32] else None
        result['总市值(亿)'] = float(fields[38]) if fields[38] else None
        result['流通市值(亿)'] = float(fields[39]) if fields[39] else None
        result['市盈率'] = float(fields[44]) if fields[44] else None
        result['市净率'] = float(fields[46]) if fields[46] else None
        
    except Exception as e:
        pass
    
    return result


def fetch_company_news(company_name, stock_code):
    """
    从搜狗新闻搜索获取公司近期新闻（3-5条标题）
    """
    news_titles = []
    
    try:
        url = f'https://news.sogou.com/news?query={company_name}&sort=1'
        resp = requests.get(url, headers=NEWS_HEADERS, timeout=10)
        resp.encoding = 'utf-8'
        html = resp.text
        
        # 搜狗新闻标题匹配
        titles = re.findall(r'<h3[^>]*class="vr-title"[^>]*>.*?<a[^>]*>(.*?)</a>', html, re.DOTALL)
        if not titles:
            titles = re.findall(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', html, re.DOTALL)
        
        for title in titles[:5]:
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            if clean_title and len(clean_title) > 5:
                news_titles.append(clean_title)
        
        time.sleep(0.3)
        
    except Exception as e:
        pass
    
    return {
        '股票代码': stock_code,
        '公司名称': company_name,
        '近期新闻': ' | '.join(news_titles) if news_titles else None,
        '新闻数量': len(news_titles)
    }


def process_company(row):
    """处理单家公司：获取市值+新闻"""
    stock_code = str(row['股票代码']).zfill(6)
    company_name = row['公司名称']
    
    market_data = fetch_market_data(stock_code, company_name)
    news_data = fetch_company_news(company_name, stock_code)
    
    result = {**market_data, **news_data}
    return result


def main():
    print("=" * 60)
    print("额外数据采集器 - 市值数据 + 近期新闻")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 未找到输入文件：{INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"📋 共 {len(df)} 家公司待采集")
    
    results = []
    success_count = 0
    market_success = 0
    news_success = 0
    
    print("\n🚀 开始采集（3线程并发）...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_company, row): idx 
                   for idx, row in df.iterrows()}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                success_count += 1
                
                if result.get('总市值(亿)'):
                    market_success += 1
                if result.get('近期新闻'):
                    news_success += 1
                
                if success_count % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  进度：{success_count}/{len(df)} | "
                          f"市值成功：{market_success} | "
                          f"新闻成功：{news_success} | "
                          f"用时：{elapsed:.1f}秒")
                    
            except Exception as e:
                print(f"  ❌ 第{idx}家公司采集失败：{e}")
    
    result_df = pd.DataFrame(results)
    result_df = result_df.drop_duplicates(subset=['股票代码'], keep='first')
    
    os.makedirs(DATA_DIR, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ 采集完成！")
    print(f"  总公司数：{len(result_df)}")
    print(f"  市值数据成功：{market_success} 家 ({market_success/len(result_df)*100:.1f}%)")
    print(f"  新闻数据成功：{news_success} 家 ({news_success/len(result_df)*100:.1f}%)")
    print(f"  总用时：{elapsed:.1f} 秒")
    print(f"  结果保存：{OUTPUT_FILE}")
    print("=" * 60)
    
    print("\n📊 前5条数据样例：")
    print(result_df[['股票代码', '公司名称', '总市值(亿)', '市盈率', '新闻数量']].head())


if __name__ == '__main__':
    main()
