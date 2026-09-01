"""
财务数据采集模块
从同花顺F10财务分析页面爬取每家公司的核心财务指标
数据来源：http://basic.10jqka.com.cn
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

MAX_WORKERS = 3

# 请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Referer': 'http://basic.10jqka.com.cn/',
}


def get_financial_data(stock_code, stock_name):
    """
    从同花顺财务分析页面获取核心财务指标（从财务诊断部分提取）
    """
    try:
        url = f'http://basic.10jqka.com.cn/{stock_code}/finance.html'
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = 'gbk'
        
        if r.status_code != 200:
            return None
        
        soup = BeautifulSoup(r.text, 'html.parser')
        text = ' '.join(soup.get_text().split())
        
        result = {
            '股票代码': stock_code,
            '公司名称': stock_name,
            '营业总收入': '',
            '营收同比增长率': '',
            '净利润': '',
            '净利润同比增长率': '',
            '总资产': '',
            '资产负债率': '',
            '毛利率': '',
            '净利率': '',
            '员工人数': '',
            '报告期': '',
        }
        
        # 从财务诊断部分提取核心指标（格式：本期毛利率60.33%,去年同期为60.38%）
        patterns = {
            '毛利率': r'本期毛利率([\d.\-]+)%',
            '净利率': r'本期净利率([\d.\-]+)%',
            '营收同比增长率': r'本期营业收入增长率([\d.\-]+)%',
            '净利润同比增长率': r'本期净利润增长率([\d.\-]+)%',
            '资产负债率': r'本期资产负债率([\d.\-]+)%',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                result[key] = match.group(1) + '%'
        
        # 提取报告期
        match = re.search(r'报告期[：:]\s*(\S+?) ', text)
        if match:
            result['报告期'] = match.group(1)
        
        # 获取员工人数（从公司概况页面）
        try:
            url2 = f'http://basic.10jqka.com.cn/{stock_code}/'
            r2 = requests.get(url2, headers=HEADERS, timeout=10)
            r2.encoding = 'gbk'
            if r2.status_code == 200:
                text2 = ' '.join(BeautifulSoup(r2.text, 'html.parser').get_text().split())
                match = re.search(r'员工人数[：:]\s*([\d,]+人?)', text2)
                if match:
                    result['员工人数'] = match.group(1)
        except:
            pass
        
        # 每个请求后小延时
        time.sleep(0.3)
        return result
        
    except Exception as e:
        return None


def main():
    print("=" * 70)
    print("📊 财务数据采集模块（多线程优化版）")
    print(f"⚡ 线程数：{MAX_WORKERS}，预计速度提升{MAX_WORKERS}倍")
    print("📋 数据来源：同花顺F10财务分析页面（公开可查）")
    print("=" * 70)
    
    # 读取已有的公司列表
    input_file = os.path.join(DATA_DIR, 'company_merged_data.csv')
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件：{input_file}")
        print("请先运行 data_collector.py 采集基本信息")
        return
    
    df_companies = pd.read_csv(input_file)
    print(f"✅ 读取到 {len(df_companies)} 家公司")
    
    # 多线程采集
    print(f"\n🔄 正在多线程采集财务数据...")
    start_time = time.time()
    
    financial_list = []
    total = len(df_companies)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_financial_data, row['股票代码'], row['公司名称']): idx
            for idx, row in df_companies.iterrows()
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    financial_list.append(result)
                if len(financial_list) % 50 == 0 or len(financial_list) == total:
                    elapsed = time.time() - start_time
                    print(f"  进度：已成功 {len(financial_list)}/{total} 家，用时 {elapsed:.1f} 秒")
            except:
                pass
    
    # 保存财务数据
    df_financial = pd.DataFrame(financial_list)
    output_file = os.path.join(DATA_DIR, 'company_financial_data.csv')
    df_financial.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 合并基本信息和财务数据
    df_final = pd.merge(df_companies, df_financial, on=['股票代码', '公司名称'], how='left')
    final_file = os.path.join(DATA_DIR, 'company_full_data.csv')
    df_final.to_csv(final_file, index=False, encoding='utf-8-sig')
    
    elapsed = time.time() - start_time
    
    # 统计各字段完整率
    print("\n" + "=" * 70)
    print(f"✅ 财务数据采集完成！总用时 {elapsed:.1f} 秒")
    print(f"   - 公司总数：{total} 家")
    print(f"   - 成功获取财务数据：{len(df_financial)} 家（{len(df_financial)/total*100:.1f}%）")
    print(f"   - 平均每秒处理：{total/elapsed:.1f} 家")
    print(f"\n📈 各字段完整率：")
    for col in ['毛利率', '净利率', '营收同比增长率', '净利润同比增长率', '资产负债率', '员工人数']:
        count = df_financial[col].apply(lambda x: x != '' and pd.notna(x)).sum()
        print(f"   - {col}：{count} 家（{count/total*100:.1f}%）")
    print(f"\n   - 数据保存路径：{DATA_DIR}")
    print("=" * 70)
    print("\n📁 生成的文件：")
    print(f"   1. company_financial_data.csv - 财务数据")
    print(f"   2. company_full_data.csv - 完整数据（基本信息+财务数据，评分模型用这个）")


if __name__ == '__main__':
    main()
