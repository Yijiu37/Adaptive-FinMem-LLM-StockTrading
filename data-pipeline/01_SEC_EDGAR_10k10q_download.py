"""
01_SEC_EDGAR_10k10q_download.py
================================

直接走 SEC 官方 EDGAR 接口下载 10-K / 10-Q 全文（完全免费、无配额限制），
作为付费 sec-api.io 的零成本替代。

接口：
  1. https://data.sec.gov/submissions/CIK{cik}.json   -> 该公司所有报表索引
  2. https://www.sec.gov/Archives/edgar/data/...      -> 报表 HTML 全文
说明：
  EDGAR 唯一硬性要求是 User-Agent 必须包含联系邮箱（不要默认 python-requests）。
  本脚本默认 UA="FinMem-Research student@example.com"，可改成你自己的。

输出 parquet 列与 sec-api 版本完全一致：
  document_url, content, ticker, cik, utc_timestamp, est_timestamp, type
"""

import os
import re
import time
import pytz
import httpx
import polars as pl
from cleantext import clean
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict


# ========= 用户参数 =========
TICKERS_CIK = {
    # AAPL CIK 是 0000320193
    # 查询其它票：https://www.sec.gov/cgi-bin/browse-edgar
    "AAPL": "0000320193",
}
START_DATE = "2014-01-01"
END_DATE = "2015-12-31"
FORMS = ["10-K", "10-Q"]
USER_AGENT = "FinMem-Research 3508639577@qq.com"  # SEC 要求格式: "组织名/用途 邮箱"
SLEEP = 0.15  # SEC 限频 10 请求/秒，留点余量
OUT_DIR = os.path.join("data", "03_primary")
# ===========================


HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def convert_utc_to_est(dt: datetime) -> datetime:
    return pytz.UTC.localize(dt).astimezone(pytz.timezone("US/Eastern")).replace(tzinfo=None)


def _parse_filings_block(block: Dict, ticker: str, cik: str) -> List[Dict]:
    """把 EDGAR 的一个 filings block（平铺的多个列表）解析为行列表"""
    rows = []
    forms = block.get("form", [])
    for i, form in enumerate(forms):
        if form not in FORMS:
            continue
        filed_at = block["filingDate"][i]
        accession = block["accessionNumber"][i].replace("-", "")
        primary_doc = block["primaryDocument"][i]
        url_doc = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession}/{primary_doc}"
        )
        dt = datetime.strptime(filed_at, "%Y-%m-%d")
        rows.append({
            "ticker": ticker,
            "cik": cik,
            "type": form,
            "utc_timestamp": dt,
            "est_timestamp": convert_utc_to_est(dt),
            "document_url": url_doc,
        })
    return rows


def get_filing_index(ticker: str, cik: str) -> List[Dict]:
    """读取 EDGAR 提交索引（含历史分页），返回 10-K/10-Q 列表

    EDGAR 的 submissions 接口对提交频繁的公司（如 AAPL）会把早年的记录
    放到 filings.files[].name 引用的历史分页文件中，这里一并抓取合并。
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = httpx.get(url, headers=HEADERS, timeout=30.0)
    r.raise_for_status()
    data = r.json()

    out: List[Dict] = []

    # 1) 近期 recent 块
    recent = data["filings"].get("recent", {})
    out.extend(_parse_filings_block(recent, ticker, cik))

    # 2) 历史分页块（files[].name -> https://data.sec.gov/submissions/{name}）
    history_files = data["filings"].get("files", []) or []
    for f in history_files:
        name = f.get("name")
        if not name:
            continue
        time.sleep(SLEEP)
        hist_url = f"https://data.sec.gov/submissions/{name}"
        try:
            hr = httpx.get(hist_url, headers=HEADERS, timeout=30.0)
            hr.raise_for_status()
            hist_block = hr.json()
            # 历史分页整个 JSON 就是一个 block（平铺字段）
            out.extend(_parse_filings_block(hist_block, ticker, cik))
        except Exception as e:
            print(f"  [warn] 历史分页 {name} 读取失败: {e}")

    # 按 accessionNumber 去重（防止极端情况下重复）
    seen = set()
    unique = []
    for r_ in out:
        key = r_["document_url"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(r_)
    return unique


def download_filing_html(url: str) -> str:
    """下载报表 HTML 全文"""
    r = httpx.get(url, headers=HEADERS, timeout=60.0, follow_redirects=True)
    if r.status_code != 200:
        return ""
    return r.text


def html_to_text(html: str) -> str:
    """简单去 HTML 标签 + 清洗"""
    txt = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    txt = re.sub(r"<style.*?</style>", " ", txt, flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"&nbsp;|&|&#\d+;", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return clean(txt, to_ascii=True, lower=True, no_line_breaks=True)


def extract_mdna(text: str, form: str) -> str:
    """
    粗粒度截取 MD&A 段（10-K Item 7 / 10-Q Part I Item 2）。
    找不到就返回前 30000 字符兜底，仍可作为长期记忆素材。
    """
    if form == "10-K":
        m = re.search(r"item\s*7\.?(.+?)item\s*7a", text, re.S)
    else:
        m = re.search(r"item\s*2\.?(.+?)item\s*3", text, re.S)
    if m:
        seg = m.group(1).strip()
        if len(seg) > 500:
            return seg[:60000]
    return text[:30000]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")

    rows = []
    for ticker, cik in TICKERS_CIK.items():
        print(f"==> 抓取 {ticker} (CIK={cik}) 的提交索引")
        idx = get_filing_index(ticker, cik)
        idx = [r for r in idx if start_dt <= r["utc_timestamp"] <= end_dt]
        print(f"   区间内 {len(idx)} 份 (10-K/10-Q)")

        for r in tqdm(idx, desc=f"下载 {ticker} 全文"):
            time.sleep(SLEEP)
            html = download_filing_html(r["document_url"])
            if not html:
                print("  [warn] 下载失败:", r["document_url"])
                continue
            text = html_to_text(html)
            mdna = extract_mdna(text, r["type"])
            r2 = dict(r)
            r2["content"] = mdna
            rows.append(r2)

    if not rows:
        print("!! 没有抓到任何文件，请检查 CIK 或日期范围")
        return

    df = pl.DataFrame(rows).select(
        ["document_url", "content", "ticker", "cik",
         "utc_timestamp", "est_timestamp", "type"]
    )
    out_path = os.path.join(OUT_DIR, "filing_data.parquet")
    df.write_parquet(out_path)
    print(f"\n✓ 已保存 {df.shape[0]} 条记录到 {out_path}")
    print(df.select(["ticker", "type", "est_timestamp"]))


if __name__ == "__main__":
    main()