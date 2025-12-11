# coding=utf-8

import os
import json
import re
import hashlib
import shutil
from typing import List, Dict, Any, Optional

from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import requests


# === 抓取内容日志工具（按日清理，仅保留当天） ===

def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _prepare_fetch_log_dir(base_dir: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    today = _today_str()
    # 清理非当日目录
    for p in base.iterdir():
        if p.is_dir() and p.name != today:
            try:
                shutil.rmtree(p, ignore_errors=True)
                print(f"Agent：已清理历史抓取内容目录 {p}")
            except Exception as e:
                print(f"Agent：清理目录失败 {p}: {e}")
    out_dir = base / today
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _safe_name(s: str, max_len: int = 40) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", s)
    return (s[:max_len] or "NA")


def _write_fetch_log(log_dir: Path, item: Dict, content: str) -> Path:
    url = (item.get("url") or "").strip()
    host = urlparse(url).netloc or "unknown"
    sha = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    fname = f"{_safe_name(host, 30)}_{sha}.txt"
    path = log_dir / fname
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"URL: {url}\n"
        f"Source: {item.get('source','')}\n"
        f"Title: {item.get('title','')}\n"
        f"SavedAt: {ts}\n"
        f"Length: {len(content)} chars\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(content)
    return path


def fetch_content(items: List[Dict], cfg: Dict[str, Any], proxy_url: Optional[str]) -> List[Dict]:
    """
    抓取候选新闻条目的正文文本，结果写回 items 的 "content" 字段并返回。
    cfg 需要包含：AI_FETCH_TIMEOUT, AI_FETCH_MAX_CHARS
    同时：
    - 将清洗后的正文按日落盘到 output/fetched/YYYYMMDD 下（可通过 AI_FETCH_DUMP/AI_FETCH_DUMP_DIR 控制）
    - 每次运行会自动清理该目录下非当天的子目录
    """
    if not items:
        return items

    try:
        fetch_timeout = float(cfg.get("AI_FETCH_TIMEOUT", 10.0))
    except Exception:
        fetch_timeout = 10.0
    try:
        fetch_max_chars = int(cfg.get("AI_FETCH_MAX_CHARS", 3000))
    except Exception:
        fetch_max_chars = 3000

    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    headers_fetch = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    # 是否开启抓取内容落盘与打印（默认开启）
    dump_enabled = (os.environ.get("AI_FETCH_DUMP", "true").lower() in ("1", "true", "yes"))
    dump_base_dir = os.environ.get("AI_FETCH_DUMP_DIR", "output/fetched")
    log_dir: Optional[Path] = _prepare_fetch_log_dir(dump_base_dir) if dump_enabled else None

    fetched = 0
    for it in items:
        url = (it.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        try:
            r = requests.get(url, headers=headers_fetch, timeout=fetch_timeout, proxies=proxies)
            r.raise_for_status()
            text = r.text or ""
            ctype = (r.headers.get("Content-Type") or "").lower()
            content_txt = ""

            if "text/plain" in ctype and text:
                content_txt = text
            else:
                try:
                    from bs4 import BeautifulSoup  # 可选依赖
                    soup = BeautifulSoup(text, "html.parser")
                    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
                        tag.decompose()
                    content_txt = soup.get_text("\n")
                except Exception:
                    content_txt = re.sub(r"<[^>]+>", " ", text)

            # 归一化
            content_txt = content_txt.replace("\r", " ")
            content_txt = "\n".join(ln.strip() for ln in content_txt.splitlines() if ln.strip())
            content_txt = re.sub(r"\n\n+", "\n", content_txt)
            orig_len = len(content_txt)
            if len(content_txt) > fetch_max_chars:
                content_txt = content_txt[:fetch_max_chars] + "..."

            if content_txt.strip():
                it["content"] = content_txt
                fetched += 1

                # 落盘与打印
                if log_dir is not None:
                    try:
                        saved_path = _write_fetch_log(log_dir, it, content_txt)
                        preview = content_txt[:200].replace("\n", " ")
                        print(f"Agent：成功抓取 {url[:80]}... ({len(content_txt)} 字符，截断前 {orig_len} 字符) 已保存: {saved_path}")
                        print(f"Agent：抓取预览：{preview}")
                    except Exception as e:
                        print(f"Agent：保存抓取内容失败: {e}")
                else:
                    preview = content_txt[:200].replace("\n", " ")
                    print(f"Agent：成功抓取 {url[:80]}... ({len(content_txt)} 字符，截断前 {orig_len} 字符)")
                    print(f"Agent：抓取预览：{preview}")
            else:
                # 即便为空也打印提示，便于定位反爬或解析问题
                print(f"Agent：页面内容为空 {url[:80]}...")
        except requests.Timeout:
            print(f"Agent 抓取超时: {url[:80]}...")
        except requests.RequestException as e:
            print(f"Agent 抓取网络错误: {url[:80]}... ({type(e).__name__})")
        except Exception as e:
            print(f"Agent 抓取失败: {url[:80]}... ({e})")

    print(f"Agent 抓取完成：成功 {fetched}/{len(items)} 条")
    return items


def relevance_llm(items: List[Dict], cfg: Dict[str, Any], proxy_url: Optional[str], mode: str = "daily") -> Optional[bool]:
    """
    使用“默认模型与默认 API”进行金融相关性二判：
    - 使用 AI_AGENT_DEFAULT_* / OPENAI_*_DEFAULT / OPENAI_* 环境变量
    - 只返回 True/False（True 表示金融相关），None 表示调用失败
    """
    try:
        # 默认模型与端点
        model = (
            (cfg.get("OPENAI_MODEL_DEFAULT") or "").strip()
            or (os.environ.get("AI_AGENT_DEFAULT_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
        )
        api_key = (
            os.environ.get("AI_AGENT_DEFAULT_API_KEY")
            or os.environ.get("OPENAI_API_KEY_DEFAULT")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        base_url = (
            os.environ.get("AI_AGENT_DEFAULT_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL_DEFAULT")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        if not api_key:
            return None

        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        # 组装简短样本
        k = min(int(cfg.get("AI_FETCH_TOPK", 8) or 8), len(items or []))
        samples = []
        for it in (items or [])[:k]:
            title = (it.get("title") or "").strip()
            content = (it.get("content") or "").strip().replace("\n", " ")
            if len(content) > 160:
                content = content[:160] + "..."
            samples.append({"title": title, "content": content})
        print(samples)
        user_prompt = (
            "请判断下面这些新闻是否与金融/证券/资本市场相关。\n\n"
            f"新闻样本（共{len(samples)}条）:\n"
            f"{json.dumps(samples, ensure_ascii=False, indent=2)}\n\n"
            "金融相关的判断标准：\n"
            "• 股票市场：A股/港股/美股、上市公司、股价涨跌、板块行情、概念股\n"
            "• 金融产品：基金、债券、期货、期权、理财产品\n"
            "• 大宗商品：有色金属（铜铝锌）、黑色金属（钢铁）、贵金属（黄金白银）、能源（原油天然气）、农产品期货\n"
            "• 行业板块：新能源、芯片半导体、医药生物、房地产、消费、科技等行业的市场表现\n"
            "• 宏观经济：GDP、CPI、PMI、利率、汇率、货币政策、财政政策\n"
            "• 公司财报：营收、利润、业绩预告、财务数据\n"
            "• 监管政策：证监会、央行、交易所的政策法规\n"
            "• 产业链：供应链、产能、产量、库存、供需关系对市场的影响\n\n"
            "判断示例：\n"
            "• '有色金属大涨' → yes（大宗商品价格影响股市）\n"
            "• '美联储降息' → yes（货币政策影响市场）\n"
            "• 'A股探底回升' → yes（股票市场）\n"
            "• '新能源板块领涨' → yes（行业板块）\n"
            "• '铜价创新高' → yes（大宗商品）\n"
            "• '明星演唱会' → no（娱乐新闻）\n"
            "• '体育比赛' → no（体育新闻）\n\n"
            "重要提示：\n"
            "1. 即使只有标题没有正文，也要基于标题关键词准确判断\n"
            "2. 只要涉及上述任一金融领域，就应判断为相关\n"
            "3. 多条新闻中只要有一条金融相关，就判断为yes\n\n"
            "请直接回答 yes 或 no（不要添加编号或其他内容）："
        )

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一个金融新闻判别器，只返回 yes 或 no，不要添加任何其他内容。"},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 10,  # 增加到10以确保有足够空间
        }

        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=body, timeout=30, proxies=proxies)
        if resp.status_code != 200:
            print(f"Agent：相关性LLM调用失败 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        content = ((data.get("choices", [{}])[0].get("message", {}) or {}).get("content") or "").strip().lower()
        ans = content.split()[:1]
        token = (ans[0] if ans else content)
        is_fin = token in ("y", "yes", "是", "相关", "yes.")
        print(f"Agent：金融相关性 LLM 判定：{is_fin}（模型: {model}，输出: {content[:24]}）")
        return bool(is_fin)
    except Exception as e:
        print(f"Agent：金融相关性 LLM 异常: {e}")
        return None


def summarize(
    report_data: Dict,
    items: List[Dict],
    cfg: Dict[str, Any],
    proxy_url: Optional[str],
    mode: str = "daily",
) -> Optional[str]:
    """
    使用 OpenAI 进行总结。使用 cfg 中的 OPENAI_TEMPERATURE、OPENAI_MAX_TOKENS、AI_MAX_LINES。
    支持为“总结模型”和“默认模型”分别配置不同的 BASE_URL 与 API_KEY：
    - 总结优先级：cfg.OPENAI_MODEL_SUMMARY > AI_AGENT_SUMMARY_MODEL > OPENAI_MODEL_SUMMARY > 默认模型
    - API KEY 选择：
        - 若使用“总结模型”：AI_AGENT_SUMMARY_API_KEY > OPENAI_API_KEY_SUMMARY > OPENAI_API_KEY
        - 若使用“默认模型”：AI_AGENT_DEFAULT_API_KEY > OPENAI_API_KEY_DEFAULT > OPENAI_API_KEY
    - BASE URL 选择：
        - 若使用“总结模型”：AI_AGENT_SUMMARY_BASE_URL > OPENAI_BASE_URL_SUMMARY > OPENAI_BASE_URL（默认 https://api.openai.com/v1）
        - 若使用“默认模型”：AI_AGENT_DEFAULT_BASE_URL > OPENAI_BASE_URL_DEFAULT > OPENAI_BASE_URL
    """
    # 模型选择
    default_model = (os.environ.get("AI_AGENT_DEFAULT_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    cfg_summary_model = (cfg.get("OPENAI_MODEL_SUMMARY") or "").strip() if isinstance(cfg, dict) else ""
    env_summary_model = (os.environ.get("AI_AGENT_SUMMARY_MODEL") or os.environ.get("OPENAI_MODEL_SUMMARY") or "").strip()
    model = (cfg_summary_model or env_summary_model or default_model)

    # 判定当前是否为“总结模型”
    summary_model_ref = (cfg_summary_model or env_summary_model)
    is_summary_model = bool(summary_model_ref) and (model == summary_model_ref)

    # 选择 API KEY 与 BASE_URL（可独立配置）
    if is_summary_model:
        api_key = (
            os.environ.get("AI_AGENT_SUMMARY_API_KEY")
            or os.environ.get("OPENAI_API_KEY_SUMMARY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        base_url = (
            os.environ.get("AI_AGENT_SUMMARY_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL_SUMMARY")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
    else:
        api_key = (
            os.environ.get("AI_AGENT_DEFAULT_API_KEY")
            or os.environ.get("OPENAI_API_KEY_DEFAULT")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        base_url = (
            os.environ.get("AI_AGENT_DEFAULT_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL_DEFAULT")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")

    if not api_key:
        return None

    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    try:
        temperature = float(cfg.get("OPENAI_TEMPERATURE", os.environ.get("OPENAI_TEMPERATURE", 0.5)))
    except Exception:
        temperature = 0.5
    try:
        max_tokens = int(cfg.get("OPENAI_MAX_TOKENS", os.environ.get("OPENAI_MAX_TOKENS", 1200)))
    except Exception:
        max_tokens = 1200
    try:
        max_lines = int(cfg.get("AI_MAX_LINES", os.environ.get("AI_MAX_LINES", 50)))
    except Exception:
        max_lines = 50

    # system prompt（可被环境变量覆盖）
    system_prompt = (os.environ.get("AI_SYSTEM_PROMPT") or "").strip()
    if not system_prompt:
        system_prompt = (
            "你是一位资深的金融市场分析师和投资研究专家。\n\n"
            "分析任务：\n"
            "基于提供的新闻数据，进行深度的金融市场分析。你可以自由地：\n"
            "1. 识别关键的市场驱动因素和趋势\n"
            "2. 分析对不同资产类别的潜在影响（股票、债券、汇率、大宗商品等）\n"
            "3. 评估行业和公司层面的机会与风险\n"
            "4. 提供可执行的投资见解和策略建议\n"
            "5. 指出信息缺口和不确定性\n\n"
            "输出风格：\n"
            "- 结构清晰，逻辑严密\n"
            "- 避免冗长的新闻摘要，重点在于分析和见解\n"
            "- 用具体的数据和事实支撑观点\n"
            "- 对不确定的信息标注风险提示\n"
            "- 可以自由选择最相关的分析维度，无需强制覆盖所有领域\n\n"
            "语言：中文，专业但易于理解"
        )

    # user content：把 items 用 JSON 结构提供给模型
    news_summary = json.dumps(items or [], ensure_ascii=False, indent=2)
    user_content = (
        f"以下是 {mode} 模式的热点新闻数据（共 {len(items or [])} 条）：\n\n"
        f"{news_summary}\n\n"
        f"请基于上述新闻数据进行深度分析，提供你的专业见解。"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        print(f"Agent：调用 {model} 进行分析...")
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=body, timeout=3660, proxies=proxies)
        if resp.status_code != 200:
            print(f"Agent：AI 调用失败 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        if "error" in data:
            print(f"Agent：AI 错误: {data.get('error', {}).get('message', '未知错误')}")
            return None
        content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()
        if not content:
            print("Agent：AI 返回为空")
            return None
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if len(lines) > max_lines:
            print(f"Agent：结果行数 {len(lines)} 超过限制 {max_lines}，进行裁剪")
            return "\n".join(lines[:max_lines])
        return "\n".join(lines)
    except requests.Timeout:
        print("Agent：AI 请求超时")
        return None
    except requests.RequestException as e:
        print(f"Agent：网络错误 ({type(e).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Agent：异常错误: {e}")
        return None
