# coding=utf-8

from typing import TypedDict, Optional, List, Dict, Any, Callable
import os

# LangGraph
from langgraph.graph import StateGraph, END


class AgentState(TypedDict, total=False):
    report_data: Dict
    mode: str
    proxy_url: Optional[str]
    news_items: List[Dict]
    is_finance: bool
    is_finance_rule: bool
    is_finance_llm: bool
    ai_summary: Optional[str]
    cfg: Dict[str, Any]


class TrendRadarAgent:
    """
    一个极简可插拔的 LangGraph Agent：
    - 通过回调注入 fetch/summarize/collect/relevance，避免与 main.py 循环依赖
    - 节点：router -> collect -> relevance -> (END | fetch -> summarize -> END)
    """

    def __init__(
        self,
        fetch_content_fn: Callable[[List[Dict], Dict[str, Any], Optional[str]], List[Dict]],
        summarize_fn: Callable[[Dict, List[Dict], Dict[str, Any], Optional[str], str], Optional[str]],
        collect_fn: Callable[[Dict, int], List[Dict]],
        relevance_fn: Callable[[List[Dict]], bool],
        relevance_llm_fn: Optional[Callable[[List[Dict], Dict[str, Any], Optional[str], str], Optional[bool]]] = None,
    ) -> None:
        self.fetch_content_fn = fetch_content_fn
        self.summarize_fn = summarize_fn
        self.collect_fn = collect_fn
        self.relevance_fn = relevance_fn
        self.relevance_llm_fn = relevance_llm_fn
        self.graph = self._build_graph()

    # 1) 路由/策略
    def _router(self, state: AgentState) -> AgentState:
        mode = state.get("mode") or "daily"
        # 模型策略：总结用更强模型（可被环境变量覆盖），其它默认模型给下游参考
        summary_model = (
            os.environ.get("AI_AGENT_SUMMARY_MODEL")
            or os.environ.get("OPENAI_MODEL_SUMMARY")
            or "gpt-4o"
        )
        default_model = (
            os.environ.get("AI_AGENT_DEFAULT_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        relevance_llm_enabled = (os.environ.get("AI_RELEVANCE_LLM_ENABLED", "true").lower() in ("1", "true", "yes"))
        cfg: Dict[str, Any] = {
            "AI_FETCH_TOPK": 15 if mode == "daily" else 8,
            "AI_FETCH_TIMEOUT": 10.0,
            "AI_FETCH_MAX_CHARS": 3000,
            "AI_FETCH_CONTENT": True,
            "OPENAI_TEMPERATURE": 0.5 if mode == "daily" else 0.4,
            "OPENAI_MAX_TOKENS": 1200 if mode == "daily" else 900,
            "AI_MAX_LINES": 50,
            # 模型配置
            "OPENAI_MODEL_SUMMARY": summary_model,
            "OPENAI_MODEL_DEFAULT": default_model,
            # 二判开关
            "RELEVANCE_LLM_ENABLED": relevance_llm_enabled,
        }
        state["cfg"] = cfg
        return state

    # 2) 收集候选条目
    def _collect(self, state: AgentState) -> AgentState:
        report = state["report_data"]
        topk = int(state["cfg"]["AI_FETCH_TOPK"]) if state.get("cfg") else 10
        items = self.collect_fn(report, topk)
        state["news_items"] = items
        return state

    # 3) 金融相关性判别
    def _relevance(self, state: AgentState) -> AgentState:
        items = state.get("news_items", [])
        is_finance_rule = bool(self.relevance_fn(items))
        state["is_finance_rule"] = is_finance_rule
        state["is_finance"] = is_finance_rule  # 默认先用规则判断
        return state

    # 3.1) LLM（二判，使用默认模型与默认 API KEY/BASE_URL）
    def _relevance_llm(self, state: AgentState) -> AgentState:
        enabled = state.get("cfg", {}).get("RELEVANCE_LLM_ENABLED", True)
        if not enabled or self.relevance_llm_fn is None:
            # 不启用二判，保留规则结果
            state["is_finance_llm"] = state.get("is_finance_rule", False)
            state["is_finance"] = state.get("is_finance_rule", False)
            return state
        try:
            result = self.relevance_llm_fn(
                state.get("news_items", []),
                state.get("cfg", {}),
                state.get("proxy_url"),
                state.get("mode") or "daily",
            )
            if result is None:
                # 二判失败，沿用规则
                state["is_finance_llm"] = state.get("is_finance_rule", False)
                state["is_finance"] = state.get("is_finance_rule", False)
            else:
                state["is_finance_llm"] = bool(result)
                state["is_finance"] = bool(result)
        except Exception:
            # 防御：任何异常都沿用规则
            state["is_finance_llm"] = state.get("is_finance_rule", False)
            state["is_finance"] = state.get("is_finance_rule", False)
        return state

    # 4) 抓取正文
    def _fetch(self, state: AgentState) -> AgentState:
        if not state.get("cfg", {}).get("AI_FETCH_CONTENT", True):
            return state
        state["news_items"] = self.fetch_content_fn(
            state.get("news_items", []),
            state.get("cfg", {}),
            state.get("proxy_url"),
        )
        return state

    # 5) 调 LLM 总结
    def _summarize(self, state: AgentState) -> AgentState:
        state["ai_summary"] = self.summarize_fn(
            state["report_data"],
            state.get("news_items", []),
            state.get("cfg", {}),
            state.get("proxy_url"),
            state.get("mode") or "daily",
        )
        return state

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("router", self._router)
        g.add_node("collect", self._collect)
        g.add_node("relevance", self._relevance)
        g.add_node("relevance_llm", self._relevance_llm)
        g.add_node("fetch", self._fetch)
        g.add_node("summarize", self._summarize)

        g.set_entry_point("router")
        g.add_edge("router", "collect")
        g.add_edge("collect", "relevance")
        g.add_edge("relevance", "relevance_llm")

        def after_relevance_llm(state: AgentState):
            return "fetch" if state.get("is_finance") else END

        g.add_conditional_edges("relevance_llm", after_relevance_llm, {"fetch": "fetch", END: END})
        g.add_edge("fetch", "summarize")
        g.add_edge("summarize", END)
        return g.compile()

    def run(self, report_data: Dict, mode: str = "daily", proxy_url: Optional[str] = None) -> Dict[str, Any]:
        """
        运行Agent进行金融新闻分析
        
        返回：
        - dict: 包含以下字段
            - ai_summary: AI生成的摘要文本（如果是金融相关）
            - is_finance: 最终判断结果（规则判断 + LLM二判）
            - is_finance_rule: 规则判断结果
            - is_finance_llm: LLM判断结果
        """
        initial: AgentState = {
            "report_data": report_data,
            "mode": mode,
            "proxy_url": proxy_url,
        }
        final_state = self.graph.invoke(initial)
        return {
            "ai_summary": final_state.get("ai_summary"),
            "is_finance": final_state.get("is_finance", False),
            "is_finance_rule": final_state.get("is_finance_rule", False),
            "is_finance_llm": final_state.get("is_finance_llm", False),
        }
