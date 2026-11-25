import glob
import json
import os
import re
from typing import Any, Dict, List, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import pandas as pd  
from langchain_core.messages import HumanMessage, SystemMessage  
from langchain_groq import ChatGroq  
from pydantic import BaseModel, Field  
from sqlalchemy import create_engine, text  

from src.state import GraphState  

CHARTS_DIR = "temp_charts"
FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE)\b"
)


class Step(BaseModel):
    """Single execution step in the analysis plan."""

    agent: Literal["sql_agent", "data_analyst_agent"] = Field(
        description="Target specialist agent: 'sql_agent' or 'data_analyst_agent'."
    )
    task: str = Field(description="Clear task description for the selected agent.")


class Plan(BaseModel):
    """Structured execution plan produced by the planner agent."""

    steps: List[Step] = Field(
        default_factory=list,
        description="Ordered list of extraction/analysis steps to run in Stage 1.",
    )
    needs_visualization: bool = Field(
        description="Whether charts should be generated in Stage 2."
    )


class Review(BaseModel):
    """Structured quality review produced by the reviewer agent."""

    is_approved: bool = Field(
        description="True if the draft report accurately and completely answers the user query."
    )
    feedback: str = Field(
        default="",
        description="Specific actionable feedback if not approved; empty string if approved.",
    )


def get_llm() -> ChatGroq:
    """Instantiate the Groq LLM configured for deterministic reasoning."""
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)


def clean_code_block(text_content: str) -> str:
    """Strip markdown code fences (e.g., ```python, ```sql, ```) from LLM output."""
    if not text_content:
        return ""
    cleaned = text_content.strip()
    match = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_agent_tasks(state: GraphState, agent_name: str) -> str:
    """Collect task instructions assigned to a specific agent from current_plan."""
    plan = state.get("current_plan") or []
    tasks = [
        step.get("task", "")
        for step in plan
        if isinstance(step, dict) and step.get("agent") == agent_name and step.get("task")
    ]
    return "\n".join(f"- {t}" for t in tasks) if tasks else state.get("user_query", "")


def planner_agent(state: GraphState) -> Dict[str, Any]:
    """Create a structured Stage 1 execution plan and decide visualization needs."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(Plan, method="json_mode")

    has_db = bool(state.get("db_uri"))
    active_file_keys = list((state.get("active_files") or {}).keys())
    extracted_keys = list((state.get("extracted_data") or {}).keys())
    feedback = state.get("reviewer_feedback") or "None"

    system_prompt = (
        "You are the Lead Data Strategy Planner in an Autonomous Data Insights Network.\n"
        "Your job is to inspect the available data sources, schema context, user question, "
        "and any prior reviewer feedback, then output a JSON execution plan.\n\n"
        "Available Stage 1 Agents:\n"
        "- 'sql_agent': Queries the live SQL database (ONLY assign if a live DB is connected).\n"
        "- 'data_analyst_agent': Analyzes uploaded DataFrames in `active_files` and/or "
        "combines data in `extracted_data` using Pandas.\n\n"
        "Rules:\n"
        "1. Only include agents needed for the user's question and supported by active sources.\n"
        "2. Set 'needs_visualization' to true when trends, comparisons, distributions, or "
        "rankings would benefit from a chart, or when the user asks for charts/plots.\n"
        "3. Respond ONLY with a valid JSON object matching this exact template:\n"
        '{"steps": [{"agent": "sql_agent", "task": "..."}, '
        '{"agent": "data_analyst_agent", "task": "..."}], "needs_visualization": true}'
    )

    user_prompt = (
        f"User Query: {state.get('user_query', '')}\n\n"
        f"Live Database Connected: {has_db}\n"
        f"Uploaded File Datasets: {active_file_keys}\n"
        f"Already Extracted Keys: {extracted_keys}\n"
        f"Prior Reviewer Feedback:\n{feedback}\n\n"
        f"Schema Context:\n{state.get('schema_context', 'No schema provided.')}"
    )

    plan = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    valid_steps: List[Dict[str, Any]] = []
    for step in plan.steps:
        if step.agent == "sql_agent" and not has_db:
            continue
        if step.agent == "data_analyst_agent" and not active_file_keys and not extracted_keys:
            continue
        valid_steps.append(step.model_dump())

    return {
        "current_plan": valid_steps,
        "needs_visualization": bool(plan.needs_visualization),
        "iteration_count": state.get("iteration_count", 0) + 1,
        "reviewer_feedback": None,
    }


def sql_agent(state: GraphState) -> Dict[str, Any]:
    """Generate, validate (read-only), and execute SQL against the connected database."""
    db_uri = state.get("db_uri")
    if not db_uri:
        return {"reviewer_feedback": "SQL Error: No active database connection URI."}

    llm = get_llm()
    assigned_tasks = _extract_agent_tasks(state, "sql_agent")
    feedback = state.get("reviewer_feedback") or "None"

    system_prompt = (
        "You are an expert SQL Data Engineer.\n"
        "Write a single, valid, read-only SELECT SQL query matching the database dialect "
        "specified in the schema context.\n"
        "Rules:\n"
        "- Output ONLY the raw SQL query (no explanation, no comments outside SQL).\n"
        "- Use ONLY read-only SELECT / WITH CTE statements. Never modify data or schema.\n"
        "- Respect exact table and column names from the schema context.\n"
        "- Limit huge raw row dumps appropriately (e.g., aggregate or LIMIT 500 where sensible)."
    )

    user_prompt = (
        f"User Query: {state.get('user_query', '')}\n"
        f"Assigned SQL Tasks:\n{assigned_tasks}\n"
        f"Prior Feedback:\n{feedback}\n\n"
        f"Schema Context:\n{state.get('schema_context', '')}"
    )

    engine = None
    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        sql_query = clean_code_block(str(response.content))
        if not sql_query:
            return {"reviewer_feedback": "SQL Error: Model returned an empty SQL query."}

        match = FORBIDDEN_SQL_PATTERN.search(sql_query.upper())
        if match:
            return {
                "reviewer_feedback": (
                    f"SQL Guardrail Violation: Forbidden keyword '{match.group(1)}' detected. "
                    "Only read-only SELECT queries are permitted."
                )
            }

        engine = create_engine(db_uri, pool_pre_ping=True)
        with engine.connect() as conn:
            df = pd.read_sql(text(sql_query), conn)

        return {"extracted_data": {"sql_data": df.to_dict(orient="records")}}
    except Exception as exc:
        return {"reviewer_feedback": f"SQL Error: {exc}"}
    finally:
        if engine is not None:
            engine.dispose()


def data_analyst_agent(state: GraphState) -> Dict[str, Any]:
    """Run Pandas analysis over in-memory active_files and extracted_data via exec()."""
    llm = get_llm()
    active_files = state.get("active_files") or {}
    extracted_data = state.get("extracted_data") or {}
    assigned_tasks = _extract_agent_tasks(state, "data_analyst_agent")
    feedback = state.get("reviewer_feedback") or "None"

    system_prompt = (
        "You are a Senior Python & Pandas Data Analyst.\n"
        "Write clean, executable Python code to answer the assigned task.\n"
        "Variables ALREADY injected into your execution namespace (do NOT recreate or reload files):\n"
        "- `pd`: pandas module\n"
        "- `active_files`: dict mapping dataset names (strings) -> pandas DataFrames\n"
        "- `extracted_data`: dict of previously extracted records (e.g. `extracted_data.get('sql_data')`)\n\n"
        "Rules:\n"
        "- Output ONLY executable Python code.\n"
        "- Do NOT read files from disk; access DataFrames via `active_files['dataset_name']`.\n"
        "- Always work on `.copy()` if mutating DataFrames.\n"
        "- Store your final summary/result DataFrame in a variable named `result_df`.\n"
        "- Ensure `result_df` is a `pd.DataFrame` (use `.reset_index()` after `groupby` if needed)."
    )

    user_prompt = (
        f"User Query: {state.get('user_query', '')}\n"
        f"Assigned Analysis Tasks:\n{assigned_tasks}\n"
        f"Available `active_files` keys: {list(active_files.keys())}\n"
        f"Available `extracted_data` keys: {list(extracted_data.keys())}\n"
        f"Prior Feedback:\n{feedback}\n\n"
        f"Schema Context:\n{state.get('schema_context', '')}"
    )

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        code = clean_code_block(str(response.content))
        if not code:
            return {"reviewer_feedback": "Data Analysis Error: Model returned empty Python code."}

        exec_namespace: Dict[str, Any] = {
            "pd": pd,
            "active_files": {k: v.copy() for k, v in active_files.items()},
            "extracted_data": dict(extracted_data),
        }
        exec(code, exec_namespace)

        result_df = exec_namespace.get("result_df")
        if result_df is None:
            return {
                "reviewer_feedback": (
                    "Data Analysis Error: Generated code did not define `result_df`."
                )
            }
        if isinstance(result_df, pd.Series):
            result_df = result_df.to_frame().reset_index()
        elif not isinstance(result_df, pd.DataFrame):
            result_df = pd.DataFrame(result_df)

        return {"extracted_data": {"analysis_data": result_df.to_dict(orient="records")}}
    except Exception as exc:
        return {"reviewer_feedback": f"Data Analysis Error: {exc}"}


def visualization_agent(state: GraphState) -> Dict[str, Any]:
    """Generate Matplotlib PNG charts in temp_charts/ using memory injection and snapshotting."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    before_files = set(glob.glob(os.path.join(CHARTS_DIR, "*.png")))

    llm = get_llm()
    active_files = state.get("active_files") or {}
    extracted_data = state.get("extracted_data") or {}

    extracted_summary = {
        key: {
            "row_count": len(records) if isinstance(records, list) else 0,
            "columns": list(records[0].keys())
            if isinstance(records, list) and len(records) > 0 and isinstance(records[0], dict)
            else [],
        }
        for key, records in extracted_data.items()
    }

    system_prompt = (
        "You are a Data Visualization Specialist using Matplotlib.\n"
        "Write clean Python code to create 1 to 3 clear, executive-ready PNG charts.\n"
        "Variables ALREADY injected into your execution namespace:\n"
        "- `plt`: matplotlib.pyplot\n"
        "- `pd`: pandas\n"
        "- `extracted_data`: dict where values are lists of record dicts "
        "(convert with `pd.DataFrame(extracted_data['key'])`)\n"
        "- `active_files`: dict of uploaded DataFrames\n\n"
        "Rules:\n"
        "- Output ONLY executable Python code.\n"
        "- Save each chart inside `temp_charts/` with a descriptive filename, e.g. "
        "`plt.savefig('temp_charts/revenue_by_region.png', dpi=150, bbox_inches='tight')`.\n"
        "- NEVER call `plt.show()`.\n"
        "- Always set clear titles, axis labels, readable tick formatting, and `plt.tight_layout()`.\n"
        "- Guard against empty DataFrames before plotting."
    )

    user_prompt = (
        f"User Query: {state.get('user_query', '')}\n"
        f"Available `extracted_data` keys and shapes: {json.dumps(extracted_summary)}\n"
        f"Available `active_files` keys: {list(active_files.keys())}\n"
        f"Schema Context:\n{state.get('schema_context', '')}"
    )

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        code = clean_code_block(str(response.content))
        if not code:
            return {"chart_file_paths": []}

        exec_namespace: Dict[str, Any] = {
            "plt": plt,
            "pd": pd,
            "extracted_data": dict(extracted_data),
            "active_files": {k: v.copy() for k, v in active_files.items()},
        }
        exec(code, exec_namespace)

        after_files = set(glob.glob(os.path.join(CHARTS_DIR, "*.png")))
        new_files = sorted(
            path.replace("\\", "/") for path in (after_files - before_files)
        )
        return {"chart_file_paths": new_files}
    except Exception as exc:
        return {"reviewer_feedback": f"Visualization Error: {exc}"}
    finally:
        plt.close("all")


def report_writer_agent(state: GraphState) -> Dict[str, Any]:
    """Synthesize extracted data and context into an executive Markdown report."""
    llm = get_llm()
    extracted_data = state.get("extracted_data") or {}
    chart_paths = state.get("chart_file_paths") or []
    feedback = state.get("reviewer_feedback") or "None"

    try:
        data_str = json.dumps(extracted_data, default=str, indent=2)
    except Exception:
        data_str = str(extracted_data)

    max_chars = 8000
    if len(data_str) > max_chars:
        data_str = (
            data_str[:max_chars]
            + "\n\n... [DATA TRUNCATED AT 8,000 CHARACTERS FOR PROMPT SAFETY. "
            "Use this representative sample together with the generated charts for insights.]"
        )

    system_prompt = (
        "You are an Executive Data Insights Writer.\n"
        "Write a clear, accurate, data-grounded Markdown report answering the user's question.\n"
        "CRITICAL RULES:\n"
        "- Base every number and finding strictly on the provided extracted data.\n"
        "- NEVER include Markdown image links (such as `![Chart](...)`). "
        "The UI renders generated charts automatically below your report.\n"
        "- If execution errors or missing data occurred, explain them transparently.\n"
        "- Structure the report with clean headings, key takeaways, metrics tables/bullets, "
        "and actionable recommendations."
    )

    user_prompt = (
        f"User Query: {state.get('user_query', '')}\n\n"
        f"Generated Charts (rendered separately by UI): {chart_paths}\n"
        f"Prior Feedback / Errors:\n{feedback}\n\n"
        f"Schema Context:\n{state.get('schema_context', '')}\n\n"
        f"Extracted Data:\n{data_str}"
    )

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    return {"draft_report": str(response.content).strip()}


def reviewer_agent(state: GraphState) -> Dict[str, Any]:
    """Review the draft report for completeness, accuracy, and alignment with the query."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(Review, method="json_mode")

    extracted_keys = list((state.get("extracted_data") or {}).keys())
    chart_paths = state.get("chart_file_paths") or []
    upstream_feedback = state.get("reviewer_feedback") or "None"

    system_prompt = (
        "You are a Strict Quality Assurance Reviewer for an Autonomous Data Insights Network.\n"
        "Evaluate whether the draft report accurately answers the user's query using real data, "
        "without unresolved SQL/Python errors or hallucinated numbers.\n"
        "Note: Charts are rendered natively by the UI, so the report must NOT contain image links.\n\n"
        "Respond ONLY with a valid JSON object matching this exact template:\n"
        '{"is_approved": true, "feedback": ""}'
    )

    user_prompt = (
        f"User Query: {state.get('user_query', '')}\n"
        f"Iteration: {state.get('iteration_count', 1)}\n"
        f"Extracted Data Keys: {extracted_keys}\n"
        f"Generated Chart Count: {len(chart_paths)}\n"
        f"Upstream Errors/Feedback: {upstream_feedback}\n\n"
        f"Draft Report:\n{state.get('draft_report', '')}"
    )

    review = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    if review.is_approved:
        return {"reviewer_feedback": None}
    return {
        "reviewer_feedback": review.feedback or "Report requires refinement to address the query."
    }
