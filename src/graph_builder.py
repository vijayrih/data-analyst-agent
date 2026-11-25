from typing import Dict, List

from langgraph.graph import END, StateGraph

from src.nodes import (
    data_analyst_agent,
    planner_agent,
    report_writer_agent,
    reviewer_agent,
    sql_agent,
    visualization_agent,
)
from src.state import GraphState


def stage_2_router(state: GraphState) -> Dict[str, object]:
    """Join node where parallel Stage 1 extractors converge before Stage 2."""
    return {}


def route_stage_1(state: GraphState) -> List[str] | str:
    """Route from planner_agent to SQL agent, Data Analyst agent, both in parallel, or report."""
    plan = state.get("current_plan") or []
    has_db = bool(state.get("db_uri"))
    has_files = bool(state.get("active_files"))
    has_extracted = bool(state.get("extracted_data"))

    targets: List[str] = []
    for step in plan:
        if not isinstance(step, dict):
            continue
        agent = step.get("agent")
        if agent == "sql_agent" and has_db and "sql_agent" not in targets:
            targets.append("sql_agent")
        elif (
            agent == "data_analyst_agent"
            and (has_files or has_extracted)
            and "data_analyst_agent" not in targets
        ):
            targets.append("data_analyst_agent")

    if not targets:
        return "report_writer_agent"
    return targets


def route_stage_2(state: GraphState) -> str:
    """Route from stage_2_router to visualization_agent or directly to report_writer_agent."""
    if state.get("needs_visualization") and bool(state.get("extracted_data")):
        return "visualization_agent"
    return "report_writer_agent"


def route_after_review(state: GraphState) -> str:
    """Exit on approval or max iterations (3); otherwise loop back to planner_agent."""
    if state.get("reviewer_feedback") is None or state.get("iteration_count", 0) >= 3:
        return END
    return "planner_agent"


def build_graph():
    """Compile and return the 2-stage Autonomous Data Insights Network StateGraph."""
    workflow = StateGraph(GraphState)

    workflow.add_node("planner_agent", planner_agent)
    workflow.add_node("sql_agent", sql_agent)
    workflow.add_node("data_analyst_agent", data_analyst_agent)
    workflow.add_node("stage_2_router", stage_2_router)
    workflow.add_node("visualization_agent", visualization_agent)
    workflow.add_node("report_writer_agent", report_writer_agent)
    workflow.add_node("reviewer_agent", reviewer_agent)

    workflow.set_entry_point("planner_agent")

    workflow.add_conditional_edges(
        "planner_agent",
        route_stage_1,
        {
            "sql_agent": "sql_agent",
            "data_analyst_agent": "data_analyst_agent",
            "report_writer_agent": "report_writer_agent",
        },
    )

    workflow.add_edge("sql_agent", "stage_2_router")
    workflow.add_edge("data_analyst_agent", "stage_2_router")

    workflow.add_conditional_edges(
        "stage_2_router",
        route_stage_2,
        {
            "visualization_agent": "visualization_agent",
            "report_writer_agent": "report_writer_agent",
        },
    )

    workflow.add_edge("visualization_agent", "report_writer_agent")
    workflow.add_edge("report_writer_agent", "reviewer_agent")

    workflow.add_conditional_edges(
        "reviewer_agent",
        route_after_review,
        {
            END: END,
            "planner_agent": "planner_agent",
        },
    )

    return workflow.compile()


app_graph = build_graph()
