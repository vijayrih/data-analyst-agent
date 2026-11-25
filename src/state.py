import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def merge_dicts(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Safely merge two dictionaries during parallel node execution."""
    return {**(a or {}), **(b or {})}


def combine_feedback(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Concatenate non-empty feedback strings, or reset when explicitly cleared with None."""
    if b is None:
        return None
    parts = [s.strip() for s in (a, b) if s and s.strip()]
    return "\n---\n".join(parts) if parts else None


class GraphState(TypedDict):
    """Shared state for the Autonomous Data Insights Network graph."""

    user_query: str
    schema_context: str
    db_uri: Optional[str]
    active_files: Dict[str, Any]
    current_plan: Optional[List[Dict[str, Any]]]
    needs_visualization: bool
    extracted_data: Annotated[Dict[str, Any], merge_dicts]
    chart_file_paths: Annotated[List[str], operator.add]
    draft_report: Optional[str]
    reviewer_feedback: Annotated[Optional[str], combine_feedback]
    iteration_count: int
