import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv  

load_dotenv()

import streamlit as st  

from src.data_connector import connect_and_inspect_db, load_uploaded_files  
from src.graph_builder import app_graph  

st.set_page_config(
    page_title="Autonomous Data Insights Network",
    page_icon="📊",
    layout="wide",
)


def _init_session_state() -> None:
    """Initialize required Streamlit session state keys."""
    defaults: Dict[str, Any] = {
        "messages": [],
        "active_files": {},
        "file_schema": "",
        "db_uri": None,
        "db_schema": "",
        "db_tables": [],
        "db_label": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_unified_schema() -> str:
    """Combine active file schemas and live database schema into one unified context."""
    parts: List[str] = []
    if st.session_state.file_schema:
        parts.append("=== UPLOADED FILE DATASETS ===\n" + st.session_state.file_schema)
    if st.session_state.db_schema:
        parts.append("=== LIVE DATABASE SCHEMA ===\n" + st.session_state.db_schema)
    return "\n\n".join(parts)


def _dedupe_existing_paths(paths: List[str]) -> List[str]:
    """Return unique chart file paths that exist on disk, preserving order."""
    seen = set()
    valid: List[str] = []
    for path in paths:
        if path and path not in seen and os.path.exists(path):
            seen.add(path)
            valid.append(path)
    return valid


_init_session_state()

st.title("Autonomous Data Insights Network")
st.caption(
    "Upload CSV / multi-sheet Excel files, connect to a live MySQL or PostgreSQL database, "
    "and ask natural language questions across any or all sources."
)

with st.sidebar:
    st.header("Data Connectors")

    # 1. File Uploader
    st.subheader("1. Upload Files (CSV / Excel)")
    uploaded_files = st.file_uploader(
        "Upload CSV or XLSX files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        try:
            dataframes, file_schema = load_uploaded_files(uploaded_files)
            st.session_state.active_files = dataframes
            st.session_state.file_schema = file_schema
            st.success(f"Loaded {len(dataframes)} dataset(s)")
            for ds_name, df in dataframes.items():
                st.caption(f"• `{ds_name}` — {df.shape[0]} rows × {df.shape[1]} cols")
        except Exception as exc:
            st.error(f"Failed to load files: {exc}")
    else:
        st.session_state.active_files = {}
        st.session_state.file_schema = ""

    st.divider()

    st.subheader("2. Database Connection")
    with st.form("db_connection_form"):
        db_type = st.selectbox("DB Type", ["MySQL", "PostgreSQL"], index=0)
        host = st.text_input("Host", value="localhost")
        default_port = 3306 if db_type == "MySQL" else 5432
        port = st.number_input("Port", min_value=1, max_value=65535, value=default_port, step=1)
        database = st.text_input("Database Name", value="")
        user = st.text_input("Username", value="root")
        password = st.text_input("Password", value="", type="password")
        connect_submitted = st.form_submit_button("Connect & Inspect", use_container_width=True)

    if connect_submitted:
        if not database.strip():
            st.warning("Please enter a database name.")
        else:
            try:
                with st.spinner("Connecting and inspecting database schema..."):
                    db_uri, db_schema, tables = connect_and_inspect_db(
                        db_type=db_type,
                        host=host.strip(),
                        port=int(port),
                        user=user.strip(),
                        password=password,
                        database=database.strip(),
                    )
                st.session_state.db_uri = db_uri
                st.session_state.db_schema = db_schema
                st.session_state.db_tables = tables
                st.session_state.db_label = f"{db_type} ({host}:{int(port)}/{database})"
            except Exception as exc:
                st.error(f"Database connection failed: {exc}")

    if st.session_state.db_uri:
        st.success(f"Connected: {st.session_state.db_label}")
        if st.session_state.db_tables:
            st.caption(
                f"Tables ({len(st.session_state.db_tables)}): "
                + ", ".join(f"`{t}`" for t in st.session_state.db_tables)
            )
        else:
            st.caption("Connected (0 tables discovered).")
        if st.button("Disconnect Database", use_container_width=True):
            st.session_state.db_uri = None
            st.session_state.db_schema = ""
            st.session_state.db_tables = []
            st.session_state.db_label = ""
            st.rerun()

    st.divider()

    # 3. Unified Schema Context Preview
    unified_schema = _build_unified_schema()
    with st.expander("Active Schema Context", expanded=False):
        if unified_schema:
            st.code(unified_schema, language="text")
        else:
            st.info("No active datasets or database connected yet.")



has_data_source = bool(st.session_state.active_files or st.session_state.db_uri)
if not has_data_source:
    st.info(
        "Connect a live database or upload at least one CSV / Excel file in the sidebar "
        "to start asking questions."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for chart_path in _dedupe_existing_paths(message.get("charts", [])):
            st.image(chart_path, use_container_width=True)

prompt = st.chat_input(
    "Ask a question across your uploaded files and/or live database...",
    disabled=not has_data_source,
)

if prompt:
    if not has_data_source:
        st.warning("Please upload a file or connect a database first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt, "charts": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    schema_context = _build_unified_schema()
    initial_state = {
        "user_query": prompt,
        "schema_context": schema_context,
        "db_uri": st.session_state.db_uri,
        "active_files": st.session_state.active_files,
        "current_plan": None,
        "needs_visualization": False,
        "extracted_data": {},
        "chart_file_paths": [],
        "draft_report": None,
        "reviewer_feedback": None,
        "iteration_count": 0,
    }

    with st.chat_message("assistant"):
        final_state: Dict[str, Any] = {}
        try:
            with st.status("🤖 Agents collaborating...", expanded=True) as status:
                for chunk in app_graph.stream(initial_state):
                    for node_name, state_update in chunk.items():
                        if not isinstance(state_update, dict):
                            continue

                        prev_charts = list(final_state.get("chart_file_paths", []))
                        prev_extracted = dict(final_state.get("extracted_data", {}))
                        final_state.update(state_update)

                        if state_update.get("chart_file_paths"):
                            final_state["chart_file_paths"] = prev_charts + list(
                                state_update["chart_file_paths"]
                            )
                        if state_update.get("extracted_data"):
                            final_state["extracted_data"] = {
                                **prev_extracted,
                                **state_update["extracted_data"],
                            }

                        if node_name == "planner_agent":
                            steps = state_update.get("current_plan") or []
                            agents_used = [s.get("agent") for s in steps]
                            st.write(
                                f"**Planner** (Iteration {state_update.get('iteration_count', 1)}): "
                                f"agents={agents_used}, visualization={state_update.get('needs_visualization')}"
                            )
                        elif node_name == "sql_agent":
                            if "extracted_data" in state_update:
                                rows = len(state_update["extracted_data"].get("sql_data", []))
                                st.write(f"**SQL Agent**: extracted {rows} row(s).")
                            else:
                                st.write(f"**SQL Agent**: {state_update.get('reviewer_feedback')}")
                        elif node_name == "data_analyst_agent":
                            if "extracted_data" in state_update:
                                rows = len(state_update["extracted_data"].get("analysis_data", []))
                                st.write(f"**Data Analyst Agent**: produced {rows} row(s).")
                            else:
                                st.write(
                                    f"**Data Analyst Agent**: {state_update.get('reviewer_feedback')}"
                                )
                        elif node_name == "stage_2_router":
                            st.write("**Stage 2 Router**: Stage 1 extraction complete.")
                        elif node_name == "visualization_agent":
                            charts_created = state_update.get("chart_file_paths") or []
                            st.write(f"**Visualization Agent**: generated {len(charts_created)} chart(s).")
                        elif node_name == "report_writer_agent":
                            st.write("**Report Writer Agent**: drafted executive report.")
                        elif node_name == "reviewer_agent":
                            fb = state_update.get("reviewer_feedback")
                            if fb is None:
                                st.write("**Reviewer Agent**: Report approved.")
                            else:
                                st.write(f"**Reviewer Agent**: Requested revision — {fb}")

                status.update(label="Analysis complete!", state="complete", expanded=False)

            report_text = final_state.get("draft_report") or "No report was generated."
            chart_paths = _dedupe_existing_paths(final_state.get("chart_file_paths", []))

            st.markdown(report_text)
            for chart_path in chart_paths:
                st.image(chart_path, use_container_width=True)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": report_text,
                    "charts": chart_paths,
                }
            )
        except Exception as exc:
            error_msg = f"Execution failed: {exc}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "charts": []}
            )
