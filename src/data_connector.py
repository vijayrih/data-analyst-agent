from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL


def _sanitize_identifier(name: str) -> str:
    """Convert a filename or sheet name into a clean, safe dataset key."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_")
    return cleaned or "dataset"


def _format_dataframe_schema(name: str, df: pd.DataFrame) -> str:
    """Format a DataFrame's shape, columns/dtypes, and first 3 sample rows."""
    cols_summary = ", ".join(f"{col} ({dtype})" for col, dtype in df.dtypes.items())
    sample_str = df.head(3).to_string(index=False) if not df.empty else "(empty dataset)"
    return (
        f"Dataset: `{name}`\n"
        f"- Shape: {len(df)} rows x {len(df.columns)} columns\n"
        f"- Columns: {cols_summary}\n"
        f"- Sample Rows (first 3):\n{sample_str}"
    )


def load_uploaded_files(uploaded_files: Iterable[Any]) -> Tuple[Dict[str, pd.DataFrame], str]:
    """Load uploaded CSV and multi-sheet Excel files into DataFrames and schema text.

    Args:
        uploaded_files: Iterable of Streamlit UploadedFile (or file-like) objects.

    Returns:
        Tuple of (dataframes_dict, file_schema_text).
    """
    dataframes: Dict[str, pd.DataFrame] = {}
    schema_sections: List[str] = []

    if not uploaded_files:
        return dataframes, ""

    for uploaded_file in uploaded_files:
        file_name = getattr(uploaded_file, "name", "uploaded_file")
        suffix = Path(file_name).suffix.lower()
        base_name = _sanitize_identifier(Path(file_name).stem)

        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        if suffix == ".csv":
            df = pd.read_csv(uploaded_file)
            dataset_key = base_name
            if dataset_key in dataframes:
                idx = 2
                while f"{base_name}_{idx}" in dataframes:
                    idx += 1
                dataset_key = f"{base_name}_{idx}"
            dataframes[dataset_key] = df
            schema_sections.append(_format_dataframe_schema(dataset_key, df))
        elif suffix == ".xlsx":
            excel_file = pd.ExcelFile(uploaded_file, engine="openpyxl")
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                clean_sheet = _sanitize_identifier(str(sheet_name))
                dataset_key = f"{base_name}_{clean_sheet}"
                if dataset_key in dataframes:
                    idx = 2
                    while f"{dataset_key}_{idx}" in dataframes:
                        idx += 1
                    dataset_key = f"{dataset_key}_{idx}"
                dataframes[dataset_key] = df
                schema_sections.append(_format_dataframe_schema(dataset_key, df))
        else:
            raise ValueError(f"Unsupported file format '{suffix}' for '{file_name}'. Only .csv and .xlsx are supported.")

    file_schema_text = "\n\n".join(schema_sections)
    return dataframes, file_schema_text


def connect_and_inspect_db(
    db_type: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> Tuple[str, str, List[str]]:
    """Connect to a live SQL database, verify connectivity, and inspect its schema.

    Args:
        db_type: Database type ('MySQL' or 'PostgreSQL').
        host: Hostname or IP address.
        port: Database port number.
        user: Database username.
        password: Database password.
        database: Database/schema name.

    Returns:
        Tuple of (db_uri_string, db_schema_text, table_list).
    """
    normalized_type = db_type.strip().lower()
    if normalized_type == "mysql":
        driver = "mysql+pymysql"
        effective_port = int(port or 3306)
        dialect_label = "MySQL"
    elif normalized_type in ("postgresql", "postgres"):
        driver = "postgresql+psycopg2"
        effective_port = int(port or 5432)
        dialect_label = "PostgreSQL"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

    url = URL.create(
        drivername=driver,
        username=user,
        password=password,
        host=host,
        port=effective_port,
        database=database,
    )
    db_uri_string = url.render_as_string(hide_password=False)

    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        inspector = inspect(engine)
        table_list = inspector.get_table_names()

        schema_lines = [f"Database Dialect: {dialect_label} | Database: `{database}`"]
        if not table_list:
            schema_lines.append("No tables found in this database.")
        else:
            for table_name in table_list:
                columns = inspector.get_columns(table_name)
                pk_info = inspector.get_pk_constraint(table_name) or {}
                pk_cols = set(pk_info.get("constrained_columns") or [])
                fks = inspector.get_foreign_keys(table_name) or []

                col_descriptions = []
                for col in columns:
                    col_name = col["name"]
                    col_type = str(col["type"])
                    pk_tag = " [PK]" if col_name in pk_cols else ""
                    col_descriptions.append(f"{col_name} ({col_type}{pk_tag})")

                fk_descriptions = []
                for fk in fks:
                    src_cols = ", ".join(fk.get("constrained_columns") or [])
                    ref_table = fk.get("referred_table", "")
                    ref_cols = ", ".join(fk.get("referred_columns") or [])
                    if src_cols and ref_table:
                        fk_descriptions.append(f"{src_cols} -> {ref_table}({ref_cols})")

                table_block = [
                    f"Table `{table_name}`:",
                    f"  Columns: {', '.join(col_descriptions)}",
                ]
                if fk_descriptions:
                    table_block.append(f"  Foreign Keys: {'; '.join(fk_descriptions)}")
                schema_lines.append("\n".join(table_block))

        db_schema_text = "\n\n".join(schema_lines)
        return db_uri_string, db_schema_text, table_list
    finally:
        engine.dispose()
