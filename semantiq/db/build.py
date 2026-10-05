"""Phase 0: build the DuckDB semantic mirror from CSVs.

Reads data/*.csv, creates the star schema, adds the DAX calculated columns
as SQL, creates one view per certified DAX measure, then runs the correctness
gate: every view must match the known Power BI expected values.

Usage:
    python -m semantiq.db.build
"""

import sys
from pathlib import Path

import duckdb

DATA = Path(__file__).resolve().parents[2] / "data"
DB_PATH = Path(__file__).resolve().parents[2] / "semantiq.duckdb"

# ----------------------------------------------------------------------------
# Band logic — direct SQL translations of powerbi/Calculated_Columns.dax
# ----------------------------------------------------------------------------
BAND_COLUMNS = """
    CASE WHEN Exited = 1 THEN 'Churned' ELSE 'Retained' END                       AS churn_label,
    CASE WHEN IsActiveMember = 1 THEN 'Active' ELSE 'Inactive' END                 AS activity_label,
    CASE WHEN HasCrCard = 1 THEN 'Card holder' ELSE 'No card' END                  AS card_label,
    CASE
        WHEN Age < 30 THEN '18-29'
        WHEN Age < 40 THEN '30-39'
        WHEN Age < 50 THEN '40-49'
        WHEN Age < 60 THEN '50-59'
        ELSE '60+'
    END                                                                           AS age_group,
    CASE
        WHEN CreditScore < 580 THEN 'Poor (<580)'
        WHEN CreditScore < 670 THEN 'Fair (580-669)'
        WHEN CreditScore < 740 THEN 'Good (670-739)'
        ELSE 'Excellent (740+)'
    END                                                                           AS credit_band,
    CASE
        WHEN Tenure <= 2 THEN 'New (0-2 yrs)'
        WHEN Tenure <= 5 THEN 'Growing (3-5 yrs)'
        WHEN Tenure <= 8 THEN 'Established (6-8 yrs)'
        ELSE 'Loyal (9-10 yrs)'
    END                                                                           AS tenure_group,
    CASE
        WHEN Balance = 0 THEN 'Zero balance'
        WHEN Balance < 50000 THEN 'Low (< \u20ac50k)'
        WHEN Balance < 100000 THEN 'Medium (\u20ac50k-100k)'
        WHEN Balance < 150000 THEN 'High (\u20ac100k-150k)'
        ELSE 'Very high (\u20ac150k+)'
    END                                                                           AS balance_band
"""

# ----------------------------------------------------------------------------
# Certified measures — SQL equivalents of docs/DAX_Measures_reference.dax
# name -> (view_sql, expected_value, tolerance)
# ----------------------------------------------------------------------------
MEASURES = {
    "Total Customers": (
        "SELECT COUNT(DISTINCT CustomerId) AS v FROM fact_churn", 10000, 0),
    "Churned Customers": (
        "SELECT COUNT(DISTINCT CASE WHEN Exited = 1 THEN CustomerId END) AS v FROM fact_churn", 2037, 0),
    "Retained Customers": (
        "SELECT COUNT(DISTINCT CASE WHEN Exited = 0 THEN CustomerId END) AS v FROM fact_churn", 7963, 0),
    "Churn Rate": (
        """SELECT COUNT(DISTINCT CASE WHEN Exited = 1 THEN CustomerId END)
                  / COUNT(DISTINCT CustomerId) AS v FROM fact_churn""", 0.2037, 0.001),
    "Retention Rate": (
        """SELECT 1 - COUNT(DISTINCT CASE WHEN Exited = 1 THEN CustomerId END)
                  / COUNT(DISTINCT CustomerId) AS v FROM fact_churn""", 0.7963, 0.001),
    "Total Balance": (
        "SELECT SUM(Balance) AS v FROM fact_churn", 764_900_000, 1_000_000),
    "Balance at Risk": (
        "SELECT SUM(CASE WHEN Exited = 1 THEN Balance ELSE 0 END) AS v FROM fact_churn", 185_600_000, 1_000_000),
    "Share of Balance at Risk": (
        """SELECT SUM(CASE WHEN Exited = 1 THEN Balance ELSE 0 END)
                  / SUM(Balance) AS v FROM fact_churn""", 0.243, 0.002),
    "Avg Balance per Customer": (
        "SELECT SUM(Balance) / COUNT(DISTINCT CustomerId) AS v FROM fact_churn", 76_486, 50),
    "Avg Credit Score": (
        "SELECT AVG(CreditScore) AS v FROM fact_churn", 651, 0.6),
    "Avg Credit Score Churned": (
        "SELECT AVG(CASE WHEN Exited = 1 THEN CreditScore END) AS v FROM fact_churn", 645, 0.6),
    "Avg Credit Score Retained": (
        "SELECT AVG(CASE WHEN Exited = 0 THEN CreditScore END) AS v FROM fact_churn", 652, 0.6),
    "Avg Age": (
        "SELECT AVG(Age) AS v FROM fact_churn", 38.9, 0.06),
    "Avg Age Churned": (
        "SELECT AVG(CASE WHEN Exited = 1 THEN Age END) AS v FROM fact_churn", 44.8, 0.06),
    "Avg Tenure Years": (
        "SELECT AVG(Tenure) AS v FROM fact_churn", 5.0, 0.06),
    "Avg Products": (
        "SELECT AVG(NumOfProducts) AS v FROM fact_churn", 1.53, 0.01),
    "Avg Salary": (
        "SELECT AVG(EstimatedSalary) AS v FROM fact_churn", 100_090, 100),
    "Active Member Share": (
        """SELECT COUNT(DISTINCT CASE WHEN IsActiveMember = 1 THEN CustomerId END)
                  / COUNT(DISTINCT CustomerId) AS v FROM fact_churn""", 0.515, 0.002),
}

QA_CHECKS = {
    "QA Row Count": ("SELECT COUNT(*) AS v FROM fact_churn", 10000, 0),
    "QA Null Check": (
        """SELECT
               COUNT(*) FILTER (WHERE age_group IS NULL)
             + COUNT(*) FILTER (WHERE credit_band IS NULL)
             + COUNT(*) FILTER (WHERE tenure_group IS NULL)
             + COUNT(*) FILTER (WHERE balance_band IS NULL) AS v
           FROM fact_churn""", 0, 0),
}

DIMS = [
    "Dim_Geography", "Dim_Gender", "Dim_AgeGroup", "Dim_BalanceBand",
    "Dim_CreditBand", "Dim_TenureGroup", "Dim_ChurnStatus",
]


def build(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))

    # Fact table with DAX calculated columns translated to SQL
    con.execute(f"""
        CREATE TABLE fact_churn AS
        SELECT *, {BAND_COLUMNS}
        FROM read_csv_auto('{DATA / 'Fact_Churn.csv'}', header=true)
    """)

    # Dimension tables
    for dim in DIMS:
        table = dim.replace("Dim_", "dim_").lower()
        con.execute(f"""
            CREATE TABLE {table} AS
            FROM read_csv_auto('{DATA / (dim + '.csv')}', header=true)
        """)

    # One view per certified measure
    for name, (sql, _expected, _tol) in MEASURES.items():
        view = "v_" + name.lower().replace(" ", "_")
        con.execute(f"CREATE OR REPLACE VIEW {view} AS {sql}")

    return con


def correctness_gate(con: duckdb.DuckDBPyConnection) -> bool:
    """Every measure must match its Power BI expected value. Returns True iff all pass."""
    all_checks = {**MEASURES, **QA_CHECKS}
    failed = []
    print(f"{'Check':<28} {'Expected':>16} {'Actual':>16}  Result")
    print("-" * 72)
    for name, (sql, expected, tol) in all_checks.items():
        actual = con.execute(sql).fetchone()[0]
        ok = abs(actual - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"{name:<28} {expected:>16,.4f} {actual:>16,.4f}  {status}")
        if not ok:
            failed.append(name)
    print("-" * 72)
    if failed:
        print(f"CORRECTNESS GATE: {len(failed)} FAILED -> {failed}")
        return False
    print(f"CORRECTNESS GATE: all {len(all_checks)} checks PASS")
    return True


def main() -> None:
    con = build()
    n_fact = con.execute("SELECT COUNT(*) FROM fact_churn").fetchone()[0]
    print(f"Built {DB_PATH.name}: fact_churn={n_fact:,} rows, "
          f"{len(DIMS)} dims, {len(MEASURES)} measure views\n")
    if not correctness_gate(con):
        sys.exit(1)


if __name__ == "__main__":
    main()
