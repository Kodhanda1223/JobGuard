"""
JobGuard Project Verification
=============================

Run this from the JobGuard project root:

    python scripts\verify_project.py

This script checks:
1. Dashboard Python syntax
2. Required dataset files
3. Required ML model files
4. Processed dataset readability
5. Target column availability
6. Basic dataset dimensions
"""

from pathlib import Path
import ast
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# REQUIRED FILES
# ============================================================

REQUIRED_FILES = [
    ROOT / "dashboard" / "app.py",

    ROOT / "data" / "processed" / "jobguard_processed.csv",
    ROOT / "data" / "raw" / "fake_job_postings.csv",

    ROOT / "src" / "models" / "jobguard_model.joblib",
    ROOT / "src" / "models" / "jobguard_preprocessor.joblib",
    ROOT / "src" / "models" / "jobguard_metadata.joblib",
]


# ============================================================
# HELPERS
# ============================================================

errors = 0


def relative(path: Path) -> str:
    """Return a clean project-relative path."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_file(path: Path) -> bool:
    """Check whether a required file exists."""
    global errors

    if path.exists() and path.is_file():
        print(f"[OK]      {relative(path)}")
        return True

    print(f"[MISSING] {relative(path)}")
    errors += 1
    return False


def check_python_syntax(path: Path) -> None:
    """Check Python syntax using AST parsing."""
    global errors

    if not path.exists():
        print(f"[SKIP]    {relative(path)} does not exist")
        return

    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)

        print(f"[OK]      Python syntax: {relative(path)}")

    except SyntaxError as exc:
        errors += 1

        print(
            f"[FAIL]    Python syntax: {relative(path)}\n"
            f"          Line: {exc.lineno}\n"
            f"          Column: {exc.offset}\n"
            f"          Error: {exc.msg}"
        )

    except Exception as exc:
        errors += 1
        print(
            f"[FAIL]    Could not read {relative(path)}\n"
            f"          Error: {exc}"
        )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("                 JOBGUARD PROJECT VERIFICATION")
print("=" * 70)
print()

print(f"Project root:")
print(f"  {ROOT}")
print()


# ============================================================
# 1. PROJECT ROOT CHECK
# ============================================================

print("-" * 70)
print("1. PROJECT STRUCTURE")
print("-" * 70)

for required_file in REQUIRED_FILES:
    check_file(required_file)

print()


# ============================================================
# 2. PYTHON SYNTAX
# ============================================================

print("-" * 70)
print("2. PYTHON SYNTAX")
print("-" * 70)

dashboard_app = ROOT / "dashboard" / "app.py"

check_python_syntax(dashboard_app)

# Also check source Python files if they exist.
src_directory = ROOT / "src"

if src_directory.exists():

    for python_file in sorted(src_directory.rglob("*.py")):
        check_python_syntax(python_file)

print()


# ============================================================
# 3. DATASET CHECK
# ============================================================

print("-" * 70)
print("3. DATASET CHECK")
print("-" * 70)

processed_dataset = (
    ROOT
    / "data"
    / "processed"
    / "jobguard_processed.csv"
)

if processed_dataset.exists():

    try:

        import pandas as pd

        df = pd.read_csv(processed_dataset)

        print("[OK]      Processed dataset can be loaded")

        print(f"[INFO]    Rows: {len(df):,}")
        print(f"[INFO]    Columns: {len(df.columns):,}")

        if "fraudulent" in df.columns:

            print("[OK]      Target column: fraudulent")

            fraud_counts = df["fraudulent"].value_counts(dropna=False)

            print("[INFO]    Target distribution:")

            for value, count in fraud_counts.items():

                percentage = (
                    count / len(df) * 100
                    if len(df) > 0
                    else 0
                )

                print(
                    f"          {value}: "
                    f"{count:,} ({percentage:.2f}%)"
                )

        else:

            print("[FAIL]    Target column 'fraudulent' not found")
            errors += 1

        # Important JobGuard feature columns.
        expected_features = [
            "description_length",
            "description_word_count",
            "money_payment",
            "bank_financial",
            "urgent_action",
            "guaranteed_income",
        ]

        print()
        print("[INFO]    Feature-column check:")

        for feature in expected_features:

            if feature in df.columns:
                print(f"          [OK] {feature}")
            else:
                print(f"          [--] {feature} not present")

    except ImportError:

        errors += 1

        print(
            "[FAIL]    pandas is not installed.\n"
            "          Install dependencies with:\n"
            "          pip install -r requirements.txt"
        )

    except Exception as exc:

        errors += 1

        print(
            "[FAIL]    Could not read processed dataset:\n"
            f"          {exc}"
        )

else:

    print("[MISSING] Processed dataset not found")
    errors += 1

print()


# ============================================================
# 4. MODEL ARTIFACT CHECK
# ============================================================

print("-" * 70)
print("4. MODEL ARTIFACT CHECK")
print("-" * 70)

model_files = {
    "Model": ROOT / "src" / "models" / "jobguard_model.joblib",
    "Preprocessor": ROOT / "src" / "models" / "jobguard_preprocessor.joblib",
    "Metadata": ROOT / "src" / "models" / "jobguard_metadata.joblib",
}

for name, path in model_files.items():

    if path.exists():

        size_kb = path.stat().st_size / 1024

        print(
            f"[OK]      {name}: "
            f"{relative(path)} "
            f"({size_kb:.1f} KB)"
        )

    else:

        print(
            f"[MISSING] {name}: "
            f"{relative(path)}"
        )

        errors += 1

print()


# ============================================================
# 5. OPTIONAL METADATA CHECK
# ============================================================

print("-" * 70)
print("5. MODEL METADATA")
print("-" * 70)

metadata_path = (
    ROOT
    / "src"
    / "models"
    / "jobguard_metadata.joblib"
)

if metadata_path.exists():

    try:

        import joblib

        metadata = joblib.load(metadata_path)

        if isinstance(metadata, dict):

            print("[OK]      Metadata loaded")

            for key in [
                "model_name",
                "threshold",
                "pr_auc",
                "roc_auc",
                "feature_count",
                "train_rows",
                "test_rows",
            ]:

                if key in metadata:

                    print(
                        f"[INFO]    "
                        f"{key}: {metadata[key]}"
                    )

                else:

                    print(
                        f"[WARN]    "
                        f"Metadata key missing: {key}"
                    )

        else:

            print(
                "[WARN]    Metadata exists but "
                "is not a dictionary"
            )

    except ImportError:

        print(
            "[WARN]    joblib is not installed; "
            "metadata check skipped"
        )

    except Exception as exc:

        print(
            f"[FAIL]    Metadata could not be loaded: {exc}"
        )

        errors += 1

else:

    print("[SKIP]    Metadata file missing")

print()


# ============================================================
# FINAL RESULT
# ============================================================

print("=" * 70)

if errors == 0:

    print("                    VERIFICATION PASSED")
    print("=" * 70)

    print()
    print("JobGuard project structure looks healthy.")
    print()
    print("Next command:")
    print()
    print("    streamlit run dashboard\\app.py")
    print()

    sys.exit(0)

else:

    print("                    VERIFICATION FAILED")
    print("=" * 70)

    print()
    print(
        f"Found {errors} issue(s). "
        "Fix the reported items before deployment."
    )

    print()

    sys.exit(1)