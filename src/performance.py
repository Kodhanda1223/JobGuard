from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# JOBGUARD MODEL PERFORMANCE MODULE
# ============================================================
#
# This module:
#
# 1. Loads the processed JobGuard dataset
# 2. Loads the saved model
# 3. Loads the saved preprocessor
# 4. Recreates the evaluation split
# 5. Handles missing values safely
# 6. Generates probability predictions
# 7. Calculates:
#       Accuracy
#       Precision
#       Recall
#       F1 Score
#       ROC-AUC
#       PR-AUC
# 8. Generates:
#       Confusion Matrix
#       ROC Curve
#       Precision-Recall Curve
#       Feature Importance
#
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobguard_processed.csv"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "jobguard_model.joblib"
)


PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "jobguard_preprocessor.joblib"
)


METADATA_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "jobguard_metadata.joblib"
)


# ============================================================
# LOAD MODEL COMPONENTS
# ============================================================

def load_model_components():

    print("Loading JobGuard model components...")

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    print(
        f"Model loaded: {metadata.get('model_name', type(model).__name__)}"
    )

    print(
        f"Saved threshold: {metadata.get('threshold', 0.5)}"
    )

    print(
        f"Saved ROC-AUC: {metadata.get('roc_auc', 'N/A')}"
    )

    print(
        f"Saved PR-AUC: {metadata.get('pr_auc', 'N/A')}"
    )

    return (
        model,
        preprocessor,
        metadata
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_processed_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATA_PATH}"
        )


    df = pd.read_csv(
        DATA_PATH
    )


    print(
        f"Processed dataset loaded: {df.shape}"
    )


    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================
#
# IMPORTANT:
#
# The saved preprocessor contains text vectorizers.
#
# Text vectorizers cannot accept NaN values.
#
# Therefore we convert:
#
#   object/string columns -> ""
#   numeric columns       -> 0
#
# before passing the dataframe to the saved preprocessor.
#
# ============================================================

def clean_for_preprocessor(df):

    cleaned = df.copy()


    # --------------------------------------------------------
    # Replace infinity values
    # --------------------------------------------------------

    cleaned = cleaned.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    # --------------------------------------------------------
    # Identify text/object columns
    # --------------------------------------------------------

    object_columns = (
        cleaned
        .select_dtypes(
            include=[
                "object",
                "string"
            ]
        )
        .columns
    )


    # --------------------------------------------------------
    # Fill text columns
    # --------------------------------------------------------

    for column in object_columns:

        cleaned[column] = (
            cleaned[column]
            .fillna("")
            .astype(str)
        )


    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = (
        cleaned
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )


    # --------------------------------------------------------
    # Fill numeric columns
    # --------------------------------------------------------

    for column in numeric_columns:

        cleaned[column] = (
            cleaned[column]
            .fillna(0)
        )


    return cleaned


# ============================================================
# PREPARE EVALUATION DATA
# ============================================================

def prepare_evaluation_data(
    df,
    test_size=0.20,
    random_state=42
):

    # --------------------------------------------------------
    # Check target
    # --------------------------------------------------------

    if "fraudulent" not in df.columns:

        raise ValueError(
            "Target column 'fraudulent' was not found."
        )


    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = (
        pd.to_numeric(
            df["fraudulent"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )


    # --------------------------------------------------------
    # Remove target and ID
    # --------------------------------------------------------

    columns_to_drop = [
        "fraudulent",
        "job_id"
    ]


    X = df.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in df.columns
        ],
        errors="ignore"
    )


    # --------------------------------------------------------
    # Clean features
    # --------------------------------------------------------

    X = clean_for_preprocessor(
        X
    )


    # --------------------------------------------------------
    # Stratified train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,

        y,

        test_size=test_size,

        random_state=random_state,

        stratify=y
    )


    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# SAFE PROBABILITY PREDICTION
# ============================================================

def get_probability_predictions(
    model,
    X_transformed
):

    # --------------------------------------------------------
    # Standard classifiers
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model
            .predict_proba(
                X_transformed
            )[:, 1]
        )

        return probabilities


    # --------------------------------------------------------
    # Models with decision_function
    # --------------------------------------------------------

    if hasattr(
        model,
        "decision_function"
    ):

        decision_scores = (
            model
            .decision_function(
                X_transformed
            )
        )


        # ----------------------------------------------------
        # Convert decision scores to sigmoid probabilities
        # ----------------------------------------------------

        probabilities = (
            1.0
            / (
                1.0
                + np.exp(
                    -decision_scores
                )
            )
        )


        return probabilities


    raise ValueError(
        "Saved model does not support "
        "predict_proba() or decision_function()."
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(
    model,
    preprocessor
):

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    if not hasattr(
        model,
        "coef_"
    ):

        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "absolute_importance"
            ]
        )


    coefficients = (
        model.coef_[0]
    )


    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        feature_names = np.array(
            [
                f"feature_{index}"
                for index
                in range(
                    len(coefficients)
                )
            ]
        )


    # --------------------------------------------------------
    # Protect against mismatch
    # --------------------------------------------------------

    count = min(
        len(feature_names),
        len(coefficients)
    )


    feature_names = (
        feature_names[:count]
    )


    coefficients = (
        coefficients[:count]
    )


    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,

            "importance": coefficients,

            "absolute_importance": np.abs(
                coefficients
            )
        }
    )


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    importance_df = (
        importance_df
        .sort_values(
            "absolute_importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    return importance_df


# ============================================================
# TOP FEATURES
# ============================================================

def get_top_features(
    performance,
    n=15
):

    feature_df = (
        performance[
            "feature_importance"
        ]
    )


    if feature_df.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )


    # --------------------------------------------------------
    # Positive coefficients
    #
    # Positive coefficients push the prediction toward
    # fraudulent = 1.
    # --------------------------------------------------------

    positive = (
        feature_df[
            feature_df["importance"] > 0
        ]
        .sort_values(
            "importance",
            ascending=False
        )
        .head(n)
        .copy()
    )


    # --------------------------------------------------------
    # Negative coefficients
    #
    # Negative coefficients push the prediction toward
    # legitimate = 0.
    # --------------------------------------------------------

    negative = (
        feature_df[
            feature_df["importance"] < 0
        ]
        .sort_values(
            "importance",
            ascending=True
        )
        .head(n)
        .copy()
    )


    return (
        positive,
        negative
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def calculate_model_performance(
    test_size=0.20,
    random_state=42,
    threshold=None
):

    print()
    print(
        "Loading model components..."
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        preprocessor,
        metadata
    ) = load_model_components()


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_processed_data()


    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    print(
        "Preparing evaluation dataset..."
    )


    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_evaluation_data(

        df,

        test_size=test_size,

        random_state=random_state
    )


    print(
        f"Training rows: {len(X_train)}"
    )


    print(
        f"Test rows: {len(X_test)}"
    )


    # --------------------------------------------------------
    # Transform test data
    # --------------------------------------------------------

    print(
        "Transforming test data..."
    )


    try:

        X_test_transformed = (
            preprocessor.transform(
                X_test
            )
        )

    except Exception as error:

        print()
        print(
            "ERROR DURING PREPROCESSING"
        )

        print(
            error
        )

        print()

        print(
            "Test data sample:"
        )

        print(
            X_test.head()
        )

        raise


    print(
        "Test data transformed successfully."
    )


    # --------------------------------------------------------
    # Probability predictions
    # --------------------------------------------------------

    print(
        "Generating model predictions..."
    )


    probabilities = (
        get_probability_predictions(
            model,
            X_test_transformed
        )
    )


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    if threshold is None:

        threshold = metadata.get(
            "threshold",
            0.5
        )


    # --------------------------------------------------------
    # Binary predictions
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    pr_auc = average_precision_score(
        y_test,
        probabilities
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )


    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )


    report_text = classification_report(
        y_test,
        predictions,
        zero_division=0
    )


    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

    fpr, tpr, roc_thresholds = (
        roc_curve(
            y_test,
            probabilities
        )
    )


    # --------------------------------------------------------
    # Precision Recall Curve
    # --------------------------------------------------------

    (
        precision_curve,
        recall_curve,
        pr_thresholds
    ) = precision_recall_curve(
        y_test,
        probabilities
    )


    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_distribution = (
        y_test
        .value_counts()
        .sort_index()
        .to_dict()
    )


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print(
        "Extracting feature importance..."
    )


    feature_importance = (
        extract_feature_importance(
            model=model,
            preprocessor=preprocessor
        )
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "model_name": metadata.get(
            "model_name",
            type(model).__name__
        ),

        "threshold": threshold,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc,

        "pr_auc": pr_auc,

        "confusion_matrix": cm,

        "classification_report": report_dict,

        "classification_report_text": report_text,

        "fpr": fpr,

        "tpr": tpr,

        "roc_thresholds": roc_thresholds,

        "precision_curve": precision_curve,

        "recall_curve": recall_curve,

        "pr_thresholds": pr_thresholds,

        "y_test": y_test,

        "probabilities": probabilities,

        "predictions": predictions,

        "class_distribution": class_distribution,

        "feature_importance": feature_importance,

        "dataset_rows": len(df),

        "dataset_columns": len(df.columns),

        "train_rows": len(X_train),

        "test_rows": len(X_test),

        "feature_count": metadata.get(
            "feature_count",
            None
        )
    }


    return results


# ============================================================
# PRINT PERFORMANCE
# ============================================================

def print_performance_report(
    results
):

    print()
    print(
        "=" * 60
    )

    print(
        "JOBGUARD MODEL PERFORMANCE"
    )

    print(
        "=" * 60
    )


    print(
        f"Model             : "
        f"{results['model_name']}"
    )


    print(
        f"Threshold         : "
        f"{results['threshold']:.2f}"
    )


    print(
        f"Accuracy          : "
        f"{results['accuracy']:.4f}"
    )


    print(
        f"Precision         : "
        f"{results['precision']:.4f}"
    )


    print(
        f"Recall            : "
        f"{results['recall']:.4f}"
    )


    print(
        f"F1 Score          : "
        f"{results['f1']:.4f}"
    )


    print(
        f"ROC-AUC           : "
        f"{results['roc_auc']:.4f}"
    )


    print(
        f"PR-AUC            : "
        f"{results['pr_auc']:.4f}"
    )


    print()
    print(
        "Confusion Matrix:"
    )


    print(
        results[
            "confusion_matrix"
        ]
    )


    print()
    print(
        "Classification Report:"
    )


    print(
        results[
            "classification_report_text"
        ]
    )


    print(
        "=" * 60
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        results = (
            calculate_model_performance()
        )


        print_performance_report(
            results
        )


    except Exception as error:

        print()
        print(
            "=" * 60
        )

        print(
            "JOBGUARD PERFORMANCE EVALUATION FAILED"
        )

        print(
            "=" * 60
        )

        print(
            f"Error: {error}"
        )

        print(
            "=" * 60
        )

        raise