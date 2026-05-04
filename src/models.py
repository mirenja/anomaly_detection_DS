import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt


FEATURE_COLS = [
    'avg_memory', 'max_memory', 'std_memory',
    'avg_page_cache', 'max_page_cache',
    'avg_cpi', 'max_cpi', 'std_cpi',
    'avg_mapi', 'max_mapi',
    'fail_count', 'event_count', 'failure_rate'
]


def train_model(window_features):
    """
    Train XGBoost on windowed features to predict pre-warning label.
    Uses time-ordered split (no shuffle) to respect temporal structure.
    Returns model, X_test, y_test for evaluation.
    """
    # drop rows where all resource features are NaN
    df = window_features.dropna(subset=FEATURE_COLS, how='all').copy()

    X = df[FEATURE_COLS]
    y = df['label']

    print(f"Training set shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts()}")
    print(f"Positive rate: {y.mean()*100:.2f}%")

    # time-ordered split — no shuffle to respect temporal order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    # handle class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1
    print(f"\nClass imbalance ratio (scale_pos_weight): {scale:.1f}")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale,  # handles imbalance
        eval_metric='aucpr',
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return all metrics and plots.
    For failure prediction we prioritise RECALL over precision
    — missing a failure is more costly than a false alarm.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Avg Precision:     {avg_precision:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)

    # feature importance plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=True)
    importance.plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Feature Importance (XGBoost)')
    axes[0].set_xlabel('Importance Score')

    # precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recall, precision, color='steelblue', lw=2)
    axes[1].axhline(y=y_test.mean(), color='red', linestyle='--',
                    label=f'Baseline (random) = {y_test.mean():.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve (AP={avg_precision:.3f})')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('../outputs/model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'report': report,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }