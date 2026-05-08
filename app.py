import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from src.preprocessing import load_and_prepare
from src.features import compute_poisson, compute_conditional_poisson
from src.features import build_window_features
from src.models import train_model, evaluate_model

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Cluster Infrastructure Analysis", layout="wide")

st.title("Google Cluster Infrastructure Anomaly & Failure Analysis")
st.markdown("""
> This dashboard explores failure patterns in Google's 2019 Borg cluster traces.  
> We model failure arrivals using a **Poisson distribution**   a standard approach  
> for counting rare, independent events over time (such as machine failures in a data centre).
""")


@st.cache_data
def get_data():
    return load_and_prepare()

df = get_data()

st.sidebar.header("Settings")
interval = st.sidebar.selectbox(
    "Time bin interval (seconds)",
    [3600, 7200],
    index=2,
    help="How wide each time window is. 3600 = 1 hour bins."
)
st.sidebar.markdown("""
**About the time bin:**  
We split the trace into fixed-width windows and count how many failures 
occur per machine per window. This gives us the data to fit a Poisson distribution.
""")

# ==============================
# SECTION 1   POISSON OVERVIEW
# ==============================
st.header("1. Failure Arrival Rate   Poisson Distribution")
st.markdown("""
**Why Poisson?**  
Machine failures in a large cluster are rare, unpredictable, and (largely) independent of each other    
exactly the conditions under which a Poisson process applies. The Poisson distribution tells us:
- **λ (lambda):** the average number of failures per time window
- Whether failures are truly random or tend to cluster in time
""")

poisson_results = compute_poisson(df, interval=interval)
lambda_ = poisson_results['lambda_']
variance = poisson_results['variance']
fail_counts = poisson_results['fail_counts']
lambda_per_machine = poisson_results['lambda_per_machine']

# METRICS 
col1, col2, col3 = st.columns(3)
col1.metric(
    "λ (mean failures/bin)", 
    f"{lambda_:.3f}",
    help="Average number of failures per machine per time window"
)
col2.metric(
    "Variance", 
    f"{variance:.3f}",
    help="Spread of failure counts across bins. For Poisson, this should equal λ."
)
ratio = variance / lambda_ if lambda_ > 0 else 0
col3.metric(
    "Variance/Mean ratio", 
    f"{ratio:.3f}",
    help="Should be ≈ 1 for a good Poisson fit. >1 means overdispersion (bursty failures)."
)

# INSIGHT BANNER
if ratio < 0.5:
    st.info("   **Observation:** Variance/Mean is well below 1   failures are very evenly spread across machines and time. This is a near-perfect Poisson process: failures are rare and independent.")
elif 0.5 <= ratio <= 1.5:
    st.success("  **Good Poisson fit:** Variance/Mean is close to 1   failure arrivals follow a Poisson distribution, meaning they are roughly random and independent across the cluster.")
else:
    st.warning("   **Overdispersion detected:** Variance/Mean > 1   failures tend to cluster together in time. This suggests failures may not be fully independent (e.g. cascading failures or correlated workloads).")

# ---- POISSON PLOT ----
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(0, fail_counts['fail_count'].max() + 1)
poisson_pmf = stats.poisson.pmf(x, lambda_)
ax.hist(fail_counts['fail_count'],
        bins=range(0, fail_counts['fail_count'].max() + 2),
        density=True, alpha=0.6, label='Observed failure counts')
ax.plot(x, poisson_pmf, 'r-o', lw=2, label=f'Poisson fit (λ={lambda_:.2f})')
ax.set_xlabel(f'Failures per {interval}s bin per machine')
ax.set_ylabel('Probability')
ax.set_title('Observed Failure Counts vs Poisson Model')
ax.legend()
st.pyplot(fig)

st.caption(f"""
**Reading this chart:** The blue bars show how often each failure count (0, 1, 2, ...) 
was observed across all machine-bin combinations. The red line is the theoretical Poisson 
distribution with λ={lambda_:.3f}. When the bars closely follow the red line, 
failures behave as a Poisson process   random and memoryless.
""")

# ==============================
# SECTION 2   λ PER MACHINE
# ==============================
st.header("2. Failure Rate (λ) per Machine")
st.markdown("""
Each machine has its own failure rate λ. Machines with higher λ fail more frequently 
within the observation window   these are the **highest-risk nodes** in the cluster.  
Identifying them early is the foundation of a proactive maintenance system.
""")

top_n = st.slider("Show top N machines", 5, 50, 20)

fig2, ax2 = plt.subplots(figsize=(12, 4))
top_machines = lambda_per_machine.head(top_n)
top_machines.plot(kind='bar', ax=ax2, color='steelblue')
ax2.axhline(y=lambda_, color='red', linestyle='--', linewidth=1.5,
            label=f'Cluster average λ={lambda_:.2f}')
ax2.set_xlabel('Machine ID')
ax2.set_ylabel('λ (avg failures per bin)')
ax2.set_title('Most Failure-Prone Machines')
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)

# ---- INSIGHT BANNER ----
top_machine = lambda_per_machine.index[0]
top_lambda = lambda_per_machine.iloc[0]
above_avg = (lambda_per_machine > lambda_).sum()

st.info(f"""
   **Observations:**
- Machine **{top_machine}** has the highest failure rate at λ={top_lambda:.2f} failures/bin   
  {(top_lambda/lambda_):.1f}x the cluster average.
- **{above_avg}** machines have a failure rate above the cluster average (λ={lambda_:.3f}).
- Machines with small integer IDs (like 6, 9, 49) appear at the top   these may be 
  heavily-loaded or older nodes in the cluster.
""")

st.caption("""
**Reading this chart:** Each bar represents one machine's average failure rate (λ) 
across all time bins. The red dashed line is the cluster-wide average. 
Machines whose bars exceed the red line are failure-prone and should be 
prioritised for inspection or workload redistribution.
""")

# ==============================
# SECTION 3   CONDITIONAL POISSON
# ==============================
st.header("3. Conditional Poisson: Does Resource Usage Affect Failure Rate?")
st.markdown("""
Here we ask: **does a machine's resource usage change how often it fails?**  
We split machines into HIGH and LOW resource groups at the median,  
then fit a separate Poisson distribution to each group.  
If λ differs significantly between groups, that resource is a **risk factor** for failure.
""")

resource_options = [
    'assigned_memory',
    'page_cache_memory',
    'cycles_per_instruction',
    'memory_accesses_per_instruction'
]
resource_labels = {
    'assigned_memory': 'Assigned Memory (normalised)',
    'page_cache_memory': 'Page Cache Memory (normalised)',
    'cycles_per_instruction': 'Cycles per Instruction (CPU efficiency)',
    'memory_accesses_per_instruction': 'Memory Accesses per Instruction'
}

selected_resource = st.selectbox(
    "Select resource to condition on:",
    resource_options,
    format_func=lambda x: resource_labels[x]
)

df_clean = df[df[selected_resource].notna()].copy()
threshold = df_clean[selected_resource].median()

st.write(f"Splitting at median value: **{threshold:.4f}**")

cond_results, thresh = compute_conditional_poisson(
    df_clean,
    resource_col=selected_resource,
    interval=interval
)

# ---- SIDE BY SIDE POISSON PLOTS ----
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['steelblue', 'coral']

lambdas = {}
for idx, (group, res) in enumerate(cond_results.items()):
    ax = axes[idx]
    fc = res['fail_counts']['fail_count']
    lam = res['lambda_']
    lambdas[group] = lam

    x = np.arange(0, fc.max() + 1)
    pmf = stats.poisson.pmf(x, lam)

    ax.hist(fc, bins=range(0, fc.max() + 2), density=True,
            alpha=0.6, color=colors[idx], label='Observed')
    ax.plot(x, pmf, 'r-o', lw=2, label=f'Poisson λ={lam:.2f}')
    ax.set_title(f'{group}\nλ={lam:.3f} | var={res["variance"]:.3f} | n={res["n"]} bins')
    ax.set_xlabel('Failures per bin')
    ax.set_ylabel('Probability')
    ax.legend()

plt.suptitle(
    f'Failure Rate conditioned on {resource_labels[selected_resource]}',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
st.pyplot(fig3)

st.caption(f"""
**Reading these charts:** Each panel shows the failure count distribution for one resource group.  
The red line is the Poisson model fitted to that group's data.  
Compare the λ values between panels   a larger λ in the HIGH group means  
higher resource usage is associated with more frequent failures.
""")

# ---- INSIGHT BANNER ----
groups = list(lambdas.keys())
if len(groups) == 2:
    lam_high = lambdas[groups[0]]
    lam_low = lambdas[groups[1]]
    diff = abs(lam_high - lam_low)
    pct = (diff / min(lam_high, lam_low)) * 100 if min(lam_high, lam_low) > 0 else 0

    if diff < 0.05:
        st.success(f"  **No meaningful difference:** Both groups have similar λ values ({lam_high:.3f} vs {lam_low:.3f}). **{resource_labels[selected_resource]}** does not appear to be a significant failure risk factor.")
    elif lam_high > lam_low:
        st.warning(f"   **Risk factor detected:** The HIGH {selected_resource} group has λ={lam_high:.3f} vs λ={lam_low:.3f} for the LOW group   a **{pct:.0f}% higher failure rate**. Machines using more {selected_resource} fail more often.")
    else:
        st.info(f"   **Inverse relationship:** The LOW {selected_resource} group has a higher failure rate (λ={lam_low:.3f} vs λ={lam_high:.3f}). This may indicate that already-failing machines are consuming fewer resources.")

# ---- SUMMARY TABLE ----
st.subheader("Summary Table")
summary_data = {
    'Group': [],
    'λ (mean failures/bin)': [],
    'Variance': [],
    'Variance/Mean': [],
    'Sample bins (n)': []
}
for group, res in cond_results.items():
    lam = res['lambda_']
    var = res['variance']
    summary_data['Group'].append(group)
    summary_data['λ (mean failures/bin)'].append(round(lam, 4))
    summary_data['Variance'].append(round(var, 4))
    summary_data['Variance/Mean'].append(round(var / lam if lam > 0 else 0, 4))
    summary_data['Sample bins (n)'].append(res['n'])

st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

st.caption("""
**Variance/Mean interpretation:** A ratio close to 1 confirms the Poisson assumption holds 
within each resource group. A ratio significantly above 1 suggests bursty, correlated failures 
within that group   which may warrant a negative binomial model instead.
""")

# ==============================
# SECTION 4   FEATURE ENGINEERING
# ==============================
st.header("4. Feature Engineering   Per-Machine Time Windows")
st.markdown("""
Rather than feeding raw row-level events to the model, we aggregate  
into **per-machine per-hour windows**   each row summarises how a machine  
behaved during that window. This gives the model temporal context.
""")

@st.cache_data
def get_window_features(_df, interval):
    return build_window_features(_df, interval=interval)

window_features = get_window_features(df, interval)

col1, col2, col3 = st.columns(3)
col1.metric("Total windows", f"{len(window_features):,}")
col2.metric("Positive windows (label=1)", 
            f"{window_features['label'].sum():,}")
col3.metric("Positive rate", 
            f"{window_features['label'].mean()*100:.2f}%")

st.subheader("Feature Summary Statistics")
feature_cols = [
    'avg_memory', 'max_memory', 'std_memory',
    'avg_page_cache', 'max_page_cache',
    'avg_cpi', 'max_cpi', 'std_cpi',
    'avg_mapi', 'max_mapi',
    'fail_count', 'event_count', 'failure_rate'
]
st.dataframe(
    window_features[feature_cols].describe().round(4),
    use_container_width=True
)

st.caption("""
**Reading this table:** Each column is one engineered feature.  
`avg_` = mean over the window, `max_` = peak value, `std_` = instability.  
High `std_memory` or `max_cpi` in a window may signal a machine under stress.
""")

# insight banner
high_fail_windows = (window_features['fail_count'] > 1).sum()
st.info(f"""
   **Observation:** {high_fail_windows:,} windows contain more than one failure    
these are machines experiencing repeated failures within the same hour,  
which the Poisson analysis flagged as overdispersed behaviour.
""")

# ==============================
# SECTION 5   XGBOOST MODEL
# ==============================
st.header("5. XGBoost Failure Prediction")
st.markdown("""
We train an XGBoost classifier on the windowed features to predict  
whether a machine will fail within the next **30 minutes**.

**Key design decisions:**
- **No shuffle** in train/test split   respects temporal order, avoids data leakage
- **scale_pos_weight**   handles class imbalance by upweighting failure events  
- **Recall prioritised over Precision**   missing a failure is more costly than a false alarm
""")

@st.cache_data
def get_model(_window_features):
    return train_model(_window_features)

with st.spinner("Training XGBoost model..."):
    model, X_test, y_test = get_model(window_features)

st.success("  Model trained successfully")

# ---- METRICS ----
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)
import pandas as pd

report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = roc_auc_score(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC-AUC", f"{roc_auc:.3f}")
col2.metric("Avg Precision (PR-AUC)", f"{avg_precision:.3f}")
col3.metric(
    "Recall (failures caught)",
    f"{report['1']['recall']:.3f}",
    help="Of all actual failures, how many did we catch?"
)
col4.metric(
    "Precision",
    f"{report['1']['precision']:.3f}",
    help="Of all predicted failures, how many were real?"
)

# ---- INSIGHT BANNER ----
recall = report['1']['recall']
precision = report['1']['precision']

if recall >= 0.7:
    st.success(f"""
      **Good recall ({recall:.2f}):** The model catches {recall*100:.0f}% of actual failures.  
    In a production system this means {recall*100:.0f}% of failures would trigger an early warning.
    """)
elif recall >= 0.4:
    st.warning(f"""
       **Moderate recall ({recall:.2f}):** The model catches {recall*100:.0f}% of failures.  
    With more training data or lag features this could improve significantly.
    """)
else:
    st.error(f"""
        **Low recall ({recall:.2f}):** The model is missing most failures.  
    The 30-minute window may be too narrow   consider widening the horizon.
    """)

# ---- PLOTS ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=True)
importance.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Feature Importance')
axes[0].set_xlabel('Importance Score')

# Precision-Recall curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
axes[1].plot(recall_curve, precision_curve, color='steelblue', lw=2)
axes[1].axhline(
    y=y_test.mean(), color='red', linestyle='--',
    label=f'Baseline = {y_test.mean():.3f}'
)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title(f'Precision-Recall Curve\n(AP={avg_precision:.3f})')
axes[1].legend()

# Confusion matrix
im = axes[2].imshow(cm, interpolation='nearest', cmap='Blues')
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(['No Failure', 'Failure'])
axes[2].set_yticklabels(['No Failure', 'Failure'])
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black',
                     fontsize=14, fontweight='bold')

plt.tight_layout()
st.pyplot(fig)

st.caption("""
**Feature Importance:** Which signals the model relies on most to predict failure.  
**Precision-Recall Curve:** The red line is a random classifier baseline   anything  
above it represents genuine predictive power.  
**Confusion Matrix:** Top-left = correctly predicted no failure,  
bottom-right = correctly predicted failure (true positives).
""")

# ---- FULL REPORT ----
with st.expander("Full Classification Report"):
    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True
    )
    st.caption("""
    **support** = number of actual samples in each class.  
    For imbalanced datasets like this, weight your interpretation  
    towards the failure class (label=1) metrics.
    """)