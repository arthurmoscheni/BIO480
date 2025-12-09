import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

# Check for scikit-optimize
try:
    from skopt import BayesSearchCV
    SKOPT_INSTALLED = True
except ImportError:
    SKOPT_INSTALLED = False
    BayesSearchCV = None

def run_classifier_analysis(df_metrics, allowed_classes=None, suffix="", use_bayes_search=True):
    """
    Run classification using LabelEncoded targets to prevent MLP/Bayes errors.
    """
    print(f"\n--- Running Classifier ({suffix}) ---")

    if use_bayes_search and not SKOPT_INSTALLED:
        print("! WARNING: 'use_bayes_search' is True, but 'scikit-optimize' is not installed.")
        print("! Falling back to standard parameters.")
        use_bayes_search = False

    features = [
        'Time_to_Peak_ms',
        'Max_Slope',
        'Latency_Jitter_ms',
        'Slope_Adaptation_Ratio',
        'Amp_Adaptation_Ratio',
        'EI_Coupling_Strength'
    ]

    # --- Robust Data Cleaning ---
    df_clf = df_metrics.copy()
    
    # 1. Ensure features exist
    missing_cols = [c for c in features + ['Cell_Type'] if c not in df_clf.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return

    # 2. Force numeric types (coerce errors to NaN)
    for f in features:
        df_clf[f] = pd.to_numeric(df_clf[f], errors='coerce')
    
    # 3. Handle Infinite/NaN
    df_clf = df_clf.replace([np.inf, -np.inf], np.nan)
    df_clf = df_clf.dropna(subset=features + ['Cell_Type'])

    # 4. Filter Classes
    if allowed_classes is not None:
        df_clf = df_clf[df_clf['Cell_Type'].isin(allowed_classes)]

    # 5. Minimum sample check
    counts = df_clf['Cell_Type'].value_counts()
    valid_types = counts[counts >= 5].index.tolist()
    
    df_clf = df_clf[df_clf['Cell_Type'].isin(valid_types)]

    if len(valid_types) < 2:
        print("Not enough data per class for classification.")
        return

    # 6. Prepare X and y (CRITICAL: ENCODE Y TO INTEGERS)
    X = df_clf[features].values.astype(float)
    y_raw = df_clf['Cell_Type'].values
    
    # Use LabelEncoder to convert 'PV', 'SST' -> 0, 1
    # This prevents the "ufunc 'isnan' not supported" error in MLP
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_ # Store names for plotting later

    # CV Setup
    min_class_n = counts[valid_types].min()
    n_splits = max(2, min(5, min_class_n))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

    # Base Models
    base_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, multi_class='multinomial', class_weight='balanced', solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=1
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', class_weight='balanced', probability=False, random_state=1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=1
        ),
        'Neural Network': MLPClassifier(
            max_iter=1000, early_stopping=True, random_state=1
        )
    }

    # Search Spaces
    search_spaces = {
        'Random Forest': {
            'randomforestclassifier__n_estimators': (50, 400),
            'randomforestclassifier__max_depth': (2, 20),
            'randomforestclassifier__min_samples_split': (2, 20)
        },
        'SVM (RBF)': {
            'svc__C': (1e-1, 1e4, 'log-uniform'),
            'svc__gamma': (1e-5, 1e0, 'log-uniform')
        },
        'Gradient Boosting': {
            'gradientboostingclassifier__n_estimators': (50, 400),
            'gradientboostingclassifier__learning_rate': (1e-3, 1.0, 'log-uniform'),
            'gradientboostingclassifier__max_depth': (2, 20)
        },
        'Neural Network': {
            'mlpclassifier__hidden_layer_sizes': (1, 50), # Integer range
            'mlpclassifier__alpha': (1e-5, 1e-1, 'log-uniform'),
            'mlpclassifier__learning_rate_init': (1e-4, 1e-1, 'log-uniform'),
            'mlpclassifier__activation': ['tanh', 'relu']
        }
    }

    model_scores = {}
    best_model = None
    best_score = -np.inf
    best_y_pred = None

    for name, base_model in base_models.items():
        print(f"\n-> Evaluating {name}")
        pipe = make_pipeline(StandardScaler(), base_model)
        clf = pipe

        # Attempt Bayesian Search
        if use_bayes_search and SKOPT_INSTALLED and name in search_spaces:
            print("   Using Bayesian hyperparameter search...")
            try:
                opt = BayesSearchCV(
                    estimator=pipe,
                    search_spaces=search_spaces[name],
                    n_iter=20,
                    cv=cv,
                    scoring='balanced_accuracy',
                    n_jobs=-1,
                    random_state=1,
                    error_score=0 
                )
                opt.fit(X, y)
                clf = opt.best_estimator_
            except Exception as e:
                print(f"   Bayes search failed ({e}). Falling back to default.")
                clf = pipe
                clf.fit(X, y) 

        try:
            # Main Evaluation
            y_pred = cross_val_predict(clf, X, y, cv=cv)
            
            bal_acc = balanced_accuracy_score(y, y_pred)
            f1_macro = f1_score(y, y_pred, average='macro')
            
            model_scores[name] = bal_acc
            print(f"   Balanced Accuracy: {bal_acc:.3f}")
            print(f"   Macro F1:          {f1_macro:.3f}")

            if f1_macro > best_score:
                best_score = f1_macro
                best_model = name
                best_y_pred = y_pred

        except Exception as e:
            print(f"   Evaluation failed: {e}")

    if best_model is None:
        print("No models succeeded.")
        return

    # ---- Visualization ----
    # 1. Bar Plot
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(8, 5))
    bars = plt.bar([x[0] for x in sorted_models], [x[1] for x in sorted_models])
    plt.ylim(0, 1)
    plt.title(f'Model Accuracy {suffix}')
    plt.ylabel('Balanced Accuracy')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f"{bar.get_height():.2f}", ha='center', fontsize=9)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, f"fig9_model_comparison{suffix}.png"), dpi=300)
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y, best_y_pred)
    # We use class_names (the string labels) for the axes
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({best_model}) {suffix}")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, f"fig10_confusion_matrix{suffix}.png"), dpi=300)
    plt.close()

    # 3. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    plt.figure(figsize=(8, 6))
    for i, c_name in enumerate(class_names):
        mask = (y == i) # y is now integers
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=c_name, alpha=0.7, edgecolors='k')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA Feature Space {suffix}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, f"fig11_pca_plot{suffix}.png"), dpi=300)
    plt.close()

    print(f"\nBest Model: {best_model} (F1={best_score:.3f})")