# E-commerce Purchase Prediction — Code-focused Documentation

This document explains, in order, what the notebook `ecommerce_purchase_prediction.ipynb` does at the code level. It deliberately omits shell/terminal instructions and focuses on the code, data shapes, inputs/outputs, and production considerations.

1) Environment & imports
- Imports numeric and data libraries (numpy, pandas), plotting libs (matplotlib, seaborn) and sklearn utilities used later (model selection, preprocessing, pipeline, metrics, classifiers).
- Sets plotting style for consistent figures.

2) Synthetic dataset generation
- Sets a reproducible random seed variable (`RANDOM_SEED`).
- Defines sample size `n` (the notebook uses ~30,000 records) and draws per-session values:
  - Identifiers: `user_id` (random int), `session_id` (sequential).
  - Numeric session features: `session_length` (exponential), `pages_viewed` (Poisson), `product_views` (Poisson), `cart_adds` (binomial w.r.t. product_views), `prev_purchases` (Poisson).
  - Categorical features: `device` (mobile/desktop/tablet), `time_of_day` (morning/afternoon/evening/night), `referral` (organic/ad/email/social), `day_of_week` (0-6), and `is_promo` (binary flag).
  - Price & discount: `price` from a lognormal distribution, `discount` from a Beta-derived fraction.
- Computes an internal logistic-style `score` combining these features, applies a logistic transform to create `prob`, injects small noise, and samples a binary `purchase` target with `np.random.binomial`.
- Assembles everything into a pandas DataFrame `df` with the columns: `user_id, session_id, session_length, pages_viewed, product_views, cart_adds, prev_purchases, device, time_of_day, price, discount, is_promo, referral, day_of_week, purchase`.

3) Data cleaning & quality checks
- Lists numeric columns: `session_length, pages_viewed, product_views, cart_adds, prev_purchases, price, discount` and categorical columns: `device, time_of_day, referral`.
- Converts categorical columns to `category` dtype and standardizes text (lowercase, trimmed whitespace).
- Detects and caps numeric outliers using the IQR method: for each numeric column, compute Q1/Q3 and clip values to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. This prevents extreme values from dominating training.
- Drops exact duplicate rows, ensures `discount` within [0,1], and clips `day_of_week` to [0,6].
- Prints or returns summary statistics so downstream cells work on the cleaned `df`.

4) Exploratory Data Analysis (plots)
- Produces a grid of visualizations (the notebook includes >=6 graphs):
  - Purchase distribution (count of 0 vs 1).
  - Session length distribution stratified by `purchase`.
  - Purchase rate by `device` (bar chart of mean purchase per device).
  - Product views vs cart adds (scatter colored by `purchase`).
  - Purchase rate by `time_of_day` (count + rate charts).
  - Purchase rate across price deciles (line plot over price quantiles).
  - Referral source impact (purchase rates per `referral`).
  - Correlation heatmap for numeric features.
- Each figure uses the cleaned `df` and is intended to surface feature relationships and potential transformations.

5) Feature selection & preprocessing
- Defines `X` as `df` without identifiers and target: X = df.drop(['user_id','session_id','purchase'], axis=1).
- Sets `y = df['purchase']`.
- Splits data into `X_train`, `X_test`, `y_train`, `y_test` using stratified sampling on the target to preserve class balance.
- Declares `numeric_features` and `categorical_features` lists used in the ColumnTransformer.
- Builds a `preprocessor` using sklearn's `ColumnTransformer`:
  - Numeric transformer: `StandardScaler()` applied to numeric features.
  - Categorical transformer: `OneHotEncoder(handle_unknown='ignore')` for categorical features.
- This preprocessor is used inside Pipelines so transformations are learned only on training data and applied consistently at inference time.

6) Modeling
- Trains a Logistic Regression baseline inside a Pipeline (preprocessor + LogisticRegression). The logistic model is configured with `class_weight='balanced'` and increased `max_iter` to ensure convergence.
- Builds a RandomForest pipeline (preprocessor + RandomForestClassifier with `class_weight='balanced'`).
- Runs `GridSearchCV` on the RandomForest pipeline with a moderate but thorough parameter grid that includes variations of:
  - `n_estimators` (e.g., 200, 300)
  - `max_depth` (e.g., 8, 12, 16)
  - `min_samples_split`, `min_samples_leaf`
  - `max_features` (`sqrt`, `log2`)
- Uses 5-fold cross-validation and `roc_auc` as the scoring metric during grid search.
- After GridSearch, selects `best_model = gs.best_estimator_` and computes cross-validated ROC AUC on the training set for reliability estimates.

7) Evaluation
- For each model (logistic baseline and best RandomForest), computes on `X_test`:
  - Predicted labels (`predict`) and predicted probabilities (`predict_proba` for the positive class).
  - Metrics: Accuracy, Precision, Recall, F1-score, ROC AUC.
  - Confusion matrix to show class-wise counts.
- Plots ROC curves for both models for a visual comparison of discriminative ability.

8) Interpretability
- Extracts feature names after preprocessing: numeric feature names plus OneHotEncoder generated categorical feature names.
- For RandomForest: reads `feature_importances_` from the fitted classifier and prints/sorts the top features.
- For LogisticRegression: reads `coef_` (the learned weights) and prints the features with largest absolute coefficients to show direction and magnitude.
- The notebook may include a short note showing how to extend interpretability with SHAP if desired (compute SHAP values for the tree model and visualize per-feature impacts).

9) Single-record inference (helper)
- Implements a helper function `predict_purchase(session_data)` that accepts a dict with the same feature keys as training `X` (excluding identifiers) and returns a dictionary with `{'probability': float, 'prediction': int}`.
- The helper wraps the model pipeline and ensures preprocessing is applied the same way as during training.

10) Production considerations (conceptual, code-focused)
- Model contract:
  - Inputs: a single-row record (JSON/dict) matching the training feature schema: numeric_features + categorical_features.
  - Outputs: a probability score (float 0..1) and a binary decision (0/1) at threshold 0.5 by default. The threshold can be tuned based on business costs.
- Serialization: while the current notebook shows an in-memory helper, production serving should persist the trained pipeline (preprocessor + model). The common approach is to serialize the pipeline artifact (for example via joblib or cloud artifact store) so the same preprocessing and model are reloaded for inference.
- Validation & testing: include unit tests that (a) validate schema mapping, (b) verify numeric columns scale similarly to training, and (c) a smoke test that model.predict_proba returns a valid probability.
- Data drift & monitoring: track distributional changes on numeric features (e.g., population statistics) and monitor model performance metrics over time to detect drift.
- Scalability: for batch scoring, apply preprocessor.transform in vectorized batches; for low-latency inference, use an optimized serving stack (lightweight API + warmed model instances).

11) Notes on adapting the notebook for production deployments
- Replace synthetic data creation with your real dataset ingestion step and ensure correct column mapping.
- Use time-based splitting or GroupKFold (group by `user_id`) if sessions are time-dependent to avoid label leakage.
- Consider model calibration (Platt or isotonic) if probability estimates are used directly for decisions.
- Add robust input validation at the API boundary (type, ranges, categorical membership).

12) Recommended next steps (code tasks)
- Add a small `tests/` folder with pytest tests for the `predict_purchase` helper and for basic end-to-end smoke tests using a held-out sample.
- Add a script that loads saved pipeline and runs a deterministic smoke test on a sample of production data.
- Optionally, add a `model/` folder and store serialized artifacts and a `manifest.json` describing model version and training metadata.

Appendix: Data schema (expected column names and types in `X`)
- session_length: int (seconds)
- pages_viewed: int
- product_views: int
- cart_adds: int
- prev_purchases: int
- price: float
- discount: float (0..1)
- is_promo: int/bool
- device: categorical (mobile|desktop|tablet)
- time_of_day: categorical (morning|afternoon|evening|night)
- referral: categorical (organic|ad|email|social)
- day_of_week: int (0..6)

This README focuses entirely on the code logic and the behavior of each notebook section. If you want, I can next (a) export a short `TEST_PLAN.md` listing unit and integration tests to add, or (b) produce a minimal `model_manifest.json` template describing how to version and store trained artifacts.
