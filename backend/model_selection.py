import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from features import combine_text_columns, build_preprocessor


CURRENT_FILE = __file__.replace('\\', '/')
BACKEND_FOLDER = CURRENT_FILE.rsplit('/', 1)[0]
DATASET_FILE = BACKEND_FOLDER + '/CEAS_08.csv'
OUTPUT_FOLDER = BACKEND_FOLDER + '/output'


def _safe_metrics(y_true, predictions):
    return {
        'accuracy': accuracy_score(y_true, predictions),
        'precision': precision_score(y_true, predictions, pos_label=1, zero_division=0),
        'recall': recall_score(y_true, predictions, pos_label=1, zero_division=0),
        'f1': f1_score(y_true, predictions, pos_label=1, zero_division=0)
    }


def _split_temporal(dataframe):
    parsed_dates = pd.to_datetime(dataframe['date'], errors='coerce', utc=True)
    has_dates = parsed_dates.notna().sum()

    # Require at least half the rows to have parseable dates to trust a temporal split.
    if has_dates < len(dataframe) * 0.5:
        return None, None

    temp_df = dataframe.copy()
    temp_df['parsed_date'] = parsed_dates
    temp_df = temp_df.dropna(subset=['parsed_date']).sort_values('parsed_date')

    if len(temp_df) < 50:
        return None, None

    split_index = int(len(temp_df) * 0.8)
    temporal_train = temp_df.iloc[:split_index]
    temporal_test = temp_df.iloc[split_index:]
    return temporal_train, temporal_test


def _leakage_report(train_text, test_text):
    train_set = set(train_text)
    test_set = set(test_text)
    duplicate_text_count = len(train_set.intersection(test_set))
    duplicate_ratio = duplicate_text_count / max(1, len(test_set))
    return duplicate_text_count, duplicate_ratio


# Load dataset
data = pd.read_csv(DATASET_FILE)

# Clean fields
data['label'] = data['label'].astype(int)
data['sender'] = data['sender'].fillna('').astype(str)
data['subject'] = data['subject'].fillna('').astype(str)
data['body'] = data['body'].fillna('').astype(str)
data['urls'] = data['urls'].fillna('').astype(str)
data['date'] = data['date'].fillna('').astype(str)

# Keep only useful columns
x = data[['sender', 'subject', 'body', 'urls']]
y = data['label']

# Split data into test and train
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Combine text
train_text = combine_text_columns(x_train)
test_text = combine_text_columns(x_test)

# Leak check between train/test
duplicate_count, duplicate_ratio = _leakage_report(train_text, test_text)
leakage_df = pd.DataFrame([{
    'split': 'stratified_random_split',
    'duplicate_combined_text_count': duplicate_count,
    'duplicate_combined_text_ratio_in_test': duplicate_ratio
}])
leakage_df.to_csv(OUTPUT_FOLDER + '/leakage_report.csv', index=False)
print('\nLeakage check (exact text overlap)')
print(leakage_df)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        random_state=42,
        n_jobs=1
    ),
    'Linear SVM': LinearSVC(random_state=42)
}

results = []
failed_models = []
best_model = None
best_model_name = ''
best_predictions = None
best_temporal_f1 = -1

temporal_train_df, temporal_test_df = _split_temporal(data)
has_temporal_split = temporal_train_df is not None and temporal_test_df is not None

for model_name, model in models.items():
    try:
        cv_pipeline = Pipeline([
            ('preprocess', build_preprocessor()),
            ('model', clone(model))
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            cv_pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            n_jobs=1
        )

        # Train model on stratified split
        trained_pipeline = Pipeline([
            ('preprocess', build_preprocessor()),
            ('model', clone(model))
        ])
        trained_pipeline.fit(x_train, y_train)

        # Predict on train and test
        train_predictions = trained_pipeline.predict(x_train)
        test_predictions = trained_pipeline.predict(x_test)

        train_metrics = _safe_metrics(y_train, train_predictions)
        test_metrics = _safe_metrics(y_test, test_predictions)

        temporal_accuracy = None
        temporal_precision = None
        temporal_recall = None
        temporal_f1 = None

        if has_temporal_split:
            temporal_y_train = temporal_train_df['label']
            temporal_y_test = temporal_test_df['label']

            temporal_x_train = temporal_train_df[['sender', 'subject', 'body', 'urls']]
            temporal_x_test = temporal_test_df[['sender', 'subject', 'body', 'urls']]

            temporal_pipeline = Pipeline([
                ('preprocess', build_preprocessor()),
                ('model', clone(model))
            ])
            temporal_pipeline.fit(temporal_x_train, temporal_y_train)
            temporal_predictions = temporal_pipeline.predict(temporal_x_test)
            temporal_metrics = _safe_metrics(temporal_y_test, temporal_predictions)

            temporal_accuracy = temporal_metrics['accuracy']
            temporal_precision = temporal_metrics['precision']
            temporal_recall = temporal_metrics['recall']
            temporal_f1 = temporal_metrics['f1']

        print(f'\nResults for {model_name}')
        print('Cross-Validation (5-fold, train split only)')
        print(f'CV Accuracy : {cv_scores["test_accuracy"].mean():.4f}')
        print(f'CV Precision: {cv_scores["test_precision"].mean():.4f}')
        print(f'CV Recall   : {cv_scores["test_recall"].mean():.4f}')
        print(f'CV F1-Score : {cv_scores["test_f1"].mean():.4f}')

        print('\nTraining Metrics (stratified split)')
        print(f'Accuracy : {train_metrics["accuracy"]:.4f}')
        print(f'Precision: {train_metrics["precision"]:.4f}')
        print(f'Recall   : {train_metrics["recall"]:.4f}')
        print(f'F1-Score : {train_metrics["f1"]:.4f}')

        print('\nTesting Metrics (stratified split)')
        print(f'Accuracy : {test_metrics["accuracy"]:.4f}')
        print(f'Precision: {test_metrics["precision"]:.4f}')
        print(f'Recall   : {test_metrics["recall"]:.4f}')
        print(f'F1-Score : {test_metrics["f1"]:.4f}')

        if has_temporal_split:
            print('\nTemporal Holdout Metrics (oldest 80% train -> newest 20% test)')
            print(f'Accuracy : {temporal_accuracy:.4f}')
            print(f'Precision: {temporal_precision:.4f}')
            print(f'Recall   : {temporal_recall:.4f}')
            print(f'F1-Score : {temporal_f1:.4f}')

        print('\nTesting Classification Report')
        print(classification_report(y_test, test_predictions, zero_division=0))

        results.append({
            'model': model_name,
            'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
            'cv_precision_mean': cv_scores['test_precision'].mean(),
            'cv_recall_mean': cv_scores['test_recall'].mean(),
            'cv_f1_mean': cv_scores['test_f1'].mean(),
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1_score': train_metrics['f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1_score': test_metrics['f1'],
            'temporal_accuracy': temporal_accuracy,
            'temporal_precision': temporal_precision,
            'temporal_recall': temporal_recall,
            'temporal_f1_score': temporal_f1,
            'accuracy_gap': train_metrics['accuracy'] - test_metrics['accuracy'],
            'f1_gap': train_metrics['f1'] - test_metrics['f1']
        })

        # Prefer temporal robustness when available, otherwise fallback to stratified test F1.
        selection_score = temporal_f1 if temporal_f1 is not None else test_metrics['f1']
        if selection_score > best_temporal_f1:
            best_temporal_f1 = selection_score
            best_model = clone(model)
            best_model_name = model_name
            best_predictions = test_predictions
    except Exception as error:
        failed_models.append({
            'model': model_name,
            'error': str(error)
        })
        print(f'\nSkipping model {model_name} due to error: {error}')

# Save results
results_df = pd.DataFrame(results)
if results_df.empty:
    raise RuntimeError('All models failed. Check failed_models.csv for details.')
sort_column = 'temporal_f1_score' if has_temporal_split else 'test_f1_score'
results_df = results_df.sort_values(by=sort_column, ascending=False).reset_index(drop=True)

print('\nModel Comparison Table')
print(results_df)

results_df.to_csv(OUTPUT_FOLDER + '/model_results.csv', index=False)

# Smaller file focused on overfitting check
overfitting_df = results_df[
    ['model', 'train_accuracy', 'test_accuracy', 'accuracy_gap', 'train_f1_score', 'test_f1_score', 'f1_gap']
]
overfitting_df.to_csv(OUTPUT_FOLDER + '/overfitting_summary.csv', index=False)
pd.DataFrame(failed_models).to_csv(OUTPUT_FOLDER + '/failed_models.csv', index=False)

# Train selected best model on all data for deployment
full_x = data[['sender', 'subject', 'body', 'urls']]
full_y = data['label']
deployment_pipeline = Pipeline([
    ('preprocess', build_preprocessor()),
    ('model', clone(best_model))
])
deployment_pipeline.fit(full_x, full_y)

# Save the best model and vectorizer together
best_bundle = {
    'model_name': best_model_name,
    'pipeline': deployment_pipeline
}
joblib.dump(best_bundle, OUTPUT_FOLDER + '/best_phishing_model.pkl')

print('\nBest Model Selected:', best_model_name)
print('Best model saved to output folder.')

# Model comparison chart using primary split test scores
x_positions = range(len(results_df))
width = 0.2

plt.figure(figsize=(11, 6))
plt.bar([x - 1.5 * width for x in x_positions], results_df['test_accuracy'], width=width, label='Test Accuracy')
plt.bar([x - 0.5 * width for x in x_positions], results_df['test_precision'], width=width, label='Test Precision')
plt.bar([x + 0.5 * width for x in x_positions], results_df['test_recall'], width=width, label='Test Recall')
plt.bar([x + 1.5 * width for x in x_positions], results_df['test_f1_score'], width=width, label='Test F1-Score')

plt.xticks(list(x_positions), results_df['model'])
plt.xlabel('Models')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.title('Model Comparison Bar Chart')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + '/model_comparison.png')
plt.close()

# Confusion matrix for best model on stratified split
matrix = confusion_matrix(y_test, best_predictions, labels=[0, 1])
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['Legit', 'Phishing'])
display.plot()
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + '/best_model_confusion_matrix.png')
plt.close()

print('\nFiles saved in output folder:')
print('- best_phishing_model.pkl')
print('- model_results.csv')
print('- overfitting_summary.csv')
print('- leakage_report.csv')
print('- failed_models.csv')
print('- model_comparison.png')
print('- best_model_confusion_matrix.png')
print('\nTo check overfitting, compare:')
print('- train_accuracy vs test_accuracy')
print('- train_f1_score vs test_f1_score')
if has_temporal_split:
    print('- temporal_f1_score vs test_f1_score')
print('Smaller gaps usually mean less overfitting.')