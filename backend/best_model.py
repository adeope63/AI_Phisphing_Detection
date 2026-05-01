import pandas as pd
import joblib
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

CURRENT_FILE = __file__.replace('\\', '/')
BACKEND_FOLDER = CURRENT_FILE.rsplit('/', 1)[0]
DATASET_FILE = BACKEND_FOLDER + '/CEAS_08.csv'
MODEL_FILE = BACKEND_FOLDER + '/output/best_phishing_model.pkl'


def _load_saved_bundle():
    saved_bundle = joblib.load(MODEL_FILE)
    if 'pipeline' in saved_bundle:
        return saved_bundle['pipeline'], None
    vectorizer = saved_bundle['vectorizer']
    model = saved_bundle['model']
    return vectorizer, model


def _build_sample_frame(sender, subject, body, urls=''):
    return pd.DataFrame([
        {
            'sender': str(sender),
            'subject': str(subject),
            'body': str(body),
            'urls': str(urls)
        }
    ])


def predict_email(sender, subject, body, urls=''):
    """
    Predict one email using the saved best model.
    """
    prediction, _ = predict_email_with_confidence(sender, subject, body, urls)
    return prediction


def predict_email_with_confidence(sender, subject, body, urls=''):
    """
    Predict one email and return (label, confidence).
    Confidence is the model's phishing-class probability when available.
    """
    preprocessor_or_vectorizer, model = _load_saved_bundle()
    sample = _build_sample_frame(sender, subject, body, urls)

    if model is None:
        pipeline = preprocessor_or_vectorizer
        prediction = int(pipeline.predict(sample)[0])
        if hasattr(pipeline, 'predict_proba'):
            probs = pipeline.predict_proba(sample)[0]
            phishing_probability = float(probs[1])
        elif hasattr(pipeline, 'decision_function'):
            score = float(pipeline.decision_function(sample)[0])
            phishing_probability = 1 / (1 + math.exp(-score))
        else:
            phishing_probability = 0.5
    else:
        vectorizer = preprocessor_or_vectorizer
        sample_text = (
            sample['sender'].fillna('').astype(str) + ' ' +
            sample['subject'].fillna('').astype(str) + ' ' +
            sample['body'].fillna('').astype(str) + ' ' +
            sample['urls'].fillna('').astype(str)
        )
        sample_vector = vectorizer.transform(sample_text)
        prediction = int(model.predict(sample_vector)[0])

        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(sample_vector)[0]
            phishing_probability = float(probs[1])
        elif hasattr(model, 'decision_function'):
            score = float(model.decision_function(sample_vector)[0])
            phishing_probability = 1 / (1 + math.exp(-score))
        else:
            phishing_probability = 0.5

    if not (0.0 <= phishing_probability <= 1.0):
        phishing_probability = 0.5

    confidence = phishing_probability if prediction == 1 else (1 - phishing_probability)
    return prediction, float(confidence)


def _predict_batch(model_input, x_frame):
    if model_input[1] is None:
        pipeline = model_input[0]
        return pipeline.predict(x_frame)

    vectorizer = model_input[0]
    model = model_input[1]
    sample_text = (
        x_frame['sender'].fillna('').astype(str) + ' ' +
        x_frame['subject'].fillna('').astype(str) + ' ' +
        x_frame['body'].fillna('').astype(str) + ' ' +
        x_frame['urls'].fillna('').astype(str)
    )
    x_vectors = vectorizer.transform(sample_text)
    return model.predict(x_vectors)


def evaluate_saved_model():
    model_input = _load_saved_bundle()

    data = pd.read_csv(DATASET_FILE)
    data['label'] = data['label'].astype(int)
    data['sender'] = data['sender'].fillna('').astype(str)
    data['subject'] = data['subject'].fillna('').astype(str)
    data['body'] = data['body'].fillna('').astype(str)
    data['urls'] = data['urls'].fillna('').astype(str)

    x = data[['sender', 'subject', 'body', 'urls']]
    y = data['label']

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_predictions = _predict_batch(model_input, x_train)
    test_predictions = _predict_batch(model_input, x_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(y_train, train_predictions, pos_label=1)
    train_recall = recall_score(y_train, train_predictions, pos_label=1)
    train_f1 = f1_score(y_train, train_predictions, pos_label=1)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions, pos_label=1)
    test_recall = recall_score(y_test, test_predictions, pos_label=1)
    test_f1 = f1_score(y_test, test_predictions, pos_label=1)

    print('\nBest Model Training Metrics')
    print(f'Accuracy : {train_accuracy:.4f}')
    print(f'Precision: {train_precision:.4f}')
    print(f'Recall   : {train_recall:.4f}')
    print(f'F1-Score : {train_f1:.4f}')

    print('\nBest Model Testing Metrics')
    print(f'Accuracy : {test_accuracy:.4f}')
    print(f'Precision: {test_precision:.4f}')
    print(f'Recall   : {test_recall:.4f}')
    print(f'F1-Score : {test_f1:.4f}')

    print('\nAccuracy Gap:', round(train_accuracy - test_accuracy, 4))
    print('F1 Gap      :', round(train_f1 - test_f1, 4))

    print('\nTesting Classification Report')
    print(classification_report(y_test, test_predictions))


if __name__ == '__main__':
    evaluate_saved_model()