import sys
import logging
from flask import Flask, jsonify, render_template, request


CURRENT_FILE = __file__.replace('\\', '/')
FRONTEND_FOLDER = CURRENT_FILE.rsplit('/', 1)[0]
PROJECT_FOLDER = CURRENT_FILE.rsplit('/', 2)[0]
BACKEND_FOLDER = PROJECT_FOLDER + '/backend'

if BACKEND_FOLDER not in sys.path:
    sys.path.insert(0, BACKEND_FOLDER)

from best_model import predict_email_with_confidence


def _setup_logging(app_instance):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    app_instance.logger.setLevel(logging.INFO)


def _validate_email_input(payload):
    errors = []

    sender = str(payload.get('sender', '')).strip()
    subject = str(payload.get('subject', '')).strip()
    body = str(payload.get('body', '')).strip()
    urls = str(payload.get('urls', '')).strip()

    if not sender:
        errors.append('sender is required')
    if not subject:
        errors.append('subject is required')
    if not body:
        errors.append('body is required')

    return {
        'sender': sender,
        'subject': subject,
        'body': body,
        'urls': urls,
        'errors': errors
    }


def create_app():
    app = Flask(
        __name__,
        template_folder=FRONTEND_FOLDER,
        static_folder=FRONTEND_FOLDER,
        static_url_path=''
    )
    _setup_logging(app)

    @app.route('/', methods=['GET', 'POST'])
    def index():
        result = None
        confidence = None
        sender = ''
        subject = ''
        body = ''

        if request.method == 'POST':
            form_payload = {
                'sender': request.form.get('sender', ''),
                'subject': request.form.get('subject', ''),
                'body': request.form.get('body', ''),
                'urls': request.form.get('urls', '')
            }
            parsed = _validate_email_input(form_payload)
            sender = parsed['sender']
            subject = parsed['subject']
            body = parsed['body']

            if not parsed['errors']:
                prediction, confidence = predict_email_with_confidence(
                    sender,
                    subject,
                    body,
                    parsed['urls']
                )
                result = 'Phishing' if prediction == 1 else 'Legit'
                app.logger.info(
                    'web_prediction sender=%s label=%s confidence=%.4f',
                    sender,
                    result,
                    confidence
                )

        return render_template(
            'index.html',
            result=result,
            confidence=confidence,
            sender=sender,
            subject=subject,
            body=body
        )

    @app.route('/predict', methods=['POST'])
    def predict():
        payload = request.get_json(silent=True) or {}
        parsed = _validate_email_input(payload)

        if parsed['errors']:
            return jsonify({'errors': parsed['errors']}), 400

        prediction, confidence = predict_email_with_confidence(
            parsed['sender'],
            parsed['subject'],
            parsed['body'],
            parsed['urls']
        )
        label = 'Phishing' if prediction == 1 else 'Legit'

        app.logger.info(
            'api_prediction sender=%s label=%s confidence=%.4f',
            parsed['sender'],
            label,
            confidence
        )

        return jsonify({
            'label': label,
            'prediction': int(prediction),
            'confidence': float(confidence)
        })

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'}), 200

    return app


app = create_app()


if __name__ == '__main__':
    app.run(debug=True, port=5001)