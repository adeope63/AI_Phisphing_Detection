FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "-c", "import importlib.util; p='/app/frontend/app.py'; s=importlib.util.spec_from_file_location('frontend_app', p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)"]
