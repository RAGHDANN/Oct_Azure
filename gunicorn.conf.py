import gunicorn

# Gunicorn config file
bind = "0.0.0.0:8000"
workers = 2
threads = 2
timeout = 300  # 5 minutes timeout for model loading
worker_class = "sync"
keepalive = 65
accesslog = "-"  # Log to stdout
