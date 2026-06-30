# --timeout 300: screener runs can take >30s; avoid gunicorn's default 30s kill.
# --workers 1 --threads 8: screens are I/O-bound (the per-request thread pool
#   does the concurrency), so one worker keeps memory low on small instances
#   while threads still keep the UI responsive during a screen.
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 8
