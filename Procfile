# --timeout 300: long screener runs (cold total-market price fetch) exceed
#   gunicorn's default 30s worker timeout and would otherwise 502.
# --workers 2 --threads 4: keep the UI responsive while a long screen runs.
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 2 --threads 4
