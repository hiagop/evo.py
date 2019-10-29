FROM python:3.7.5-slim

ADD ./evo.py /app/evo.py

ENTRYPOINT /app/entrypoint.sh
