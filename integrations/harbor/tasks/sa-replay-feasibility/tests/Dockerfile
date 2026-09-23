FROM python:3.12-slim

# The verifier image owns /tests. Private evaluator inputs live here and are
# never uploaded to the agent container, which builds from environment/.
COPY . /tests

WORKDIR /app
