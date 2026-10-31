# Same pinned base as environment/Dockerfile; see the provenance note there.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

# The verifier image owns /tests. Private evaluator inputs live here and are
# never uploaded to the agent container, which builds from environment/.
COPY . /tests

WORKDIR /app
