# python:3.12-slim pinned by its multi-platform index digest. Resolved 2026-10-31
# with a read-only registry query, `docker buildx imagetools inspect python:3.12-slim`
# (linux/amd64 manifest: 3.12.16-slim-trixie, created 2026-10-28). No digest for this
# image was recorded anywhere in sharpebench or sharpearena before. Bump both task
# Dockerfiles together; scripts/test_harbor_grader.py fails on an unpinned FROM.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

WORKDIR /app
