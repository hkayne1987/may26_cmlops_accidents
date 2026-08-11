FROM python:3.10-slim

# System deps: git is required because DVC piggybacks on Git for metadata tracking
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install DVC (pinned to keep image builds reproducible)
RUN pip install --no-cache-dir "dvc[http]==3.67.1"

# The repo is bind-mounted at /app at run time, so DVC metadata and pulled
# data live on the host. Nothing is copied in: a COPY here would be hidden
# by the mount anyway.

# DVC needs a Git repo to resolve the project root. The mounted repo already
# has .git, but `git init` keeps this working when only the DVC files are
# mounted. Credentials go to .dvc/config.local, which is gitignored, so the
# token never lands in a tracked file.
ENTRYPOINT ["sh", "-c", "git rev-parse --git-dir >/dev/null 2>&1 || git init -q; dvc remote modify origin --local auth basic && dvc remote modify origin --local user \"$DAGSHUB_USER\" && dvc remote modify origin --local password \"$DAGSHUB_TOKEN\" && exec dvc \"$@\"", "sh"]
CMD ["pull"]