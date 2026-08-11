FROM python:3.10-slim

# System deps: git is required because DVC piggybacks on Git for metadata tracking
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install DVC
RUN pip install --no-cache-dir dvc

# Copy DVC files
COPY .dvc/ .dvc/
COPY dvc.yaml dvc.lock ./
COPY .dvcignore ./

RUN git init -q

# Configure remote auth from env vars at container start, then run dvc
# with whatever args CMD/docker run passes in ("pull", "push", "status"...)
ENTRYPOINT ["sh", "-c", "dvc remote modify origin --local auth basic && dvc remote modify origin --local user \"$DAGSHUB_USER\" && dvc remote modify origin --local password \"$DAGSHUB_TOKEN\" && exec dvc \"$@\"", "sh"]
CMD ["pull"]