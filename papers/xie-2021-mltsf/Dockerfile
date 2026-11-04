FROM ghcr.io/sign-language-processing/reproduction@sha256:305b6165d306192996358ca312d9a751fa409f43063a76dc7758880a8f905291
ENV HF_HOME=/cache/huggingface HF_HUB_CACHE=/cache/huggingface/hub OMP_NUM_THREADS=8
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git perl && rm -rf /var/lib/apt/lists/*
# Benchmark scorer; immutable revisions, executed only in this container.
RUN git clone https://github.com/usnistgov/SCTK.git /opt/SCTK && cd /opt/SCTK && git checkout 9688a26882a688132a5e414cadcb4c19b6fffaba && make config && make all && make install && test -x bin/sclite
RUN git clone https://github.com/hulianyuyy/CorrNet.git /opt/CorrNet && cd /opt/CorrNet && git checkout a812814f4deefda0b75ab1dcc90b106c62912263
ENV PATH=/opt/SCTK/bin:$PATH
WORKDIR /workspace
COPY scripts /workspace/scripts
