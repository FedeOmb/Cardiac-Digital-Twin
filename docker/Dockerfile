FROM python:3.12-slim

# Dipendenze di sistema per multiprocessing e librerie scientifiche
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ \
    libopenmpi-dev openmpi-bin \
    libhdf5-dev libnetcdf-dev \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Crea utente non-root
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} cdt && \
    useradd -m -u ${USER_ID} -g cdt -s /bin/bash cdt

# Directory di lavoro
#WORKDIR /Cardiac-Digital-Twin

# Copia e installa le dipendenze Python del progetto
COPY --chown=cdt:cdt requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Installa JupyterLab
RUN pip install --no-cache-dir jupyterlab ipywidgets ipykernel

# Registra il kernel Jupyter con il nome corretto
RUN python -m ipykernel install --name cdt-kernel --display-name "CDT kernel Python 3.12"

# Copia e rendi eseguibile l'entrypoint
COPY --chown=cdt:cdt entrypoint.sh /tmp/entrypoint.sh
RUN chmod +x /tmp/entrypoint.sh

# Passa all'utente non-root
USER cdt

ENTRYPOINT ["/tmp/entrypoint.sh"]