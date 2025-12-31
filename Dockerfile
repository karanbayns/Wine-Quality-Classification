FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

USER root

# Install GNU Make
RUN apt-get update && \
    apt-get install -y --no-install-recommends make && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch back to the default notebook user
USER ${NB_USER}

COPY conda-lock.yml /tmp/conda-lock.yml

RUN conda env update --quiet --file /tmp/conda-lock.yml --name base \
    && conda clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

RUN pip install deepchecks==0.18.1 click==8.3.1
