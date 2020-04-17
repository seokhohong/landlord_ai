gcloud beta dataproc clusters create $1 --enable-component-gateway --region us-central1 \
--subnet default --zone us-central1-f --master-machine-type n1-standard-16 --master-boot-disk-size 500 \
--num-workers 2 --worker-machine-type e2-highcpu-16 --worker-boot-disk-size 50 --image-version 1.4-ubuntu18 \
--optional-components ANACONDA,JUPYTER --project cps-ds-dev-vke8 \
--initialization-actions 'gs://goog-dataproc-initialization-actions-us-central1/python/conda-install.sh','gs://hseokho-lai/init/pip3-install.sh' \
--metadata CONDA_PACKAGES="tensorflow keras numpy scipy" --metadata PIP_PACKAGES="landlord-ai tqdm pathlib" --project='dao-aa-poc-uyim' \
--region=us-central1 --properties spark:spark.speculation=True,spark:spark.speculation.multiplier=2,spark:spark.dynamicAllocation.executorIdleTimeout=15s,spark:spark.executor.memory=1g\
