export PYSPARK_PYTHON=./lai/bin/python
export PYSPARK_DRIVER_PYTHON=./lai/bin/python
export SPARK_HOME=/usr/lib/spark
export HADOOP_CONF_DIR=/etc/hadoop/conf
#export PYSPARK_DRIVER_PYTHON_OPTS='notebook'
pyspark \
	--master yarn \
	--conf spark.dynamicAllocation.enabled=True \
	--conf spark.shuffle.service.enabled=True \
	--conf spark.dynamicAllocation.minExecutors=1 \
	--conf spark.dynamicAllocation.maxExecutors=100 \
	--conf spark.dynamicAllocation.initialExecutors=1 \
	--archives lai.zip#lai
