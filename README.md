# General Information
Anomaly Detection is an independent system that provides Intelligent Controller with information about status of each node. It can detect anomalies based on cpu temperatures, power and number of allocated cpus. In case of detected anomaly system provides information about it to Intelligent Controller so the node is not considered during scheduling new jobs.

System is easily scalable to train on more than 4 metrics as long as the data feeded is collected in the same manner.
# Installation
In case of testing Anomaly Detection system outside of Torino HPC one has to prepare virtual environment with Python >= 3.10. 
```
python -m venv /path/to/dir
source /path/to/dir/bin/activate
pip install -r /path/to/requirements.txt
```


Requirements.txt file will help with installation of proper versions of needed libraries.

# Running
In order to test run Anomaly Detection system we have prepared starting script localized in /scripts/run.sh. The script is ready to launch a test job on Torino HPC cluster thanks to ready virtual environment and paths properly set up. All files are ready in /beegfs/home/Shared/admire/AnomalyDetection folder.

# Architecture
The main file orchestrating work in Anomaly Detection system is RTDataHandler.py.
It is responsible for : 
- reading data from database
- invoking prediction pipeline 
- updating history
- logging to Redis database

The other important part of our system is RTMetricsEvaluator.py which handles calculating reconstruction errors for a window, history, entire time series and respective metrics.

Whole architecture is shown on graphic below.

![alt text](https://gitlab.pcss.pl/deti/admire-applications/AnomalyDetection/-/raw/develop/Images/architecture2.png)

Both architecture and results of our systems are shown in more detail in a power point presentation.

# Example results

Graphic below shows each node reconstruction error with respect to mean and anomaly threshold.
![alt text](https://gitlab.pcss.pl/deti/admire-applications/AnomalyDetection/-/raw/develop/Images/results1.png)

On the second figure we can see all time-series of anomalous nodes (colours) with respect to the others (grey background)

![alt text](https://gitlab.pcss.pl/deti/admire-applications/AnomalyDetection/-/raw/develop/Images/results2.png)

# Summary
In summary, our system not only provides information whether anomaly is present but also allows administrators to inspect more closely what is the ongoing problem with various visualisation methods allowing more in-depth exploration. 