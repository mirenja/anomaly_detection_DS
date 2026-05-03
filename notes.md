In real life, each machine:
* runs tasks (jobs / containers)
* reports resource usage (CPU, memory)
* can degrade over time
* eventually fails or gets removed

what we have is  
* Telemetry (signals) or kpi like CPU spikes or task failures increasing 
* Events machine removed,machine restarted,machine failed

the task is to correlate signals and future failure.given the current state of the machine will it failmsoon?

first we group the raw logs into windows, since the laogs are too granular:
 like 5min windows or 10min:

| machine | 5-min window | avg CPU |
| ------- | ------------ | ------- |
| A       | window_1     | 30%     |
| A       | window_2     | 85%     |

faeture aggregation.encoding system behaviour

CPU mean: How busy was the machine overall?
CPU max: Did it spike suddenly?
CPU std deviation: Was it unstable?
task failure rate: is the workload already breaking things?

WHAT IS A “FAILURE” IN THIS SYSTEM?
REMOVE = machine taken out
FAIL = machine crashed

we are predicting Will this machine fail SOON?

WHY WE DEFINE A “FUTURE WINDOW”
 something like 30 minutes into the future
 we do this since we need time to react and 'remediate?'
 
So we define If failure happens in next 30 minutes:
* label = 1

we are turning time into a prediction problem.

WHAT THE MODEL IS REALLY LEARNING:patterns that look like this tend to fail soon, defining system degradation behaviour
like:
* CPU rising steadily
* memory increasing
* task failures increasing


###### WHY WE USE ANOMALY DETECTION
maybe Isolation Forest
Is this machine currently behaving abnormally compared to normal machines?
prediction answers “will it fail soon?”
anomaly looks into “does it already look weird?”


##### Eral warning system,
the key idea is HOW EARLY did I know something was wrong?
we look for :
* Detection rate : Did we detect failure at all?
* Lead time : How early did we detect it?
* Stability: Did we spam false alerts?

##### general steps:
raw logs
   then
time windows (summarize system state)
    then
features (system health signals)
    v
model (learn failure patterns)
   V
risk score over time
   V
early warning trigger