## Correlation Analysis Folder's Structure
- data/
   - `fork_entropy_dataset.csv` - raw data file
- Differential-Based_CoxAnalysis.R implement the Correlation Analysis with Time-shift cross-correlation.
- The lightweight and exemplified difference time-shift correlation analysis conducted in Plot_Difference_threepro_TimeShift_CoxValueVariation.R.
- The lightweight and exemplified linear time-shift correlation analysis conducted in Plot_Linear_threepro_TimeShift_slopeVariation.R.
- Stationarity and AutoCorrelation analysis incorporated in Statistical_MultiTest.R.

## Trans_Prediction Folder's Structure
- dataset/
   - `fork_entropy_dataset.csv` - raw data file
- models/
   - `model` - Main architecture of the model
   - `layers` - Implementation of multi-level structures
   - ...
- training_process.py
   - Overall design for RQ2 training
- transformer_trainer_numBugReportIssues.py
   - Training process for `numBugReportIssues`
- transformer_trainer_numIntegratedCommits.py
   - Training process for predicting `numIntegratedCommit`
- transformer_trainer_ratioMergedPrs.py
   - Training process for predicting `ratioMergedPrs`
- test_model.py
   - Testing design for RQ2.
- LICENSE
   - License file

In the README, we provide the file structure for conducting the experiments in this paper. By adjusting the parameters in the training_process.py file, you can train all three response variables representing received contribution of an OSS project.  The test_model.py will then execute predictions and store the results in the results directory. The model's implementation architecture is found in the model directory, while all the information used for training is located in the datasets directory.
The analysis and prediction Results have put in Results Folder.
