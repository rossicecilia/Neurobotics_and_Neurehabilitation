# BCI decoding, using Error Potentials

Neurobotics and Neurehabilitation course Final Project.

My groupmates and I were asked to analyze the EEG data collected during an experiment with 5 healthy subjects controlling the avatar of a wheelchair during a virtual race via joystick. 

Finally, we were asked to investigate the presence of error potentials during the game and classify them. 
In particular, two types of analyses were requested:

  1. Grand average analyses on the whole population and on representative subjects
     
    a. Process the data and apply the convenient filters;
    b. Identifyandextractthemostsuitablefeatures;
    c. Report the achieved results.
    
  2. Analyses on BMI decoding on each subject (use a leave-on-out strategy [run-based])
     
    a. Calibrationphase:
      ▪ In the trainset: process the data, compute the features, select the most disciminant features;
      ▪ Create a classifier based on those features.
    b. Evaluationphase:
      ▪ In the testset: process the data, compute the features, and extract those already selected during the calibration phase;
      ▪ Use this data to evaluate the classifier created during the calibration phase;
    c. Report on the achieved results in terms of (but not limited to): trial accuracy (trainset/testset), ROC curve and AUC (trainset/testset)
