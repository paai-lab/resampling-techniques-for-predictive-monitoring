# resampling-techniques-for-predictive-monitoring

## -The code uses Deep Neural Network, Logistic Regression, Naive Bayes, Random Forest and Support Vector Machine for training BPIC 2013 dataset.
## -The code implements following resampling techniques: ADASYN, ALLKNN, CNN, ENN, IHT, NCR, NM, OSS, RENN, ROS, RUS, SMOTE and TOMEK
## -This code is implemented in a little naive way, so the code is long, but the overall structure is pretty simple as follows: 
Resampling -> One-versus-All Training -> Testing -> Aggregating the results from the testing results -> Classification based on the aggregated results
