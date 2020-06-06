# Kaggle_titanic_comp
My first forray into making a kaggle submission, using the infamous Titanic challange. Accuracy of submission: 0.74641

Structure:
Data processing must be completed first so that the models will actually work correctly. There is then a logistical regression and random forest model modelling survival on the data available in at 70:30 train to test split. This was to check if one model performed better. Surprisingly both models gave the same results. 

The a logistical regression model was then used for the full data set to create the submission.

Lessons learnt:
One of the following two things should really have been done:
1) look at the test data and process with the training data, rather than having to do secondary processing
or
2) write a function that contains the processing steps to not re-process test data in the same way as training data.

Additionally:
I am not convinced the data was used in the best way, more experiece required
