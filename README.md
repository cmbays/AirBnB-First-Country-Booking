AirBnB Data Science Report: Country of First Booking
Introduction:
AirBnB operates as a distributed dwelling services across many countries. Users can
rent out their property for short stays and can purchase rentals from other users. There is a
marginal cost for advertising to any user and an opportunity cost for not advertising. AirBnB
can increase its profit by reducing theses costs by making use of available data. Particularly,
AirBnB needs to predict the country of first booking for new users, and advertise appropriately.
This report will explore models to classify the country of first booking for users. These models
will be evaluated for performance of accuracy and precision to determine the best model for
predicting new users’ first destination booked.

Data:
The user dataset was obtained from the Kaggle.com and came split into a training
dataset and test dataset. The training dataset has 213451 entries and the test dataset has
62096 entries. Table 1 and Table 2 display information about the datasets loaded into a Python
dataframe. The target class has 12 values including NDF(No Destination Found), US(United
States), other(all other countries), FR(France), IT(Italy), GB(Great Britain), ES(Spain),
CA(Canada), DE(Germany), NL(Netherlands), AU(Australia), and PT(Portugal). A composite
dataframe was created by combining the test and train dataframes for exploratory feature
analysis and feature engineering. Missing data is presented in Table 3 as percentage of overall
entries for each feature. Figure 1 shows a histogram and Table 4 describes the age data. The
age data was cleaned as described in methods and the distribution of cleaned data is illustrated
in Figure 2. Figure 3 presents the first destination booked with complete age data compared to
missing age data. There is a significantly greater proportion of NDF bookings and significantly
smaller proportion of US bookings in the missing age data compared to the complete age data.
Figure 4 illustrates a bar graph of users first bookings split by age 65. The younger than 65
group had more first booked destinations in the US than the 65 and older group. Figure 5
displays a bar and whiskers plot for each first booked destination distributed by age. The
distributions are similar between the destinations with 33 as the average age. Great Britain
(GB), France (FR) and no destination found (NDF) had slightly larger box upper bounds (75%
quartile) averaging around 43 years old compared to most countries averaging around 40 years
old. In Figure 6, the relationship between first booked location and primary language is
illustrated. English is the most common language among AirBnB users as 96.37% identified
English as the primary language. English speakers have approximately 10 % less NDF first
bookings than all other speakers combined. English speakers had approximately 7% more US
first bookings compared to all other speakers combined. Figure 7 displays the categorical
gender data in a bar graph. Note nan represents null values. Figure 8 displays the first
destination booked with missing gender data as a bar graph. There is a greater quantity of NDF
bookings among missing gender data users compared to completed gender data users. Figure 9
illustrates the first destination booked by gender. Other gender users have a greater
proportion of NDF and non-US destination bookings compared to male and female users.
Figure 10 displays count and proportion bar graphs for first destination booked. The data
follows the same trend as when segmented by age gender and language. NDF bookings
account for nearly 60% of all bookings. Ignoring NDF bookings, over 70% of users booked the
first destination in the US. Figure 11 illustrates a line graph for date account created. The
amount of accounts created steadily increases from 2010 to 2014 with smaller dips and peaks.
There does not appear to be a seasonal trend as the data peaks and troughs at the same time
of year across different years. Figure 12 displays a bar graph of the weekday accounts are
created. The x-axis represents indexed weekdays beginning at 0=Monday and ending at
6=Sunday. There is a greater number of accounts created on weekdays compared to
weekends.

Methods:
The data was loaded in from csv files into python dataframes. The ID and Date First
Booking features were dropped from the dataframes. ID was dropped as it is irrelevant to the
problem. Date First Booking was dropped because the test dataframe only had null values for
this feature. The gender data was transformed to replace ‘-unknown-‘ as null. The age data
was examined to have values assumed to be year of birth. The data was cleaned such that
values between 1919 and 2000 were transformed to age at 2015. Next, all values over 100 and
under 14 were replaced with null values. A new feature was created by transforming the Date
Account Created feature into Weekday Account Created.
The features were engineered for classification by converting all categorical features
using one hot encoding. All null values were converted to -1. The date account created feature
was sliced into month account created. Note the day and year account created was dropped as
they are irrelevant to the problem. The Weekday Account Created feature was used to replace
the day account created. The test data has users from only 2014 so the year account created
data was assumed to be irrelevant. The Timestamp First Active feature was sliced into month
first active. The day and year sliced data were dropped for the same reason as account created
data. Feature engineering resulted in 158 features to model the data.
After feature engineering, the dataframe was split into the original train and test
dataframes. For this report, the training dataframe was split 80/20-train/test because the test
dataframe does not have the target class data. The target class data from the training
dataframe was converted to integers for the classifiers. The scikitlearn python module was
imported to use the Naïve Bayes Classifier, Multi-Layer Perceptron (MLP) Classifier, and the KNearest
Neighbors (KNN) Classifier. Three different KNN classifiers were used including k=1,
k=5, and k=5 weight=distance. The scikitlearn metrics precision_score(average=’weighted’),
accuracy_score, and confusion matrix were utilized to report performance of each classifier.
Results:
Six models were used to classify the AirBnB user data based on first destination booked.
The accuracy score represents the percentage of users’ first destination booked that were
classified correctly. The precision score represents the weighted average of the precision of
each class, which is weighted by the number of instance per class. The error matrix represents
actual class for each user against the predicted class.
Naïve Bayes Classifier performed had the quickest execution time at a couple of
seconds. Gaussian Naïve Bayes reported the worse performance with an accuracy of 0.623%
and precision of 56.6%. Bernoulli Naïve Bayes reported an accuracy of 56.8% and precision of
51.9%. The performance output is printed in Figure 13 and Figure 14 for Gaussian NB and
Bernoulli NB respectively.
The MLP Neural Network classifier had an execution time of approximately a few
minutes. The NN outperformed the Naïve Bayes with an accuracy of 63.5% and precision of
55%. The performance output is printed in Figure 15.
The KNN classifiers had an execution time of approximately a few minutes for each
model. The KNN models outperformed Naïve Bayes but did not outperform the Neural
Network. The KNN models provided accuracy between 50% to 59% and precision between
50.2% to 51.3%. The performance out is printed in Figure 16, Figure 17, and Figure 18.
Discussion:
The Gaussian Naïve Bayes performed exceptionally low. According to the confusion
matrix, many mis-classifications were made towards Australia and Portugal which were the two
lowest ranked countries. The feature engineering transformed each feature to binary values, so
this would not be the ideal model. The Bernoulli Naïve Bayes performed much better and
serves as an excellent baseline classifier. Bernoulli NB is optimized for binary values, so the
difference in performance is sensible.
The Neural Network out performed Naïve Bayes and all the K-Nearest Neighbor models.
The neural network can weigh the dependencies between features unlike Naïve Bayes. KNN
has a lower level of complexity and is not capable to utilize the relationships between 158
features for classification with the accuracy and precision of a multi-layered neural network.
Conclusion:
 
The Neural Network Classifier outperformed both Naïve Bayes models and all three
models of KNN. The execution time for Naïve Bayes was significantly faster, while the NN and
KNN models had similar execution times. The Neural Network model was the optimal model
explored for this report. This is not surprising as there are many complex features that may
have dependencies. Future work may include integration the session data and age/gender
buckets data for additional features. Furthermore, a layered model that uses several classifiers
and averages the guess between them, or votes on the class may improve accuracy and
precision of the model.
 
Reference:
The following public kernels available on the Kaggle.com competition page for AirBnB inspired
the work in this report.
• David Gasquez. User Data Exploration. Available at:
https://www.kaggle.com/davidgasquez/user-data-exploration
• Sandro. Script_0.8655. Available at: https://www.kaggle.com/svpons/script-0-8655
• kwu2u. Airbnb Exploratory Analysis. Available at:
https://www.kaggle.com/kevinwu06/airbnb-exploratory-analysis