#RESEARCH QUESTION:
#Can early detection using factors like cholesterol levels, blood pressure and age help predict 
#heart disease risk for early intervention before the disease progresses to more severe stages?

#APPROACH:
#STEP 1: SUMMARY OF THE DATASET
# - Get the statistical summary of the dataset and check for any inconcsitencies.
# - Clean the dataset. 
# STEP 2: VISUALIZATION
# - Explore the relationship between key factors (age, cholesterol, and blood pressure) and heart disease severity (num).
# - Use summary statistics and visualizations to understand trends and distributions.
# STEP 2: STATISTICAL ANALYSIS
# - Perform ANOVA and logistic regression tests to determine if age, cholesterol, and blood pressure vary significantly across severity levels.
# STEP 4: MODELING FOR EARLY DETECTION
# - Develop a predictive model to assess how well the selected factors can predict heart disease severity.


# Install and load packages
install.packages("ggplot2")
install.packages("readr")
install.packages("dplyr")
install.packages("MASS")
install.packages("rpart.plot")
library(ggplot2)
library(readr)
library(dplyr)
library(MASS)
library(lattice)
library(caret)
library(e1071) 
library(rpart) 
library(rpart.plot)
library(cluster)

# Load dataset
heart <- read.csv('heart_disease_cleaned.csv')

# View the structure of the dataset
View(heart)
str(heart)

# Check for missing values
colSums(is.na(heart))

# Observations:
# The dataset has no missing values. We can proceed with the analysis.

# Total number of rows (people) in the dataset
n_total <- nrow(heart)
n_total

# Observations:
# Total number of records: 919. Each row represents one individual.

# Statistical summary of the dataset
summary(heart)

#Observation:
#There are cholesterol values of 0, which are not realistic and likely represent missing or erroneous data. 
#Therefore, we will remove these zero values to ensure the analysis is based on valid and meaningful data.

#Remove any zero cholesterol values
heart <- heart %>%
  filter(chol > 0) 

# Total number of rows (people) in the updated dataset
n_total <- nrow(heart)
n_total

# Observations:
# Updated total number of records: 748.

# Check the distribution of heart disease severity ('num') - target variable
severity_counts <- table(heart$num)
print(severity_counts)

# Visualize the distribution of heart disease severity levels
barplot(table(heart$num), 
        main = "Distribution of Heart Disease Severity", 
        xlab = "Severity Levels: 0 = No Heart Disease, 1 = Mild, 
        2 = Moderate, 3 = Severe, 4 = Critical", 
        ylab = "Count", 
        col = "skyblue", 
        border = "black")


# Observations:
# - Severity Level 0 (No Disease): 319 individuals.
# - Severity Level 1 (Mild Disease): 203 individuals.
# - Severity Level 2 (Moderate Disease): 63 individuals.
# - Severity Level 3 (Severe Disease): 69 individuals.
# - Severity Level 4 (Critical Disease): 22 individuals.
# - The dataset is skewed right with the majority of individuals having no heart disease (num = 0).
# - Higher severity levels have progressively fewer cases, especially critical disease (num = 4), with only 22 cases.
# - This indicates that severe heart disease is less common in the dataset.

# Ensure 'num' is treated as an ordinal factor for later analysis
heart$num <- factor(heart$num, levels = 0:4, ordered = TRUE)


#FACTOR 1: AGE

# Summary statistics for Age by Severity Level
age_summary <- heart %>%
  group_by(num) %>%
  summarise(
    mean_age = mean(age, na.rm = TRUE),
    median_age = median(age, na.rm = TRUE),
    min_age = min(age, na.rm = TRUE),
    max_age = max(age, na.rm = TRUE),
    q1_age = quantile(age, 0.25, na.rm = TRUE),
    q3_age = quantile(age, 0.75, na.rm = TRUE)
  )

print(age_summary)

# Visualization through Boxplot for Age by Heart Disease Severity
ggplot(heart, aes(x = num, y = age, fill = num)) +
  geom_boxplot() +
  labs(title = "Age Distribution by Heart Disease Severity", 
       x = "Severity Levels (0 = No Disease, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Critical)", 
       y = "Age") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")

# Observations:
# - Severity Level 0 (No Disease): Mean Age = 50.2, Median Age = 51, Age Range = 28-76.
# - Severity Level 1 (Mild Disease): Mean Age = 52.9, Median Age = 54, Age Range = 31-75.
# - Severity Level 2 (Moderate Disease): Mean Age = 59.8, Median Age = 60, Age Range = 42-74.
# - Severity Level 3 (Severe Disease): Mean Age = 59.4, Median Age = 59, Age Range = 39-77.
# - Severity Level 4 (Critical Disease): Mean Age = 61, Median Age = 61.5, Age Range = 38-77.

# - Age tends to increase with the severity of the disease.
# - The mean age for severity levels 2, 3 and 4 is higher than for 0 and 1, 
#suggesting that older individuals are more likely to experience more severe stages of heart disease.


#FACTOR 2: CHOLESTEROL

# Summary statistics for Cholesterol by Severity Level
cholesterol_summary <- heart %>%
  group_by(num) %>%
  summarise(
    mean_chol = mean(chol, na.rm = TRUE),
    median_chol = median(chol, na.rm = TRUE),
    min_chol = min(chol, na.rm = TRUE),
    max_chol = max(chol, na.rm = TRUE),
    q1_chol = quantile(chol, 0.25, na.rm = TRUE),
    q3_chol = quantile(chol, 0.75, na.rm = TRUE)
  )

print(cholesterol_summary)


# Visualization through Boxplot for Cholesterol by Heart Disease Severity
ggplot(heart, aes(x = num, y = chol, fill = num)) +
  geom_boxplot() +
  labs(title = "Cholesterol Distribution by Heart Disease Severity", 
       x = "Severity Levels (0 = No Disease, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Critical)", 
       y = "Cholesterol Level") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")  

# Observations:
# - Severity Level 0 (No Disease): Mean Cholesterol = 240, Median Cholesterol = 230, Cholesterol Range = 85-564.
# - Severity Level 1 (Mild Disease): Mean Cholesterol = 258, Median Cholesterol = 249, Cholesterol Range = 100-603.
# - Severity Level 2 (Moderate Disease): Mean Cholesterol = 253, Median Cholesterol = 246, Cholesterol Range = 153-409.
# - Severity Level 3 (Severe Disease): Mean Cholesterol = 249, Median Cholesterol = 254, Cholesterol Range = 131-369..
# - Severity Level 4 (Critical Disease): Mean Cholesterol = 247, Median Cholesterol = 241, Cholesterol Range = 166-407.

# - Some individuals without heart disease have cholesterol as low as 85.
# - The highest level of cholesterol for a person with heart disease is 603 while the lowest is 100.
# - Overall, the cholesterol levels across different severity levels show no clear upward or downward trend.
# - This suggests that while cholesterol may play a role in heart disease, it is not the sole determinant of disease severity.


#FACTOR 3: BLOOD PRESSURE

# Summary statistics for Blood Pressure by Severity Level
blood_pressure_summary <- heart %>%
  group_by(num) %>%
  summarise(
    mean_bp = mean(trestbps, na.rm = TRUE),
    median_bp = median(trestbps, na.rm = TRUE),
    min_bp = min(trestbps, na.rm = TRUE),
    max_bp = max(trestbps, na.rm = TRUE),
    q1_bp = quantile(trestbps, 0.25, na.rm = TRUE),
    q3_bp = quantile(trestbps, 0.75, na.rm = TRUE)
  )

print(blood_pressure_summary)

# Visualization through Boxplot for Blood Pressure by Heart Disease Severity
ggplot(heart, aes(x = num, y = trestbps, fill = num)) +
  geom_boxplot() +
  labs(title = "Blood Pressure Distribution by Heart Disease Severity", 
       x = "Severity Levels (0 = No Disease, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Critical)", 
       y = "Blood Pressure (mmHg)") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")

# Observations:
# - Severity Level 0 (No Disease): Mean Blood Pressure = 130, Median = 130, Range = 94–190, IQR = 120–140.
# - Severity Level 1 (Mild Disease): Mean = 135, Median = 132, Range = 92–200, IQR = 120–145.
# - Severity Level 2 (Moderate Disease): Mean = 137, Median = 136, Range = 100–180, IQR = 128–146.
# - Severity Level 3 (Severe Disease): Mean = 138, Median = 137, Range = 100–200, IQR = 125–147.
# - Severity Level 4 (Critical Disease): Mean = 143, Median = 145, Range = 104–190, IQR = 131–151.

# - Mean and median blood pressure increase as heart disease severity progresses.
# - Severity level 4 (Critical Disease) has the highest mean(143) and median(145) blood pressure.
# - This suggests that the blood pressure tends to increase with the disease severity.


# Statistical test using ANOVA for age
anova_age <- aov(age ~ num, data = heart)
summary(anova_age)


# Observations:
# - A high F-value of 33.43 indicates that the variance in age between severity levels
#is significantly greater than the variance within severity levels.
# - P-value (< 2e-16) is extremely small, much less than 0.05, 
#indicating that age significantly differs across the levels of heart disease severity.

# Statistical test using ANOVA for choleterol
anova_chol <- aov(chol ~ num, data = heart)
summary(anova_chol)

# Observations:
# - A moderate F-value of 3.652 indicates that the variance in cholesterol 
#between severity levels is statistically noticeable, though not as strong. 
# - P-value (0.00589) is  less than 0.05, indicating that cholesterol levels 
#significantly differ across the levels of heart disease severity.

# Statistical test using ANOVA for blood pressure
anova_trestbps <- aov(trestbps ~ num, data = heart)
summary(anova_trestbps)

# Observations:
# - A moderate F-value of 7.427 indicates that that resting blood pressure varies
#between different severity levels of heart disease. 
# - P-value (7.23e-06) is much smaller than 0.05, suggesting that the differences 
#in resting blood pressure across heart disease severity levels are statistically significant.


# Statistical test using ordinal logistic regression for all 3 factors  
model <- polr(num ~ age + chol + trestbps, data = heart, Hess = TRUE)
summary(model)


#Observations
# 1. Age:  
# - The coefficient for age is 0.0774 with a high t-value of 8.966, indicating that age is a highly significant predictor of heart disease severity.  
# - The odds ratio for age is calculated as  e^{0.0774} which equals approx 1.0804 , meaning for every additional year of age, the odds of having more severe heart disease increase by approximately 8.04%.

# 2. Cholesterol:  
# - The coefficient for cholesterol is 0.0026 with a t-value of 2.130, suggesting that cholesterol is a significant predictor of heart disease severity.  
# - The odds ratio for cholesterol is e^{0.0026} equals approx 1.0026, indicating that for every unit increase in cholesterol, the odds of having more severe heart disease increase by 0.26%.

# 3. Resting Blood Pressure:  
# - The coefficient for resting blood pressure is 0.0112 with a t-value of 2.640, indicating a statistically significant relationship.  
# - The odds ratio for resting blood pressure is e^{0.0112} equals approx 1.0113, meaning that for every unit increase in resting blood pressure, the odds of having more severe heart disease increase by 1.13%.

# - Intercepts: The intercepts represent the cut-off points between different severity levels of heart disease, and all have high t-values and low p-values, suggesting the model effectively distinguishes between severity levels.
# - Model Fit: The Residual Deviance is 1711.64, and the AIC is 1725.64, indicating a good fit for the model, but these values should be compared with other models for further assessment.

#Overall, the odds ratios for these predictors indicated that age has the largest effect, followed by cholesterol and resting blood pressure, with each additional unit of these factors increasing the odds of having more severe heart disease.
#Therefore, age has the largest effect among predictors, with cholesterol and resting blood pressure contributing smaller but significant effects.


#Supervised Machine Learning Models: SVM

#Data Preparation
set.seed(123) 
data <- heart
#Create a column for whether the person has diease or not
data$has_disease <- as.factor(heart$has_disease)
data$has_disease <- ifelse(data$num > 0, 1, 0)

data$has_disease <- as.factor(data$has_disease) 

# Split data into training (80%) and testing (20%)
train_index <- createDataPartition(data$has_disease, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Step 2: SVM Model: Linear Kernel
svm_model <- svm(has_disease ~ age + chol + trestbps + cp, data = train_data, kernel = "linear")

# Step 3: Model Evaluation
svm_pred <- predict(svm_model, test_data)
confusionMatrix(svm_pred, test_data$has_disease)

# Observations:
# 1. The model achieved an accuracy of 77.85%, with a balanced accuracy of 77.90%.
# 2. Sensitivity: 76.92% - Correctly identified "no disease" cases.
# 3. Specificity: 78.87% - Correctly identified "has disease" cases.
# 4. Evaluation:
#    - Confusion matrix and performance metrics show good balance between precision and recall.
#    - McNemar's test (P = 0.7277) indicates balanced error rates.
# 5. Performance:
#    - The model significantly outperforms random guessing (P < 0.001).
#    - Some false positives (18) and false negatives (15) remain, suggesting room for improvement.


#Supervised Machine Learning Models: Decision Tree

# Decision Tree Model
tree_model <- rpart(has_disease ~ age + chol + trestbps , data = train_data, method = "class")

# Visualize the Decision Tree
rpart.plot(tree_model)

# Step 3: Model Evaluation
tree_pred <- predict(tree_model, test_data, type = "class")
confusionMatrix(tree_pred, test_data$has_disease)

# Observations for Decision Tree Model:
# 1. Accuracy: 68.46% - The model correctly predicted 68.46% of cases.
# 2. Sensitivity: 73.08% - Correctly identified "no disease" cases.
# 3. Specificity: 63.38% - Correctly identified "has disease" cases.
# 4. Balanced Accuracy: 68.23% - Indicates moderate performance across both classes.
# 5. Evaluation:
#    - The model significantly outperforms random guessing (P < 0.001).
#    - McNemar's test (P = 0.5596) shows balanced error rates.
# 6. Performance:
#    - Positive Predictive Value (68.67%) and Negative Predictive Value (68.18%) are moderate.
#    - Further tuning and additional features may improve performance.

# Unsupervised Machine Learning Models: Clustering

#Perform K-means clustering based on 'age', 'chol', 'trestbps', and 'num' (disease severity)
set.seed(123)  

# K-means clustering on selected features (age, cholesterol, blood pressure, and disease severity)
kmeans_model_severity <- kmeans(data[, c("age", "chol", "trestbps", "num")], centers = 3)

# Step 2: Check the distribution of disease severity in each cluster
table(kmeans_model_severity$cluster, data$num)

# Step 3: Examine the mean values of features for each cluster
aggregate(data[, c("age", "chol", "trestbps")], by = list(kmeans_model_severity$cluster), FUN = mean)

# Step 4: Calculate Silhouette Score to evaluate the quality of clustering
silhouette_score_severity <- silhouette(kmeans_model_severity$cluster, dist(data[, c("age", "chol", "trestbps", "num")]))
summary(silhouette_score_severity)

# Optional: Visualize the clustering result (2D plot using PCA for visualization)
pca <- prcomp(data[, c("age", "chol", "trestbps", "num")], center = TRUE, scale. = TRUE)
pca_data <- as.data.frame(pca$x)
pca_data$cluster <- as.factor(kmeans_model_severity$cluster)

ggplot(pca_data, aes(PC1, PC2, color = cluster)) +
  geom_point() +
  labs(title = "K-means Clustering of Disease Severity", x = "PC1", y = "PC2")

# Observations:
# 1. Cluster Distribution
# Cluster 1 contains mostly individuals with mild to no disease severity (0, 1).
# Cluster 2 is a mix of mild, moderate, and some severe cases (1, 2, 3).
# Cluster 3 primarily includes individuals with severe to critical heart disease (3, 4).

# 2. Cluster Characteristics
# Cluster 1: Mean age = 52.2 years, Chol = 204.27, BPS = 131.85
# Cluster 2: Mean age = 53.5 years, Chol = 273.32, BPS = 133.56
# Cluster 3: Mean age = 53.8 years, Chol = 378.24, BPS = 139.23

# 3. Silhouette Scores
# Average silhouette score for Cluster 1: 0.445 (moderate separation)
# Average silhouette score for Cluster 2: 0.420 (moderate clustering)
# Average silhouette score for Cluster 3: 0.280 (weaker separation, possible outliers)

# 4. PCA Visualization
# Clusters are visually separated, with Cluster 3 (severe cases) more distinct.
# Clusters 1 and 2 show some overlap, indicating similarity in features (age, cholesterol, blood pressure).



# Conclusion: 
#Overall, the analysis suggests that age is the most significant predictor, followed by cholesterol and resting blood pressure. 
#Early detection using factors like age, cholesterol levels, and resting blood pressure significantly predicts the progression of heart disease. 
#Moreover, the use of machine learning model demonstrates the potential of data-driven approaches in personalized healthcare.
#Finally, by monitoring these factors, healthcare professionals can identify individuals at higher risk and implement timely interventions to prevent the disease from advancing to more severe stages.



