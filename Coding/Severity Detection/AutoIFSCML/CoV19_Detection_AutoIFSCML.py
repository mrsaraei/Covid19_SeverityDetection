# -*- coding: utf-8 -*-
print("------------------------------------------------------")
print("---------------- Metadata Information ----------------")
print("------------------------------------------------------")
print("")

print("In the name of God")
print("Project: AutoIFSCML: Automatic Comparative Machine Learning based on Important Features in COVID-19 Severity Detection Model")
print("Creator: Mohammad Reza Saraei")
print("Contact: m.r.saraei@seraj.ac.ir")
print("University: Seraj Institute of Higher Education")
print("Supervisor: Dr. Saman Rajebi")
print("Created Date: May 20, 2022")
print("") 

print("----------------------------------------------------")
print("------------------ Import Libraries ----------------")
print("----------------------------------------------------")
print("")

# Import Libraries for Python
import pandas as pd
import numpy as np
import random
import os
from pandas import set_option
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, f1_score, recall_score
import warnings
warnings.filterwarnings("ignore")

print("----------------------------------------------------")
print("------------------ Data Ingestion ------------------")
print("----------------------------------------------------")
print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Annotated Data\CoV19_Data_Annotated.csv")

# [0 = No Sign/Symptom]
# [1 = Mild Level]
# [2 = Moderate Level]
# [3 = Severe Level]
# [4 = Critical Level]

# print("----------------------------------------------------")
# print("----------------- Set Option -----------------------")
# print("----------------------------------------------------")
# print("")

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 1000)


# print("------------------------------------------------------")
# print("-------------- Tune-up Seed for ML Models ------------")
# print("------------------------------------------------------")
# print("")

# Set a Random State value
RANDOM_STATE = 42

# Set Python random a fixed value
random.seed(RANDOM_STATE)

# Set environment a fixed value
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

# Set numpy random a fixed value
np.random.seed(RANDOM_STATE)

print("------------------------------------------------------")
print("------------ Initial Data Understanding --------------")
print("------------------------------------------------------")
print("")

print("Initial General Information:")
print("****************************")
print(df.info())
print("")

print("------------------------------------------------------")
print("------------------ Data Spiliting --------------------")
print("------------------------------------------------------")
print("")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values     

print("------------------------------------------------------")
print("---------------- Data Normalization ------------------")
print("------------------------------------------------------")
print("")

# Normalization [0, 1] of Data
scaler = MinMaxScaler(feature_range = (0, 1))
f = scaler.fit_transform(f)
print(f)
print("")

print("------------------------------------------------------")
print("---------------- Data Label Encoding -----------------")
print("------------------------------------------------------")
print("")

# Label_Encoder Object Knows How to Understand Word Labels
LE = preprocessing.LabelEncoder()

# Encode Labels in Target Column
t = LE.fit_transform(t)
print(LE.classes_)
print(list(t))
print("")
print("")

print("------------------------------------------------------")
print("----------- Save Features and Target Data ------------")
print("------------------------------------------------------")
print("")

# Save DataFrame (f, t) After Munging
pd.DataFrame(f).to_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\f.csv", index = False)
pd.DataFrame(t).to_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\t.csv", index = False)

print("------------------------------------------------------")
print("-------- Features and Target Data Combination --------")
print("------------------------------------------------------")
print("")

# Import Again DataFrames (f, t) by Pandas Library
df_f = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\f.csv")
df_t = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\t.csv")

# Rename t Column
df_t.rename(columns = {'0': 'Diagnosis'}, inplace = True)

# Combination of DataFrames
df = pd.concat([df_f, df_t], axis = 1)

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\df.csv", index = False)

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\df.csv")

print("------------------------------------------------------")
print("---------------- Data Preprocessing ------------------")
print("------------------------------------------------------")
print("")

# Replace Question Mark to NaN:
df.replace("?", np.nan, inplace = True)

# Remove Duplicate Samples
df = df.drop_duplicates()
print("Duplicate Records After Removal:", df.duplicated().sum())
print("")

# Replace Mean instead of Missing Values
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp.fit(df)
df = imp.transform(df)
print("Mean Value For NaN Value:", "{:.3f}".format(df.mean()))
print("")

# Reordering Records / Samples / Rows
print("Reordering Records:")
print("*******************")
df = pd.DataFrame(df).reset_index(drop = True)
print(df)
print("")

print("------------------------------------------------------")
print("------------------ Data Respiliting ------------------")
print("------------------------------------------------------")
print("")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1].values
t = df.iloc[:, -1].values     

print("------------------------------------------------------")
print("----------------- Outliers Detection -----------------")
print("------------------------------------------------------")
print("")

# Identify Outliers in the Training Data
ISF = IsolationForest(n_estimators = 100, contamination = 0.1, bootstrap = True, n_jobs = -1)

# Fitting Outliers Algorithms on the Training Data
ISF = ISF.fit_predict(f, t)

# Select All Samples that are not Outliers
Mask = ISF != -1
f, t = f[Mask, :], t[Mask]

print('nFeature:', f.shape)
print('nTarget:', t.shape)
print("")

print("------------------------------------------------------")
print("------------- Data Balancing By SMOTE ----------------")
print("------------------------------------------------------")
print("")

# Summarize Targets Distribution
print('Targets Distribution Before SMOTE:', sorted(Counter(t).items()))

# OverSampling (OS) Fit and Transform the DataFrame
OS = SMOTE()
f, t = OS.fit_resample(f, t)

# Summarize the New Targets Distribution
print('Targets Distribution After SMOTE:', sorted(Counter(t).items()))
print("")

print('nFeature:', f.shape)
print('nTarget:', t.shape)
print("")

print("------------------------------------------------------")
print("----------- Save Features and Target Data ------------")
print("------------------------------------------------------")
print("")

# Save DataFrame (f, t) After Munging
pd.DataFrame(f).to_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\f.csv", index = False)
pd.DataFrame(t).to_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\t.csv", index = False)

print("------------------------------------------------------")
print("-------- Features and Target Data Combination --------")
print("------------------------------------------------------")
print("")

# Import Again DataFrames (f, t) by Pandas Library
df_f = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\f.csv")
df_t = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\t.csv")

# Rename t Column
df_t.rename(columns = {'0': 'Diagnosis'}, inplace = True)

# Combination of DataFrames
df = pd.concat([df_f, df_t], axis = 1)

# Save Combination f and t DataFrames After Munging
pd.DataFrame(df).to_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\df.csv", index = False)

print("------------------------------------------------------")
print("----------------- Data Understanding -----------------")
print("------------------------------------------------------")
print("")

print("Dataset Overview:")
print("*****************")
print(df.head(10))
print("")

print("General Information:")
print("********************")
print(df.info())
print("")

print("Statistics Information:")
print("***********************")
print(df.describe(include="all"))
print("")

print("nSample & (nFeature + Target):", df.shape)
print("")

print("Samples Range:", df.index)
print("")

print(df.columns)
print("")

print("Missing Values (NaN):")
print("*********************")
print(df.isnull().sum())                                         
print("")

print("Duplicate Records:", df.duplicated().sum())
print("")   

print("Features Correlations:")
print("**********************")
print(df.corr(method='pearson'))
print("")

print("------------------------------------------------------")
print("--------------- Data Distribution --------------------")
print("------------------------------------------------------")
print("")

print("nSample & (nFeature + Target):", df.shape)
print("")

print("Skewed Distribution of Features:")
print("********************************")
print(df.skew())
print("")
print(df.dtypes)
print("")

print("Target Distribution:")
print("********************")
print(df.groupby(df.iloc[:, -1].values).size())
print("")

print("------------------------------------------------------")
print("----------- Plotting Distribution of Data ------------")
print("------------------------------------------------------")
print("")

# Plot the Scores by Descending
plt.hist(df)
plt.xlabel('Data Value', fontsize = 11)
plt.ylabel('Data Frequency', fontsize = 11)
plt.title('Data Distribution After Preparation')
plt.show()

# print("----------------------------------------------------")
# print("------------------ Data Ingestion ------------------")
# print("----------------------------------------------------")
# print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\df.csv")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, 0: -1]
t = df.iloc[:, -1]

# Computing the Number of Features in Dataset
nFeature = len(f.columns)
print('The Number of Features:', nFeature)
print("")

print("----------------------------------------------------")
print("------------ Select K Best (ANOVA F) ---------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
BF = SelectKBest(score_func = f_classif, k = 'all')    
fit_ANOVAF = BF.fit(f, t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_ANOVAF = pd.DataFrame(fit_ANOVAF.scores_)

# Concatenate DataFrames
feature_ANOVAF_scores = pd.concat([df_columns, df_ANOVAF], axis = 1)
feature_ANOVAF_scores.columns = ['Features', 'Score']

print(feature_ANOVAF_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("-------------- Select K Best (Chi2) ----------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
BF = SelectKBest(score_func = chi2, k = 'all')
fit_Chi2 = BF.fit(f, t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_Chi2 = pd.DataFrame(fit_Chi2.scores_)

# Concatenate DataFrames
feature_Chi2_scores = pd.concat([df_columns, df_Chi2], axis = 1)
feature_Chi2_scores.columns = ['Feature', 'Score']  

print(feature_Chi2_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------- Select K Best (Mutual Info Classif) --------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
BF = SelectKBest(score_func = mutual_info_classif, k = 'all')    
fit_MICIF = BF.fit(f, t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_MICIF = pd.DataFrame(fit_MICIF.scores_)

# Concatenate DataFrames
feature_MICIF_scores = pd.concat([df_columns, df_MICIF], axis = 1)
feature_MICIF_scores.columns = ['Features', 'Score']

print(feature_MICIF_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------------ PI KNeighbors Classifier --------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
KNN = KNeighborsClassifier()
KNN.fit(f,t)

# Perform Permutation Importance
results = permutation_importance(KNN, f, t, scoring = 'accuracy')

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_PIKNN = pd.DataFrame(results.importances_mean)

# Concatenate DataFrames
feature_PIKNN_scores = pd.concat([df_columns, df_PIKNN], axis = 1)
feature_PIKNN_scores.columns = ['Features', 'Score']

print(feature_PIKNN_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("---------------------- LASSO -----------------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
reg = LassoCV()
reg.fit(f, t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_LASSO = pd.DataFrame(reg.coef_)

# Concatenate DataFrames
feature_LASSO_scores = pd.concat([df_columns, df_LASSO], axis = 1)
feature_LASSO_scores.columns = ['Features', 'Score']

print(feature_LASSO_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------------- Decision Trees Classifier ------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
DTC = DecisionTreeClassifier()
DTC.fit(f,t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_DTC = pd.DataFrame(DTC.feature_importances_)

# Concatenate DataFrames
feature_DTC_scores = pd.concat([df_columns, df_DTC], axis = 1)
feature_DTC_scores.columns = ['Features', 'Score']

print(feature_DTC_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------------- Extra Trees Classifier ---------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
ETC = ExtraTreesClassifier()
ETC.fit(f,t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_ETC = pd.DataFrame(ETC.feature_importances_)

# Concatenate DataFrames
feature_ETC_scores = pd.concat([df_columns, df_ETC], axis = 1)
feature_ETC_scores.columns = ['Features', 'Score']

print(feature_ETC_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------------ Random Forest Classifier --------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
RFC = RandomForestClassifier()
RFC.fit(f,t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_RFC = pd.DataFrame(RFC.feature_importances_)

# Concatenate DataFrames
feature_RFC_scores = pd.concat([df_columns, df_RFC], axis = 1)
feature_RFC_scores.columns = ['Features', 'Score']

print(feature_RFC_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("--------------- XGBoost Classifier -----------------")
print("----------------------------------------------------")
print("")

# Determine the most Important Features
XGB = XGBClassifier()
XGB.fit(f,t)

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_XGB = pd.DataFrame(XGB.feature_importances_)

# Concatenate DataFrames
feature_XGB_scores = pd.concat([df_columns, df_XGB], axis = 1)
feature_XGB_scores.columns = ['Features', 'Score']

print(feature_XGB_scores.nlargest(nFeature, 'Score'))
print("")

print("----------------------------------------------------")
print("------------------- Make DataFrame -----------------")
print("----------------------------------------------------")
print("")

# Create DataFrames
df_columns = pd.DataFrame(f.columns)
df_ANOVAF = pd.DataFrame(fit_ANOVAF.scores_)
df_Chi2 = pd.DataFrame(fit_Chi2.scores_)
df_MICIF = pd.DataFrame(fit_MICIF.scores_)
df_PIKNN = pd.DataFrame(results.importances_mean)
df_LASSO = pd.DataFrame(reg.coef_)
df_DTC = pd.DataFrame(DTC.feature_importances_)
df_ETC = pd.DataFrame(ETC.feature_importances_)
df_RFC = pd.DataFrame(RFC.feature_importances_)
df_XGB = pd.DataFrame(XGB.feature_importances_)

# Concatenate DataFrames
feature_scores = pd.concat([df_columns, df_ANOVAF, df_Chi2, df_MICIF, df_PIKNN, df_LASSO, df_DTC, df_ETC, df_RFC, df_XGB], axis = 1)
feature_scores.columns = ['Features', 'ANOVAF', 'SKB-Chi2', 'SKB-MICIF', 'PIKNN', 'LASSO', 'DTC', 'ETC', 'RFC', 'XGB']

print(feature_scores)
print("")

# Adding 'Mean Scores' Column to Feature Scores DataFrame
df_Mean = pd.DataFrame(feature_scores.iloc[:, 1: -1].mean(axis = 1))
feature_scores['Mean Scores'] = df_Mean

# Prioritized Features
prioritized_features = feature_scores.nlargest(nFeature, 'Mean Scores')
print(prioritized_features)
print("")

print("----------------------------------------------------")
print("---------------- Plotting Outputs ------------------")
print("----------------------------------------------------")
print("")

# Plot the Scores by Descending
plt.bar(prioritized_features['Features'], prioritized_features['Mean Scores'], color = 'black')
plt.xlabel('Feature', fontsize = 12)
plt.ylabel('Score', fontsize = 12)
plt.legend(['Importance Bar'])
plt.title('Prioritized Features based on AutoFS')
plt.show()

print("----------------------------------------------------")
print("----------------- Saving Outputs -------------------")
print("----------------------------------------------------")
print("")

# Export Selected Features to .CSV
df_feat = feature_scores.nlargest(nFeature, 'Mean Scores')
df_feat.to_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\Results\Severity Detection\Important Features\AutoFS.csv", index = False)

print("----------------------------------------------------")
print("------------ Important Features Selection ----------")
print("----------------------------------------------------")
print("")

# Display Top 50% Important Features
top_feat = feature_scores.nlargest(int(0.7*nFeature), 'Mean Scores')
important_feat = (top_feat.index).values
featureList = list(important_feat)
print(featureList)
print("")

print("----------------------------------------------------")
print("------ Data Ingestion with Important Features ------")
print("----------------------------------------------------")
print("")

# Import DataFrame (.csv) by Pandas Library
df = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Prepared Data [IF]\df.csv")

print("------------------------------------------------------")
print("----------------- Data Respiliting -------------------")
print("------------------------------------------------------")
print("")

# Select Features (as "f") and Target (as "t") Data
f = df.iloc[:, featureList].values
t = df.iloc[:, -1].values   

print("------------------------------------------------------")
print("------------------ Test & Train Data -----------------")
print("------------------------------------------------------")
print("")

# Split Train and Test Data in Proportion of 70:30 %
f_train, f_test, t_train, t_test = train_test_split(f, t, test_size = 0.33, random_state = RANDOM_STATE)

print('Feature Train Set:', f_train.shape)
print('Feature Test Set:', f_test.shape)
print('Target Train Set:', t_train.shape)
print('Target Test Set:', t_test.shape)
print("")

print("------------------------------------------------------")
print("----------------- ML Models Building -----------------")
print("------------------------------------------------------")
print("")

print("KNN = KNeighborsClassifier")
print("DTC = DecisionTreeClassifier")
print("GNB = GaussianNBClassifier")
print("SVM = SupportVectorMachineClassifier")
print("LRG = LogisticRegressionClassifier")
print("MLP = MLPClassifier")
print("RFC = RandomForestClassifier")
print("GBC = GradientBoostingClassifier")
print("XGB = XGBClassifier")
print("ADB = AdaBoostClassifier")
print("ETC = ExtraTreesClassifier")
print("CBC = CatBoostClassifier")
print("")

# Creating Machine Learning Models
KNN = KNeighborsClassifier(n_neighbors = 6, p = 2)
DTC = DecisionTreeClassifier(random_state = RANDOM_STATE)
GNB = GaussianNB()
SVM = SVC(decision_function_shape = "ovo", probability = True, random_state = RANDOM_STATE)
LRG = LogisticRegression(solver ='lbfgs', random_state = RANDOM_STATE)
MLP = MLPClassifier(max_iter = 500, solver = 'lbfgs', random_state = RANDOM_STATE)
RFC = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = RANDOM_STATE)
GBC = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.5, random_state = RANDOM_STATE)
XGB = XGBClassifier(n_estimators = 100, eval_metric = 'error', objective = 'binary:logistic')
ADB = AdaBoostClassifier(n_estimators = 100, random_state = RANDOM_STATE)
ETC = ExtraTreesClassifier(n_estimators = 100, random_state = RANDOM_STATE)
CBC = CatBoostClassifier(iterations = 10, learning_rate = 0.5, loss_function = 'MultiClass')

# Fitting Machine Learning Models on Train Data 
KNN.fit(f_train, t_train)
DTC.fit(f_train, t_train)
GNB.fit(f_train, t_train)
SVM.fit(f_train, t_train)
LRG.fit(f_train, t_train)
MLP.fit(f_train, t_train)
RFC.fit(f_train, t_train)
GBC.fit(f_train, t_train)
XGB.fit(f_train, t_train)
ADB.fit(f_train, t_train)
ETC.fit(f_train, t_train)
CBC.fit(f_train, t_train)

# Prediction of Test Data by Machine Learning Models 
t_pred0 = KNN.predict(f_test)
t_pred1 = DTC.predict(f_test)
t_pred2 = GNB.predict(f_test)
t_pred3 = SVM.predict(f_test)
t_pred4 = LRG.predict(f_test)
t_pred5 = MLP.predict(f_test)
t_pred6 = RFC.predict(f_test)
t_pred7 = GBC.predict(f_test)
t_pred8 = XGB.predict(f_test)
t_pred9 = ADB.predict(f_test)
t_pred10 = ETC.predict(f_test)
t_pred11 = CBC.predict(f_test)

# Prediction of Test Data by Machine Learning Models for ROC_AUC_Score
t_pred0_prob = KNN.predict_proba(f_test)
t_pred1_prob = DTC.predict_proba(f_test)
t_pred2_prob = GNB.predict_proba(f_test)
t_pred3_prob = SVM.predict_proba(f_test)
t_pred4_prob = LRG.predict_proba(f_test)
t_pred5_prob = MLP.predict_proba(f_test)
t_pred6_prob = RFC.predict_proba(f_test)
t_pred7_prob = GBC.predict_proba(f_test)
t_pred8_prob = XGB.predict_proba(f_test)
t_pred9_prob = ADB.predict_proba(f_test)
t_pred10_prob = ETC.predict_proba(f_test)
t_pred11_prob = CBC.predict_proba(f_test)

print("")
print("------------------------------------------------------")
print("----------------- Accessed Results -------------------")
print("------------------------------------------------------")
print("")

# Machine Learning Models Overfitting-Underfitting Values
print("KNN Overfitting-Underfitting Value:", "{:.3f}".format(((KNN.score(f_train, t_train))-(KNN.score(f_test, t_test)))))
print("DTC Overfitting-Underfitting Value:", "{:.3f}".format(((DTC.score(f_train, t_train))-(DTC.score(f_test, t_test)))))
print("GNB Overfitting-Underfitting Value:", "{:.3f}".format(((GNB.score(f_train, t_train))-(GNB.score(f_test, t_test)))))
print("SVM Overfitting-Underfitting Value:", "{:.3f}".format(((SVM.score(f_train, t_train))-(SVM.score(f_test, t_test)))))
print("LRG Overfitting-Underfitting Value:", "{:.3f}".format(((LRG.score(f_train, t_train))-(LRG.score(f_test, t_test)))))
print("KNN Overfitting-Underfitting Value:", "{:.3f}".format(((MLP.score(f_train, t_train)) - (MLP.score(f_test, t_test)))))
print("RFC Overfitting-Underfitting Value:", "{:.3f}".format(((RFC.score(f_train, t_train)) - (RFC.score(f_test, t_test)))))
print("GBC Overfitting-Underfitting Value:", "{:.3f}".format(((GBC.score(f_train, t_train)) - (GBC.score(f_test, t_test)))))
print("XGB Overfitting-Underfitting Value:", "{:.3f}".format(((XGB.score(f_train, t_train)) - (XGB.score(f_test, t_test)))))
print("ADB Overfitting-Underfitting Value:", "{:.3f}".format(((ADB.score(f_train, t_train)) - (ADB.score(f_test, t_test)))))
print("ETC Overfitting-Underfitting Value:", "{:.3f}".format(((ETC.score(f_train, t_train)) - (ETC.score(f_test, t_test)))))
print("CBC Overfitting-Underfitting Value:", "{:.3f}".format(((CBC.score(f_train, t_train)) - (CBC.score(f_test, t_test)))))
print("")

print("------------------------------------------------------")
print("----------------- Confusion Matrix -------------------")
print("------------------------------------------------------")
print("")

# Calculating Confusion Matrix for Machine Learning Models
print("KNN CM:")
print(confusion_matrix(t_test, t_pred0))
print("DTC CM:")
print(confusion_matrix(t_test, t_pred1))
print("GNB CM:")
print(confusion_matrix(t_test, t_pred2))
print("SVM CM:")
print(confusion_matrix(t_test, t_pred3))
print("LRG CM:")
print(confusion_matrix(t_test, t_pred4))
print("MLP CM:")
print(confusion_matrix(t_test, t_pred5))
print("RFC CM:")
print(confusion_matrix(t_test, t_pred6))
print("GBC CM:")
print(confusion_matrix(t_test, t_pred7))
print("XGB CM:")
print(confusion_matrix(t_test, t_pred8))
print("ADB CM:")
print(confusion_matrix(t_test, t_pred9))
print("ETC CM:")
print(confusion_matrix(t_test, t_pred10))
print("CBC CM:")
print(confusion_matrix(t_test, t_pred11))

print("")

print("------------------------------------------------------")
print("----------------- Assessment Report ------------------")
print("------------------------------------------------------")
print("")

# Create DataFrames of Machine Learning Models
Models = ['KNN', 'DTC', 'GNB', 'SVM', 'LRG', 'MLP', 'RFC', 'GBC', 'XGB', 'ADB', 'ETC', 'CBC']

Accuracy = [accuracy_score(t_test, t_pred0),
            accuracy_score(t_test, t_pred1),
            accuracy_score(t_test, t_pred2),
            accuracy_score(t_test, t_pred3),
            accuracy_score(t_test, t_pred4),
            accuracy_score(t_test, t_pred5),
            accuracy_score(t_test, t_pred6),
            accuracy_score(t_test, t_pred7),
            accuracy_score(t_test, t_pred8),
            accuracy_score(t_test, t_pred9),
            accuracy_score(t_test, t_pred10),
            accuracy_score(t_test, t_pred11)]

ROC_AUC = [roc_auc_score(t_test, t_pred0_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred1_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred2_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred3_prob, multi_class = 'ovo'),               
           roc_auc_score(t_test, t_pred4_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred5_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred6_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred7_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred8_prob, multi_class = 'ovo'),               
           roc_auc_score(t_test, t_pred9_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred10_prob, multi_class = 'ovo'),
           roc_auc_score(t_test, t_pred11_prob, multi_class = 'ovo')]

Precision = [precision_score(t_test, t_pred0, average = 'macro'),
             precision_score(t_test, t_pred1, average = 'macro'),
             precision_score(t_test, t_pred2, average = 'macro'),
             precision_score(t_test, t_pred3, average = 'macro'),
             precision_score(t_test, t_pred4, average = 'macro'),
             precision_score(t_test, t_pred5, average = 'macro'),
             precision_score(t_test, t_pred6, average = 'macro'),
             precision_score(t_test, t_pred7, average = 'macro'),
             precision_score(t_test, t_pred8, average = 'macro'),
             precision_score(t_test, t_pred9, average = 'macro'),
             precision_score(t_test, t_pred10, average = 'macro'),
             precision_score(t_test, t_pred11, average = 'macro')]

F1_Score = [f1_score(t_test, t_pred0, average = 'macro'),
            f1_score(t_test, t_pred1, average = 'macro'),
            f1_score(t_test, t_pred2, average = 'macro'),
            f1_score(t_test, t_pred3, average = 'macro'),
            f1_score(t_test, t_pred4, average = 'macro'),
            f1_score(t_test, t_pred5, average = 'macro'),
            f1_score(t_test, t_pred6, average = 'macro'),
            f1_score(t_test, t_pred7, average = 'macro'),
            f1_score(t_test, t_pred8, average = 'macro'),
            f1_score(t_test, t_pred9, average = 'macro'),
            f1_score(t_test, t_pred10, average = 'macro'),
            f1_score(t_test, t_pred11, average = 'macro')]

Recall = [recall_score(t_test, t_pred0, average = 'macro'),
          recall_score(t_test, t_pred1, average = 'macro'),
          recall_score(t_test, t_pred2, average = 'macro'),
          recall_score(t_test, t_pred3, average = 'macro'),
          recall_score(t_test, t_pred4, average = 'macro'),
          recall_score(t_test, t_pred5, average = 'macro'),
          recall_score(t_test, t_pred6, average = 'macro'),
          recall_score(t_test, t_pred7, average = 'macro'),
          recall_score(t_test, t_pred8, average = 'macro'),
          recall_score(t_test, t_pred9, average = 'macro'),
          recall_score(t_test, t_pred10, average = 'macro'),
          recall_score(t_test, t_pred11, average = 'macro')]

KFCV = [np.mean(cross_val_score(KNN, f, t, cv = 10)),
        np.mean(cross_val_score(DTC, f, t, cv = 10)),
        np.mean(cross_val_score(GNB, f, t, cv = 10)),
        np.mean(cross_val_score(SVM, f, t, cv = 10)),
        np.mean(cross_val_score(LRG, f, t, cv = 10)),
        np.mean(cross_val_score(MLP, f, t, cv = 10)),
        np.mean(cross_val_score(RFC, f, t, cv = 10)),
        np.mean(cross_val_score(GBC, f, t, cv = 10)),
        np.mean(cross_val_score(XGB, f, t, cv = 10)),
        np.mean(cross_val_score(ADB, f, t, cv = 10)),
        np.mean(cross_val_score(ETC, f, t, cv = 10)),
        np.mean(cross_val_score(CBC, f, t, cv = 10))]

model_scores = {'Model': Models,
                'Accuracy': Accuracy,
                'ROC-AUC': ROC_AUC,
                'Precision': Precision,
                'F1-Score': F1_Score,
                'Recall': Recall,
                'KFCV': KFCV}

print("")

df_models = pd.DataFrame(model_scores)
print(df_models)
print("")

# Adding 'Mean Scores' Column to Machine Learning Model Scores DataFrame
nModel = len(Models)
df_Mean = pd.DataFrame(df_models.iloc[:, 1: -1].mean(axis = 1))
df_models['Mean Scores'] = df_Mean

# Prioritized Machine Learning Models
prioritized_models = df_models.nlargest(nModel, 'Mean Scores')
print(prioritized_models)
print("")

print("----------------------------------------------------")
print("---------------- Plotting Outputs ------------------")
print("----------------------------------------------------")
print("")

# Plot the Scores by Descending
plt.barh(prioritized_models['Model'], prioritized_models['Mean Scores'], color = 'black')
plt.xlabel('Model', fontsize = 12)
plt.ylabel('Score', fontsize = 12)
plt.title('Prioritized Machine Learning Models based on AutoCML')
plt.show()

print("------------------------------------------------------")
print("-------------- The Best AutoIFSCML Model -------------")
print("------------------------------------------------------")
print("")

# Selection of the Best Machine Learning Model by Hybrid Descended Mean Scores
print(df_models.nlargest(1, 'Mean Scores')) 
print("")

print("----------------------------------------------------")
print("----------------- Saving Outputs -------------------")
print("----------------------------------------------------")
print("")

# Export Selected Features to .CSV
df_CML = df_models.nlargest(nModel, 'Mean Scores')
df_CML.to_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\Results\Severity Detection\Important Features\AutoIFSCML.csv", index = False)
print("")

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")