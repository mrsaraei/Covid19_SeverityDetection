# -*- coding: utf-8 -*-
print("------------------------------------------------------")
print("---------------- Metadata Information ----------------")
print("------------------------------------------------------")
print("")

print("In the name of God")
print("Project: AutoCML: Automatic Comparative Machine Learning in COVID-19 Severity Detection Model")
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

print("----------------------------------------------------")
print("------------------ Data Ingestion ------------------")
print("----------------------------------------------------")
print("")

# Import DataFrames (.csv) by Pandas Library
df_PLS = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Encoded Data\CoV19_PLS_Encoded.csv")
df_CLS = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Encoded Data\CoV19_CLS_Encoded.csv")
df_EHR = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Encoded Data\CoV19_EHR_Encoded.csv")
df_PCR = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Encoded Data\CoV19_PCR_Encoded.csv")
df_CT = pd.read_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Encoded Data\CoV19_CT_Encoded.csv")

# Fusion of DataFrames (.csv) 
df1 = pd.concat([df_PLS, df_CLS, df_EHR, df_PCR, df_CT], axis = 1)         
# df1 = np.concatenate((df_PLS, df_CLS, df_EHR, df_PCR, df_CT), axis = 1)

# Get rid of Poor Features from DataFrame 
df = df1.drop(['Gender', 'GGO', 'Consolid', 'Paving', 'Halo'], axis = 1)

print("------------------------------------------------------")
print("----------------- Data Annotation --------------------")
print("------------------------------------------------------")
print("")

# [0 = No Sign/Symptom]
# [1 = Mild Level]
# [2 = Moderate Level]
# [3 = Severe Level]
# [4 = Critical Level]

df['Target'] = 0     # range(0, len(df))

def target(row):
    if  (row['RT_PCR'] == 0):
        return "0"
    elif ((row['RT_PCR'] == 1) | (row['RT_PCR'] == 2)) & (row['Hypoxemia'] == 3) & (row['LoS_M_C'] == 1):
        return "4" 
    elif ((row['RT_PCR'] == 1) | (row['RT_PCR'] == 2)) & ((row['Hypoxemia'] == 3) & (row['Dyspnea'] == 1)) & ((row['CSS'] == 4) | (row['CSS'] == 5)):
        return "3"
    elif ((row['RT_PCR'] == 1) | (row['RT_PCR'] == 2)) & ((row['Hypoxemia'] == 2) & ((row['Dyspnea'] == 1) | (row['Ch_P'] == 1)) & ((row['CSS'] == 1) | (row['CSS'] == 2) | (row['CSS'] == 3))):
        return "2"
    elif ((row['RT_PCR'] == 1) | (row['RT_PCR'] == 2)) & ((row['Pyrexia'] == 1) & (row['Hypoxemia'] == 1) & ((row['Cough'] == 2) | (row['Fatigue'] == 1) | (row['Headache'] == 1) | (row['LoT_S'] == 1) | (row['GI'] == 1) | (row['Dyspnea'] == 1) | (row['LoS_M_C'] == 1) | (row['Ch_P'] == 1))):
        return "1"
    elif ((row['RT_PCR'] == 1) | (row['RT_PCR'] == 2)) & ((row['Pyrexia'] == 1) | (row['Cough'] == 2) | (row['Fatigue'] == 1) | (row['Headache'] == 1) | (row['LoT_S'] == 1) | (row['GI'] == 1) | (row['Dyspnea'] == 1) | (row['LoS_M_C'] == 1) | (row['Ch_P'] == 1)) & ((row['Age'] == 1) | (row['Anamnesis'] == 3)):     
        return "1"    
    elif ((row['RT_PCR'] == 1) | (row['RT_PCR'] == 2)) & ((row['Pyrexia'] == 1) | (row['Hypoxemia'] == 1) | (row['Cough'] == 2) | (row['Fatigue'] == 1) | (row['Headache'] == 1) | (row['LoT_S'] == 1) | (row['GI'] == 1) | (row['Dyspnea'] == 1) | (row['LoS_M_C'] == 1) | (row['Ch_P'] == 1)) & (row['Contact'] == 1):
        return "1"
    elif ((row['RT_PCR'] == 1) | (row['RT_PCR'] == 2)) & ((row['Pyrexia'] == 0) & (row['Hypoxemia'] == 0) & (row['Cough'] == 0) & (row['Fatigue'] == 0) & (row['Headache'] == 0) & (row['LoT_S'] == 0) & (row['GI'] == 0) & (row['Dyspnea'] == 0) & (row['LoS_M_C'] == 0) | (row['Ch_P'] == 0)):
        return "0" 
    else:
        return "0"

df = df.assign(Target=df.apply(target, axis = 1))

print("")

print("----------------------------------------------------")
print("----------------- Saving Outputs -------------------")
print("----------------------------------------------------")
print("")

# Save DataFrame After Munging
df.to_csv(r"D:\My Papers\2020 - 2030\Journal Paper\Journal of Communications Technology and Electronics [IF=0.5] (2022)\CoV19 Dataset\Severity Detection\Annotated Data\CoV19_Data_Annotated.csv", index = False)

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")
