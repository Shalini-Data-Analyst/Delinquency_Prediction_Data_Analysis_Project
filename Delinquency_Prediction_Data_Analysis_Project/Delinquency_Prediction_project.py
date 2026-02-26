import pandas as pd
df=pd.read_csv(r"C:\Users\shalu\OneDrive\Documents\Delinquency_prediction_dataset (1).csv")
# Data Analysis
print(df.shape)
print(df.head())
print(df.columns)
print(df.dtypes)
print(df.info)
print(df.nunique())
print(df.describe())
print(df.drop_duplicates())
print(df.isnull().sum())

## Handaling missing values
df['Income']=df['Income'].fillna(df['Income'].mean())
print(df['Income'])
df['Loan_Balance']=df['Loan_Balance'].fillna(df['Loan_Balance'].median())
print(df['Loan_Balance'])
df['Credit_Score']=df['Credit_Score'].fillna(df['Credit_Score'].median())
print(df['Credit_Score'])
print(df.isnull().sum())
print(df.describe())
print(df['Customer_ID'].unique)


##Exploratary Data Analysis

#Delinquency Distribution
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Delinquent_Account',data=df,color='blue')
plt.title("Delinquency Distribution",color='black')
plt.show()

#Income VS Delinquency
sns.boxplot(x='Delinquent_Account',y='Income',data=df,color='purple')
plt.title("Income VS Delinquency",color='black')
plt.show()

#Creadit Score VS Delinquency
sns.histplot(df['Credit_Score'],kde=True)
sns.boxplot(x='Delinquent_Account',y='Credit_Score',color='violet',linewidth=4,data=df)
plt.title("Creadit Score VS Delinquency",color='black',fontsize=12)
plt.show()

#Creadit Utilization  VS Delinquency
sns.boxplot(x='Delinquent_Account',y='Credit_Utilization',color='violet',linewidth=4,data=df)
plt.title("Creadit Utilization VS Delinquency",color='black',fontsize=12)
plt.show()

#Debit income ratio
sns.boxplot(x='Delinquent_Account',y='Debt_to_Income_Ratio',color='yellow',linewidth=4,data=df)
plt.title("Debt_to_Income_Ratio  VS Delinquency",color='black',fontsize=12)
plt.show()

#Employement status Count
sns.countplot(x='Employment_Status',data=df,color='orange',linewidth=4)
plt.title("Employement status Count",color='black',fontsize=16)
plt.show()

#Behavioural risk(Missed_Payments)
sns.barplot(x='Delinquent_Account',y='Missed_Payments',data=df,color='orange',linewidth=4)
plt.title("Missed_Payments vs Delinquency",color='black',fontsize=16)
plt.show()



#Employement VS Delinquency
sns.barplot(x='Employment_Status',y='Delinquent_Account',data=df,color='blue',linewidth=4)
plt.title("Employement Status VS Delinquency",color='black')
plt.show()

#Correaltion B/W  Missed payments and delinquency

correlation=df[['Missed_Payments','Delinquent_Account']].corr()
print(correlation)
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt=".2f")
plt.title("Correlation heatmap",color='black')
plt.show()

#Correaltion B/W  Debt_to_Income_Ratio and delinquency
correlation=df[['Delinquent_Account','Debt_to_Income_Ratio']].corr()
print(correlation)
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt=".2f")
plt.title("Correlation heatmap",color='black')
plt.show()



