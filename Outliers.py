################## IRQ Method  

# Typically defined as 1.5 times the IQR above the third quartile or below the first quartile)

Q1 = df['cog$'].quantile(0.25)
Q3 = df['cog$'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['cog$'] < lower_bound) | (df['cog$'] > upper_bound)]
print("\nOutliers detected using IQR:")

# Seeing the outliers
outliers_iqr

# Removing outliers from the main DataFrame
cleaned_df = df[~df.index.isin(outliers_iqr.index)]

################### Boxplot Method
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x=df['cog$'])
plt.title('Boxplot for cog$')
plt.show()

#################### DBSCAN
from sklearn.cluster import DBSCAN
df['outliers flag'] = DBSCAN(eps=1.0, min_samples=2).fit_predict(df[['cog$']]) # Create Model, Train and Test Model -> It flags each outlier as -1
outliers_dbscan = df[df['dbscan_labels'] == -1]

# See outliers
print("\nOutliers detected using DBSCAN:")
outliers_dbscan

# Removing outliers from the main DataFrame
cleaned_df = df[df['dbscan_labels'] != -1]

###################### Isolation Forest
from sklearn.ensemble import IsolationForest
df['isolation_forest'] = IsolationForest(contamination=0.05).fit_predict(df[['cog$']]) #Training ML Model
df[df['isolation_forest'] == -1] -- Flag for outliers

##################### Local Outlier Factor (LOF)
from sklearn.neighbors import LocalOutlierFactor
df['lof_scores'] = LocalOutlierFactor(n_neighbors=20, contamination=0.05).fit_predict(df[['cog$']])
outliers_lof = df[df['lof_scores'] == -1] -- Flag for outliers
print("\nOutliers detected using LOF:")
outliers_lof

# Remove the outlier rows
df = df.drop([1, 2, 3],axis=0)

# Remove the outlier columns
df = df.drop(['Branch', 'City'], axis=1)