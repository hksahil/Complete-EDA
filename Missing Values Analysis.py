
######################################### Finding Missing Values #########################################
# Select rows where name is missing
df['City'].isnull()

# Count how many missing data are in name column
df['City'].isnull().sum()

# Count how many missing data are in multiple cols
df[['City','Gender']].isnull().sum()

# Select rows where name is not missing
df['City'].notnull()

# Find missing values in entire df
df.isnull().sum()

# Visualising the missing values
df.isnull().sum().plot(kind='bar')

# Find missing values percentage
data.isnull().mean()

# Find columns which is having more than 20% NaN values
columns = []
for col in df.columns:
    if df[col].isnull().mean() > 0.20:
        columns.append(feature)

for i in columns:
    print(('{} having     {}% null values').format(i,df[i].isnull().mean()))


# Some useful viz for outlier detection
import seaborn as sns
import matplotlib.pyplot as plt
sns.displot(df['cog$'],color='r')
sns.boxplot(df['Rating'],color='y')


# Doing all the viz in one shot
numeric_df = df.select_dtypes(include=['number'])
categorical_columns = df.select_dtypes(include=['object']).columns

# Create a grid of boxplots for all numeric columns
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

for i, column in enumerate(numeric_df.columns):
    plt.subplot(3, 4,i+1)  # Adjust the subplot arrangement based on the number of columns
    #numeric_df.boxplot(column)
    #sns.countplot(data=df, x=column)
    plt.title(column)

plt.tight_layout()  # Adjust subplot layout
plt.show()


# Corelation of vars
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), cmap='Blues', annot=True)


################################################# Removing NaNs #########################################
############## Remove the rows containing NaNs
# Remove row if atleast one NaN is there
df.dropna(inplace=True)

# Remove row if all values are NaN
df.dropna(axis=0, how='all',inplace=True)

# Remove row if NaNs only in specific columns
df.dropna(subset=['Customer type'],inplace=True)

############## Remove the columns containing NaNs
# Remove the column if Atleast one NaN is there
df.dropna(axis=1,inplace=True)

# Remove the column if  all values are NaN
df.dropna(axis=1, how='all', inplace=True)

################################################# Replacing/Imputing NaNs #########################################

################## Using Pandas - fillna() Techniques
# Replace nulls with 0
df.fillna(0,inplace=True)

# Replace nulls with anything else
df.fillna("any value",inplace=True)

# Replacing with mathematical functions
df.fillna(df.mean(), inplace=True)
df.fillna(df.median(), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Above cell’s value will be copied in NaN
df.fillna(method="ffill",inplace=True)

# Below cell’s value will be copied in NaN
df.fillna(method="bfill",inplace=True)

################## Using Interpolation Techniques <--
df.interpolate() -> (above+below/2)
df.interpolate(method='linear', inplace=True)
df.interpolate(method='quadratic', inplace=True)
df.interpolate(method='polynomial', order=2, inplace=True)

################## Using Scikit-learn Techniques
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

################### Using map
country_to_code = {'India': 'IN', 'Pak': 'PK', 'US': 'US'}
df['Code'] =df['Code'].fillna(df['Country'].map(country_to_code))


code_to_country = {'IN': 'India', 'PK': 'Pak', 'US': 'US'}
df['Country'] = df['Country'].fillna(df['Code'].map(code_to_country))


# Real life example Filling missing values
# Dimension
df['Customer type'] = df['Customer type'].fillna(df['Customer type'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Payment'] = df['Payment'].fillna(df['Payment'].mode()[0])

# Measure
df['cog$'] = df['cog$'].fillna(df['cog$'].mean())
df['gross margin percentage'] = df['gross margin percentage'].fillna(df['gross margin percentage'].mean())
df['gross income'] = df['gross income'].fillna(df['gross income'].mean())