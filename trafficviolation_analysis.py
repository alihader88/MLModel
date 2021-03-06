import pandas as pd
data = pd.read_csv("Traffic_Violations.csv")
from scipy import stats
from statsmodels.stats import weightstats as stests

df = pd.read_csv('Traffic_Violations.csv')

df

numerics = ['float64']
numeric_df = df.select_dtypes(include= numerics)
len(numeric_df.columns)

missing_percentages = df.isna().sum().sort_values(ascending= False)/len(df)
missing_percentages[missing_percentages != 0]

missing_percentages[missing_percentages != 0].plot(kind= 'barh')

df.columns

locations = df.Location.unique()
len(locations)
Vehicletype = df.VehicleType.unique()
len(Vehicletype)

Vehicletype = df.VehicleType.unique()
len(Vehicletype)

make =df.Make.unique()
len(make)

Accident = df.Accident.value_counts()
Accident

locationcount[:20]

type(locationcount)

locationcount[:20].plot(kind='barh')

import seaborn as sns
sns.set_style("darkgrid")

sns.histplot(locationcount, log_scale=True)

locationcount[locationcount == 1]

color = df.Color.value_counts()
color

color.plot(kind='barh')

df.Latitude

df.Longitude

sample_df = df.sample(int(0.1 * len(df)))

sns.scatterplot(x=sample_df.Longitude, y=sample_df.Latitude, size=0.001)

wrokzone = df["Work Zone"].unique()
len(wrokzone)

sub_df = pd.DataFrame(zip(df['Article'],df['Alcohol'],df['Belts'],df['Fatal']), columns=['violation','alcohol','belts','fatal'])
sub_df['alcohol'] = sub_df.alcohol.eq('Yes').mul(1)
sub_df['belts'] = sub_df.belts.eq('Yes').mul(1)
sub_df['fatal'] = sub_df.fatal.eq('Yes').mul(1)
sub_df.set_index('violation').describe()

import numpy as np
sub_df1 = pd.DataFrame(zip(df['Violation Type'],df['Alcohol'],df['Belts'],df['Fatal']), columns=['violation','alcohol','belts','fatal'])
sub_df1['alcohol'] = sub_df1.alcohol.eq('Yes').mul(1)
sub_df1['belts'] = sub_df1.belts.eq('Yes').mul(1)
sub_df1['fatal'] = sub_df1.fatal.eq('Yes').mul(1)
table1 = pd.pivot_table(sub_df1, values=['alcohol','belts','fatal'], columns='violation', aggfunc=np.mean)
table1

df_chi = pd.read_csv('/content/traffic-violations-in-maryland-county/Traffic_Violations.csv')

contingency_table=pd.crosstab(df_chi["Gender"],df_chi["Belts"])
print('contingency_table :-\n',contingency_table)

Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)

b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)

no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
df11=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",df)
alpha = 0.05

from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)

critical_value=chi2.ppf(q=1-alpha,df=df11)
print('critical_value:',critical_value)

#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=df11)
print('p-value:',p_value)

print('Significance level: ',alpha)
print('Degree of Freedom: ',df11)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)

if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

#x-axis ranges from 0 to 20 with .001 steps
x = np.arange(0, 20, 0.001)

#define multiple Chi-square distributions
plt.plot(x, chi2.pdf(x, df=4), label='df: 4')
plt.plot(x, chi2.pdf(x, df=8), label='df: 8') 
plt.plot(x, chi2.pdf(x, df=12), label='df: 12') 

#add legend to plot
plt.legend()

df.columns

cdf = df[['Alcohol','Belts','Gender','Fatal','Accident']]
cdf.head(9)



df.info()

dummy_var = ['Accident', 'Alcohol', 'Belts', 'Gender', 'Fatal']
df_oh = pd.get_dummies( df, columns= dummy_var, drop_first= True)
df_oh.describe()

import numpy as np
Accident = pd.DataFrame(
    {
 "Accident": pd.Series([True, False , np.nan], dtype=np.dtype("O"))
    }
)
accident = df.convert_dtypes()
accident

import numpy as np
df = pd.DataFrame(
    {
 "Alcohol": pd.Series([True, False , np.nan], dtype=np.dtype("O"))
    }
)
Alcohol = df.convert_dtypes()

import numpy as np
df = pd.DataFrame(
    {
 "Belts": pd.Series([True, False , np.nan], dtype=np.dtype("O"))
    }
)
Belts = df.convert_dtypes()

import numpy as np
df = pd.DataFrame(
    {
 "Fatal": pd.Series([True, False , np.nan], dtype=np.dtype("O"))
    }
)
Fatal = df.convert_dtypes()

import numpy as np
df = pd.DataFrame(
    {
 "Gender": pd.Series(['M', 'F' , 'U'], dtype=np.dtype("O"))
    }
)
Gender= df.convert_dtypes()
Gender

cdf = df[['Alcohol','Belts','Gender','Fatal','Accident']]
cdf.head(9)