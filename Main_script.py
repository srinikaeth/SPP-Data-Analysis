import numpy as np
import pandas as pd
from IPython.display import display
from time import time

data = pd.read_csv('SPP_data.csv')

display(data.head(n=5))

# Size of the data
size_df = data.size

print (size_df)

# Count the number of unique Price Nodes
qty_nodes = data['Price Node Name'].nunique()

print "No of nodes is: {} ".format(qty_nodes)

# Extracting only the necessary columns
data_reqd = data[['Price Node Name','Local Datetime (Hour Ending)','Price $/MWh','Price Node ID']]

display(data_reqd.head(n=5))

# Checking for NaN values in any column
data_reqd.isnull().sum().sum()

# Listing out all the Node names
node_names = data_reqd['Price Node Name'].unique()
print (node_names)

# Converting to a MultiIndex DataFrame grouped based on Price Node Names
data_multilevel = data_reqd.set_index(['Price Node Name', 'Local Datetime (Hour Ending)'])
display(data_multilevel.head())

hour_count = []

# Calculating no of rows (i.e data points) for each node based on node name
for x in node_names:
    hour_count.append(len(data_multilevel.xs(x).index))

# Just confirming that the hour_count has calculated hours for all nodes    
print(len(hour_count))

# Converting the list into array
hour_array = np.array(hour_count, dtype=float)

# Converting to number of days
day_array = hour_array/24

# Converting the index in data_reqd variable as datetime variables
data_mod = data_reqd

data_mod['Local Datetime (Hour Ending)'] = pd.to_datetime(data_mod['Local Datetime (Hour Ending)'], format='%m/%d/%Y %H:%M')

# Converting the data to it's final usable form, by indexing according to the node names and the datetime variables
data_final = data_mod.set_index(['Price Node Name','Local Datetime (Hour Ending)'])

data_final = data_final.drop('Price Node ID', axis=1)

# Applying the first technique

min_max = {}

for name in node_names:
    max_node = data_final.xs(name).resample('D').max()
    min_node = data_final.xs(name).resample('D').min()
    
    max_node.columns = ['Max Price $/MWh']
    min_node.columns = ['Min Price $/MWh']

    min_max[name] = pd.concat([max_node, min_node], axis=1)
    
# Calculating the number of hours for which the peak is present
    
check_data = data_final.copy()
check_min = min_max.copy()
       
for name in node_names:
    df = check_data.xs(name)
    dict_vals = check_min[name]
    
    total_days = np.arange(1, len(dict_vals) + 1, dtype=np.int16)
    
    hours = np.zeros(len(total_days))
    
    for i in total_days:
        end = 24*i - 1
        start = max(0, end - 24)
        
        max_val = dict_vals['Max Price $/MWh'].iloc[i-1]
        
        df_check = df.iloc[start:end]
        
        hours[i-1] = (df_check >= max_val-2).apply(np.count_nonzero)
        
    dict_vals['Hours'] = hours
    
# Computing the RAP as Max - Min

for name in node_names:
    df = check_min[name]
    
    df['Dprice'] = df['Max Price $/MWh'] - df['Min Price $/MWh']
    df['Factor'] = df['Dprice']*df['Hours']
    

# Calculating the sum of the RAP over time for each node      
val_tot = []

for name in node_names:
    val_tot.append(check_min[name]['Factor'].sum())
    
# Storing the summed up values in val_tot    
val_tot = np.hstack(val_tot)

# Normalizing by the number of days
norm_eas = val_tot/day_array

# Sorting the values to choose the top 12 nodes
method_1 = np.argsort(norm_eas)[-12:]

# Printing out the top 12 values
print('The top 12 values according to method 1 are: ')
print(method_1)

# Better area calculation based on taking mean of values (except from 12-6pm) as the minimum

check_data2 = data_final.copy()
check_min2 = min_max.copy()

for name in node_names:
    df2 = check_data2.xs(name)
    dict_vals2 = check_min2[name]
    
    total_days2 = np.arange(1, len(dict_vals2) + 1, dtype=np.int16)
    
    hours2 = np.zeros(len(total_days2))
    mean2 = np.zeros(len(total_days2))
    
    for i in total_days2:
        end = 24*i - 1
        start = max(0, end - 24)
        
        max_val2 = dict_vals2['Max Price $/MWh'].iloc[i-1]
        
        df_check2 = df2.iloc[start:end]
        df_mean2 = df_check2.drop(df_check2.index[11:19])
        
        hours2[i-1] = (df_check2 >= max_val2-2).apply(np.count_nonzero)
        
        mean2[i-1] = df_mean2.mean()
        
    dict_vals2['Hours'] = hours2
    dict_vals2['Mean off peak'] = mean2
    
# Calculating the RAP as Max-Avg

for name in node_names:
    df = check_min2[name]
    
    df['Dprice2'] = df['Max Price $/MWh'] - df['Mean off peak']
    df['Factor2'] = df['Dprice2']*df['Hours']
    
# Calculating the sum of the RAP over time for each node          
val_tot2 = []

for name in node_names:
    val_tot2.append(check_min2[name]['Factor2'].sum())

# Storing the summed up values in val_tot    
val_tot2 = np.hstack(val_tot2)

# Normalizing by number of days
norm_eas2 = val_tot2/day_array

# Sorting the values to choose the top 12 nodes
method_2 = np.argsort(norm_eas2)[-12:]

# Printing out the top 12 values
print('The top 12 values according to method 2 are: ')
print(method_2)