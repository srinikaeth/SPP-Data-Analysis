import numpy as np
import pandas as pd
from time import time

data = pd.read_csv('SPP_data.csv')

data.head()

# Size of the data
size_df = data.size

print (size_df)

# Count the number of unique Price Nodes
qty_nodes = data['Price Node Name'].nunique()

print "No of nodes is: {} ".format(qty_nodes)

# Extracting only the necessary columns
data_reqd = data[['Price Node Name','Local Datetime (Hour Ending)','Price $/MWh','Price Node ID']]

data_reqd.head()

# Checking for NaN values in any column
data_reqd.isnull().sum().sum()

# Listing out all the Node names
node_names = data_reqd['Price Node Name'].unique()
print (node_names)

# Converting to a MultiIndex DataFrame grouped based on Price Node Names
data_multilevel = data_reqd.set_index(['Price Node Name', 'Local Datetime (Hour Ending)'])
data_multilevel.head()

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

val_tot = []
val_tot2 = []

for name in node_names:    
    
    df = check_data.xs(name)
    dict_vals = check_min[name]
       
    total_days = np.arange(1, len(dict_vals) + 1, dtype=np.int16)
    
    hours = np.zeros(len(total_days))
    mean2 = np.zeros(len(total_days))
    
    for i in total_days:
        end = 24*i - 1
        start = max(0, end - 24)
        
        max_val = dict_vals['Max Price $/MWh'].iloc[i-1]
        
        df_check = df.iloc[start:end]
        df_mean2 = df_check.drop(df_check.index[11:19])       
        
        hours[i-1] = (df_check >= max_val-2).apply(np.count_nonzero)
        mean2[i-1] = df_mean2.mean()
           
    dict_vals['Hours'] = hours
    dict_vals['Mean off peak'] = mean2
    
    # Computing the RAP as Max - Min
    
    dict_vals['Dprice'] = dict_vals['Max Price $/MWh'] - dict_vals['Min Price $/MWh']
    dict_vals['Factor'] = dict_vals['Dprice']*dict_vals['Hours']
    
    # Computing RAP as Max - Avg
    
    dict_vals['Dprice2'] = dict_vals['Max Price $/MWh'] - dict_vals['Mean off peak']
    dict_vals['Factor2'] = dict_vals['Dprice2']*dict_vals['Hours']
     
    # Calculating the sum of the first RAP over time for each node     
    val_tot.append(dict_vals['Factor'].sum())
    
    # Calculating the sum of second RAP over time for each node
    val_tot2.append(dict_vals['Factor2'].sum())
    
# Storing the summed up values in val_tot    
val_tot = np.hstack(val_tot)
val_tot2 = np.hstack(val_tot2)

# Normalizing by the number of days
norm_eas = val_tot/day_array
norm_eas2 = val_tot2/day_array
    
# Sorting the values to choose the top 12 nodes
method_1 = np.argsort(norm_eas)[-12:]
method_2 = np.argsort(norm_eas2)[-12:]

# Printing out the top 12 values
print('The top 12 values according to method 1 are: ')
print(method_1)

print('The top 12 values according to method 2 are: ')
print(method_2)