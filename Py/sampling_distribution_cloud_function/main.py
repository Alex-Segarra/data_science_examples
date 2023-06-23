def function_pearson_rs_lmp_channel_analysis(data,context):  
  from google.cloud import bigquery
  import pandas as pd
  import numpy as np
  #import scipy.stats as stats
  from scipy import stats
  import re
  import datetime
  import warnings
  warnings.filterwarnings("ignore")

# Declare functions
  def sampling_dist_builder_sliced(df,n,cols,groupby_col='',conv_cols=''):
        
    """
    Purpose: This function takes in a dataframe and returns a list of dataframes to be used for sampling distribution analysis.
    
    Args:
    df: A dataframe
    n: An integer referring to the number of iterations.
    cols: A list of columns to split by.
    groupby_col: A string referring to the column to group by.
    conv_cols: A list of target columns to run analysis against.
    
    returns:
    A list of dataframes
    """
    
    base_mean_df = pd.DataFrame()
    
    conv_mean_df = pd.DataFrame()
    
    seg_mean_df = pd.DataFrame()
    
    agg_mean_df = pd.DataFrame()
    
    # A loop that runs sampling for n iterations to bootstrap a larger dataset
    
    for x in range(0,n):
        
        if x%100 == 0:
            
            print(f"On iteration {x}")
            
        # Sampling occurs here.
        
        wdf = df.sample(frac=.01, replace=True)
        
        #print(wdf.columns)
        
        base_mean_df = base_mean_df.append(pd.DataFrame(wdf[cols].mean()).T)
        
        conv_mean_df = conv_mean_df.append(wdf[cols+conv_cols].groupby(conv_cols).mean().reset_index())
        
        seg_mean_df = seg_mean_df.append(wdf[cols+groupby_col].groupby(groupby_col).mean().reset_index())
        
        agg_mean_df = agg_mean_df.append(wdf[cols+groupby_col+conv_cols].groupby(groupby_col+conv_cols).mean().reset_index())
        
    return([base_mean_df,conv_mean_df,seg_mean_df,agg_mean_df]) 

  def significance_test(df,X,boolean,conv_cols, table_name, agg_var='agg_var'):
        
    """
    Purpose: This function takes in a dataframe and returns a list of dataframes to be used for significance test.  It outputs the results to a bigquery table.
    
    Args:
    df: A list of dataframes
    X: A string referring to the independent variable column.
    boolean: A string referring a the boolean column that determines a status.
    conv_cols: A list of target columns to run analysis against.
    table_name: A string referring to the table name to output to.
    agg_var: A string referring to the column to group by.
    
    returns:
    Nothing
    
    """
    
    #parses through a list of dataframes
    
    agg_str = 'agg_string'
    
    base_mean_df = df[0]
        
    conv_mean_df = df[1]
    
    seg_mean_df = df[2]

    agg_mean_df = df[3]
    
    # Creates an aggregated dataframe and assigns a mean and standard deviation
    sdf = agg_mean_df.assign(metric=X,conversion_stage=conv_cols).groupby(['metric','conversion_stage',agg_var,boolean])\
    .agg(avg=(X,'mean'), std = (X,'std'))\
    .reset_index()
    
    #Fills in other columns that will be filled in later
    sdf['mean_overall'] = base_mean_df[X].dropna().mean()
    sdf['std_overall'] = base_mean_df[X].dropna().std()
    sdf['tstat'] = 0
    sdf['pvalue'] = 0
    sdf['significance'] = ''
    
    sdf[f'mean_{agg_str}'] = 0
    sdf[f'std_{agg_str}'] = 0
    sdf[f'tstat_{agg_str}'] = 0
    sdf[f'pvalue_{agg_str}'] = 0
    sdf[f'significance_{agg_str}'] = ''
    
    sdf['mean_conv'] = 0
    sdf['std_conv'] = 0
    sdf['tstat_conv'] = 0
    sdf['pvalue_conv'] = 0
    sdf['significance_conv'] = ''
    
    sdf['mean_comp'] = 0
    sdf['std_comp'] = 0
    sdf['tstat_comp'] = 0
    sdf['pvalue_comp'] = 0
    sdf['significance_comp'] = ''
    
    #Finds the mean and standard deviation for each status (boolean)
    
    sdf.loc[(sdf[boolean] == 1), 'mean_conv'] = conv_mean_df.loc[(conv_mean_df[boolean] == 1), X].dropna().mean()
    sdf.loc[(sdf[boolean] == 1), 'std_conv'] = conv_mean_df.loc[(conv_mean_df[boolean] == 1), X].dropna().std()
    sdf.loc[(sdf[boolean] == 0), 'mean_conv'] = conv_mean_df.loc[(conv_mean_df[boolean] == 0), X].dropna().mean()
    sdf.loc[(sdf[boolean] == 0), 'std_conv'] = conv_mean_df.loc[(conv_mean_df[boolean] == 0), X].dropna().std()
    
    # Loops over aggregation variables and finds the mean and standard deviation for each status (boolean)
    for c in sdf[agg_var].unique():
        
        y_all = base_mean_df[X].astype(float).dropna().values
        
        sdf.loc[sdf[agg_var] == c,f'mean_{agg_str}'] = seg_mean_df.loc[(seg_mean_df[agg_var] == c), X].dropna().mean()
        sdf.loc[sdf[agg_var] == c,f'std_{agg_str}'] = seg_mean_df.loc[(seg_mean_df[agg_var] == c), X].dropna().std()
        
        for k in [0,1]:
        
            #Declaring the value by conversion status and agg_var
            x = agg_mean_df.loc[(agg_mean_df[agg_var] == c) & (agg_mean_df[boolean] == k), X ].astype(float).dropna().values
            
            #sdf.loc[(sdf[agg_var] == c) & (sdf[boolean] == k), 'ct'] = df.loc[(df[agg_var] == c) & (df[boolean] == k), 'ct'].sum()
            
            #Are the converted/non-converted means by agg_var different from the overall mean?

            tstat,pvalue = stats.ttest_ind(x,y_all,equal_var=False) ##T-Test

            sdf.loc[sdf[agg_var] == c, 'tstat'] = tstat 
            sdf.loc[sdf[agg_var] == c, 'pvalue'] = pvalue

            if np.abs(pvalue) > .025:
                sdf.loc[sdf[agg_var] == c, 'significance'] = 'Not Significant'
            else:
                sdf.loc[sdf[agg_var] == c, 'significance'] = 'Significant'
                
            #Are the converted/non-converted means by agg_var different from the agg_var mean?
            
            y_seg = seg_mean_df.loc[(seg_mean_df[agg_var] == c), X ].dropna().values  #Declaring the agg_var specific values for the t test
            
            tstat,pvalue = stats.ttest_ind(x,y_seg,equal_var=False)

            sdf.loc[sdf[agg_var] == c, f'tstat_{agg_str}'] = tstat
            sdf.loc[sdf[agg_var] == c, f'pvalue_{agg_str}'] = pvalue

            if np.abs(pvalue) > .025:
                sdf.loc[sdf[agg_var] == c, f'significance_{agg_str}'] = 'Not Significant'
            else:
                sdf.loc[sdf[agg_var] == c, f'significance_{agg_str}'] = 'Significant'
                
            #Are the converted/non-coverted means by agg_var different other conv/non-conv?
            
            y_conv = conv_mean_df.loc[(conv_mean_df[boolean] == k), X].dropna().values  #Declaring the agg_var specific values for the t test
            
            tstat,pvalue = stats.ttest_ind(x,y_conv,equal_var=False)

            sdf.loc[sdf[agg_var] == c, 'tstat_conv'] = tstat
            sdf.loc[sdf[agg_var] == c, 'pvalue_conv'] = pvalue

            if np.abs(pvalue) > .025:
                sdf.loc[sdf[agg_var] == c, 'significance_conv'] = 'Not Significant'
            else:
                sdf.loc[sdf[agg_var] == c, 'significance_conv'] = 'Significant'
                
            #Are the converted/non-coverted means by agg_var from each other by agg_var?
            
            y_comp = agg_mean_df.loc[(agg_mean_df[agg_var] == c) & (agg_mean_df[boolean] != k), X ].dropna()
                        
            tstat,pvalue = stats.ttest_ind(x,y_comp,equal_var=False)

            sdf.loc[sdf[agg_var] == c, 'tstat_comp'] = tstat
            sdf.loc[sdf[agg_var] == c, 'pvalue_comp'] = pvalue
            
            sdf.loc[(sdf[agg_var] == c) & (sdf[boolean] == k), 'mean_comp'] = y_comp.mean()
            sdf.loc[(sdf[agg_var] == c) & (sdf[boolean] == k), 'std_comp'] = y_comp.std()

            if np.abs(pvalue) > .025:
                sdf.loc[sdf[agg_var] == c, 'significance_comp'] = 'Not Significant'
            else:
                sdf.loc[sdf[agg_var] == c, 'significance_comp'] = 'Significant'
    
    project_id = 'project-1'
    
    schema = 'schema'
    
    sdf.rename(columns={boolean:"conversion_status"}, inplace=True)
    
    try:
        client.get_table(table)
        sdf.to_gbq(destination_table = f"{schema}.{table_name}", project_id=project_id,if_exists='append')
    except:
        sdf.to_gbq(destination_table = f"{schema}.{table_name}", project_id=project_id,if_exists='replace')

  cols=[
  #'sum_total_time_between_opp_and_app',
  'avg_total_time_between_opp_and_app',
  #'median_total_time_between_opp_and_app',
  #'sum_total_time_between_app_and_cf',
  'avg_total_time_between_app_and_cf',
  #'median_total_time_between_app_and_cf',
  #'sum_total_time_between_cf_and_acc',
  'avg_total_time_between_cf_and_acc',
  #'median_total_time_between_cf_and_acc',
  #'sum_total_time_between_acc_and_enr',
  'avg_total_time_between_acc_and_enr',
  #'median_total_time_between_acc_and_enr',
  #'sum_total_time_between_enr_and_str',
  'avg_total_time_between_enr_and_str',
  #'median_total_time_between_enr_and_str',
  'total_contacts_for_opp',
  'contacts_between_opp_and_app',
  'contacts_between_app_and_cf',
  'contacts_between_cf_and_acc',
  'contacts_between_acc_and_enr',
  'contacts_between_enr_and_str',
  #'contacts_before_app',
  #'contacts_before_cf',
  #'contacts_before_acc',
  #'contacts_before_enr',
  #'contacts_before_str',
  'avg_days_between_contacts',
  #'std_days_between_contacts',
  #'min_days_between_contacts',
  #'median_days_between_contacts',
  #'max_days_between_contacts',
  'avg_days_between_contacts_between_opp_and_app',
  'avg_days_between_contacts_between_app_and_cf',
  'avg_days_between_contacts_between_cf_and_acc',
  'avg_days_between_contacts_between_acc_and_enr',
  'avg_days_between_contacts_between_enr_and_str',
  #'avg_days_contact_before_app',
  #'avg_days_contact_before_cf',
  #'avg_days_contact_before_acc',
  #'avg_days_contact_before_enr',
  #'avg_days_contact_before_str',
  #'avg_days_contact_after_opp',
  #'avg_days_contact_after_app',
  #'avg_days_contact_after_cf',
  #'avg_days_contact_after_acc',
  #'avg_days_contact_after_enr',
  #'avg_days_contact_after_str'
  ]

  client = bigquery.Client()

  base_project='project-1'
  base_dataset='schema'
  base_table_list= ['prd_data_analysis_agg', 'prd_data_analysis_agg_90_day']
 
 # Looping over tables
  for b in base_table_list:
    
      # Pulling the appropriate pre-aggregated table
      full_name=f"{base_project}.{base_dataset}.{b}"

      print(f'Calling {full_name} for dataframe.')

      work_df = client.query(f'SELECT * FROM {full_name}').to_dataframe()

      conv_vars = ['Stage_1_conversion', 'Stage_2_conversion', 'Stage_3_conversion']

      metric_vars = ['avg_days_between_contacts_between_stg1_and_stg2','contacts_between_stg1_and_stg2','avg_total_time_between_stg1_and_stg2']

      project_id = 'project-2'
      schema = 'schema'
        
      if b == 'prd_data_analysis_agg':
          table_name = f"prd_inf_stats_analysis"
      elif b == 'prd_data_analysis_agg_90_day':
          table_name = f"prd_inf_stats_analysis_90_day"
            
      table = project_id+'.'+schema+'.'+table_name

      try:
        client.get_table(table)
        job = client.query(f'DELETE FROM {table} WHERE true;')
      except:
        print(f'{table} does not exist.')

      for c in conv_vars:

          print(f'Working on {c}')

          samp_df = sampling_dist_builder_sliced(df=work_df,n=500,cols=cols,groupby_col=['group_var'],conv_cols=[c])

          for m in metric_vars:

              print(f'Working on {m} for {c}')

              significance_test(df = samp_df,X = m ,boolean = c,agg_var='agg_var',conv_cols=c, table_name=table_name)
