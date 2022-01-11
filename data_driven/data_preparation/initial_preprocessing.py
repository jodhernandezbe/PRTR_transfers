# Importing libraries
from data_driven.data_preparation.rdkit_descriptors import information_for_set_of_chems
from data_engineering.extract.nlm_scraper import looking_for_structure_details as nlm
from data_engineering.extract.pubchem_scraper import looking_for_structure_details as pubchem
from data_driven.data_preparation.opening_dataset import opening_dataset

import random
from random import seed
import pandas as pd
import numpy as np
from scipy.stats import zscore, mode
import os
import pickle
import dask.dataframe as dd

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path
seed(1) # seed random number generator

def chemical_group_descriptors(group, generic_substance_id, grouping_type=1):
    '''
    Function to calculate the descriptors for chemical groups
    
    Options:

    (1) mean value without outliers (default)
    (2) mean value with outliers
    (3) median value
    (4) min value
    (5) max value
    (6) random value
    (7) random chemical
    (8) keep chemicals (keep all chemicals having non-null records (95%))
    '''

    if (grouping_type == 1) or (grouping_type == 2):
        if (group.shape[0] != 1) and (grouping_type == 1):
            # Removing outliers
            z_scores = zscore(group)
            abs_z_scores = z_scores.abs()
            # Filling NaN zscore values
            abs_z_scores.fillna(0, inplace=True)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            group = group[filtered_entries]
        group = group.mean(skipna=True)
    elif grouping_type == 3:
        group = group.median(skipna=True)
    elif grouping_type == 4:
        group = group.min(skipna=True)
    elif grouping_type == 5:
        group = group.max(skipna=True)
    elif grouping_type == 6:
        group.reset_index(drop=True, inplace=True)
        group = pd.Series(group.apply(lambda x: pd.Series(random.choice(x)), axis=0).to_dict('records')[0])
    elif grouping_type == 7:
        group = pd.Series(group.sample(1, random_state=0).to_dict('records')[0])
    elif grouping_type == 8:
        group = group[group.count(axis=1)/group.shape[1] >= 0.95]
        group['generic_substance_id'] = generic_substance_id

    return group



def looking_for_smiles(cas_number):
    '''
    Function to look for the SMILES using both the NLM and PubChem databases
    '''
    smiles = pubchem(cas_number)
    if not smiles:
        smiles = nlm(cas_number)

    return smiles


def obtaining_intervals(df, vals_for_intervals, flow_handling, save_info, id):
    '''
    Function to obtain the intervals for the flows
    '''

    num_different_elements = len(vals_for_intervals)
    #if flow_handling == 3:
    #    vals_for_intervals[-1] = vals_for_intervals[-1] + 2
    intervals = pd.DataFrame({'From': vals_for_intervals[0:num_different_elements-1],
                                'To': vals_for_intervals[1:]})
    intervals['Value'] = pd.Series(intervals.index.tolist()) + 1
    intervals = intervals.set_index(pd.IntervalIndex.from_arrays(intervals['From'], intervals['To'], closed='left'))['Value'] 
    df['transfer_amount_kg'] = df['transfer_amount_kg'].map(intervals)
    df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(object)
    
    # Saving equal-width intervals 
    intervals = intervals.reset_index()
    intervals.rename(columns={'index': 'Flow rate interval [kg]'}, inplace=True)
    if save_info == 'Yes':
        intervals.to_csv(f'{dir_path}/output/input_features/flow_handling_discretizer_id_{id}.csv',
                        index=False)

    return df


def transfer_flow_rates(df, id, flow_handling=1, number_of_intervals=10, save_info='No'):
    '''
    Function to organize the transfer flow rates

    Options:

    (1) Float values (default)
    (2) Integer values
    (3) m balanced intervals split by quantiles
    (4) m non-balanced equal-width intervals
    '''

    if flow_handling == 1:
        pass
    elif flow_handling == 2:
        df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(int)
    else:

        if flow_handling == 3:
            df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(int)
            quantiles = np.linspace(start=0, stop=1,
                                    num=number_of_intervals+1)
            quantile_values = df['transfer_amount_kg'].quantile(quantiles).astype(int).unique().tolist()
            df = obtaining_intervals(df, quantile_values,
                                flow_handling, save_info, id)
        else:
            df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(int)
            max_value = df['transfer_amount_kg'].max()
            linear = np.linspace(start=0,
                                stop=max_value+2,
                                num=number_of_intervals+1,
                                dtype=int).tolist()
            df = obtaining_intervals(df, linear, 
                    flow_handling, save_info, id)

    return df


def initial_data_preprocessing(logger, args):
    '''
    Function for a preliminary preprocessing of the data
    '''

    db_name = args.db_name
    including_groups = args.including_groups
    grouping_type = args.grouping_type
    cas_dict = {'chemical': 'chemical_in_category_cas',
                'substance': 'cas_number'}
    datasets = ['chemical', 'substance', 'record']

    df_chem = pd.DataFrame()
    for dataset in datasets:

        # Opening and/or creating the dataset
        logger.info(f' Fetching the needed information for the {dataset} dataset from the {db_name} database')

        if os.path.isfile(f'{dir_path}/output/data/raw/{dataset}.csv'):

            df = pd.read_csv(f'{dir_path}/output/data/raw/{dataset}.csv',
                            dtype={'generic_substance_id': object})
        else:

            df = opening_dataset(args, dataset)

            if (dataset in cas_dict.keys()):
                
                # Looking for SMILES
                logger.info(f' Looking for SMILES for compounds belonging to the {dataset}s list')
                df['smiles'] = df[cas_dict[dataset]].apply(lambda cas: looking_for_smiles(cas))

                # Looking for chemical descriptors
                df = df[pd.notnull(df['smiles'])]
                logger.info(f' Looking for descriptors for compounds belonging to the {dataset}s list')
                df = information_for_set_of_chems(cas_dict[dataset], df)
                df.reset_index(drop=True, inplace=True)
                df.drop(columns=['smiles'], inplace=True)

            # Saving information for further use and speeding up ML pipeline
            df.to_csv(f'{dir_path}/output/data/raw/{dataset}.csv', index=False, sep=',')

        # Organizing descriptors for chemicals belonging to the groups
        fcols = None
        icols = None
        if (dataset == 'chemical') and (including_groups):
            df.drop(columns=['chemical_in_category_cas'], inplace=True)
            descriptors = [col for col in df.columns if col != 'generic_substance_id']
            fcols = df[descriptors].select_dtypes('float').columns
            icols = df[descriptors].select_dtypes('integer').columns
            df = df.groupby(['generic_substance_id'], as_index=False)\
                            .apply(lambda group: chemical_group_descriptors(
                                    group[descriptors],
                                    group.generic_substance_id.unique()[0],
                                    grouping_type=grouping_type)
                                    )
            df_chem = pd.concat([df_chem, df], ignore_index=True, axis=0)
        elif (dataset == 'substance'):
            df.drop(columns=['cas_number'], inplace=True)
            if not fcols:
                descriptors = [col for col in df.columns if col != 'generic_substance_id']
                fcols = df[descriptors].select_dtypes('float').columns
                icols = df[descriptors].select_dtypes('integer').columns
            df_chem = pd.concat([df_chem, df], ignore_index=True, axis=0)

            # Dropping columns with a lot missing values
            to_drop = df_chem.columns[pd.isnull(df_chem).sum(axis=0)/df_chem.shape[0] > 0.8].tolist()
            df_chem.drop(columns=to_drop, inplace=True)

            # Missing values imputation
            to_impute = df_chem.columns[pd.isnull(df_chem).any(axis=0)].tolist()
            fcols_i = [col for col in to_impute if col in fcols]
            icols_i = [col for col in to_impute if col in icols]
            df_chem[fcols_i] = df_chem[fcols_i].fillna({fcols_i[idx]: val for idx, val in enumerate(np.nanmedian(df_chem[fcols_i].values, axis=0))})
            df_chem[fcols] = df_chem[fcols].round(2)
            df_chem[icols_i] = df_chem[icols_i].fillna({icols_i[idx]: val for idx, val in enumerate(mode(df_chem[icols_i].values, nan_policy='omit')[0])})
            df_chem[fcols] = df_chem[fcols].apply(pd.to_numeric, downcast='float')
            df_chem[icols] = df_chem[icols].apply(pd.to_numeric, downcast='integer')
            del to_impute, fcols_i, icols_i, fcols, icols
        elif (dataset == 'record'):

            # Keeping the column selected by the user as the model output
            if args.output_column == 'generic':
                df.drop(columns=['transfer_class_wm_hierarchy_name'],
                        inplace=True)
                target_colum = 'generic_transfer_class_id'
            else:
                df.drop(columns=['generic_transfer_class_id'],
                        inplace=True)
                target_colum = 'transfer_class_wm_hierarchy_name'

            # Data before 2005 or not (green chemistry and engineering boom!)
            if args.before_2005:
                pass
            else:
                df = df[df.reporting_year >= 2005]

            # Organazing transfers flow rates
            logger.info(' Organizing transfer flows')
            df = transfer_flow_rates(df, args.id,
                        flow_handling=args.flow_handling,
                        number_of_intervals=args.number_of_intervals,
                        save_info=args.save_info)

            df = df.sample(10000, random_state=0)

            # Grouping generic transfer classes
            if args.classification_type == 'multi-label classification':

                logger.info(f' Organizing the target column {target_colum} for multi-label classification')

                filepath = f'{dir_path}/output/data/raw/record_dataset_for_multi_label_classification.csv'
                if not os.path.isfile(filepath):
                    grouping_columns = ['reporting_year',
                                        'transfer_amount_kg',
                                        'generic_substance_id',
                                        'generic_sector_code',
                                        'prtr_system']
                    ddf = dd.from_pandas(df, npartitions=10)
                    metadata = df.dtypes.apply(lambda x: x.name).to_dict()
                    metadata = {col: metadata[col] for col in grouping_columns + [target_colum]}
                    df = ddf.groupby(grouping_columns).apply(multi_label_classification_target,
                                                            target_colum,
                                                              meta=metadata).compute(scheduler='processes').sort_index().reset_index(drop=True)
                    del ddf
                    df.to_csv(filepath, index=False, sep=',')
                else:
                    df = pd.read_csv(filepath, dtype={'generic_substance_id': object})

            # Obtaining the Environmental Policy Stringency Index (EPSI)
            logger.info(f' Adding the Environmental Policy Stringency Index')
            df_epsi = pd.read_csv(f'{dir_path}/../../ancillary/OECD_EPSI.csv', index_col=0).round(2)
            df_epsi.columns = [int(col) for col in df_epsi.columns]
            fun_epsi = lambda year, list_years, list_epsis: list_epsis[list_years.index(year)] if year in list_years else (list_epsis[np.argmin(list_years)] if year < 1990 else list_epsis[np.argmax(list_years)])
            df['epsi'] = df[['prtr_system', 'reporting_year']].apply(lambda row: fun_epsi(row['reporting_year'],
                                                                                        df_epsi.columns.tolist(),
                                                                                        df_epsi.loc[row['prtr_system']].tolist()),
                                                                    axis=1)
            del df_epsi

            # Obtaining the Gross Value Added (GVA)
            logger.info(f' Adding the Gross Value Added')
            df_to_search = df[['reporting_year', 'generic_sector_code', 'prtr_system']].drop_duplicates(keep='first')
            df_to_search.reset_index(drop=True, inplace=True)
            df_to_search['gva'] = df_to_search.apply(lambda row: round(getting_gva(row['reporting_year'],
                                                                            row['generic_sector_code'],
                                                                            row['prtr_system']
                                                                            ), 2),
                                                    axis=1)
            df_to_search['gva'] = df_to_search['gva'].astype('float32')
            df = pd.merge(df, df_to_search,
                        on=['reporting_year', 'generic_sector_code', 'prtr_system'],
                        how='left')
            del df_to_search            

            # Dropping columns that are not needed more
            df.drop(columns=['prtr_system', 'reporting_year'],
                        inplace=True)
            df_ml = pd.merge(df, df_chem, on='generic_substance_id', how='inner')
            df_ml.drop(columns=['generic_substance_id'], inplace=True)
            del df_chem

        del df

    return df_ml


def getting_gva(year, sector_code, prtr_system):
    '''
    Function to get the Gross Value Added (GVA)
    '''

    # Opening the GVA
    df_gva = pd.read_csv(f'{dir_path}/../../ancillary/OECD_GVA.csv',
                        usecols=['prtr_system',
                                'reporting_year',
                                'value_usd',
                                'isic_code',
                                'aggregate'])

    # Filtering the GVA by prtr_system
    df_gva = df_gva[df_gva.prtr_system == prtr_system]
    df_gva.drop(columns=['prtr_system'], inplace=True)

    # Filtering the GVA by the closest year
    df_gva['diff_year'] = df_gva['reporting_year'] - year
    min_difference = df_gva['diff_year'].min()
    df_gva = df_gva[df_gva['diff_year'] == min_difference]
    df_gva.drop(columns=['diff_year', 'reporting_year'], inplace=True)

    # Filtering the GVA by sector_code
    df_aux = df_gva[df_gva.isic_code == str(sector_code)]
    if df_aux.empty:

        df_aux = df_gva[df_gva.isic_code.str.contains(str(sector_code))]

        if df_aux.empty:

            df_isic_to_activity = pd.read_csv(f'{dir_path}/../../ancillary/isic_to_activity.csv')
            activity = df_isic_to_activity[df_isic_to_activity.isic_code == int(sector_code)].activity_code.values[0]
            del df_isic_to_activity
            
            df_aux = df_gva[df_gva.isic_code == str(activity)]

            if df_aux.empty:

                return df_gva.loc[df_gva.isic_code == 'TOT', 'value_usd'].values[0] / df_gva.loc[df_gva.isic_code == 'TOT', 'aggregate'].values[0]

            else:

                return df_aux['value_usd'].values[0] / df_aux['aggregate'].values[0]

        else:

            return (df_aux['value_usd'] / df_aux['aggregate']).mean()

    else:

        return (df_aux['value_usd'] / df_aux['aggregate']).mean()


def multi_label_classification_target(group, target_colum):
    '''
    Function to group the target column in the dataframe for multi-label classification
    '''

    if target_colum == 'generic_transfer_class_id':
        existing_classes = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']
    else:
        existing_classes = ['Disposal', 'Sewerage', 'Treatment', 'Energy recovery', 'Recycling']

    result = [1 if (group[target_colum] == class_t).any() else 0 for class_t in existing_classes]
    result = ' '.join([str(elem) for elem in result])

    group.drop(columns=[target_colum], inplace=True)
    group.drop_duplicates(keep='first', inplace=True)
    group.reset_index(drop=True, inplace=True)

    group[target_colum] = None
    group[target_colum].iloc[0] = result

    return group