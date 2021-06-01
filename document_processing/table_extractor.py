import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from collections import defaultdict

from .structure_extractor import StructureExtractor

TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y, \
BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, \
BOTTOM_LEFT_Y, TEXT = 'top_left_x', 'top_left_y', 'top_right_x', \
'top_right_y', 'bottom_right_x', 'bottom_right_y', \
'bottom_left_x', 'bottom_left_y', 'text'

__all__ = ['TableExtractor']

class TableExtractor:
    def __init__(self, document_filepath=None, endpoint=None, subscription_key=None, 
                 operation_url=None, ocr_outputs=None, api_type='azure', 
                 api='read', vertical_columns=None, horizontal_columns=None,
                 horizontal_keywords=None):
        self.ocr_outputs = ocr_outputs
        self.operation_url = operation_url
        self.vertical_columns = vertical_columns
        self.horizontal_columns = horizontal_columns
        self.horizontal_keywords = horizontal_keywords
        
        structure_extractor = StructureExtractor(
            document_filepath=document_filepath, 
            endpoint=endpoint, 
            subscription_key=subscription_key,
            operation_url = self.operation_url,
            ocr_outputs = self.ocr_outputs,
            api=api
        )
        self.word_dataframe = structure_extractor.word_dataframe
        self.ocr_outputs = structure_extractor.ocr_outputs
        self.is_scanned = structure_extractor.is_scanned
        # self.line_dataframe = structure_extractor.structure_extraction(self.line_dataframe)
        self.line_dataframe = structure_extractor.structure_extraction(structure_extractor.line_dataframe)
        self.ocr_outputs = structure_extractor.ocr_outputs
        self.operation_url = structure_extractor.operation_url

    def _close_matches(self, word, possibilities):
        lower_possibilities = [str(item).lower() for item in possibilities]    
        lower_word = str(word).lower()
        lower_possibilities_maxout = []
        [lower_possibilities_maxout.extend(j) for j in [i.split() for i in lower_possibilities]]        
        if len(lower_word.split()) and lower_word.split()[0] in [i.strip() for i in lower_possibilities_maxout]:
            present_bool = [lower_word in l.strip() for l in lower_possibilities]
            # if True not in present_bool:
            #     present_bool = [lower_word.split()[0] in l.strip() for l in lower_possibilities]
            if True in present_bool:
                match = lower_word  and possibilities[present_bool.index(True)]
                if match:
                    return word
        else:
            return None
    
    def _nearest_matches(self, given_value, possibilities):
        """
        *Author: Vaibhav Hiwase
        *Details: This function gives closest match number form possibilities 
                  of number such that the number given should be in a range 
                  of maximum difference between any two consecutive numbers 
                  in a sequence of possibilities.
        """
        absolute_difference_function = lambda list_value : abs(list_value - given_value)
        closest_value = min(possibilities, key=absolute_difference_function)
        if len(possibilities)==1 and abs(given_value-closest_value)  > 0.9:
            return None
        elif len(possibilities)==1 and abs(given_value-closest_value)  <= 0.9:
            return possibilities[0] 
        if abs(given_value-closest_value) < max(map(lambda x, y: abs(x-y), possibilities[1:], possibilities[:-1])):
            return closest_value
    
    def _get_key_val(self, x, horizontal_keywords=None):
        """
        *Author: Vaibhav Hiwase
        *Details: Functioning creating mapping of horizontal text splitted by
        any one word in horizantal_keyword.
        """
        if horizontal_keywords is None:
            horizontal_keywords = self.horizontal_keywords
        for i in horizontal_keywords:
            if x.startswith(i):
                return (i, x.replace(i, ''))
            elif x in horizontal_keywords:
                return x
        else:
            return None
    
    def _filter_val(self, x):
        """
        *Author: Vaibhav Hiwase
        *Details Filter for removing ":" 
        """
        if type(x)==tuple:
            return x and {x[0]: x[1].replace(':', '').strip()} 
        else:
            return x
        
    def _vertical_tables(self, line_dataframe, vertical_columns=None, horizontal_columns=None, horizontal_keywords=None):
        """
        *Author: Vaibhav Hiwase
        *Details: Mapping tables based on table column names
        """
        if vertical_columns is None:
            vertical_columns = self.vertical_columns
        if horizontal_columns is None:
            horizontal_columns = self.horizontal_columns
        if horizontal_keywords is None:
            horizontal_keywords = self.horizontal_keywords
        key_dict = {}
        MIN_TABLE_COLUMN = 2
        MIN_TABLE_PAGE_BREAK = 1
        table_count = 0
        prev_page = sorted(set(line_dataframe['page']))[0]
        line_dataframe['parent_table_number'] = None
        for para_num in sorted(set(line_dataframe['paragraph_number'])):
            df_para = line_dataframe[line_dataframe['paragraph_number']==para_num]
            df_para['horizontal_mapping'] = df_para[TEXT].apply(self._close_matches, possibilities=horizontal_columns)
            for line_num in sorted(set(df_para['line_number'])):
                temp_df = df_para[df_para['line_number'] ==line_num]
                temp_df['horizontal_mapping'] = temp_df[TEXT].apply(self._get_key_val, horizontal_keywords=horizontal_keywords)
                temp_df['horizontal_mapping'] = temp_df['horizontal_mapping'].apply(self._filter_val)
                df_para.loc[temp_df.index, 'horizontal_mapping'] = temp_df['horizontal_mapping'].tolist()    
            line_dataframe.loc[df_para.index, 'horizontal_mapping'] = df_para['horizontal_mapping'].tolist()    
            df_para['vertical_mapping'] = df_para[TEXT].apply(self._close_matches, possibilities=vertical_columns)
            line_dataframe.loc[df_para.index, 'vertical_mapping'] = df_para['vertical_mapping'].tolist()
            if abs(df_para['page'].unique()[0] - prev_page) > MIN_TABLE_PAGE_BREAK:
                key_dict = {}
            df_table = df_para[~pd.isna(df_para['vertical_mapping'])]
            if not df_table.empty and not any(pd.isna(df_table['table_number'])):
                df_table = df_para[df_para['table_number']==df_table['table_number'].mode().unique()[0]]
            key_df = df_table[~pd.isna(df_table['vertical_mapping'])]            
            if len(key_df) < 2:
                key_df = pd.DataFrame()
            if not key_df.empty :
                line = key_df['line_number'].unique()[0]        
                if all(~pd.isna(df_table[df_table['line_number']==line]['vertical_mapping'])):
                    key_dict = {}
                    table_count += 1
                    for k, v in zip(key_df['column'], key_df[TEXT]):
                        left_x_column = key_df[key_df['column']==k][TOP_LEFT_X].mean()
                        key_dict[k]=(v, left_x_column)
                    df_para.loc[key_df.index, 'parent_table_number']= table_count
                    line_dataframe.loc[key_df.index, 'parent_table_number']= table_count
                    prev_page = df_para['page'].unique()[0]           
            value_df = df_para[pd.isna(df_para['vertical_mapping'])]
            value_df = value_df[pd.isna(df_para['horizontal_mapping'])]            
            for table_num in sorted(set(value_df['table_number'])):
                if not key_dict:
                    break
                table = value_df[value_df['table_number'] == table_num]
                # define data
                table_row_left_x = []
                for column_num in sorted(set(table['column'])):
                    column = table[table['column']==column_num]
                    left_x_row = column[TOP_LEFT_X].mean()
                    table_row_left_x.append(left_x_row)                
                table_column_available = [None for _ in range(int(max(key_dict))+1)]
                for k, v in key_dict.items():
                    table_column_available[int(k)] = v[1]        
                table_column = list(filter(None, table_column_available))                
                table_row = [None for _ in range(len(table_column))]
                for given_value in table_row_left_x:
                    matched = self._nearest_matches(given_value=given_value, possibilities=table_column)
                    if matched:
                        matched_index = table_column.index(matched)
                        table_row[matched_index] = matched
                table_row_without_none = list(filter(None, table_row))
                if len(table_row_without_none) >= MIN_TABLE_COLUMN:
                    table_count+= 1
                    df_para.loc[table.index, 'parent_table_number']= table_count
                    line_dataframe.loc[table.index, 'parent_table_number']= table_count
                    prev_page = df_para['page'].unique()[0]       
        for page_num in sorted(set(line_dataframe['page'])):
            df_page = line_dataframe[line_dataframe['page']==page_num]
            #check first 20 line 
            first_five_lines = sorted(set(df_page['line_number']))[:20]
            extract_lines = []
            for line_num in first_five_lines:
                temparary_df = df_page[df_page['line_number']==line_num]
                if sorted(filter(None, temparary_df['vertical_mapping'].tolist())):
                    break
                extract_lines.append(line_num)           
            for line_num in extract_lines:
                temparary_df = df_page[df_page['line_number']==line_num]
                if min(temparary_df['column']):
                    df_page.loc[temparary_df.index, 'table_number']=None
                    line_dataframe.loc[temparary_df.index, 'table_number']=None
                else:
                    break
            starting_indexes = []
            for i in df_page.index:
                if pd.isna(df_page['table_number'][i]) and not df_page['is_header'][i]:
                    starting_indexes.append(i)
            if page_num > 1:
                prev_page_df = line_dataframe[line_dataframe["page"] == page_num-1]
                distinct_table_numbers = prev_page_df[~pd.isna(prev_page_df['table_number'])]['table_number'].tolist()
                if distinct_table_numbers:
                    line_dataframe.loc[starting_indexes, 'table_number']  = max(distinct_table_numbers)
        my_table_number = []
        prev = 0
        for table_num in line_dataframe['table_number']:
            if table_num < prev:
                my_table_number.append(None)
            else:
                my_table_number.append(table_num)
                prev = table_num
        line_dataframe['table_number'] = my_table_number    
        for table_num in sorted(set(line_dataframe['table_number'])):
            df_para = line_dataframe[line_dataframe['table_number']==table_num]
            key_df = df_para[~pd.isna(df_para['vertical_mapping'])]    
            parent_num_redefined = list(filter(None, key_df['parent_table_number'].unique()))
            if parent_num_redefined:
                parent_num_redefined = parent_num_redefined.pop(0)
            else:
                parent_num_redefined = None
            df_para.loc[key_df.index, 'parent_table_number'] = parent_num_redefined
            line_dataframe.loc[key_df.index, 'parent_table_number'] = parent_num_redefined
        tab_number = 0
        for tab_num in sorted(set(line_dataframe[~pd.isna(line_dataframe['table_number'])]["table_number"])):
            tab_df = line_dataframe[line_dataframe['table_number'] == tab_num]
            val = tab_df[tab_df['sum_of_column_up_space']==max(tab_df['sum_of_column_up_space'])]['table_identifier'].iloc[0,]
            truth_values = tab_df['table_identifier']==val
            truth_values = truth_values.tolist()
            flag = truth_values and truth_values.pop(0)
            table_number = []
            while flag :
                tab_number += 1
                while flag is True:
                    flag = truth_values and truth_values.pop(0)
                    table_number.append(tab_number)
                while flag is False:
                    flag = truth_values and truth_values.pop(0)
                    table_number.append(tab_number)                                
            if table_number:
                line_dataframe.loc[tab_df.index, 'table_number2'] = table_number                
        line_dataframe['table_number'] = line_dataframe['table_number2']
        for table_number in sorted(set(line_dataframe['table_number'])):
            table_dataframe = line_dataframe[line_dataframe['table_number']==table_number]
            key_dict = defaultdict(list)
            for k in zip(sorted(set(table_dataframe['column']))):
                left_x_column = table_dataframe[table_dataframe['column']==k][TOP_LEFT_X].mean()
                index = table_dataframe[table_dataframe['column']==k].index.tolist()
                key_dict[left_x_column].extend(index)
            key_dict = dict(key_dict)
            key_dict.values()    
            key = list(key_dict.keys())
            X = np.array(key).reshape(-1,1)
            model = MeanShift(n_jobs=-1)
            if np.any(X):
                xhat = model.fit_predict(X)
                my_tuple = []
                for k, xkey in zip(key, xhat):
                    my_tuple.append((k, xkey, key_dict[k]))    
                my_tuple = sorted(my_tuple, key=lambda x: x[0])    
                my_final_dict = defaultdict(list)
                for i, my_tup in enumerate(my_tuple):
                    k, xkey, klist = my_tup
                    my_final_dict[xkey].extend(klist)        
                my_dict = {}
                for i, v in enumerate(my_final_dict.values()):
                    my_dict[i] = v        
                for col, index in my_dict.items():
                    table_dataframe.loc[index, 'column'] = col    
                line_dataframe.loc[table_dataframe.index, 'column'] = table_dataframe['column'].tolist()
        line_data = line_dataframe[line_dataframe['is_header']==False]
        line_data = line_data[line_data['is_footer']==False]
        table_dictionary = {}
        for parent_num in sorted(filter(None, set(line_data['parent_table_number']))):
            df_parent = line_data[line_data['parent_table_number']==parent_num]
            key_df = df_parent[~pd.isna(df_parent['vertical_mapping'])]
            if len(key_df) < 2:
                key_df = pd.DataFrame()
            if not key_df.empty :
                line = key_df['line_number'].unique()[0]        
                if all(~pd.isna(line_data[line_data['line_number']==line]['vertical_mapping'])):
                    key_dict = {}
                    for k, v in zip(key_df['column'], key_df[TEXT]):
                        left_x_column = key_df[key_df['column']==k][TOP_LEFT_X].mean()
                        key_dict[k]=(v, left_x_column)                        
                    my_dict = {}
                    for column in sorted(set(key_df['column'])):
                        col_df = key_df[key_df['column']==column]
                        left_x_column = col_df[TOP_LEFT_X].mean()
                        text = ' '.join(col_df[TEXT].tolist())
                        my_dict[left_x_column]=text           
            value_df = df_parent[pd.isna(df_parent['vertical_mapping'])]
            for para_num in sorted(set(value_df['paragraph_number'])):
                df_para = value_df[value_df['paragraph_number']==para_num]
                for table_num in sorted(set(df_para['table_number'])):
                    if not key_dict:
                        break
                    table = line_data[line_data['table_number'] == table_num]
                    # define data
                    table_row_left_x = []
                    table_row_text = []
                    for column_num in sorted(set(table['column'])):
                        column = table[table['column']==column_num]
                        left_x_row = column[TOP_LEFT_X].mean()
                        cell_text = ' '.join(column[TEXT].tolist())
                        table_row_left_x.append(left_x_row)
                        table_row_text.append(cell_text)                    
                    table_column_available = [None for _ in range(int(max(key_dict))+1)]
                    for k, v in key_dict.items():
                        table_column_available[int(k)] = v[1]        
                    table_column = list(filter(None, table_column_available))                    
                    table_row = [None for _ in range(len(table_column))]
                    for given_value in table_row_left_x:
                        matched = self._nearest_matches(given_value=given_value, possibilities=table_column)            
                        if matched:
                            matched_index = table_column.index(matched)
                            table_row[matched_index] = matched
                    table_row_without_none = list(filter(None, table_row))
                    table_row_dict = {}
                    for text, left_x in zip(table_row_text, table_row_without_none):
                        key = my_dict[left_x]
                        value = text
                        table_row_dict[key]= value
                    for col_name in my_dict.values():
                        if col_name not in table_row_dict.keys():
                            table_row_dict[col_name] = ''    
                    table_dictionary[parent_num] = table_row_dict                    
        return table_dictionary, line_dataframe                
    
    def _horizontal_tables(self, line_dataframe, horizontal_columns=None, horizontal_keywords=None):
        """
        *Author: Vaibhav Hiwase
        *Details: Mapping tables based on horizontal alignment in horizontal_columns and horizontak_keywords
        """

        if horizontal_columns is None:
            horizontal_columns = self.horizontal_columns
        if horizontal_keywords is None:
            horizontal_keywords = self.horizontal_keywords
        horizontal_mapped_dict = {}
        for para_num in sorted(set(line_dataframe['paragraph_number'])):
            df_para = line_dataframe[line_dataframe['paragraph_number']==para_num]    
            df_para['horizontal_mapping'] = df_para[TEXT].apply(self._close_matches, possibilities=horizontal_columns)
            for line_num in sorted(set(df_para['line_number'])):
                temp_df = df_para[df_para['line_number'] ==line_num]
                temp_df['horizontal_mapping'] = temp_df[TEXT].apply(self._get_key_val, horizontal_keywords=horizontal_keywords)
                temp_df['horizontal_mapping'] = temp_df['horizontal_mapping'].apply(self._filter_val)
                df_para.loc[temp_df.index, 'horizontal_mapping'] = temp_df['horizontal_mapping'].tolist()    
            horizontal_mapping_lines = sorted(set(df_para[~pd.isna(df_para['horizontal_mapping'])]['line_number']))    
            for line_num in horizontal_mapping_lines :
                temp_df = df_para[df_para['line_number'] ==line_num]
                key_df = temp_df[~pd.isna(temp_df['horizontal_mapping'])]
                value_df = temp_df[pd.isna(temp_df['horizontal_mapping'])]
                if not key_df.empty and not value_df.empty:
                    key = ' '.join(key_df[TEXT].tolist())
                    value = ' '.join(value_df[TEXT].tolist())
                elif not key_df.empty and value_df.empty:
                    if key_df['horizontal_mapping'].tolist() and type(key_df['horizontal_mapping'].tolist()[0]) == dict:
                        item = key_df['horizontal_mapping'].tolist()[0]
                        for k, v in item.items():
                            key = k
                            value = v
                    else:
                        key = ' '.join(key_df[TEXT].tolist())
                        value = ''
                else:
                    continue 
                key_value_dict = {}
                key_value_dict[key]=value    
                horizontal_mapped_dict[line_num] = key_value_dict
        return horizontal_mapped_dict, line_dataframe
    
    def table_extraction(self, line_dataframe=None):
        """
        *Author: Vaibhav Hiwase
        *Details: Extracting tables of vertical mapping and horizontal mapping.
        """
        if line_dataframe is False:
            return {}
        if line_dataframe is None:
            line_dataframe = self.line_dataframe.copy()
        table_dictionary, line_dataframe = self._vertical_tables(line_dataframe)
        horizontal_line_dictionary, line_dataframe = self._horizontal_tables(line_dataframe)
        prev_parent_table_no = None
        for line_num, parent_table_no in zip(line_dataframe['line_number'], line_dataframe['parent_table_number']):
            if parent_table_no:
                prev_parent_table_no = parent_table_no
            elif prev_parent_table_no and line_num in horizontal_line_dictionary.keys():
                table_dictionary[prev_parent_table_no].update(horizontal_line_dictionary[line_num])
        count = 0
        my_table_dict = {}
        for k, v in table_dictionary.items():
            my_table_dict[count] = v
            count += 1
        return my_table_dict


if __name__ == '__main__':
    import json 
    endpoint = ''
    subscription_key = ''
    document_filepath = ''
    json_path = '/'.join(document_filepath.split('/')[:-1]) + '/' + document_filepath.split('/')[-1].split('.')[0] + '.json'    
    VERTICAL_COLUMNS=[]    
    HORIZONTAL_COLUMNS=[]
    HORIZONTAL_KEYWORDS=[]
    table_extractor = TableExtractor(
        document_filepath=document_filepath, 
        endpoint=endpoint, 
        subscription_key=subscription_key,
        operation_url = None,
        ocr_outputs = None,
        api_type='azure', 
        api='read',vertical_columns=VERTICAL_COLUMNS,
        horizontal_columns=HORIZONTAL_COLUMNS,
        horizontal_keywords=HORIZONTAL_KEYWORDS
    )
    word_dataframe = table_extractor.word_dataframe
    line_dataframe = table_extractor.line_dataframe
    ocr_outputs = table_extractor.ocr_outputs
    is_scanned = table_extractor.is_scanned
    table_dict = table_extractor.table_extraction()    
    with open(json_path, 'w') as f:
        f.write(json.dumps(table_dict,indent=4, ensure_ascii=False, sort_keys=False))

