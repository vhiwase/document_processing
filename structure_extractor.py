import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, AgglomerativeClustering, estimate_bandwidth

from .azure_api import AzureOCR

TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y, \
BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, \
BOTTOM_LEFT_Y, TEXT = 'top_left_x', 'top_left_y', 'top_right_x', \
'top_right_y', 'bottom_right_x', 'bottom_right_y', \
'bottom_left_x', 'bottom_left_y', 'text'

__all__ = ['StructureExtractor']

class StructureExtractor:
    def __init__(self, document_filepath, endpoint=None, subscription_key=None, operation_url=None, ocr_outputs=None, api_type='azure', api='read'):
        self.ocr_outputs = ocr_outputs
        self.operation_url = operation_url
        if api_type=='azure':
            azure_ocr = AzureOCR(
                document_filepath=document_filepath, 
                endpoint=endpoint, 
                subscription_key=subscription_key,
                operation_url = self.operation_url,
                ocr_outputs = self.ocr_outputs,
                api=api
            )
            read_api_ocr = azure_ocr.get_api_ocr()
            self.word_dataframe = read_api_ocr.word_dataframe
            self.line_dataframe = read_api_ocr.line_dataframe
            self.is_scanned = read_api_ocr.is_scanned
            self.ocr_outputs = read_api_ocr.ocr_outputs
            self.operation_url = read_api_ocr.operation_url
        else:
            self.word_dataframe = None
            self.line_dataframe = None
            self.is_scanned = None
            self.document_binaries = None
    
    def calculating_paragraph_and_column_per_page(self, line_dataframe, page_number):
        """
        *Author: Vaibhav Hiwase
        *Details: Creating paragraph attribute for calculating paragraph number of the text
                  present in given dataframe using clustering on coordiantes.
        """
        MIN_LINE_SPACE = 0.09
        line_dataframe = line_dataframe.reset_index(drop=True)        
        # Operation on page
        page_df = line_dataframe[line_dataframe['page']==page_number]
        # Calculating vertical text
        page_df['x_diff'] = page_df[TOP_RIGHT_X]-page_df[TOP_LEFT_X]
        page_df['y_diff'] = page_df[TOP_RIGHT_Y]-page_df[TOP_LEFT_Y]
        temp_page_df = page_df[page_df['x_diff']==0]    
        v_df = pd.DataFrame(index=temp_page_df[TOP_LEFT_X], columns=[TEXT, 'line_number'])
        v_df[TEXT] = temp_page_df[TEXT].tolist()
        v_df['line_number'] = temp_page_df['line_number'].tolist()    
        my_line_num_text_dict = v_df.T.to_dict()
        page_df.loc[temp_page_df.index, 'vertical_text_lines'] = [my_line_num_text_dict for _ in range(len(temp_page_df))]
        line_dataframe.loc[temp_page_df.index, 'vertical_text_lines'] = [my_line_num_text_dict for _ in range(len(temp_page_df))]    
        dd = pd.DataFrame(index = temp_page_df.index)
        dd[TOP_LEFT_X] = temp_page_df[TOP_RIGHT_X].tolist()
        dd[TOP_LEFT_Y] = temp_page_df[TOP_RIGHT_Y].tolist()    
        dd[TOP_RIGHT_X] = temp_page_df[BOTTOM_RIGHT_X].tolist()
        dd[TOP_RIGHT_Y] = temp_page_df[BOTTOM_RIGHT_Y].tolist()    
        dd[BOTTOM_RIGHT_X] = temp_page_df[BOTTOM_LEFT_X].tolist()
        dd[BOTTOM_RIGHT_Y] = temp_page_df[BOTTOM_LEFT_Y].tolist()    
        dd[BOTTOM_LEFT_X] = temp_page_df[TOP_LEFT_X].tolist()
        dd[BOTTOM_LEFT_Y] = temp_page_df[TOP_LEFT_Y].tolist()
        if not dd.empty:
            dd[TOP_LEFT_X] = min(dd[TOP_LEFT_X])
        page_df.loc[dd.index, [TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y,
           BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, BOTTOM_LEFT_Y]] = dd.loc[dd.index, [TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y,
           BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, BOTTOM_LEFT_Y]]                                                                                                              
        line_dataframe.loc[dd.index, [TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y,
           BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, BOTTOM_LEFT_Y]] = dd.loc[dd.index, [TOP_LEFT_X, TOP_LEFT_Y, TOP_RIGHT_X, TOP_RIGHT_Y,
           BOTTOM_RIGHT_X, BOTTOM_RIGHT_Y, BOTTOM_LEFT_X, BOTTOM_LEFT_Y]]
        # Assigning approprate value for coordinated belonging to same line
        for li in sorted(set(page_df.line_number)):
            df_li = page_df[page_df['line_number']==li]
            page_df.loc[df_li.index, BOTTOM_RIGHT_Y] = max(df_li[BOTTOM_RIGHT_Y])
            page_df.loc[df_li.index, TOP_LEFT_Y] = min(df_li[TOP_LEFT_Y])
            page_df.loc[df_li.index, BOTTOM_LEFT_Y] = max(df_li[BOTTOM_LEFT_Y])
            page_df.loc[df_li.index, TOP_RIGHT_Y] = min(df_li[TOP_RIGHT_Y])
        # Calculating y-coordinates space above and below line
        page_df['bottom'] = [0] + page_df[BOTTOM_RIGHT_Y].tolist()[:-1]
        page_df['up_space'] = page_df[TOP_LEFT_Y] - page_df['bottom']
        page_df['down_space'] = page_df['up_space'][1:].tolist()+ [0]    
        # Assigning approprate value for coordinated belonging to same line
        for li in sorted(set(page_df.line_number)):
            df_li = page_df[page_df['line_number']==li]
            page_df.loc[df_li.index, 'up_space'] = max(df_li['up_space'])
            page_df.loc[df_li.index, 'down_space'] = max(df_li['down_space'])        
        # Filter for eliminating large bottom blank space before clustering
        page_df1 = page_df[page_df['up_space'] < 1.8]
        page_df2 = page_df[page_df['up_space'] >= 1.8]    
        if page_df1.empty:
            return line_dataframe    
        # MeanShift Clustering in space between two lines
        X = np.array(page_df1.loc[:, ['up_space']])
        model = MeanShift(n_jobs=-1)    
        # fit model and predict clusters
        yhat = model.fit_predict(X)    
        # Adding -1 cluster number for ignored words below large bottom blank space
        page_df['yhat'] = list(yhat) + [-1 for _ in range(len(page_df2))]    
        # Sorting clustering number bases on upper space of line
        page_df = page_df.sort_values(by=['up_space'])
        # Reordering clustering in ascending order based on height of upper blank space of line
        yhat_ascending_sequence = []
        count = 0
        prev_cluster_no = page_df['yhat'].tolist() and page_df['yhat'].tolist()[0]
        for cluster_no in page_df['yhat']:
            if prev_cluster_no != cluster_no:
                count += 1
            yhat_ascending_sequence.append(count)
            prev_cluster_no = cluster_no    
        page_df['yhat'] = yhat_ascending_sequence
        page_df = page_df.sort_index()    
        # Creating paragraph sequence by combining 0 with non-zerp values and lines whose upper space is less than MIN_LINE_SPACE
        paragraph_seq = []
        count = 0
        prev_line = page_df['line_number'].tolist() and page_df['line_number'].tolist()[0]
        for y, line, up_space in zip(page_df['yhat'], page_df['line_number'], page_df['up_space']):
            if y and line != prev_line:
                if up_space > MIN_LINE_SPACE:
                    count += 1
            prev_line = line
            paragraph_seq.append(count)
        # Adding paragraph number and sorting results
        page_df['paragraph'] = paragraph_seq
        page_df= page_df.sort_values(by=['line_number', TOP_LEFT_X])
        # MeanShift Clustering in top left x coordinates
        X = np.array(page_df.loc[:, [TOP_LEFT_X]])
        bandwidth = estimate_bandwidth(X, quantile=0.16, n_samples=500, n_jobs=-1)
        if bandwidth:
            model = MeanShift(bandwidth=bandwidth, n_jobs=-1)
        else:
            model = MeanShift(n_jobs=-1)
        xhat = model.fit_predict(X)
        cluster_centers = model.cluster_centers_
        page_df['xhat'] = xhat     
        # Sorting clustering number bases on Top left x of line
        page_df = page_df.sort_values(by=[TOP_LEFT_X])    
        # Reordering clustering in ascending order based on height of upper blank space of line
        xhat_ascending_sequence = []
        count = 0
        prev_cluster_no = page_df['xhat'].tolist() and page_df['xhat'].tolist()[0]
        for cluster_no in page_df['xhat']:
            if prev_cluster_no != cluster_no:
                count += 1
            xhat_ascending_sequence.append(count)
            prev_cluster_no = cluster_no
        
        page_df['column'] = xhat_ascending_sequence
        page_df = page_df.sort_index()    
        # Assignment of value to line_dataframe
        line_dataframe.loc[page_df.index, 'up_space'] = page_df['up_space']
        line_dataframe.loc[page_df.index, 'down_space'] = page_df['down_space']
        line_dataframe.loc[page_df.index, 'xhat'] = page_df['xhat']
        line_dataframe.loc[page_df.index, 'yhat'] = page_df['yhat']
        line_dataframe.loc[page_df.index, 'paragraph'] = page_df['paragraph']
        line_dataframe.loc[page_df.index, 'column'] = page_df['column']        
        return line_dataframe

    def paragraph_extraction(self, line_dataframe=None):  
        """
        *Author: Vaibhav Hiwase
        *Details: Creating paragraph number in line_dataframe.
        """
        if line_dataframe is None:
            line_dataframe = self.line_dataframe
        line_dataframe ['vertical_text_lines'] = None
        for page_number in sorted(set(line_dataframe ['page'])):    
            line_dataframe = self.calculating_paragraph_and_column_per_page(line_dataframe , page_number)
        # Calculating paragraph_number column for complete PDF
        paragraph_number = []
        count = 0
        prev_para_num = line_dataframe['paragraph'].tolist() and line_dataframe['paragraph'].tolist()[0]
        for para_num in line_dataframe['paragraph']:
            if para_num==prev_para_num or pd.isna(para_num):
                pass
            else:
                count += 1
                prev_para_num = para_num        
            paragraph_number.append(count)
        line_dataframe['paragraph_number'] = paragraph_number
        return line_dataframe
        
    def structure_extraction(self, line_dataframe=None):
        """
        *Author: Vaibhav Hiwase
        *Details: Identifying page header, page footer, 
        table rows (i.e `table_number` attribute) and 
        table columns (i.e `column` attribute)
        """
        if line_dataframe is None:
            line_dataframe = line_dataframe
        line_dataframe = self.paragraph_extraction(line_dataframe)
        # Calculating table identified in a paragraph
        for para_num in sorted(set(line_dataframe['paragraph_number'])):
            df_para = line_dataframe[line_dataframe['paragraph_number']==para_num]
            for col in sorted(set(df_para[~pd.isna(df_para['column'])]['column'])):
                col_df = df_para[df_para['column']==col]
                col_df['column_up_space'] = col_df[TOP_LEFT_Y].diff().tolist()
                df_para.loc[col_df.index, 'column_up_space'] = col_df['column_up_space'].tolist()
                line_dataframe.loc[col_df.index, 'column_up_space'] = col_df['column_up_space'].tolist()
                
        df_nan = line_dataframe[pd.isna(line_dataframe['column_up_space'])]
        df_nan = df_nan.sort_values(by=['column'])
        df_nan["column_up_space"] = df_nan[TOP_LEFT_Y].diff()
        df_nan["column_up_space"] = df_nan["column_up_space"].apply(lambda x: abs(x))
        line_dataframe.loc[df_nan.index, 'column_up_space'] = df_nan['column_up_space'].tolist()       
        prev_para_num = sorted(set(line_dataframe['paragraph_number'])) and sorted(set(line_dataframe['paragraph_number']))[0]
        for para_num in sorted(set(line_dataframe['paragraph_number'])):
            df_para = line_dataframe[line_dataframe['paragraph_number']==para_num]    
            for line in sorted(set(df_para['line_number'])):
                temp_df = df_para[df_para['line_number']==line]
                my_sum = 0
                for val in temp_df['column_up_space']:
                    if not pd.isna(val):
                        my_sum+=val
                df_para.loc[temp_df.index, 'sum_of_column_up_space'] = my_sum * len(temp_df) + len(temp_df)*100
                line_dataframe.loc[temp_df.index, 'sum_of_column_up_space'] = my_sum * len(temp_df) + len(temp_df)*100
        # Identify Table Rows
        for page in sorted(set(line_dataframe['page'])):
            df_page = line_dataframe[line_dataframe['page']==page]
            # min_col = min(df_page['column'])
            X = np.array(df_page.loc[:, ['sum_of_column_up_space']])
            if len(X) != 1:
                model = AgglomerativeClustering(n_clusters=2)
                # fit model and predict clusters
                yhat = model.fit_predict(X)
                df_page['table_identifier'] = yhat
                line_dataframe.loc[df_page.index, 'table_identifier'] = yhat
            row = []
            count = 0
            table_starter = df_page[df_page['sum_of_column_up_space']==max(df_page['sum_of_column_up_space'])]['table_identifier'].unique()[0]
            first_identifier = df_page['table_identifier'].tolist() and df_page['table_identifier'].tolist()[0]
            prev_identifier = df_page['table_identifier'].tolist() and df_page['table_identifier'].tolist()[0]
            flag = True
            for identifier in df_page['table_identifier']:
                if pd.isna(identifier):
                    row.append(identifier)
                    continue
                if flag :
                    if identifier == table_starter:
                        flag = False
                        count += 1
                elif prev_identifier != identifier:
                    if identifier != first_identifier:
                        count += 1
                prev_identifier = identifier
                row.append(count)
            df_page['row'] = row
            starting_table_identifier = df_page[df_page['sum_of_column_up_space'] == max(df_page['sum_of_column_up_space'])]['table_identifier'].unique()[0]
            for r in sorted(set(row)):
                df_page_row = df_page[df_page['row'] == r]
                # starting_table_identifier = df_page_row[df_page_row['sum_of_column_up_space'] == max(df_page_row['sum_of_column_up_space'])]['table_identifier'].unique()[0]
                table_expected_column_table_identifier = set(df_page_row[df_page_row['table_identifier']== starting_table_identifier]['column'])
                table_column_checker = df_page_row[df_page_row['table_identifier'] != starting_table_identifier]['column']
                vertical_text_lines = df_page_row[df_page_row['table_identifier'] != starting_table_identifier]['vertical_text_lines']
                for index, column_no, vertical_text_line in zip(table_column_checker.index, table_column_checker, vertical_text_lines):
                    if not table_expected_column_table_identifier:
                        df_page.loc[index, 'row'] = -1
                    elif column_no not in table_expected_column_table_identifier:
                        if pd.isna(vertical_text_line):
                            df_page.loc[index, 'row'] = None
            line_dataframe.loc[df_page.index, 'row'] = df_page['row'].tolist()
        table_number = []
        count = 0
        flag = False
        prev_row = line_dataframe['row'].tolist() and line_dataframe['row'].tolist()[0]
        prev_page = line_dataframe['page'].tolist() and line_dataframe['page'].tolist()[0]
        for r, p in zip(line_dataframe['row'], line_dataframe['page']):
            if flag:
                if pd.isna(r) or (prev_row == r and prev_page == p):
                    flag = True
                    table_number.append(None)
                else:
                    flag = False
                    count += 1
                    table_number.append(count)
                    prev_row = r
                    prev_page = p
                continue
            if pd.isna(r):
                table_number.append(None)
                flag = True
                continue
            if r != prev_row and r != -1:
                count += 1
            if not pd.isna(r):
                prev_row = r
            table_number.append(count)
            prev_page = p
        line_dataframe['table_number']=table_number
        # Identifying header and footers by Clustering
        header_para = []
        footer_para = []
        for page in sorted(set(line_dataframe['page'])):
            page_df = line_dataframe[line_dataframe['page']==page]
            page_numbers = sorted(page_df['paragraph_number'])
            if not page_numbers:
                continue
            elif len(page_numbers)==1:
                header_para.append(page_numbers[0])
            else:
                header_para.append(page_numbers[0])
                footer_para.append(page_numbers[-1])
        header_df = pd.DataFrame()
        for h_para in header_para:
            h_df = line_dataframe[line_dataframe['paragraph_number']==h_para]
            header_df = header_df.append(h_df)
        # MeanShift Clustering in space between two lines
        X = np.array(header_df.loc[:, [TOP_LEFT_X, TOP_RIGHT_X, BOTTOM_RIGHT_X, BOTTOM_LEFT_X]])
        model = MeanShift(n_jobs=-1)
        hhat = model.fit_predict(X)
        cluster_centers = model.cluster_centers_
        header_df['header_clusters'] = hhat 
        header_cluster_number_list = header_df['header_clusters'].mode().tolist()
        header_cluster_number = header_cluster_number_list and sorted(header_cluster_number_list)[0]
        header_df = header_df[header_df["header_clusters"]==header_cluster_number]
        line_dataframe['is_header'] = False
        line_dataframe.loc[header_df.index, 'is_header'] = True
        footer_df = pd.DataFrame()
        for f_para in footer_para:
            f_df = line_dataframe[line_dataframe['paragraph_number']==f_para]
            footer_df = footer_df.append(f_df)                
        # MeanShift Clustering in space between two lines
        X = np.array(footer_df.loc[:, [TOP_LEFT_X, TOP_RIGHT_X, BOTTOM_RIGHT_X, BOTTOM_LEFT_X]])
        model = MeanShift(n_jobs=-1)
        fhat = model.fit_predict(X)
        cluster_centers = model.cluster_centers_
        footer_df['footer_clusters'] = fhat 
        footer_cluster_number_list = footer_df['footer_clusters'].mode().tolist()
        footer_cluster_number = footer_cluster_number_list and sorted(footer_cluster_number_list)[0]    
        footer_df = footer_df[footer_df['footer_clusters'] == footer_cluster_number]
        line_dataframe['is_footer'] = False
        line_dataframe.loc[footer_df.index, 'is_footer'] = True
        return line_dataframe

    # def structure_extraction_combine(self, line_dataframe=None):
    #     if line_dataframe is None:
    #         line_dataframe = self.line_dataframe
    #     line_dataframe = self.structure_extraction(line_dataframe)        
    #     for page in sorted(set(line_dataframe['page'])):
    #         page_df = line_dataframe[line_dataframe["page"] == page]
    #         starting_indexes = []
    #         for i in page_df.index:
    #             if pd.isna(page_df['table_number'][i]) and not page_df['is_header'][i]:
    #                 starting_indexes.append(i)
    #         if page > 1:
    #             prev_page_df = line_dataframe[line_dataframe["page"] == page-1]
    #             distinct_table_numbers = prev_page_df[~pd.isna(prev_page_df['table_number'])]['table_number'].tolist()
    #             if distinct_table_numbers:
    #                 line_dataframe.loc[starting_indexes, 'table_number']  = max(distinct_table_numbers)
    #         self.line_dataframe = line_dataframe
    #     return line_dataframe
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            