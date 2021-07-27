from collections import namedtuple
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from difflib import get_close_matches 

import requests
import time

endpoint = 'https://technomilecognitivieservices.cognitiveservices.azure.com/'

subscription_key = '0766c7b636c04a2fad9c90d68f008b8d'

text_recognition_url = endpoint + "/vision/v3.1/read/analyze"
# text_recognition_url = endpoint + "/vision/v3.1/read/syncAnalyze"

CHECK_N_PAGES = 7
PAGE_SPLIT =  None
NUM_LINE_HEADER = 5
NUM_LINE_FOOTER = 5
HEADER_FOOTER_THRESHOLD = 10

vertical_possibilities=['CLIN', 'ITEM NO', 'SUPPLIES/SERVICES', 'ESTIMATED', 'QUANTITY', 'ESTIMATED QUANTITY', 'EST.', 'AMOUNT', 'EST. AMOUNT', 'UNIT', 'UNIT PRICE', 'DESCRIPTION OF SUPPLIES/SERVICES',
                        "Section", "F Item #", 'PWS', 'Deliverable', 'Description', 'Due Date', 'Addressee and', 'F Item #', 'cross-', 'Number/type', 'reference', 'of copies']
horizontal_possibilities = ['NET AMT', 'ESTIMATED COST', 'FIXED FEE', 'TOTAL EST COST + FEE']
horizontal_in_word_keys = ['ACRN', 'CIN']


def line_numbers(df:pd.DataFrame, by_pixel_value:int = 16.6) -> pd.DataFrame:
    """
    *Author: Vaibhav
    *Details: Logic for getting accurate line number irrespective of any noise 
              in word coordinates coming from hocr file.     
              Input: DataFrame
              Output: Updated DataFrame with a new constants.MISC_LINE_NUMBER column in it.
    *@param: DataFrame, float
    *@return: DataFrame
    """
    is_scanned = True
    for index in range(1,9):
        df_type = df.iloc[:,index].dtype
        if df_type == float:
            continue
        else:
            break
    else:
        is_scanned = False

    if is_scanned:
        by_pixel_value = by_pixel_value
    else:
        by_pixel_value = 0.048

    merged_dataframe = pd.DataFrame()
    for page in sorted(set(df['page'])):
        df_per_page = df[df['page'] == page]
        df_per_page = df_per_page.sort_values(by=['Top_left_y', 'Top_left_x'], ascending = [True, True]).reset_index(drop = True)    
        # y_avg = df_per_page.loc[: , [constants.MISC_LEFT_Y,constants.MISC_RIGHT_Y]]
        y_avg = df_per_page.loc[: , ["Top_left_y"]]
        data = pd.DataFrame()
        data['Top_y_avg'] = y_avg.mean(axis=1)
        data['Top_y_difference'] = data['Top_y_avg'].diff()
        data['IS_PIXEL'] = data['Top_y_difference'] > by_pixel_value
        df_copy = data[:]
        df_copy = df_copy.reset_index(drop = True)
        line_number = []
        class_number, loop = 0, 0
        while(loop < len(df_per_page)):
            if(df_copy['IS_PIXEL'][loop] == False):
                line_number.append(class_number)
            elif(df_copy['IS_PIXEL'][loop] == True):
                class_number+=1
                line_number.append(class_number)
            loop+=1
 
        previously_total_lines = not merged_dataframe.empty and max(set(merged_dataframe['line_number']))
        df_per_page['line_number'] = [line + previously_total_lines+1 for line in line_number]
            
        df_per_page = df_per_page.sort_values(by=['line_number', 'Top_left_x', 'Bottom_right_y'], ascending = [True, True, True]).reset_index(drop = True)
        merged_dataframe = merged_dataframe.append(df_per_page)
    
    return merged_dataframe, is_scanned


def calculating_paragraph(df_line, page_number):
    """
    Creating paragraph attribute for calculating paragraph number of the text
    present in given dataframe using clustering on coordiantes.
    
    Input : 
        DataFrame
        Page Number to calculates paragraphs on that page
    """
    MIN_LINE_SPACE = 0.09

    df_line = df_line.reset_index(drop=True)    
    
    # Operation on page
    page_df = df_line[df_line['page']==page_number]

    # Calculating vertical text
    page_df['x_diff'] = page_df['Top_right_x']-page_df['Top_left_x']
    page_df['y_diff'] = page_df['Top_right_y']-page_df['Top_left_y']
    temp_page_df = page_df[page_df['x_diff']==0]    
    v_df = pd.DataFrame(index=temp_page_df['Top_left_x'], columns=['Description', 'line_number'])
    v_df['Description'] = temp_page_df['Description'].tolist()
    v_df['line_number'] = temp_page_df['line_number'].tolist()
    
    my_line_num_text_dict = v_df.T.to_dict()
    page_df.loc[temp_page_df.index, 'vertical_text_lines'] = [my_line_num_text_dict for _ in range(len(temp_page_df))]
    df_line.loc[temp_page_df.index, 'vertical_text_lines'] = [my_line_num_text_dict for _ in range(len(temp_page_df))]
    
    page_df = page_df[pd.isna(page_df['vertical_text_lines'])]
    
    
    # Assigning approprate value for coordinated belonging to same line
    for li in sorted(set(page_df.line_number)):
        df_li = page_df[page_df['line_number']==li]
        page_df.loc[df_li.index, 'Bottom_right_y'] = max(df_li['Bottom_right_y'])
        page_df.loc[df_li.index, 'Top_left_y'] = min(df_li['Top_left_y'])
        page_df.loc[df_li.index, 'Bottom_left_y'] = max(df_li['Bottom_left_y'])
        page_df.loc[df_li.index, 'Top_right_y'] = min(df_li['Top_right_y'])
        
    # Calculating y-coordinates space above and below line
    page_df['bottom'] = [0] + page_df['Bottom_right_y'].tolist()[:-1]
    page_df['up_space'] = page_df['Top_left_y'] - page_df['bottom']
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
        return df_line
    
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
    page_df= page_df.sort_values(by=['line_number', "Top_left_x"])

    # MeanShift Clustering in top left x coordinates
    X = np.array(page_df.loc[:, ['Top_left_x']])
    bandwidth = estimate_bandwidth(X, quantile=0.16, n_samples=500, n_jobs=-1)
    if bandwidth:
        model = MeanShift(bandwidth=bandwidth, n_jobs=-1)
    else:
        model = MeanShift(n_jobs=-1)
    xhat = model.fit_predict(X)
    cluster_centers = model.cluster_centers_
    page_df['xhat'] = xhat 
    
    # Sorting clustering number bases on Top left x of line
    page_df = page_df.sort_values(by=['Top_left_x'])
    
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
    
    # Assignment of value to df_line
    df_line.loc[page_df.index, 'up_space'] = page_df['up_space']
    df_line.loc[page_df.index, 'down_space'] = page_df['down_space']
    df_line.loc[page_df.index, 'xhat'] = page_df['xhat']
    df_line.loc[page_df.index, 'yhat'] = page_df['yhat']
    df_line.loc[page_df.index, 'paragraph'] = page_df['paragraph']
    df_line.loc[page_df.index, 'column'] = page_df['column']
        
    return df_line


def smart_response(response, headers):            
    # Extracting text requires two API calls: One call to submit the
    # image for processing, the other to retrieve the text found in the image.
    
    # Holds the URI used to retrieve the recognized text.
    operation_url = response.headers["Operation-Location"]
    
    # The recognized text isn't immediately available, so poll to wait for completion.
    analysis = {}
    poll = True
    tic = time.perf_counter()
    while (poll):
        response_final = requests.get(
            response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        
        # print(json.dumps(analysis, indent=4))
    
        time.sleep(1)
        if ("analyzeResult" in analysis):
            poll = False
        if ("status" in analysis and analysis['status'] == 'failed'):
            poll = False
    else:
        toc = time.perf_counter()
        total_time_of_get_request = toc-tic
        print("{} seconds required for Azure READ API GET request\n".format(total_time_of_get_request))
    
    tic = time.time()
    details = []
    details_line_by_line = []
    if ("analyzeResult" in analysis):
        # Extract the recognized text, with bounding boxes. 
        for read_result in analysis["analyzeResult"]["readResults"]:
            for key, value in read_result.items():
                if key == 'page':
                    page = value
                if key == 'angle':
                    angle = value
                if key == 'width':
                    width = value
                if key == 'height':
                    height = value
                if key == 'unit':
                    unit = value
                if key == 'lines':
                    lines = value
                    
            for line in lines:
                Top_left_x, Top_left_y, Top_right_x, Top_right_y, Bottom_right_x, Bottom_right_y, Bottom_left_x, Bottom_left_y = line['boundingBox']
                Description = line['text']       
                details_line_by_line.append([Description, Top_left_x, Top_left_y, Top_right_x, Top_right_y, Bottom_right_x, Bottom_right_y, Bottom_left_x, Bottom_left_y ] + [page, angle, width, height, unit])
                inner_details = []
                for word in line['words']:
                    inner_details.append([word['text'], *word["boundingBox"], word['confidence']] + [page, angle, width, height, unit])
                else:
                    details.extend(inner_details)
            
    
    df = pd.DataFrame(details, columns = ['Description', 
                            'Top_left_x','Top_left_y' ,
                            'Top_right_x', 'Top_right_y' ,
                            'Bottom_right_x','Bottom_right_y',
                            'Bottom_left_x','Bottom_left_y', "Confidence",
                            "page", "angle", "width", "height", "unit"])
    df, is_scanned = line_numbers(df)
    
    df_line = pd.DataFrame(details_line_by_line, columns = ['Description', 
                            'Top_left_x','Top_left_y' ,
                            'Top_right_x', 'Top_right_y' ,
                            'Bottom_right_x','Bottom_right_y',
                            'Bottom_left_x','Bottom_left_y',
                            "page", "angle", "width", "height", "unit"])
    df_line, is_scanned = line_numbers(df_line)
    toc = time.time()
    ocr_parse_time = toc-tic

    return df, df_line, is_scanned, total_time_of_get_request, ocr_parse_time


def df_line_ocr(file_name):    
    file_data = open(file_name, "rb").read()
    # Set Content-Type to octet-stream
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    # put the byte array into your post request
    tic = time.perf_counter()
    ocr_response = requests.post(text_recognition_url, headers=headers, params=None, data = file_data)
    toc = time.perf_counter()
    total_time_of_post_request = toc-tic
    print("{} seconds required for Azure READ API POST request\n".format(total_time_of_post_request))
    while not ocr_response.ok:
        ocr_response = requests.post(text_recognition_url, headers=headers, params=None, data = file_data)
        ocr_response.raise_for_status()

    df, df_line, is_scanned, total_time_of_get_request, ocr_parse_time = smart_response(ocr_response, headers)
    
    tic = time.perf_counter()
    # df_line = merging_coordinates(df_line)
    df_line['vertical_text_lines'] = None
    for page_number in sorted(set(df_line['page'])):    
        df_line = calculating_paragraph(df_line, page_number)
        
    # Calculating paragraph_number column for complete PDF
    paragraph_number = []
    count = 0
    prev_para_num = df_line['paragraph'].tolist() and df_line['paragraph'].tolist()[0]
    for para_num in df_line['paragraph']:
        if para_num==prev_para_num or pd.isna(para_num):
            pass
        else:
            count += 1
            prev_para_num = para_num        
        paragraph_number.append(count)
    df_line['paragraph_number'] = paragraph_number
    
    
    
    # Calculating table identified in a paragraph
    for para_num in sorted(set(df_line['paragraph_number'])):
        df_para = df_line[df_line['paragraph_number']==para_num]
        for col in sorted(set(df_para[~pd.isna(df_para['column'])]['column'])):
            col_df = df_para[df_para['column']==col]
            col_df['column_up_space'] = col_df['Top_left_y'].diff().tolist()
            df_para.loc[col_df.index, 'column_up_space'] = col_df['column_up_space'].tolist()
            df_line.loc[col_df.index, 'column_up_space'] = col_df['column_up_space'].tolist()
        
        for line in sorted(set(df_para['line_number'])):
            temp_df = df_para[df_para['line_number']==line]
            my_sum = 0
            for val in temp_df['column_up_space']:
                if not pd.isna(val):
                    my_sum+=val
            df_para.loc[temp_df.index, 'sum_of_column_up_space'] = my_sum
            df_line.loc[temp_df.index, 'sum_of_column_up_space'] = my_sum

        X = np.array(df_para.loc[:, ['sum_of_column_up_space']])
        if len(X) != 1:
            model = AgglomerativeClustering(n_clusters=2)
            # fit model and predict clusters
            yhat = model.fit_predict(X)
            df_para['table_identifier'] = yhat
            df_line.loc[df_para.index, 'table_identifier'] = yhat
        
        table = []
        count = 0
        first_identifier = df_para['table_identifier'].tolist() and df_para['table_identifier'].tolist()[0]
        prev_identifier = df_para['table_identifier'].tolist() and df_para['table_identifier'].tolist()[0]
        for index, column_no, table_identifier in zip(df_para.index, df_para['column'], df_para['table_identifier']):
            if column_no!= min(df_para['column']):
                table.append(count)
            else:
                break
        
        if table:
            count += 1
        
        for identifier in df_para.loc[index:, 'table_identifier']:
            if pd.isna(identifier):
                table.append(identifier)
                continue
            if prev_identifier != identifier:
                if identifier != first_identifier:
                    count += 1
            prev_identifier = identifier
            table.append(count)
        
        df_para['table'] = table
        df_line.loc[df_para.index, 'table'] = df_para['table'].tolist()


    # Merging close rows in table2
    MIN_VERTICAL_SPACE_MERGE_DISTANCE = 0.3
    for para_num in sorted(set(df_line['paragraph_number'])):
        df_para = df_line[df_line['paragraph_number']==para_num]
        df_para = df_para[df_para['table']!=-1]
        if df_para.empty:
            continue
        table_num_list = []
        for t in set(df_para['table']):
            if t is not None and t != -1:
                table_num_list.append(t)
        table_num_list = sorted(table_num_list)

        df_para['table2'] = None
        df_line.loc[df_para.index, 'table2'] = None    

        count=0        
        if not table_num_list[:-1]:
            for df_para_index in df_para.index:
                if not pd.isna(df_para.loc[df_para_index, 'table']):
                    df_para.loc[df_para_index, 'table2'] = count
                    df_line.loc[df_para.index, 'table2'] = count
                    count+= 1
                    continue
        
        for i, table_num in enumerate(table_num_list[:-1]):
            first_table_num = table_num 
            second_table_num = table_num_list[i+1]
            first_df = df_para[df_para['table'] == first_table_num]
            second_df = df_para[df_para['table'] == second_table_num]
            first_column = first_df.loc[first_df.index[0], 'column']
            second_column = second_df.loc[second_df.index[0], 'column']
            if first_column  == second_column:
                first_value = first_df[first_df["column"]==first_column]['Top_left_y'].mean()
                second_value = second_df[second_df["column"]==second_column]['Top_left_y'].mean()
                value = abs(second_value-first_value)
                if value < MIN_VERTICAL_SPACE_MERGE_DISTANCE:
                    df_para.loc[first_df.index, 'table2'] = count
                    df_para.loc[second_df.index, 'table2'] = count
                    df_line.loc[first_df.index, 'table2'] = count
                    df_line.loc[second_df.index, 'table2'] = count
                else:
                    df_para.loc[first_df.index, 'table2'] = count
                    df_line.loc[first_df.index, 'table2'] = count
                    count+=1
                    df_para.loc[second_df.index, 'table2'] = count    
                    df_line.loc[second_df.index, 'table2'] = count    
            else:
                df_para.loc[first_df.index, 'table2'] = count
                df_line.loc[first_df.index, 'table2'] = count
                count+=1
                df_para.loc[second_df.index, 'table2'] = count
                df_line.loc[second_df.index, 'table2'] = count
                
        df_t = df_para[df_para["column"]==min(df_para['column'])]
        x_min_coor_value = df_t['Top_left_x'].mean() 
        temp_df = df_para[df_para['table2']==0]
        
        our_line=0
        our_max = 0
        for l in sorted(set(df_para['line_number'])):
            dff = df_para[df_para['line_number']==l]
            if len(set(dff['column'])) > our_max:
                our_line = l
                our_max = len(set(dff['column'])) 
        
        if not temp_df.empty:
            my_given_list = df_para[df_para['line_number']==our_line]['Top_left_x'].tolist()
            if len(my_given_list)>1 :
                x_threshold = min(map(lambda x, y: abs(x-y), my_given_list[1:], my_given_list[:-1]))
            elif len(my_given_list) == 1:
                x_threshold = my_given_list.pop()
            else:
                x_threshold = 0       
            max_table_num = max(df_para['table2'])    
            my_list = []
            x_mean_val = temp_df[temp_df['column']== min(temp_df['column'])]['Top_left_x'].mean()
            if x_mean_val-x_min_coor_value <= x_threshold:
                my_list = [0 for _ in range(len(temp_df))]
            else:
                my_list = [max_table_num+1 for _ in range(len(temp_df))] 
            df_para.loc[temp_df.index, 'table3'] = my_list
            df_line.loc[temp_df.index, 'table3'] = my_list
        
    
    # Calculating table number column for complete PDF
    my_table_list=[]
    count = 0
    prev_val = df_line['table2'].tolist() and df_line['table2'].tolist()[0]
    prev_para = df_line['paragraph_number'].tolist() and df_line['paragraph_number'].tolist()[0]
    for para_num in sorted(set(df_line['paragraph_number'])):
        df_para = df_line[df_line['paragraph_number']==para_num]
        for table2, table_3, para in zip(df_para['table2'], df_para['table3'], df_para['paragraph_number']):
            if pd.isna(table2):
                my_table_list.append(None)
                continue
            if table2:
                if prev_val != table2:
                    count += 1
                my_table_list.append(count)
            elif table_3==0 and table2==0:
                if prev_val != table2 or prev_para != para:
                    count += 1
                my_table_list.append(count)
            else:
                my_table_list.append(count)
            prev_val = table2
            prev_para = para
    
    df_line['table_number']=my_table_list
    toc = time.perf_counter()
    total_table_identification_time = toc-tic
    return df_line, total_table_identification_time 


def close_matches(word, possibilities=['CLIN', 'ITEM NO', 'SUPPLIES/SERVICES', 'ESTIMATED', 'QUANTITY', 'ESTIMATED QUANTITY', 
            'EST.', 'AMOUNT', 'EST. AMOUNT', 'UNIT', 'UNIT PRICE']):
    lower_possibilities = [str(item).lower() for item in possibilities]    
    lower_word = str(word).lower()
    lower_match = get_close_matches(word=lower_word, possibilities=lower_possibilities, n=1, cutoff=0.9)
    lower_match = lower_match and lower_match.pop(0)
    match = lower_match  and possibilities[lower_possibilities.index(lower_match)]
    if match:
        return word
    else:
        return None
    

def nearest_matches(given_value, possibilities):
    """
    This function gives closest match number form possibilities of number such
    that the number given should be in a range of maximum difference between any 
    two consecutive numbers in a sequence of possibilities.
    """
    absolute_difference_function = lambda list_value : abs(list_value - given_value)
    closest_value = min(possibilities, key=absolute_difference_function)
    if len(possibilities)==1 and abs(given_value-closest_value)  > 0.9:
        return None
    elif len(possibilities)==1 and abs(given_value-closest_value)  <= 0.9:
        return possibilities[0] 
    if abs(given_value-closest_value) < max(map(lambda x, y: abs(x-y), possibilities[1:], possibilities[:-1])):
        return closest_value


def get_key_val(x):
    for i in horizontal_in_word_keys:
        if x.startswith(i):
            return (i, x.replace(i, ''))
        elif x in horizontal_possibilities:
            return x
    else:
        return None

def filter_val(x):
    if type(x)==tuple:
        return x and {x[0]: x[1].replace(':', '').strip()} 
    else:
        return x
    

def redefine_table_number_by(df_para, primary_column_number=0):
    if primary_column_number:
        df_temp_para = df_para[df_para['column']<=primary_column_number]
        temp_df = df_temp_para [df_temp_para ["column"]==primary_column_number]
        prev_y_val = temp_df['Top_left_y'].tolist() and temp_df['Top_left_y'].tolist()[0]
        prev_c_val = temp_df['column'].tolist() and temp_df['column'].tolist()[0]

        table_number = []        
        count = 0
        for c_val, y_val in zip(df_temp_para ["column"], df_temp_para ["Top_left_y"]):
            if c_val == prev_c_val:
                diff = abs(prev_y_val-y_val)
                if diff > 0.4:
                    count += 1
                prev_y_val=y_val
            table_number.append(count)
        
        df_temp_para['angle'] = table_number[1:] + [table_number[-1]]
                        
    return df_para


def vertical_tables(df_line, vertical_possibilities=vertical_possibilities):
    key_dict = {}
    value_dict = {}
    MIN_TABLE_COLUMN = 2
    MIN_TABLE_PAGE_BREAK = 1
    table_count = 0
    
    prev_page = sorted(set(df_line['page']))[0]
    df_line['parent_table_number'] = None
    for para_num in sorted(set(df_line['paragraph_number'])):
        df_para = df_line[df_line['paragraph_number']==para_num]

        df_para = df_line[df_line['paragraph_number']==para_num]    
        df_para['horizontal_mapping'] = df_para['Description'].apply(close_matches, possibilities=horizontal_possibilities)
        for line_num in sorted(set(df_para['line_number'])):
            temp_df = df_para[df_para['line_number'] ==line_num]
            temp_df['horizontal_mapping'] = temp_df['Description'].apply(get_key_val)
            temp_df['horizontal_mapping'] = temp_df['horizontal_mapping'].apply(filter_val)
            df_para.loc[temp_df.index, 'horizontal_mapping'] = temp_df['horizontal_mapping'].tolist()    
        
        df_line.loc[df_para.index, 'horizontal_mapping'] = df_para['horizontal_mapping'].tolist()    
                
        df_para['vertical_mapping'] = df_para['Description'].apply(close_matches, possibilities=vertical_possibilities)
        df_line.loc[df_para.index, 'vertical_mapping'] = df_para['vertical_mapping'].tolist()
        
        if abs(df_para['page'].unique()[0] - prev_page) > MIN_TABLE_PAGE_BREAK:
            key_dict = {}
        
        df_table = df_para[~pd.isna(df_para['vertical_mapping'])]
        if not df_table.empty and not any(pd.isna(df_table['table_number'])):
            df_table = df_para[df_para['table_number']==df_table['table_number'].mode().unique()[0]]
        key_df = df_table[~pd.isna(df_table['vertical_mapping'])]
        if not key_df.empty :
            line = key_df['line_number'].unique()[0]        
            if all(~pd.isna(df_table[df_table['line_number']==line]['vertical_mapping'])):
                key_dict = {}
                table_count += 1
                for k, v in zip(key_df['column'], key_df['Description']):
                    left_x_column = key_df[key_df['column']==k]['Top_left_x'].mean()
                    key_dict[k]=(v, left_x_column)
                df_para.loc[key_df.index, 'parent_table_number']= table_count
                df_line.loc[key_df.index, 'parent_table_number']= table_count
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
                left_x_row = column['Top_left_x'].mean()
                table_row_left_x.append(left_x_row)
            
            table_column_available = [None for _ in range(int(max(key_dict))+1)]
            for k, v in key_dict.items():
                table_column_available[int(k)] = v[1]        
            table_column = list(filter(None, table_column_available))
            
            table_row = [None for _ in range(len(table_column))]
            for given_value in table_row_left_x:
                matched = nearest_matches(given_value=given_value, possibilities=table_column)
                if matched:
                    matched_index = table_column.index(matched)
                    table_row[matched_index] = matched
                
            table_row_without_none = list(filter(None, table_row))
            if len(table_row_without_none) >= MIN_TABLE_COLUMN:
                table_count+= 1
                df_para.loc[table.index, 'parent_table_number']= table_count
                df_line.loc[table.index, 'parent_table_number']= table_count
                prev_page = df_para['page'].unique()[0]
    
    for page_num in sorted(set(df_line['page'])):
        df_page = df_line[df_line['page']==page_num]
        #check first 5 line 
        first_five_lines = sorted(set(df_page['line_number']))[:5]
        extract_lines = []
        for line_num in first_five_lines:
            temparary_df = df_page[df_page['line_number']==line_num]
            if sorted(filter(None, temparary_df['vertical_mapping'].tolist())):
                break
            extract_lines.append(line_num)
        
        for line_num in extract_lines:
            temparary_df = df_page[df_page['line_number']==line_num]
            df_page.loc[temparary_df.index, 'table_number']=None
            df_line.loc[temparary_df.index, 'table_number']=None        


    for table_num in sorted(set(df_line['table_number'])):
        df_para = df_line[df_line['table_number']==table_num]
        key_df = df_para[~pd.isna(df_para['vertical_mapping'])]    
        parent_num_redefined = list(filter(None, key_df['parent_table_number'].unique()))
        if parent_num_redefined:
            parent_num_redefined = parent_num_redefined.pop(0)
        else:
            parent_num_redefined = None
        df_para.loc[key_df.index, 'parent_table_number'] = parent_num_redefined
        df_line.loc[key_df.index, 'parent_table_number'] = parent_num_redefined
    
    
    table_dictionary = {}
    for parent_num in sorted(filter(None, set(df_line['parent_table_number']))):
        df_parent = df_line[df_line['parent_table_number']==parent_num]
        
        key_df = df_parent[~pd.isna(df_parent['vertical_mapping'])]    
        if not key_df.empty :
            line = key_df['line_number'].unique()[0]        
            if all(~pd.isna(df_table[df_table['line_number']==line]['vertical_mapping'])):
                key_dict = {}
                for k, v in zip(key_df['column'], key_df['Description']):
                    left_x_column = key_df[key_df['column']==k]['Top_left_x'].mean()
                    key_dict[k]=(v, left_x_column)
                    
                my_dict = {}
                for column in sorted(set(key_df['column'])):
                    col_df = key_df[key_df['column']==column]
                    left_x_column = col_df['Top_left_x'].mean()
                    text = ' '.join(col_df['Description'].tolist())
                    my_dict[left_x_column]=text
       
        value_df = df_parent[pd.isna(df_parent['vertical_mapping'])]
        
        for para_num in sorted(set(value_df['paragraph_number'])):
            df_para = value_df[value_df['paragraph_number']==para_num]
            for table_num in sorted(set(df_para['table_number'])):
                if not key_dict:
                    break
                
                table = df_para[df_para['table_number'] == table_num]
            
                # define data
                table_row_left_x = []
                table_row_text = []
                for column_num in sorted(set(table['column'])):
                    column = table[table['column']==column_num]
                    left_x_row = column['Top_left_x'].mean()
                    cell_text = ' '.join(column['Description'].tolist())
                    table_row_left_x.append(left_x_row)
                    table_row_text.append(cell_text)
                
                table_column_available = [None for _ in range(int(max(key_dict))+1)]
                for k, v in key_dict.items():
                    table_column_available[int(k)] = v[1]        
                table_column = list(filter(None, table_column_available))
                
                table_row = [None for _ in range(len(table_column))]
                for given_value in table_row_left_x:
                    matched = nearest_matches(given_value=given_value, possibilities=table_column)            
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
                
    return table_dictionary, df_line
                    

def horizontal_tables(df_line, horizontal_possibilities, horizontal_in_word_keys):
    horizontal_mapped_dict = {}
    for para_num in sorted(set(df_line['paragraph_number'])):
        df_para = df_line[df_line['paragraph_number']==para_num]    
        df_para['horizontal_mapping'] = df_para['Description'].apply(close_matches, possibilities=horizontal_possibilities)
        for line_num in sorted(set(df_para['line_number'])):
            temp_df = df_para[df_para['line_number'] ==line_num]
            temp_df['horizontal_mapping'] = temp_df['Description'].apply(get_key_val)
            temp_df['horizontal_mapping'] = temp_df['horizontal_mapping'].apply(filter_val)
            df_para.loc[temp_df.index, 'horizontal_mapping'] = temp_df['horizontal_mapping'].tolist()    
        
        horizontal_mapping_lines = sorted(set(df_para[~pd.isna(df_para['horizontal_mapping'])]['line_number']))    
        for line_num in horizontal_mapping_lines :
            temp_df = df_para[df_para['line_number'] ==line_num]
            
            key_df = temp_df[~pd.isna(temp_df['horizontal_mapping'])]
            value_df = temp_df[pd.isna(temp_df['horizontal_mapping'])]
            
            if not key_df.empty and not value_df.empty:
                key = ' '.join(key_df['Description'].tolist())
                value = ' '.join(value_df['Description'].tolist())
            elif not key_df.empty and value_df.empty:
                if key_df['horizontal_mapping'].tolist() and type(key_df['horizontal_mapping'].tolist()[0]) == dict:
                    item = key_df['horizontal_mapping'].tolist()[0]
                    for k, v in item.items():
                        key = k
                        value = v
                else:
                    key = ' '.join(key_df['Description'].tolist())
                    value = ''
            else:
                continue 
            key_value_dict = {}
            key_value_dict[key]=value    
            horizontal_mapped_dict[line_num] = key_value_dict
    return horizontal_mapped_dict, df_line


def table_extraction(pdf_path):
    df_line, total_table_identification_time  = df_line_ocr(pdf_path)
    tic = time.time()
    table_dictionary, df_line = vertical_tables(df_line, vertical_possibilities=vertical_possibilities)
    horizontal_line_dictionary, df_line = horizontal_tables(df_line=df_line, horizontal_possibilities=horizontal_possibilities, horizontal_in_word_keys=horizontal_in_word_keys )
    prev_parent_table_no = None
    for line_num, parent_table_no in zip(df_line['line_number'], df_line['parent_table_number']):
        if parent_table_no:
            prev_parent_table_no = parent_table_no
        elif prev_parent_table_no and line_num in horizontal_line_dictionary.keys():
            table_dictionary[prev_parent_table_no].update(horizontal_line_dictionary[line_num])
    count = 0
    my_table_dict = {}
    for k, v in table_dictionary.items():
        my_table_dict[count] = v
        count += 1
    toc = time.time()
    processing_time = toc-tic
    return my_table_dict, total_table_identification_time, processing_time



# import json 
# pdf_path = '/home/vaibhav/Documents/Technomile/Dataset/merged_pages.pdf'
# json_path = '/'.join(pdf_path.split('/')[:-1]) + '/' + pdf_path.split('/')[-1].split('.')[0] + '.json'
# my_table_json = table_extraction(pdf_path)
# with open(json_path, 'w') as f:
#     f.write(json.dumps(my_table_json,indent=4, ensure_ascii=False, sort_keys=False))

if __name__ == '__main__':
    import json 
    pdf_path = '/home/vaibhav/Documents/Technomile/Dataset/merged_pages.pdf'
    json_path = '/'.join(pdf_path.split('/')[:-1]) + '/' + pdf_path.split('/')[-1].split('.')[0] + '.json'
    my_table_json, total_table_identification_time, processing_time = table_extraction(pdf_path)
    with open(json_path, 'w') as f:
        f.write(json.dumps(my_table_json,indent=4, ensure_ascii=False, sort_keys=False))


