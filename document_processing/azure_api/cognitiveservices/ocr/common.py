import pandas as pd

class LineNumber:
    def __init__(self, df:pd.DataFrame=None):
        self.df = df
        self.is_scanned = None

    def _is_scanned_doc_dataframe(self, df: pd.DataFrame=None) -> bool:
        """
        *Author: Vaibhav Hiwase
        *Details: Checked whether dataframe containes values taken from 
                  scanned docuement or not
        """
        if df is None:
            df = self.df
        is_scanned = True
        for index in range(8):
            df_type = df.iloc[:,index].dtype
            if df_type == float:
                continue
            else:
                break
        else:
            is_scanned = False    
        return is_scanned

    def _get_line_number_threshold(self, df: pd.DataFrame=None) -> float:
        """
        *Author: Vaibhav Hiwase
        *Details: Decide threshold for correct line number identification based on
                  whether document is scanned or not.
        """
        if df is None:
            df = self.df
        if self.is_scanned:
            threshold = 16.6
        else:
            threshold = 0.048
        return threshold

    def add_line_numbers(self, df:pd.DataFrame=None) -> tuple:
        """
        *Author: Vaibhav Hiwase
        *Details: Logic for getting accurate line number irrespective of any noise 
                  in word coordinates coming from hocr file.     
        """
        if df is None:
            df = self.df
        self.is_scanned = self._is_scanned_doc_dataframe(df)
        line_number_threshold = self._get_line_number_threshold(df)
        merged_dataframe = pd.DataFrame()
        for page in sorted(set(df['page'])):
            df_per_page = df[df['page'] == page]
            df_per_page = df_per_page.sort_values(by=['top_left_y', 'top_left_x'], ascending = [True, True]).reset_index(drop = True)    
            # y_avg = df_per_page.loc[: , [constants.MISC_LEFT_Y,constants.MISC_RIGHT_Y]]
            y_avg = df_per_page.loc[: , ["top_left_y"]]
            data = pd.DataFrame()
            data['top_y_avg'] = y_avg.mean(axis=1)
            data['top_y_difference'] = data['top_y_avg'].diff()
            data['is_line_number'] = data['top_y_difference'] > line_number_threshold
            df_copy = data[:]
            df_copy = df_copy.reset_index(drop = True)
            line_number = []
            class_number, loop = 0, 0
            while(loop < len(df_per_page)):
                if(df_copy['is_line_number'][loop] == False):
                    line_number.append(class_number)
                elif(df_copy['is_line_number'][loop] == True):
                    class_number+=1
                    line_number.append(class_number)
                loop+=1
            previously_total_lines = not merged_dataframe.empty and max(set(merged_dataframe['line_number']))
            df_per_page['line_number'] = [line + previously_total_lines+1 for line in line_number]
            df_per_page = df_per_page.sort_values(
                by=['line_number', 'top_left_x', 'bottom_right_y'], 
                ascending = [True, True, True]
            ).reset_index(drop = True)
            merged_dataframe = merged_dataframe.append(df_per_page)
        return merged_dataframe
    

