from azure.ai.formrecognizer import FormRecognizerClient
from azure.core.credentials import AzureKeyCredential
import pandas as pd

from .ds import AzureOCR, OCRText
from .common import LineNumber

class AzureFormApiOCR:
    def __init__(self, document_filepath, endpoint=None, subscription_key=None, ocr_outputs=None):
        self.document_filepath = document_filepath
        azure_ocr = AzureOCR(self.document_filepath, endpoint, subscription_key)
        self.document_binaries = azure_ocr.document_binaries        
        if ocr_outputs is not None:            
            self.ocr_outputs = ocr_outputs
        else:
            # upload document
            form_recognizer_client = FormRecognizerClient(endpoint=azure_ocr.endpoint,credential= AzureKeyCredential(azure_ocr.subscription_key))
            poller = form_recognizer_client.begin_recognize_content(azure_ocr.document_binaries)
            print("'{}' document binaries uploaded".format(self.document_filepath))
            # download ocr output 
            self.ocr_outputs = poller.result()
            print("OCR of '{}' is downloaded".format(self.document_filepath))

        line_dataframe, word_dataframe = self._dataframes(self.ocr_outputs)
        self.line_dataframe = line_dataframe
        self.word_dataframe = word_dataframe
    
    def _dataframes(self, ocr_outputs:dict = None) -> tuple:        
        """
        *Author: Vaibhav Hiwase
        *Details: Converting OCR output into word and line dataframes
        """
        if ocr_outputs is None:
            ocr_outputs = self.ocr_outputs
        word_details = []
        line_details = []
        for page in ocr_outputs:
            for num, line in enumerate(page.lines):
                for word in line.words:
                    bb_word_list = []
                    [bb_word_list.extend(j) for j in [[*i] for i in word.bounding_box]]
                    ocr_words = OCRText(*bb_word_list, word.page_number, page.text_angle,
                            page.width, page.height, page.unit, word.text)
                    word_details.append(ocr_words)
                bb_line_list = []
                [bb_line_list.extend(j) for j in [[*i] for i in line.bounding_box]]
                ocr_lines = OCRText(*bb_line_list, word.page_number, page.text_angle,
                                    page.width, page.height, page.unit, line.text)
                line_details.append(ocr_lines)           
        line_number = LineNumber()
        line_dataframe = pd.DataFrame(line_details)
        line_dataframe = line_number.add_line_numbers(line_dataframe) 
        word_dataframe = pd.DataFrame(word_details)
        word_dataframe = line_number.add_line_numbers(word_dataframe)
        self.is_scanned = line_number.is_scanned
        return line_dataframe, word_dataframe

if __name__ == '__main__':
    document_filepath = ''
    endpoint =''
    subscription_key = ''    
    form_api_ocr = AzureFormApiOCR(document_filepath=document_filepath, endpoint=endpoint, subscription_key=subscription_key)
    line_dataframe= form_api_ocr.line_dataframe
    word_dataframe = form_api_ocr.word_dataframe
    ocr_outputs = form_api_ocr.ocr_outputs
    is_scanned = form_api_ocr.is_scanned
    form_api_ocr = AzureFormApiOCR(document_filepath=document_filepath, 
                                   endpoint=endpoint, 
                                   subscription_key=subscription_key,
                                   ocr_outputs=ocr_outputs)
