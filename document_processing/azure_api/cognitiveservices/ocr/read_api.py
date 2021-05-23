import requests
import time
from dataclasses import dataclass
import pandas as pd

from .ds import AzureOCR, OCRText, RequestInputs
from .common import LineNumber

class AzureReadApiOCR:
    def __init__(self, document_filepath, endpoint=None, subscription_key=None, operation_url=None, ocr_outputs=None):
        self.document_filepath = document_filepath
        azure_ocr = AzureOCR(self.document_filepath, endpoint, subscription_key)
        self.operation_url = operation_url
        self.document_binaries = azure_ocr.document_binaries
        request_inputs = RequestInputs(header={
            'Ocp-Apim-Subscription-Key': azure_ocr.subscription_key, 
            'Content-Type': 'application/octet-stream'
            },
            text_recognition_url=azure_ocr.endpoint + "/vision/v3.1/read/analyze"
        )        
        # Set Content-Type to octet-stream
        if ocr_outputs is not None:            
            self.ocr_outputs = ocr_outputs
        else:
            if self.operation_url is None:
                self._upload(request_inputs)
            if self.operation_url is not None:
                self.ocr_outputs = self._download(request_inputs)
        
        line_dataframe, word_dataframe = self._dataframes(self.ocr_outputs)
        self.line_dataframe = line_dataframe
        self.word_dataframe = word_dataframe
        
    def _upload(self, request_inputs):
        """
        *Author: Vaibhav Hiwase
        *Details: Uploading document to Azure READ API to generate operation URL.
        """
        # put the byte array into your post request
        response = requests.post(
            request_inputs.text_recognition_url, 
            headers=request_inputs.header, 
            params=None, 
            data = self.document_binaries
        )
        while not response.ok:
            response = requests.post(
                request_inputs.text_recognition_url, 
                headers=request_inputs.header, 
                params=None, 
                data = self.document_binaries
            )
            response.raise_for_status()
        # Holds the URI used to retrieve the recognized text.
        self.operation_url = response.headers["Operation-Location"]
        print("'{}' document binaries uploaded".format(self.document_filepath))
        
    def _download(self, request_inputs):
        """
        *Author: Vaibhav Hiwase
        *Details: Downloading OCR results from Azure READ API.
        """
        # The recognized text isn't immediately available, so poll to wait for completion.
        ocr_outputs = {}
        poll = True
        while (poll):
            response = requests.get(
                self.operation_url, 
                headers=request_inputs.header
            )
            ocr_outputs = response.json()
            time.sleep(1)
            if ("analyzeResult" in ocr_outputs):
                poll = False
            if ("status" in ocr_outputs and ocr_outputs['status'] == 'failed'):
                poll = False    
        print("OCR of '{}' is downloaded".format(self.document_filepath))
        return ocr_outputs
    
    def _dataframes(self, ocr_outputs:dict = None) -> tuple:        
        """
        *Author: Vaibhav Hiwase
        *Details: Converting OCR output into word and line dataframes
        """
        if ocr_outputs is None:
            ocr_outputs = self.ocr_outputs
        word_details = []
        line_details = []
        if ("analyzeResult" in ocr_outputs):
            # Extract the recognized text, with bounding boxes. 
            for read_result in ocr_outputs["analyzeResult"]["readResults"]:
                for key, values in read_result.items():
                    if key=='lines':
                        for value in values:
                            for k, v in value.items():
                                if k == 'words':
                                    for item in v:
                                        ocr_words = OCRText(
                                            *item['boundingBox'],
                                            *list(read_result.values())[:-1],
                                            item['text'], 
                                            item['confidence'],
                                        )
                                        word_details.append(ocr_words)

                            ocr_lines = OCRText(
                                *value['boundingBox'],
                                *list(read_result.values())[:-1],
                                value['text'], 
                            )
                            line_details.append(ocr_lines)

        line_number = LineNumber()
        line_dataframe = pd.DataFrame(line_details)
        line_dataframe = not line_dataframe.empty and line_number.add_line_numbers(line_dataframe) 
        word_dataframe = pd.DataFrame(word_details)
        word_dataframe = not word_dataframe.empty and line_number.add_line_numbers(word_dataframe)
        self.is_scanned = line_number.is_scanned
        return line_dataframe, word_dataframe
    
if __name__ == '__main__':
    endpoint = ''
    subscription_key = ''
    document_filepath = ''
    read_api_ocr = AzureReadApiOCR(document_filepath=document_filepath, endpoint=endpoint, subscription_key=subscription_key)
    line_dataframe= read_api_ocr.line_dataframe
    word_dataframe = read_api_ocr.word_dataframe
    operation_url = read_api_ocr.operation_url
    text_recognition_url = read_api_ocr.text_recognition_url
    ocr_outputs = read_api_ocr.ocr_outputs
    is_scanned = read_api_ocr.is_scanned
    read_api_ocr = AzureReadApiOCR(document_filepath=document_filepath, 
                                   endpoint=endpoint, 
                                   subscription_key=subscription_key,
                                   ocr_outputs=ocr_outputs)
    read_api_ocr = AzureReadApiOCR(document_filepath=document_filepath, 
                                   endpoint=endpoint, 
                                   subscription_key=subscription_key,
                                   operation_url=operation_url)
    
