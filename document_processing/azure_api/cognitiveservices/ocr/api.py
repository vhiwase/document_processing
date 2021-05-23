from .ds import OCRInputs
from .read_api import AzureReadApiOCR
from .form_api import AzureFormApiOCR

__all__ = ['AzureOCR']

class AzureOCR:
    def __init__(self,
                 document_filepath,
                 endpoint=None,
                 subscription_key=None,
                 operation_url=None,
                 ocr_outputs=None,
                 api='read'):
        
        self.api = api
        self.ocr_inputs = OCRInputs(document_filepath=document_filepath, 
                               endpoint=endpoint, 
                               subscription_key=subscription_key, 
                               operation_url=operation_url, 
                               ocr_outputs=ocr_outputs)

    
    def get_api_ocr(self):
        if self.api =='read':
            api_ocr = AzureReadApiOCR(
                document_filepath=self.ocr_inputs.document_filepath, 
                endpoint=self.ocr_inputs.endpoint, 
                subscription_key=self.ocr_inputs.subscription_key,
                operation_url=self.ocr_inputs.operation_url,
                ocr_outputs=self.ocr_inputs.ocr_outputs
            )
        else:
            
            api_ocr = AzureFormApiOCR(
                document_filepath=self.ocr_inputs.document_filepath, 
                endpoint=self.ocr_inputs.endpoint, 
                subscription_key=self.ocr_inputs.subscription_key,
                ocr_outputs=self.ocr_inputs.ocr_outputs
            )
        return api_ocr
            


