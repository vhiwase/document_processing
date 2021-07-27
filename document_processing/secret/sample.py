from ..azure_api import AzureReadApiOCR, AzureFormApiOCR

endpoint = 'https://technomilecognitivieservices.cognitiveservices.azure.com'
subscription_key = '0766c7b636c04a2fad9c90d68f008b8d'
document_filepath = '/home/vaibhav/Documents/Technomile/Dataset/A TechnoMile Data/PDFs/155b5e08acf54c0194a1b9259dc7c220.pdf'
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

document_filepath = '/home/vaibhav/Documents/Technomile/Dataset/A TechnoMile Data/PDFs/155b5e08acf54c0194a1b9259dc7c220.pdf'
endpoint ='https://tmocr.cognitiveservices.azure.com/'
subscription_key = 'c0be54398c784c68aab460ea9a7e4fef'    
form_api_ocr = AzureFormApiOCR(document_filepath=document_filepath, endpoint=endpoint, subscription_key=subscription_key)
line_dataframe= form_api_ocr.line_dataframe
word_dataframe = form_api_ocr.word_dataframe
ocr_outputs = form_api_ocr.ocr_outputs
is_scanned = form_api_ocr.is_scanned
read_api_ocr = AzureFormApiOCR(document_filepath=document_filepath, 
                                endpoint=endpoint, 
                                subscription_key=subscription_key,
                                ocr_outputs=ocr_outputs)