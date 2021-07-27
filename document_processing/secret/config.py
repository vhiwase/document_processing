import pathlib

__all__ = ["AZURE_READ_API_ENDPOINT", 
           "AZURE_READ_API_SUBSCRIPTION_KEY", 
           "AZURE_FORM_API_ENDPOINT", 
           "AZURE_FORM_API_SUBSCRIPTION_KEY", 
           "DOCUMENT_FILE_PATH",
           "VERTICAL_COLUMNS",
           "HORIZONTAL_COLUMNS",
           "HORIZONTAL_KEYWORDS"]

FILE_PATH = pathlib.PosixPath(__file__)
BASE_PATH = FILE_PATH.parent
AZURE_READ_API_ENDPOINT = 'https://technomilecognitivieservices.cognitiveservices.azure.com'
AZURE_READ_API_SUBSCRIPTION_KEY = '0766c7b636c04a2fad9c90d68f008b8d'
AZURE_FORM_API_ENDPOINT = 'https://tmocr.cognitiveservices.azure.com/'
AZURE_FORM_API_SUBSCRIPTION_KEY = 'c0be54398c784c68aab460ea9a7e4fef'

DOCUMENT_FILE_PATH = (BASE_PATH/'doc4.pdf').absolute().as_posix()

VERTICAL_COLUMNS=[
    'CLIN', 'ITEM NO', 'SUPPLIES/SERVICES', 'ESTIMATED QUANTITY', 'EST. AMOUNT', 
    'UNIT PRICE', 'DESCRIPTION OF SUPPLIES/SERVICES', "Section", "F Item #", 
    'PWS', 'Deliverable', 'Due Date', 'Addressee and', 'F Item #', 
    'cross-', 'Number/type', 'reference', 'of copies', 'CONTRACT', 'QTY', 'TOTAL']
HORIZONTAL_COLUMNS=['NET AMT', 'ESTIMATED COST', 'FIXED FEE', 'TOTAL EST COST + FEE']
HORIZONTAL_KEYWORDS=['ACRN', 'CIN']