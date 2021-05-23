from document_processing import AzureOCR
from document_processing import StructureExtractor
from document_processing import TableExtractor
import click

from secret import AZURE_READ_API_ENDPOINT, \
    AZURE_READ_API_SUBSCRIPTION_KEY, \
    AZURE_FORM_API_ENDPOINT, \
    AZURE_FORM_API_SUBSCRIPTION_KEY, \
    DOCUMENT_FILE_PATH, \
    VERTICAL_COLUMNS, \
    HORIZONTAL_COLUMNS, \
    HORIZONTAL_KEYWORDS 

__all__ = ["main"]

def azure_api_ocr(document_filepath=DOCUMENT_FILE_PATH, 
            endpoint=AZURE_READ_API_ENDPOINT, 
            subscription_key=AZURE_READ_API_SUBSCRIPTION_KEY,
            operation_url=None,
            ocr_outputs=None,
            api_type='azure', 
            api='read',
            vertical_columns=VERTICAL_COLUMNS,
            horizontal_columns=HORIZONTAL_COLUMNS,
            horizontal_keywords=HORIZONTAL_KEYWORDS
        ):
    table_extractor = TableExtractor(
          document_filepath=document_filepath, 
          endpoint=endpoint, 
          subscription_key=subscription_key,
          operation_url=operation_url,
          ocr_outputs=ocr_outputs,
          api_type=api_type, 
          api=api,
          vertical_columns=vertical_columns,
          horizontal_columns=horizontal_columns,
          horizontal_keywords=horizontal_keywords
      )
    word_dataframe = table_extractor.word_dataframe
    line_dataframe = table_extractor.line_dataframe
    ocr_outputs = table_extractor.ocr_outputs
    operation_url = table_extractor.operation_url
    is_scanned = table_extractor.is_scanned
    if line_dataframe is not False or line_dataframe is not None:
        line_dataframe_copy = line_dataframe.copy()
    print(line_dataframe_copy)
    print(operation_url)
    print(is_scanned)
    table_dict = table_extractor.table_extraction(line_dataframe_copy)
    return table_dict

@click.command()
@click.option('--document_filepath', '-DOC', prompt="Write document path",
help="write document path eg. /usr/share/doc/1.pdf")
@click.option('--endpoint', '-ENDPOINT', prompt="Write Azure ENDPOINT",
help="eg. 1.'https://something.cognitiveservices.azure.com/'")
@click.option('--subscription_key', '-KEY', prompt="Write Azure subscription Key",
help="eg.1234a5b678c98d7efd6g54h32i101j2k")
@click.option('--vertical_columns', '-TABLE_COLUMNS', prompt="Write column names of table in a list",
help="Write column names of table eg. ['ITEM NO', 'SUPPLIES/SERVICES', 'ESTIMATED QUANTITY', 'EST. AMOUNT']")
@click.option('--horizontal_columns', '-HORIZONTAL_MAPPING', prompt="Write horizontally mapped columns in a list")
@click.option('--horizontal_keywords', '-WORD_SPLIT', prompt="Write horizontally mapped keys for close words in a list")
def azure_read_api_calling(
        document_filepath,
        endpoint,
        subscription_key,
        vertical_columns,
        horizontal_columns,
        horizontal_keywords):
    table_dict = azure_api_ocr(document_filepath=document_filepath, 
            endpoint=endpoint, 
            subscription_key=subscription_key,
            api_type='azure', 
            api='read',
            vertical_columns=eval(vertical_columns),
            horizontal_columns=eval(horizontal_columns),
            horizontal_keywords=eval(horizontal_keywords)
        )
    return table_dict

@click.command()
@click.option('--document_filepath', '-DOC', prompt="Write document path",
help="write document path eg. /usr/share/doc/1.pdf")
@click.option('--endpoint', '-ENDPOINT', prompt="Write Azure ENDPOINT",
help="eg. 1.'https://something.cognitiveservices.azure.com/'")
@click.option('--subscription_key', '-KEY', prompt="Write Azure subscription Key",
help="eg.1234a5b678c98d7efd6g54h32i101j2k")
@click.option('--vertical_columns', '-TABLE_COLUMNS', prompt="Write column names of table in a list",
help="Write column names of table eg. ['ITEM NO', 'SUPPLIES/SERVICES', 'ESTIMATED QUANTITY', 'EST. AMOUNT']")
@click.option('--horizontal_columns', '-HORIZONTAL_MAPPING', prompt="Write horizontally mapped columns in a list")
@click.option('--horizontal_keywords', '-WORD_SPLIT', prompt="Write horizontally mapped keys for close words in a list")
def azure_form_api_calling(
        document_filepath,
        endpoint,
        subscription_key,
        vertical_columns,
        horizontal_columns,
        horizontal_keywords):
    table_dict = azure_api_ocr(document_filepath=document_filepath, 
            endpoint=endpoint, 
            subscription_key=subscription_key,
            api_type='azure', 
            api='form',
            vertical_columns=eval(vertical_columns),
            horizontal_columns=eval(horizontal_columns),
            horizontal_keywords=eval(horizontal_keywords)
        )
    return table_dict

@click.command()
@click.option('--endpoint', '-ENDPOINT', prompt="Write Azure ENDPOINT",
help="eg. 1.'https://something.cognitiveservices.azure.com/'")
@click.option('--subscription_key', '-KEY', prompt="Write Azure subscription Key",
help="eg.1234a5b678c98d7efd6g54h32i101j2k")
@click.option('--operation_url', '-URL', prompt="Write operation url for file upload in Azure. Default is None",
default=None, show_default='None')
@click.option('--vertical_columns', '-TABLE_COLUMNS', prompt="Write column names of table in a list",
help="Write column names of table eg. ['ITEM NO', 'SUPPLIES/SERVICES', 'ESTIMATED QUANTITY', 'EST. AMOUNT']")
@click.option('--horizontal_columns', '-HORIZONTAL_MAPPING', prompt="Write horizontally mapped columns in a list")
@click.option('--horizontal_keywords', '-WORD_SPLIT', prompt="Write horizontally mapped keys for close words in a list")
def azure_read_api_calling_operation_url(endpoint, subscription_key, operation_url=None, vertical_columns=None, horizontal_columns=None, horizontal_keywords=None):
    if operation_url:
        if vertical_columns is None:
            vertical_columns = []
        if horizontal_columns is None:
            horizontal_columns = []
        if horizontal_keywords is None:
            horizontal_keywords = []
        table_dict = azure_api_ocr(
                document_filepath=None,
                endpoint=endpoint,
                subscription_key=subscription_key,
                operation_url=operation_url,
                api_type='azure', 
                api='read',
                vertical_columns=vertical_columns,
                horizontal_columns=horizontal_columns,
                horizontal_keywords=horizontal_keywords
            )
    else:
        table_dict = {}
    return table_dict

@click.command()
@click.option('--endpoint', '-ENDPOINT', prompt="Write Azure ENDPOINT",
help="eg. 1.'https://something.cognitiveservices.azure.com/'")
@click.option('--subscription_key', '-KEY', prompt="Write Azure subscription Key",
help="eg.1234a5b678c98d7efd6g54h32i101j2k")
@click.option('--ocr_outputs', '-OUTPUT', prompt="Please provide ocr_output if exist. Default is False",
default=False, show_default='False')
@click.option('--vertical_columns', '-TABLE_COLUMNS', prompt="Write column names of table in a list",
help="Write column names of table eg. ['ITEM NO', 'SUPPLIES/SERVICES', 'ESTIMATED QUANTITY', 'EST. AMOUNT']",)
@click.option('--horizontal_columns', '-HORIZONTAL_MAPPING', prompt="Write horizontally mapped columns in a list")
@click.option('--horizontal_keywords', '-WORD_SPLIT', prompt="Write horizontally mapped keys for close words in a list")
def azure_read_api_calling_ocr_outputs(endpoint, subscription_key, ocr_outputs=None, vertical_columns=None, horizontal_columns=None, horizontal_keywords=None):
    if ocr_outputs:
        if vertical_columns is None:
            vertical_columns = []
        if horizontal_columns is None:
            horizontal_columns = []
        if horizontal_keywords is None:
            horizontal_keywords = []
        table_dict = azure_api_ocr(
                document_filepath=None,
                endpoint=endpoint,
                subscription_key=subscription_key,
                ocr_outputs=ocr_outputs,
                api_type='azure', 
                api='read',
                vertical_columns=vertical_columns,
                horizontal_columns=horizontal_columns,
                horizontal_keywords=horizontal_keywords
            )
    else:
        table_dict = {}
    return table_dict

@click.command()
@click.option('--is_ocr_output', '-IS_OUTPUT',
prompt="Do you have ocr_output? [Y/N]")
@click.option('--is_operational_url', '-IS_URL',
prompt="Do you have operational_url? [Y/N]")
def read_api_calling(is_ocr_output, is_operational_url):
    if is_ocr_output.lower()=='y':
        table_dict=azure_read_api_calling_ocr_outputs()
    elif is_operational_url.lower()=='y':
        table_dict=azure_read_api_calling_operation_url()
    else:
        table_dict = azure_read_api_calling()
    return table_dict

@click.command()
@click.option('--whichapi', '-W',
prompt="You have selected 'azure' api type. Please select any of the following \
Azure API service\n1. READ \n2. FORM\n",
default='READ', show_default='READ',
help="Azure API service default is 'read'")    
def azureapi(whichapi='read'):
    table_dict = {}
    if whichapi and whichapi.lower()=='read':
        table_dict=read_api_calling()
    elif whichapi and whichapi.lower()=='form':
        table_dict=azure_form_api_calling()
    return table_dict

@click.command()
@click.option('--api_type', '-T', default='azure', show_default='azure', 
prompt="Enter type of OCR service you want to use. Please write any of the following type \
\n1. azure",
help="Type of API default is 'azure'")
def custom_main(api_type):
    print(api_type)
    if api_type == 'azure':
        table_dict = azureapi()
    else:
        table_dict = dict()
    return table_dict
    
@click.command()
@click.option('--default', '-D', default='N', show_default='N',
prompt='Do you want to use default settings from secter folder [Y/N]')
def main(default=True):
    table_dict = dict()
    if default and default.lower()=='y':
        table_dict=azure_api_ocr()
    else:
        table_dict = custom_main()
    print(table_dict)

if __name__ == '__main__':
    main()