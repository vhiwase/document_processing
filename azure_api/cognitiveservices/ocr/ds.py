from dataclasses import dataclass

@dataclass
class OCRInputs:
    document_filepath: str
    endpoint: str = None
    subscription_key: str = None
    operation_url: str = None
    ocr_outputs: str = None        

@dataclass
class AzureOCR:
    document_filepath: str
    endpoint: str
    subscription_key: str
    
    def read_document_binaries(self):
        with open(self.document_filepath, 'rb') as f:
            self.document_binaries = f.read()
        return self.document_binaries 
    
    def __post_init__(self):
        self.read_document_binaries()

@dataclass
class OCRCoordinates:
    top_left_x: float
    top_left_y: float 
    top_right_x: float 
    top_right_y: float 
    bottom_right_x: float 
    bottom_right_y: float 
    bottom_left_x: float 
    bottom_left_y: float

@dataclass
class ReadResults:
    page: int 
    angle: float 
    width: float
    height: float 
    unit: str 

@dataclass
class OCRText(ReadResults, OCRCoordinates):
    text: str
    confidence: float = None

@dataclass    
class RequestInputs:
    header: dict = None
    text_recognition_url: str = None