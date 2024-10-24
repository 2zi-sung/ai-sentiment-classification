import re

def clean_text(text):
    cleaned_text = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', text)
    cleaned_text = cleaned_text.replace('<br','').replace('>','')

    return cleaned_text