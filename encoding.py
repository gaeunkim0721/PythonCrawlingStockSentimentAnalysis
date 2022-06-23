import chardet
import pandas as pd
filename = r"C:\Users\Admin\Desktop\team3\기사모음.csv"
with open(filename, 'rb') as f:
    result = chardet.detect(f.read())  # or read() if the file is small.
    print(result['encoding'])