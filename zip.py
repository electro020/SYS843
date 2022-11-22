import zipfile
import os

with zipfile.ZipFile(".\database.zip", 'r') as zip_ref:
    zip_ref.extractall(os.curdir)