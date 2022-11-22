import zipfile
import os

with zipfile.ZipFile("/home/ens/AP69690/database.zip", 'r') as zip_ref:
    zip_ref.extractall(os.curdir)