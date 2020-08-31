# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 01:17:44 2020

@author: Diego
"""


import base64
from io import BytesIO 
import cPickle



with open("cnn_best_model_v3.pth", "rb") as f:
    byte = f.read(1)

#zlib.compress(byte)
# Base64 Encode the bytes
data_e = base64.b64encode(byte)

#filename ='base64_checkpoint.txt'

print(data_e)


#with open(filename, "wb") as output:
    #output.write(data_e)

# Save file to Dropbox

# Download file on server
#b64_str= self.Download('url')

# String Encode to bytes
#byte_data = b64_str.encode("UTF-8")

# Decoding the Base64 bytes
#str_decoded = base64.b64decode(byte_data)

# String Encode to bytes
#byte_decoded = str_decoded.encode("UTF-8")

# Decoding the Base64 bytes
#decoded = base64.b64decode(byte_decoded)

#torch.load(BytesIO(decoded))