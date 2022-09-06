#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 22 Aug, 2022

#Get rid of temp files

import os

locations = [
    "temp/"
]

def cleanup():
    for directory in locations:
        files = os.listdir(directory)
        for f in files:
            try:
                os.remove(f'{directory}{f}')
            except:
                sub_files = os.listdir(f'{directory}{f}')
                for g in sub_files:
                    os.remove(f'{directory}{f}/{g}')
                os.rmdir(f'{directory}{f}')
