
import json

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='merge datasets.')
parser.add_argument('input', type=str,help='Path to json file.')
args = parser.parse_args()
def main(args):
    with open (args.input,'r')as f: 
        file=json.load(f) 
    for i in file['images']:
        s=i['file_name']
        s=s.split('/')
        s=s[-1]
        i['file_name']=str(s)
    #output='annotations/'+args.input
    with open(args.input,'w')as ff:
        #print('output path=',output)
        json.dump(file,ff,sort_keys=True,indent=4)

if __name__ == "__main__":
    main(args)


