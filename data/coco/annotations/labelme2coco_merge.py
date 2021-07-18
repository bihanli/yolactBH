
import labelme2coco
import argparse

parser = argparse.ArgumentParser(description='merge datasets.')
parser.add_argument('input', type=str,help='Path to image folder.')
parser.add_argument('output', type=str, help='Path to annotations file.')
args = parser.parse_args()
def main(args):
    labelme2coco.convert(args.input,args.output)

if __name__ == "__main__":
    main(args)
