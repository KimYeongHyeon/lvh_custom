'''
source for filter video 
'''

import os
import argparse
import glob
import shutil

def main(args):
    src_file_list = glob.glob(os.path.join(args.src, '*.png'))
    patient_list = [filename.split('/')[-1].split('.')[0].split('_')[0] for filename in src_file_list]
    
    for patient in patient_list:
        try:
            shutil.copy(
                os.path.join(args.video, patient + '.avi'),
                os.path.join(args.dest, patient + '.avi')
            )
        except E as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True, help='directory to video dataset')
    parser.add_argument('-s', '--src', required=True, help='directory to source')
    parser.add_argument('-d', '--dest', required=True, help='directory to destination')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    main(args)
