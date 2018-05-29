'''
All needed data downloader
'''

import os
import subprocess
from tqdm import tqdm
import argparse

IMGS_URL = 'http://mcs2018-competition.visionlabs.ru/imgs.zip'
STUDENT_MODEL_IMGS_URL = 'http://mcs2018-competition.visionlabs.ru/student_model_imgs.zip'
SUBMIT_LIST_URL = 'http://mcs2018-competition.visionlabs.ru/submit_list.csv'
PAIR_LIST_URL = 'http://mcs2018-competition.visionlabs.ru/pairs_list.csv'

parser = argparse.ArgumentParser(description='download script')
parser.add_argument('--root',
                    required=True,
                    type=str, 
                    help='data root path, where files will be saved')
parser.add_argument('--main_imgs',
                    action='store_true',
                    help='download main imgs in directory $root/imgs')
parser.add_argument('--student_model_imgs',
                    action='store_true',
                    help='download student_model_imgs in directory'\
                         '$root/student_model_imgs')
parser.add_argument('--submit_list',
                    action='store_true',
                    help='download submit_list in $root/submit_list.csv')
parser.add_argument('--pairs_list',
                    action='store_true',
                    help='download pairs_list in $root/pairs_list.csv')
args = parser.parse_args()

def downloader(url, path):
	subprocess.call('wget -P {path} --verbose {url}'.format(path=path,
			               	                        url=url), 
	                 shell=True)
	# Or use cURL
	'''
	subprocess.call('curl -o {path} -v {url}'.format(path=path,
			               	                 url=url), 
	                 shell=True)
	'''
def main(args):
	if not os.path.isdir(args.root):
		os.makedirs(args.root)

	if args.main_imgs:
		print ('==> Main images downloading')
		downloader(IMGS_URL, args.root)
		print ('==> Main images downloaded')
		zipfile_path = os.path.join(args.root, 'imgs.zip')
		dir_path = os.path.join(args.root, 'imgs')
		
		if not os.path.isdir(dir_path):
			os.makedirs(dir_path)
		
		with zipfile.ZipFile(file_path,'r') as myzip:
			myzip.extractall(path=dir_path)

		print ('==> Main imgs extracted')

	if args.student_model_imgs:
		print ('==> Student model images downloading')
		downloader(STUDENT_MODEL_IMGS, args.root)
		print ('==> Student model images downloaded')
		ipfile_path = os.path.join(args.root, 'student_model_imgs.zip')
		dir_path = os.path.join(args.root, 'student_model_imgs')
		
		if not os.path.isdir(dir_path):
			os.makedirs(dir_path)
		
		with zipfile.ZipFile(file_path,'r') as myzip:
			myzip.extractall(path=dir_path)

		print ('==> Student model images extracted')

	if args.submit_list:
		print ('==> Submit list downloading')
		downloader(SUBMIT_LIST_URL, args.root)
		print ('==> Submit list downloaded')

	if args.pairs_list:
		print ('==> Pairs list downloading')
		downloader(PAIR_LIST_URL, args.root)
		print ('==> Pairs list downloaded')

if __name__ == '__main__':
	main(args)
