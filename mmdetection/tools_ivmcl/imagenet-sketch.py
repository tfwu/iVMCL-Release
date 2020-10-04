"""Modified from Gluon CV"""
"""Prepare ImageNet Sketch datasets: https://github.com/HaohanWang/ImageNet-Sketch"""
import os
import shutil
import argparse
import tarfile
import pathlib

from download import download
from filesystem import makedirs

try:
  from google_drive_downloader import GoogleDriveDownloader as gdd
except ImportError:
    raise ImportError("Please: pip install googledrivedownloader")


where_am_I = pathlib.Path(__file__).parent.absolute()
_TARGET_DIR = os.path.expanduser(os.path.join(where_am_I, '../data/'))


def parse_args():
  parser = argparse.ArgumentParser(
      description='Initialize ImageNet V2 dataset.',
      epilog='Example: python imagenet_v2.py --download-dir path_to_your_dir',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--download-dir', type=str, required=True,
                      default='~/datasets/ImageNet-Sketch/',
                      help='dataset directory on disk')
  parser.add_argument('--overwrite', action='store_true',
                      help='overwrite downloaded files if set, in case they are corrupted')
  args = parser.parse_args()
  return args

#####################################################################################
# Download and extract the file into ``path``

def download_me(path, overwrite=False):
  # https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA
  dest_path=os.path.join(path, 'imagenetsketch.zip')
  if not os.path.exists(dest_path) or args.overwrite:
    gdd.download_file_from_google_drive(file_id='1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA',
                                      dest_path=os.path.join(path, 'imagenetsketch.zip'),
                                      unzip=True)



if __name__ == '__main__':
  args = parse_args()
  path = os.path.expanduser(args.download_dir)
  makedirs(path)
  download_me(path, overwrite=args.overwrite)

  # make symlink
  os.symlink(os.path.join(path, 'sketch'),
              os.path.join(_TARGET_DIR, 'imagenet-sketch'))

