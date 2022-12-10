"""Modified from Gluon CV"""
"""Prepare ImageNet-adv datasets: https://github.com/hendrycks/natural-adv-examples"""
import os
import shutil
import argparse
import tarfile
import pathlib

from download import download
from filesystem import makedirs

where_am_I = pathlib.Path(__file__).parent.absolute()
_TARGET_DIR = os.path.expanduser(os.path.join(where_am_I, '../data/'))


def parse_args():
  parser = argparse.ArgumentParser(
      description='Initialize ImageNet V2 dataset.',
      epilog='Example: python imagenet_v2.py --download-dir path_to_your_dir',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--download-dir', type=str, required=True,
                      default='~/datasets/ImageNet-Adv/',
                      help='dataset directory on disk')
  parser.add_argument('--overwrite', action='store_true',
                      help='overwrite downloaded files if set, in case they are corrupted')
  args = parser.parse_args()
  return args

#####################################################################################
# Download and extract the datasets into ``path``

def download_me(path, overwrite=False):
  _DOWNLOAD_URLS = [
    'https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar',
    'https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar'
    ]
  for url in _DOWNLOAD_URLS:
    filename = download(url, path=path, overwrite=overwrite)
    # extract
    with tarfile.open(filename) as tar:
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(tar, path=path)


if __name__ == '__main__':
  args = parse_args()
  path = os.path.expanduser(args.download_dir)
  makedirs(path)
  download_me(path, overwrite=args.overwrite)

  # make symlink
  os.symlink(os.path.join(path, 'imagenet-a'),
              os.path.join(_TARGET_DIR, 'imagenet-adv-a'))
  os.symlink(os.path.join(path, 'imagenet-o'),
              os.path.join(_TARGET_DIR, 'imagenet-adv-o'))

