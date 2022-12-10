"""Modified from Gluon CV"""
"""Prepare ImageNet V2 datasets: https://github.com/modestyachts/ImageNetV2"""


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
                      default='~/datasets/ImageNetV2/',
                      help='dataset directory on disk')
  parser.add_argument('--overwrite', action='store_true',
                      help='overwrite downloaded files if set, in case they are corrupted')
  args = parser.parse_args()
  return args

#####################################################################################
# Download and extract the datasets into ``path``


def download_me(path, overwrite=False):
  _DOWNLOAD_URLS = [
      'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz',
      'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz',
      'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-topimages.tar.gz'
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


# rename the folder names so torch ImageFolder works
def rename_folder_names(path):
  imagenetv2 = ['matched-frequency', 'threshold0.7', 'topimages']

  for c in imagenetv2:
    folder = 'imagenetv2-{}/'.format(c)
    root = os.path.join(path, folder)

    for d in os.scandir(root):
      if d.is_dir():
        if len(d.name) == 4:
          break
        os.rename(d, os.path.join(root, "{:04d}".format(int(d.name))))


if __name__ == '__main__':
  args = parse_args()
  path = os.path.expanduser(args.download_dir)
  makedirs(path)
  download_me(path, overwrite=args.overwrite)

  # rename
  rename_folder_names(path)

  # make symlink
  os.symlink(os.path.join(path, 'imagenetv2-matched-frequency'),
             os.path.join(_TARGET_DIR, 'imagenetv2-matched-frequency'))
  os.symlink(os.path.join(path, 'imagenetv2-threshold0.7'),
             os.path.join(_TARGET_DIR, 'imagenetv2-threshold0.7'))
  os.symlink(os.path.join(path, 'imagenetv2-topimages'),
             os.path.join(_TARGET_DIR, 'imagenetv2-topimages'))
