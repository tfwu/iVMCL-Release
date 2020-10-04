import argparse
import subprocess

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        # if logger is not None:
        #     logger.info(f'{k}')
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and \
                not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    state = {
        'state_dict': state_dict
    }
    torch.save(state, out_file)
    # sha = subprocess.check_output(['sha256sum', out_file]).decode()
    # if out_file.endswith('.pth'):
    #     out_file_name = out_file[:-4]
    # else:
    #     out_file_name = out_file
    # final_file = out_file_name + f'-{sha[:8]}.pth'
    # subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
