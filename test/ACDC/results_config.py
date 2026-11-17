import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nunet', type=int, default=3)
parser.add_argument('--base_filters', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--data_path', type=str, default='../../../../../acdcd/data/ACDC')
parser.add_argument('--flod', type=int, default=0)
args = parser.parse_args()
nunet = args.nunet
base_filters = args.base_filters
batch_size = args.batch_size
flod = args.flod
data_path = args.data_path