from train import *

parser = ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--model', default='./save_models/epoch=31-step=8976.ckpt', type=str, help='model weight file')

args = parser.parse_args()
trainer.test(ckpt_path=args.model)

print('Done')