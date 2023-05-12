from train import *

parser = ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--model', default='/save_models/cifar10_resnet18best.pth', type=str, help='model weight file')

args = parser.parse_args()

model = LitResnet()
model.load_from_checkpoint(args.model)
model.eval()

trainer.test(model, dataloaders=cifar10_dm.test_dataloader())

metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
del metrics["step"]
metrics.set_index("epoch", inplace=True)
display(metrics.dropna(axis=1, how="all").head())
sns.relplot(data=metrics, kind="line")

print('Done')