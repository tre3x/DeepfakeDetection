from training.datasets.classifier_dataset import DeepFakeClassifierDataset
from tqdm import tqdm
from torch.utils.data import DataLoader

data_train = DeepFakeClassifierDataset(mode='train', folds_csv="./folds.csv")
data_train.reset(1, 333)

train_data_loader = DataLoader(data_train, 8)

pbar = tqdm(enumerate(train_data_loader), total=40, desc="Epoch {}".format(1), ncols=0)
for i, sample in pbar:
     imgs = sample["image"]
     labels = sample["labels"]
     print(imgs.shape, labels.shape)
