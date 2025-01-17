from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import glob


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class VOCDataset(data.Dataset):
    """Dataset class for the VOC dataset."""
    def __init__(self, root_A, root_B=None, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.files_A = sorted(glob.glob(os.path.join(root_A, '*.*')))
        self.files_B = sorted(glob.glob(os.path.join(root_B, '*.*'))) if root_B else None
    
    def __getitem__(self, index):
        # 加载A域图像
        image_A = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        if self.transform:
            image_A = self.transform(image_A)
            
        if self.mode == 'train' and self.files_B:
            # 加载B域图像
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B)-1)]).convert('RGB')
            if self.transform:
                image_B = self.transform(image_B)
            # 返回两个域的图像和标签
            return image_A, image_B, torch.tensor([0]), torch.tensor([1])
        
        return image_A, torch.tensor([0])
    
    def __len__(self):
        return len(self.files_A)


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    
    if dataset == 'VOC':
        transform = []
        transform.append(T.Resize(crop_size))
        transform.append(T.RandomCrop(crop_size))
        transform.append(T.RandomHorizontalFlip())
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)

        if mode == 'train':
            trainA = os.path.join(image_dir, 'trainA')
            trainB = os.path.join(image_dir, 'trainB')
            dataset = VOCDataset(root_A=trainA, root_B=trainB, transform=transform, mode='train')
        else:
            testA = os.path.join(image_dir, 'testA')
            dataset = VOCDataset(root_A=testA, transform=transform, mode='test')
    
    data_loader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=(mode=='train'),
                                num_workers=num_workers)
    return data_loader