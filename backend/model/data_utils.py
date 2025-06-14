# backend/model/data_utils.py

import os
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((155, 220)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class UnifiedSignatureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        datasets = {
            "CEDAR": ["forgeries", "original"],
            "BHSig260-Bengali": ["F", "G"],
            "BHSig260-Hindi": ["F", "G"]
        }

        for dataset_name, label_keys in datasets.items():
            dataset_path = os.path.join(root_dir, dataset_name, dataset_name)
            if not os.path.exists(dataset_path):
                continue

            for person_folder in os.listdir(dataset_path):
                person_path = os.path.join(dataset_path, person_folder)
                if not os.path.isdir(person_path): continue

                for filename in os.listdir(person_path):
                    filepath = os.path.join(person_path, filename)
                    fname_lower = filename.lower()

                    if dataset_name == "CEDAR":
                        if fname_lower.startswith("original"):
                            label = 1
                        elif fname_lower.startswith("forgeries"):
                            label = 0
                        else: continue
                    else:
                        if "-g-" in fname_lower:
                            label = 1
                        elif "-f-" in fname_lower:
                            label = 0
                        else: continue

                    self.samples.append((filepath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


class SignaturePairDataset(Dataset):
    def __init__(self, signature_dataset, transform=None, max_pairs=10000):
        self.transform = transform
        self.pairs = []

        genuine = [s for s in signature_dataset.samples if s[1] == 1]
        forged = [s for s in signature_dataset.samples if s[1] == 0]

        user_to_genuine = {}
        user_to_forged = {}

        for path, label in signature_dataset.samples:
            user_folder = path.split("/")[-2]
            if label == 1:
                user_to_genuine.setdefault(user_folder, []).append(path)
            else:
                user_to_forged.setdefault(user_folder, []).append(path)

        user_ids = list(user_to_genuine.keys())

        for _ in range(max_pairs):
            if random.random() < 0.5:
                user = random.choice(user_ids)
                if len(user_to_genuine[user]) < 2:
                    continue
                p1, p2 = random.sample(user_to_genuine[user], 2)
                label = 1
            else:
                user = random.choice(user_ids)
                if user in user_to_forged and user_to_genuine[user]:
                    p1 = random.choice(user_to_genuine[user])
                    p2 = random.choice(user_to_forged[user])
                else:
                    u1, u2 = random.sample(user_ids, 2)
                    p1 = random.choice(user_to_genuine[u1])
                    p2 = random.choice(user_to_genuine[u2])
                label = 0

            self.pairs.append((p1, p2, label))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = Image.open(path1).convert("L")
        img2 = Image.open(path2).convert("L")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label
