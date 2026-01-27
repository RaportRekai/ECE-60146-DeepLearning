"""
ECE 60146 - Homework 2
Name : [ YOUR NAME ]
Email : [ YOUR EMAIL ]

ImageNet Data Loading and Augmentation
"""
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time

# REQUIRED CLASSES
# Labels match both the provided subset AND Hugging Face
REQUIRED_CLASSES = {
    1: "goldfish",
    151: "Chihuahua",
    281: "tabby cat",
    291: "lion",
    325: "sulphur butterfly",
    386: "African elephant",
    430: "basketball",
    466: "bullet train",
    496: "Christmas stocking",
    950: "orange",
}


def get_class_name(label):
    """ Get class name from integer label. """
    return REQUIRED_CLASSES.get(label, f"class_{label}")


# TRANSFORMS
transform_custom = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Resize((224, 224)),
    transforms.GaussianBlur(3, (0.3, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_basic = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# OPTION A : Dataset class for provided subset
class ImageNetSubset(Dataset):
    """
    Dataset for loading the provided ImageNet subset.

    Folder structure:
    imagenet_subset/
        1/ (goldfish)
            00001.JPEG
            ...
        15/ (robin)
            ...

    Args:
        root (str): Path to imagenet_subset folder
        class_labels (list): List of integer labels to load (e.g., [1, 15, 151])
        images_per_class (int): Number of images to load per class
        transform (callable): Transform to apply to images
    """

    def __init__(self, root, class_labels, images_per_class=5, transform=None):
        self.root = root
        self.class_labels = class_labels
        self.images_per_class = images_per_class
        self.transform = transform
        self.samples = []  # List of (image_path, label)
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} images from {len(class_labels)} classes")

    def _load_samples(self):
        """ Load image paths for each requested class. """
        # TODO : Implement this method
        # For each label in self.class_labels:
        # 1. Build path to class folder: os.path.join(self.root, str(label))
        # 2. List all .JPEG/.jpg/.png files in that folder
        # 3. Take first self.images_per_class images
        # 4. Append (image_path, label) to self.samples
        for label in self.class_labels:
            class_dir = os.path.join(self.root,str(label))
            files = os.listdir(class_dir)
            ext = r'\.(jpg|jpeg|png)$'
            image_files = [f for f in files if re.search(ext, f, re.IGNORECASE)]
            print("The image files are:", image_files)
            for img in image_files[:self.images_per_class]:
                self.samples.append((os.path.join(class_dir, img), label))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            image (Tensor): Transformed image
            label (int): Class label (e.g., 1 for goldfish)
        """
        # TODO : Implement this method
        # 1. Get image_path, label from self.samples[index]
        # 2. Load image using PIL: Image.open(path).convert('RGB')
        # 3. Apply self.transform if not None
        # 4. Return (image, label)
        img = Image.open(self.samples[index][0]).convert('RGB')
        if self.transform!=None:
            trnsfrm_img = self.transform(img)
            return (trnsfrm_img, self.samples[index][1])
        else:
            return (img, self.samples[index][1]) 


# OPTION B : Dataset class for Hugging Face ImageNet
class ImageNetHuggingFace(Dataset):
    """
    Dataset for loading ImageNet from Hugging Face.
    
    Requires: pip install datasets huggingface_hub
    And: huggingface-cli login

    Args:
        class_labels (list): List of integer labels to load (e.g., [1, 15, 151])
        images_per_class (int): Number of images per class
        transform (callable): Transform to apply
        split (str): 'train' or 'validation'
    """

    def __init__(self, class_labels, images_per_class=5, transform=None, split='train'):
        self.class_labels = set(class_labels)
        self.images_per_class = images_per_class
        self.transform = transform
        self.samples = []  # List of (PIL.Image, label)
        self._load_from_huggingface(split)
        
        print(f"Loaded {len(self.samples)} images from {len(class_labels)} classes")

    def _load_from_huggingface(self, split):
        """ Load images from Hugging Face dataset. """
        from datasets import load_dataset
        
        print("Loading from Hugging Face (this may take a few minutes)...")
        
        # Load with streaming to avoid downloading entire dataset
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            streaming=True,
            trust_remote_code=True
        )
        
        # TODO : Implement this method
        # Track counts per class
        counts = {label: 0 for label in self.class_labels}
        
        for sample in dataset:
            label = sample['label'] # Integer 0-999
            
            if label in self.class_labels and counts[label] < self.images_per_class:
                image = sample['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                self.samples.append((image, label))
                counts[label] += 1
                
                # Progress
                total = sum(counts.values())
                if total % 10 == 0:
                    print(f"Collected {total} images...")
                
                # Stop when done
                if all(c >= self.images_per_class for c in counts.values()):
                    break
        pass # Remove when implementing

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """ Returns (image_tensor, label) """
        # TODO : Implement
        pass


# Custom Dataset for your own images
class CustomDataset(Dataset):
    """ Dataset for loading custom images from a folder. """
    
    def __init__(self, root, transform=None,sample_len=0):
        self.root = root
        self.transform = transform
        
        # TODO : Load all image paths from root folder
        # Filter for : .jpg, .jpeg, .png, .bmp, .gif
        self.image_paths = []
        files = os.listdir(self.root)
        ext = r'\.(jpg|jpeg|png|bmp|gif|webp)$'
        image_files = [f for f in files if re.search(ext, f)]
        sample_len = len(image_files) if sample_len==0 else sample_len
        for i in range(sample_len):
            self.image_paths.append(os.path.join(self.root,image_files[i%len(image_files)]))

        print(f"Found {len(self.image_paths)} images in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # TODO : Implement
        img = Image.open(self.image_paths[index%len(self.image_paths)]).convert('RGB')
        if self.transform!=None:
            trnsfrm_img = self.transform(img)
            return trnsfrm_img
        else:
            return img


# Utility Functions
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Denormalize a tensor image for display. """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def show_images(images, titles, rows, cols, figsize=(15, 10), save_path=None):
    """ Display a grid of images. """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i >= len(axes):
            break
            
        if isinstance(img, torch.Tensor):
            if img.min() < 0 or img.max() > 1:
                img = denormalize(img)
            img = img.permute(1, 2, 0).numpy()
            
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def set_seed(seed=60146):
    """ Set random seed for reproducibility. """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Main
if __name__ == "__main__":
    
    # Required class labels
    required_labels = list(REQUIRED_CLASSES.keys())
    print("Required classes:", required_labels)
    for label in required_labels:
        print(f"{label}: {get_class_name(label)}")
        
    # Task 1 : Load ImageNet (50 images: 10 classes x 5 images)
    print("\n" + "=" * 60)
    print("Task 1 : Loading ImageNet")
    print("=" * 60)
    
    # TODO : Create dataset using Option A or B
    imagenet_dataset = ImageNetSubset(
        root='./imagenet_subset',
        class_labels=required_labels,
        images_per_class=5,
        transform=transform_custom
    )
    
    # Task 2 : Visualize ImageNet (1 image per class = 10 images)
    print("\n" + "=" * 60)
    print("Task 2 : Visualizing ImageNet")
    print("=" * 60)
    
    
    
    
    images = []
    labels = []
    imagenet_dataset.samples.sort(key=lambda x: x[1])
    for i in range(0, len(imagenet_dataset.samples), imagenet_dataset.images_per_class):
        img = Image.open(imagenet_dataset.samples[i][0]).convert('RGB')
        images.append(np.array(img) / 255.0)
        labels.append(imagenet_dataset.samples[i][1])
    print(len(images))
    show_images(images, labels, rows=2, cols=5, figsize=(15, 6), save_path='imagenet_samples.png')
    
    
    
    
    
    # Task 3 : Show augmentation effects
    print("\n" + "=" * 60)
    print("Task 3 : Augmentation comparison")
    print("=" * 60)
    


    # TODO : For 3 images, show original + 3 augmented versions
    for i in range(3):
        tr_imgs = []
        orig_img = Image.open(imagenet_dataset.samples[i][0]).convert('RGB')
        tr_imgs.append(transforms.ToTensor()(orig_img))
        labels = []
        labels.append("Original")
        for j in range(3):
            tr_img = imagenet_dataset[i][0]
            tr_imgs.append(tr_img)
            labels.append(f"Augmented {j+1}")
        show_images(tr_imgs, labels, rows=1, cols=4, figsize=(15, 5), save_path='augmentation_comparison.png')
    
    
    
    
    # Task 4 : Custom dataset
    print("\n" + "=" * 60)
    print("Task 4 : Custom dataset")
    print("=" * 60)
    
    
    
    custom_dataset = CustomDataset(
        root='./my_images',
        transform=transform_custom
    )
    augmented_imgs = []
    titles = []
    for i in range(50):
        augmented_imgs.append(custom_dataset[i % 20])
        titles.append(f"Augmented {i+1}")
    show_images(augmented_imgs[:10], titles[:10], rows=2, cols=5, figsize=(15, 6), save_path='custom_samples.png')
    
    
    
    
    
    # Task 5 : DataLoader performance
    print("\n" + "=" * 60)
    print("Task 5 : DataLoader performance")
    print("=" * 60)
    dumb_img = []
    perf_img = CustomDataset(root='./my_images',transform=transform_custom,sample_len=1000)
    
    # Raw loading time
    start = time.perf_counter()
    for i in range(1000):
        tempr = perf_img[i]
    elapsed = time.perf_counter() - start
    print(f"Raw loading time (1000 images): {elapsed:.4f}s")

    # DataLoader configurations
    print(f"Time elapsed for different DataLoader configurations (1000 images):")
    configs = [(16, 0), (16, 4), (64, 0), (64, 4)]
    pin = torch.cuda.is_available()
    for batch_size, num_workers in configs:
        
        loader = DataLoader(
            perf_img,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin
        )
        start = time.perf_counter()
        for _ in loader:
            pass
        elapsed = time.perf_counter() - start
        print(f"batch_size={batch_size}, num_workers={num_workers}: {elapsed:.4f}s")
    
    # Task 6 : RGB statistics
    print("\n" + "=" * 60)
    print("Task 6 : RGB statistics")
    print("=" * 60)
    print(f"Calculating RGB channel min/max before and after normalization for the first batch of \
           each batch config that comes out of the data loader...")
    
    # "Before" transform: Just converts to Tensor (0-1 range)
    transform_before = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # "After" transform: Converts AND Normalizes
    transform_after = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    # 3. Create a temporary loader just for this task
    # We use batch_size=16 to satisfy the "at least one batch" requirement
    stats_dataset = CustomDataset(root='./my_images', transform=transform_before) 
    stats_loader = DataLoader(stats_dataset, batch_size=16, shuffle=False)
    
    # 4. Compute "Before" Stats
    # Get just ONE batch (the first one)
    batch_before = next(iter(stats_loader))
    
    # Note: dim=(0, 2, 3) because batch shape is [Batch, Channel, Height, Width]
    # We collapse Batch(0), Height(2), and Width(3), leaving Channel(1)
    min_before = batch_before.amin(dim=(0, 2, 3))
    max_before = batch_before.amax(dim=(0, 2, 3))
    
    print(f"Before Norm Min (R,G,B): {min_before.tolist()}")
    print(f"Before Norm Max (R,G,B): {max_before.tolist()}")

    # 5. Compute "After" Stats
    # Update dataset transform to include Normalization
    stats_dataset.transform = transform_after
    
    # Re-fetch the same batch (now normalized)
    # We re-create the iterator to start from the beginning
    stats_loader = DataLoader(stats_dataset, batch_size=16, shuffle=False)
    batch_after = next(iter(stats_loader))
    
    min_after = batch_after.amin(dim=(0, 2, 3))
    max_after = batch_after.amax(dim=(0, 2, 3))
    
    print(f"After Norm Min (R,G,B):  {min_after.tolist()}")
    print(f"After Norm Max (R,G,B):  {max_after.tolist()}")    
    
    
    # Task 7 : Reproducibility
    print("\n" + "=" * 60)
    print("Task 7 : Reproducibility")
    print("=" * 60)
    
    # TODO : Test with and without set_seed(60146)
    # without seed
    repro_dataset = ImageNetSubset(
        root='./imagenet_subset', 
        class_labels=required_labels, 
        transform=transform_basic
    )

    # starting loader for the first time
    no_seed_load = DataLoader(
        repro_dataset,
        batch_size=2,
        shuffle=True,
    )

    batch_ns_1, labels_ns_1 = next(iter(no_seed_load))
    show_images(batch_ns_1, labels_ns_1, rows=1, cols=2, figsize=(15, 6), save_path='random_samples_1.png')

    # starting loader for the second time
    no_seed_load = DataLoader(
        repro_dataset,
        batch_size=2,
        shuffle=True,
    )

    batch_ns_1, labels_ns_1 = next(iter(no_seed_load))
    show_images(batch_ns_1, labels_ns_1, rows=1, cols=2, figsize=(15, 6), save_path='random_samples_2.png')

    print("Both displayed images should be unidentical without using the seed.")
    

    # with seed
    set_seed(60146)
    
    # starting loader for the first time
    seed_load = DataLoader(
        repro_dataset,
        batch_size=2,
        shuffle=True,
    )

    batch_s_1, labels_s_1 = next(iter(seed_load))
    show_images(batch_s_1, labels_s_1, rows=1, cols=2, figsize=(15, 6), save_path='seed_samples_1.png')

    set_seed(60146)
    # starting loader for the second time
    seed_load = DataLoader(
        repro_dataset,
        batch_size=2,
        shuffle=True,
    )

    batch_s_1, labels_s_1 = next(iter(seed_load))
    show_images(batch_s_1, labels_s_1, rows=1, cols=2, figsize=(15, 6), save_path='seed_samples_2.png')
    print("Both displayed images should be identical when using the seed.")

    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
