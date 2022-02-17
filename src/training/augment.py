from torchvision import transforms
import random
from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

SIMCLR_AUGMENT = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    NORMALIZE,
                ])