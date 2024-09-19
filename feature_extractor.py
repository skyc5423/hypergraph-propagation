from torchvision.models import resnet18
import torch
import torchvision.transforms as transforms
from PIL import Image


class FeatureExtractor:

    def __init__(self, model='vit'):
        self.model = model

        self.transform = transforms.Compose([
            # ResNet18 expects a 224x224 image
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            # Normalize as ResNet18 was trained on ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        if self.model == 'vit':
            self.feature_extractor = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14')
            self.feature_extractor.eval()
        elif self.model == 'cnn':
            resnet = resnet18(pretrained=True)
            self.feature_extractor = torch.nn.Sequential(
                *list(resnet.children())[:-3])
            self.global_extractor = torch.nn.Sequential(
                *list(resnet.children())[-3:-1]
            )
        elif self.model == 'sift':
            pass
        else:
            raise ValueError('Invalid model')

    def extract_features(self, image_path) -> dict:
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            if self.model == 'vit':
                features = self.feature_extractor.forward_features(image)
                return {'local': features['x_norm_patchtokens'],
                        'global': features['x_norm_clstoken']}
            elif self.model == 'cnn':
                return self.feature_extractor(image)
