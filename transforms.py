import torch
import torchvision.transforms.functional as F


class Resize:

    def __init__(self, size):

        self.height = size[0]
        self.width = size[1]

    def __call__(self, image, target):

        orig_w, orig_h = image.size

        image = F.resize(
            image,
            (self.height, self.width)
        )


        boxes = target["boxes"]

        scale_x = self.width / orig_w
        scale_y = self.height / orig_h

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        target["boxes"] = boxes

        return image, target


class ToTensor:

    def __call__(self, image, target):

        image = F.to_tensor(image)

        return image, target


class Compose:

    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, image, target):

        for t in self.transforms:

            image, target = t(image, target)

        return image, target