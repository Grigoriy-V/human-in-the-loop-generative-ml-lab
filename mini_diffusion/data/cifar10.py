from torchvision import datasets, transforms


def build_cifar10(
    root: str,
    train: bool = True,
    download: bool = True,
    mirror_url: str | None = None,
    fake_data: bool = False,
    fake_size: int = 128,
):
    if mirror_url:
        datasets.CIFAR10.url = mirror_url
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if fake_data:
        return datasets.FakeData(
            size=fake_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
        )
    return datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
