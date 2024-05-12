import random

import torch


class MockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        # available on linux/mac
        words_file: str = "/usr/share/dict/words",
        num_words: tuple[int, int] | int = (1, 100),
        num_iters: int = 100,
        # allow variable sized images of h between 500 and 1000, and w between 500 and 1000
        # if a tuple, then the range is fixed, otherwise the range is random
        img_size: tuple[tuple[int, int], tuple[int, int]] | tuple[int, int] = (
            (500, 1000),  # h_min, h_max
            (500, 1000),  # h_min, h_max
        ),
    ):
        self.num_iters = num_iters

        # just repeat it so random.randint can work for both words and size
        if len(num_words) == 1:
            num_words = (num_words[0], num_words[0])

        if isinstance(img_size[0], int):
            img_size = [[img_size[0], img_size[0]], [img_size[1], img_size[1]]]

        self.num_words = num_words
        self.image_height, self.image_width = img_size[0], img_size[1]

        with open(words_file, "r") as f:
            self.words = [word.strip() for word in f.readlines()]

    def __len__(self) -> int:
        return self.num_iters

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        return self.generate()

    def _make_image_dim(self) -> tuple[int, int]:
        return [
            random.randint(*self.image_height),
            random.randint(*self.image_width),
        ]

    def generate(self):
        num_words = random.randint(*self.num_words)
        text = " ".join([random.choice(self.words) for _ in range(num_words)])
        image = torch.rand(3, *self._make_image_dim())
        # use 'images' to allow unpacking to processor
        return {"text": text, "images": image}
