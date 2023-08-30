import random
import hashlib
import numpy as np
import cv2

def make_random_rectangle_mask(
    height, width, margin=10, bbox_min_size=100, bbox_max_size=512, min_times=1, max_times=2
):
    mask = np.zeros((height, width), np.float32)

    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)

    times = np.random.randint(min_times, max_times + 1)

    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)

        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)

        mask[start_y : start_y + box_height, start_x : start_x + box_width] = 1

    return mask


def make_random_irregular_mask(height, width, max_angle=4, max_len=60, max_width=256, min_times=1, max_times=2):
    mask = np.zeros((height, width), np.float32)

    times = np.random.randint(min_times, max_times + 1)

    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)

        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)

            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle

            length = 10 + np.random.randint(max_len)

            brush_w = 5 + np.random.randint(max_width)

            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)

            choice = random.randint(0, 2)

            if choice == 0:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif choice == 1:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1.0, thickness=-1)
            elif choice == 2:
                radius = brush_w // 2
                mask[start_y - radius : start_y + radius, start_x - radius : start_x + radius] = 1
            else:
                assert False

            start_x, start_y = end_x, end_y

    return mask


class OutpaintingMaskGenerator:
    def __init__(
        self,
        min_padding_percent: float = 0.04,
        max_padding_percent: int = 0.25,
        left_padding_prob: float = 0.5,
        top_padding_prob: float = 0.5,
        right_padding_prob: float = 0.5,
        bottom_padding_prob: float = 0.5,
        is_fixed_randomness: bool = False,
    ):
        """
        is_fixed_randomness - get identical paddings for the same image if args are the same
        """
        self.min_padding_percent = min_padding_percent
        self.max_padding_percent = max_padding_percent
        self.probs = [left_padding_prob, top_padding_prob, right_padding_prob, bottom_padding_prob]
        self.is_fixed_randomness = is_fixed_randomness

        assert self.min_padding_percent <= self.max_padding_percent
        assert self.max_padding_percent > 0
        assert (
            len([x for x in [self.min_padding_percent, self.max_padding_percent] if (x >= 0 and x <= 1)]) == 2
        ), "Padding percentage should be in [0,1]"
        assert sum(self.probs) > 0, f"At least one of the padding probs should be greater than 0 - {self.probs}"
        assert (
            len([x for x in self.probs if (x >= 0) and (x <= 1)]) == 4
        ), f"At least one of padding probs is not in [0,1] - {self.probs}"

    def apply_padding(self, mask, coord):
        mask[
            int(coord[0][0] * self.img_h) : int(coord[1][0] * self.img_h),
            int(coord[0][1] * self.img_w) : int(coord[1][1] * self.img_w),
        ] = 1
        return mask

    def get_padding(self, size):
        n1 = int(self.min_padding_percent * size)
        n2 = int(self.max_padding_percent * size)
        return self.rnd.randint(n1, n2) / size

    @staticmethod
    def _img2rs(img):
        arr = np.ascontiguousarray(img.astype(np.uint8))
        str_hash = hashlib.sha1(arr).hexdigest()
        res = hash(str_hash) % (2**32)
        return res

    def __call__(self, height, width, channles=3, iter_i=None, raw_image=None):
        _, self.img_h, self.img_w = channles, height, width
        mask = np.zeros((self.img_h, self.img_w), np.float32)
        at_least_one_mask_applied = False

        if self.is_fixed_randomness:
            assert raw_image is not None, "Cant calculate hash on raw_image=None"
            rs = self._img2rs(raw_image)
            self.rnd = np.random.RandomState(rs)
        else:
            self.rnd = np.random

        coords = [
            [(0, 0), (1, self.get_padding(size=self.img_h))],
            [(0, 0), (self.get_padding(size=self.img_w), 1)],
            [(0, 1 - self.get_padding(size=self.img_h)), (1, 1)],
            [(1 - self.get_padding(size=self.img_w), 0), (1, 1)],
        ]

        for pp, coord in zip(self.probs, coords):
            if self.rnd.random() < pp:
                at_least_one_mask_applied = True
                mask = self.apply_padding(mask=mask, coord=coord)

        if not at_least_one_mask_applied:
            idx = self.rnd.choice(range(len(coords)), p=np.array(self.probs) / sum(self.probs))
            mask = self.apply_padding(mask=mask, coord=coords[idx])
        return mask