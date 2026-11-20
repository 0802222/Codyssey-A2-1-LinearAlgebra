"""SVD 압축 실험용 64x64 흑백 이미지를 만드는 스크립트.

원본은 Matplotlib에 들어 있는 공개(퍼블릭 도메인) 예제 사진 grace_hopper.jpg이다.
컬러 사진을 흑백으로 바꾸고, 가운데를 정사각형으로 자른 뒤,
픽셀 블록 평균으로 64x64로 줄인다.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook

from src.constants import IMAGE_SIZE_PIXELS, SAMPLE_IMAGE_PATH, FloatArray

# 사람 눈이 느끼는 밝기에 맞춘 RGB 가중치 (ITU-R BT.601 표준)
LUMINANCE_WEIGHTS = np.array([0.299, 0.587, 0.114])

# 8비트 이미지의 최대 밝기 값
MAX_PIXEL_VALUE = 255.0


# 컬러 이미지를 0~1 범위의 흑백 이미지로 바꾼다
def to_grayscale(color_image: FloatArray) -> FloatArray:
    """RGB 이미지를 밝기 가중합으로 흑백 이미지로 바꾼다.

    수식: gray = 0.299·R + 0.587·G + 0.114·B

    Args:
        color_image: (높이, 너비, 3) 모양의 0~255 RGB 이미지

    Returns:
        (높이, 너비) 모양의 0~1 흑백 이미지
    """
    # 채널 축(마지막 축)으로 가중합을 구하고 0~1로 나눈다
    return np.asarray(color_image[..., :3] @ LUMINANCE_WEIGHTS / MAX_PIXEL_VALUE)


# 가운데를 정사각형으로 자르고 블록 평균으로 줄인다
def downsample_square(gray_image: FloatArray, size_pixels: int) -> FloatArray:
    """이미지 가운데를 정사각형으로 자른 뒤 블록 평균으로 size x size로 줄인다.

    Args:
        gray_image: (높이, 너비) 모양의 흑백 이미지
        size_pixels: 결과 이미지 한 변의 픽셀 수

    Returns:
        (size_pixels, size_pixels) 모양의 흑백 이미지
    """
    # 높이와 너비 중 짧은 쪽에 맞춘다
    side_length = min(gray_image.shape)

    # 블록 하나의 크기 (원본 픽셀 몇 개를 하나로 합칠지)
    block_size = side_length // size_pixels

    # 블록 크기로 나누어떨어지게 자를 길이
    crop_length = block_size * size_pixels

    # 위쪽 자르기 시작 위치 (가운데 정렬)
    top_offset = (gray_image.shape[0] - crop_length) // 2

    # 왼쪽 자르기 시작 위치 (가운데 정렬)
    left_offset = (gray_image.shape[1] - crop_length) // 2

    # 가운데 정사각형을 잘라 낸다
    cropped_image = gray_image[
        top_offset : top_offset + crop_length, left_offset : left_offset + crop_length
    ]

    # (64, 블록, 64, 블록) 모양으로 바꿔 블록마다 평균을 낸다
    blocks = cropped_image.reshape(size_pixels, block_size, size_pixels, block_size)

    # 블록 축(1, 3)으로 평균을 내 64x64를 만든다
    return np.asarray(blocks.mean(axis=(1, 3)))


# 원본 사진을 읽어 64x64 흑백 PNG로 저장한다
def main() -> None:
    """예제 사진을 64x64 흑백 PNG로 저장한다."""
    # Matplotlib 예제 사진 경로를 찾는다
    source_path = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)

    # 사진을 0~255 RGB 배열로 읽는다
    color_image = plt.imread(source_path).astype(np.float64)

    # 흑백으로 바꾼다
    gray_image = to_grayscale(color_image)

    # 64x64로 줄인다
    small_image = downsample_square(gray_image, IMAGE_SIZE_PIXELS)

    # 0~1 값을 그대로 회색조 PNG로 저장한다
    plt.imsave(SAMPLE_IMAGE_PATH, small_image, cmap="gray", vmin=0.0, vmax=1.0)

    # 저장 결과를 알려준다
    print(f"저장: data/{SAMPLE_IMAGE_PATH.name}, 크기 {small_image.shape}")


# 파일을 직접 실행했을 때만 아래 코드를 실행한다
if __name__ == "__main__":
    # 스크립트로 실행하면 이미지를 만든다
    main()
