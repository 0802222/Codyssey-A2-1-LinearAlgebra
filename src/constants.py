"""여러 파일이 같이 쓰는 공통 상수와 타입을 모아 둔 파일."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

# 실수 NumPy 배열의 타입 별칭 (모든 모듈의 타입 힌트에 쓴다)
FloatArray = npt.NDArray[np.float64]

# 재현성을 위한 난수 시드 (과제 요구: np.random.seed(42))
RANDOM_SEED = 42

# 프로젝트 최상위 폴더 (이 파일의 두 단계 위)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 시각화 PNG를 저장하는 폴더 (과제 요구: outputs/)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# SVD 압축 실험에 쓰는 64x64 흑백 이미지 경로
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "data" / "sample_image_64.png"

# 샘플 이미지 한 변의 픽셀 수 (과제 조건: 64x64 이하)
IMAGE_SIZE_PIXELS = 64

# 그림 저장 해상도 (dots per inch)
FIGURE_DPI = 120


# outputs/ 폴더가 없으면 만들고 파일 경로를 돌려준다
def output_path(file_name: str) -> Path:
    """저장할 파일의 outputs/ 아래 경로를 만들어 돌려준다.

    Args:
        file_name: 저장할 파일 이름 (예: "rotation.png")

    Returns:
        outputs/file_name 절대 경로
    """
    # 폴더가 없으면 만든다
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 폴더 경로와 파일 이름을 합쳐 돌려준다
    return OUTPUTS_DIR / file_name
