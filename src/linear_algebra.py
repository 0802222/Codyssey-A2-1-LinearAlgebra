"""선형대수 모듈: 행렬 변환 시각화, Power Iteration, SVD 이미지 압축.

다루는 수식
    회전 R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]   det = 1
    스케일링 S(sx, sy) = [[sx, 0], [0, sy]]           det = sx · sy
    전단 Sh(k) = [[1, k], [0, 1]]                     det = 1
    변환 후 면적 = |det(A)| × 변환 전 면적
    Power Iteration: v ← A v / ||A v||, λ = vᵀ A v (레일리 몫)
    SVD: A = U Σ Vᵀ, 상위 k개만 남긴 A_k = U_k Σ_k V_kᵀ
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.constants import (
    FIGURE_DPI,
    RANDOM_SEED,
    SAMPLE_IMAGE_PATH,
    FloatArray,
    output_path,
)

# 단위 원을 그릴 때 쓰는 점의 개수 (과제 가이드: np.linspace(0, 2π, 100))
CIRCLE_POINT_COUNT = 100

# 배경 격자 한 줄에 놓을 점의 개수
GRID_POINT_COUNT = 9

# 배경 격자의 범위 (-1.5 ~ 1.5)
GRID_LIMIT = 1.5

# 시각화에 쓰는 회전 각도 (45도)
ROTATION_ANGLE_RADIANS = np.pi / 4

# 시각화에 쓰는 스케일링 배율 (과제 요구: S(2, 0.5))
SCALE_X, SCALE_Y = 2.0, 0.5

# 시각화에 쓰는 전단 계수 k
SHEAR_FACTOR = 1.0

# 면적이 바뀌는 경우를 확인하는 일반 행렬 (det = 2·3 - 1·1 = 5)
GENERAL_MATRIX = np.array([[2.0, 1.0], [1.0, 3.0]])

# 과제 예시의 대칭 행렬 A = [[4, 1], [1, 3]]
EXAMPLE_SYMMETRIC_MATRIX = np.array([[4.0, 1.0], [1.0, 3.0]])

# Power Iteration 수렴 판정 기준 (고유값 변화량)
POWER_ITERATION_TOLERANCE = 1e-6

# Power Iteration 최대 반복 횟수
POWER_ITERATION_MAX_ITERATIONS = 1000

# SVD 압축에서 비교할 k 값 (과제 요구: 10, 50, 100)
SVD_RANKS = (10, 50, 100)

# 퍼센트 변환 배율
PERCENT = 100.0

# 면적비 검증의 허용 오차 (과제 기준 1%)
AREA_ERROR_LIMIT_PERCENT = 1.0

# 고유값 검증의 허용 오차 (과제 기준 5%)
EIGENVALUE_ERROR_LIMIT_PERCENT = 5.0


@dataclass
class AreaCheck:
    """행렬식과 실제 면적비를 비교한 결과."""

    # 행렬식 det(A)
    determinant: float

    # 변환 전 도형 넓이
    area_before: float

    # 변환 후 도형 넓이
    area_after: float

    # 면적비 = 변환 후 넓이 / 변환 전 넓이
    area_ratio: float

    # |면적비 - |det|| / |det| × 100
    error_percent: float


@dataclass
class PowerIterationResult:
    """Power Iteration으로 구한 최대 고유값과 고유벡터."""

    # 최대 고유값 λ
    eigenvalue: float

    # 길이 1로 맞춘 고유벡터 v
    eigenvector: FloatArray

    # 수렴할 때까지 반복한 횟수
    iteration_count: int

    # 최대 반복 전에 수렴했는지 여부
    is_converged: bool


@dataclass
class EigenComparison:
    """Power Iteration 결과와 np.linalg.eig 결과의 비교."""

    # Power Iteration 고유값
    power_eigenvalue: float

    # np.linalg.eig 최대 고유값
    numpy_eigenvalue: float

    # np.linalg.eig 고유벡터 (부호를 맞춘 것)
    numpy_eigenvector: FloatArray

    # 고유값 상대 오차 (%)
    eigenvalue_error_percent: float

    # 두 고유벡터 사이 코사인의 절댓값 (1이면 같은 방향)
    eigenvector_cosine: float


@dataclass
class SvdCompression:
    """SVD로 상위 k개 특이값만 남겨 복원한 결과."""

    # 복원한 이미지
    reconstructed_image: FloatArray

    # 실제로 쓴 k (이미지 rank보다 크면 rank로 잘린다)
    used_rank: int

    # 상대 오차 = ||원본 - 복원||_F / ||원본||_F
    relative_error: float

    # 저장 비율 = k(m + n + 1) / (m · n)
    storage_ratio: float


# 단위 원 위의 점들을 만든다
def make_unit_circle(point_count: int = CIRCLE_POINT_COUNT) -> FloatArray:
    """단위 원 위의 점을 (2, point_count) 배열로 만든다.

    수식: (x, y) = (cos θ, sin θ), θ ∈ [0, 2π]

    Args:
        point_count: 점의 개수

    Returns:
        첫 행이 x, 둘째 행이 y인 (2, point_count) 배열
    """
    # 0부터 2π까지 각도를 고르게 나눈다
    angles = np.linspace(0, 2 * np.pi, point_count)

    # 각도마다 (cos, sin)을 세로로 쌓는다
    return np.vstack([np.cos(angles), np.sin(angles)])


# 배경 격자 점들을 만든다
def make_grid_points(
    limit: float = GRID_LIMIT, point_count: int = GRID_POINT_COUNT
) -> FloatArray:
    """np.meshgrid로 2D 격자 점 집합을 만든다.

    Args:
        limit: 격자 범위 (-limit ~ limit)
        point_count: 한 축의 점 개수

    Returns:
        (2, point_count²) 모양의 격자 점
    """
    # 한 축의 좌표를 고르게 나눈다
    axis_values = np.linspace(-limit, limit, point_count)

    # x, y 좌표 격자를 만든다
    grid_x, grid_y = np.meshgrid(axis_values, axis_values)

    # 두 격자를 한 줄로 펴서 (2, N) 모양으로 쌓는다
    return np.vstack([grid_x.ravel(), grid_y.ravel()])


# 회전 행렬 R(θ)
def rotation_matrix(angle_radians: float) -> FloatArray:
    """반시계 방향 회전 행렬 R(θ)를 만든다.

    수식: R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]], det(R) = 1

    Args:
        angle_radians: 회전 각도 (라디안)

    Returns:
        (2, 2) 회전 행렬
    """
    # 코사인 값
    cosine = np.cos(angle_radians)

    # 사인 값
    sine = np.sin(angle_radians)

    # 회전 행렬을 만든다
    return np.array([[cosine, -sine], [sine, cosine]])


# 스케일링 행렬 S(sx, sy)
def scaling_matrix(scale_x: float, scale_y: float) -> FloatArray:
    """축 방향 스케일링 행렬 S(sx, sy)를 만든다.

    수식: S = [[sx, 0], [0, sy]], det(S) = sx · sy

    Args:
        scale_x: x축 배율
        scale_y: y축 배율

    Returns:
        (2, 2) 스케일링 행렬
    """
    # 대각선에 배율을 둔 행렬을 만든다
    return np.array([[scale_x, 0.0], [0.0, scale_y]])


# 전단 행렬 Sh(k)
def shear_matrix(shear_factor: float) -> FloatArray:
    """전단 행렬 Sh(k)를 x 방향으로 만든다.

    수식: Sh(k) = [[1, k], [0, 1]] → (x, y)가 (x + k·y, y)로 간다, det = 1

    Args:
        shear_factor: 전단 계수 k

    Returns:
        (2, 2) 전단 행렬
    """
    # y에 비례해 x를 미는 행렬을 만든다
    return np.array([[1.0, shear_factor], [0.0, 1.0]])


# 다각형 넓이를 신발끈 공식으로 구한다
def polygon_area(points: FloatArray) -> float:
    """꼭짓점을 순서대로 이은 다각형 넓이를 신발끈(Shoelace) 공식으로 구한다.

    수식: 넓이 = ½ |Σ (x_i · y_{i+1} - x_{i+1} · y_i)|

    Args:
        points: (2, N) 꼭짓점 배열

    Returns:
        다각형 넓이
    """
    # x 좌표와 y 좌표를 나눈다
    x_values, y_values = points[0], points[1]

    # 다음 꼭짓점 좌표 (마지막 다음은 처음으로 돌아간다)
    next_x_values, next_y_values = np.roll(x_values, -1), np.roll(y_values, -1)

    # 신발끈 공식으로 넓이를 구한다
    return float(0.5 * abs(np.sum(x_values * next_y_values - next_x_values * y_values)))


# 행렬식과 실제 면적비를 비교한다
def check_area_ratio(matrix: FloatArray, points: FloatArray) -> AreaCheck:
    """변환 전후 도형 넓이의 비율이 |det(A)|와 같은지 확인한다.

    수식: 오차(%) = |면적비 - |det(A)|| / |det(A)| × 100

    Args:
        matrix: (2, 2) 변환 행렬
        points: (2, N) 도형 꼭짓점

    Returns:
        행렬식, 넓이, 면적비, 오차를 담은 AreaCheck
    """
    # 행렬식을 구한다
    determinant = float(np.linalg.det(matrix))

    # 변환 전 넓이를 구한다
    area_before = polygon_area(points)

    # 변환 후 넓이를 구한다 (점마다 A를 곱한다)
    area_after = polygon_area(matrix @ points)

    # 면적비를 구한다
    area_ratio = area_after / area_before

    # 면적비와 |det|의 상대 오차를 퍼센트로 구한다
    error_percent = abs(area_ratio - abs(determinant)) / abs(determinant) * PERCENT

    # 결과를 묶어 돌려준다
    return AreaCheck(determinant, area_before, area_after, area_ratio, error_percent)


# 변환 전후를 한 그림에 겹쳐 그리고 저장한다
def plot_transformation(matrix: FloatArray, title: str, file_name: str) -> AreaCheck:
    """단위 원과 격자에 행렬을 적용해 변환 전후를 하나의 Figure에 겹쳐 그린다.

    Args:
        matrix: (2, 2) 변환 행렬
        title: 그림 제목
        file_name: outputs/에 저장할 파일 이름

    Returns:
        면적비 검증 결과 AreaCheck
    """
    # 단위 원을 만든다
    circle_points = make_unit_circle()

    # 배경 격자 점을 만든다
    grid_points = make_grid_points()

    # 단위 원을 변환한다
    transformed_circle = matrix @ circle_points

    # 격자를 변환한다
    transformed_grid = matrix @ grid_points

    # 면적비를 검증한다
    area_check = check_area_ratio(matrix, circle_points)

    # 그림판과 축을 만든다
    figure, axis = plt.subplots(figsize=(6, 6))

    # 변환 전 격자를 연한 파란 점으로 그린다
    axis.scatter(grid_points[0], grid_points[1], s=8, color="tab:blue", alpha=0.3)

    # 변환 후 격자를 연한 빨간 점으로 그린다
    axis.scatter(
        transformed_grid[0], transformed_grid[1], s=8, color="tab:red", alpha=0.3
    )

    # 변환 전 단위 원을 파란 선으로 그린다
    axis.plot(*circle_points, color="tab:blue", label="before: unit circle")

    # 변환 후 도형을 빨간 선으로 그린다
    axis.plot(*transformed_circle, color="tab:red", label="after: A @ circle")

    # 기저 벡터 e1=(1,0)이 옮겨 간 곳 (A의 첫 열)을 화살표로 그린다
    axis.quiver(
        0,
        0,
        *matrix[:, 0],
        color="darkred",
        angles="xy",
        scale_units="xy",
        scale=1,
        label="A e1 (1st column)",
    )

    # 기저 벡터 e2=(0,1)이 옮겨 간 곳 (A의 둘째 열)을 화살표로 그린다
    axis.quiver(
        0,
        0,
        *matrix[:, 1],
        color="darkorange",
        angles="xy",
        scale_units="xy",
        scale=1,
        label="A e2 (2nd column)",
    )

    # 가로세로 비율을 1:1로 맞춘다 (원이 찌그러져 보이지 않게)
    axis.set_aspect("equal")

    # 보조 격자선을 켠다
    axis.grid(True, alpha=0.3)

    # 제목에 행렬식과 면적비를 함께 적는다
    axis.set_title(
        f"{title}\ndet(A)={area_check.determinant:.4f}, "
        f"area ratio={area_check.area_ratio:.4f}, "
        f"error={area_check.error_percent:.4f}%"
    )

    # 범례를 표시한다
    axis.legend(loc="upper left")

    # 그림을 outputs/에 저장한다
    figure.savefig(output_path(file_name), dpi=FIGURE_DPI, bbox_inches="tight")

    # 메모리를 위해 그림을 닫는다
    plt.close(figure)

    # 검증 결과를 돌려준다
    return area_check


# 부호를 맞춘다 (가장 큰 성분이 양수가 되게)
def align_sign(vector: FloatArray) -> FloatArray:
    """고유벡터의 부호를 가장 큰 성분이 양수가 되도록 맞춘다.

    고유벡터는 v와 -v가 같은 고유벡터이므로 비교 전에 부호를 맞춘다.

    Args:
        vector: 고유벡터

    Returns:
        부호를 맞춘 고유벡터
    """
    # 절댓값이 가장 큰 성분의 부호를 곱한다
    return np.asarray(vector * np.sign(vector[np.argmax(np.abs(vector))]))


# Power Iteration으로 최대 고유값과 고유벡터를 구한다
def power_iteration(
    matrix: FloatArray,
    tolerance: float = POWER_ITERATION_TOLERANCE,
    max_iterations: int = POWER_ITERATION_MAX_ITERATIONS,
) -> PowerIterationResult:
    """Power Iteration으로 절댓값이 가장 큰 고유값과 고유벡터를 구한다.

    수식
        v_{t+1} = A v_t / ||A v_t||
        λ_t = v_tᵀ A v_t (레일리 몫, v_t의 길이는 1)
        ||v_t - v_{t-1}|| < tolerance 이면 수렴으로 본다
        (고유값은 고유벡터보다 빨리 맞으므로 더 엄격한 고유벡터 변화량으로 판정한다)

    Args:
        matrix: (n, n) 정사각 행렬
        tolerance: 수렴 판정 기준 (고유벡터 변화량)
        max_iterations: 최대 반복 횟수

    Returns:
        고유값, 고유벡터, 반복 횟수, 수렴 여부를 담은 PowerIterationResult
    """
    # 재현성을 위해 시드를 고정한다 (과제 요구)
    np.random.seed(RANDOM_SEED)

    # 무작위 시작 벡터를 만든다
    vector = np.random.rand(matrix.shape[0])

    # 길이를 1로 맞춘다
    vector = vector / np.linalg.norm(vector)

    # 최대 반복 횟수만큼 반복한다
    for iteration_number in range(1, max_iterations + 1):
        # 행렬을 곱해 최대 고유벡터 방향으로 끌어당긴다
        multiplied_vector = matrix @ vector

        # 길이를 1로 맞춘다
        next_vector = multiplied_vector / np.linalg.norm(multiplied_vector)

        # 고유벡터가 얼마나 변했는지 구한다
        vector_change = float(np.linalg.norm(next_vector - vector))

        # 다음 반복을 위해 벡터를 갱신한다
        vector = next_vector

        # 변화량이 기준보다 작으면 수렴으로 본다
        if vector_change < tolerance:
            # 레일리 몫으로 고유값을 구해 수렴 결과를 돌려준다
            return PowerIterationResult(
                float(vector @ matrix @ vector),
                align_sign(vector),
                iteration_number,
                True,
            )

    # 최대 반복까지 수렴하지 못한 결과를 돌려준다
    return PowerIterationResult(
        float(vector @ matrix @ vector), align_sign(vector), max_iterations, False
    )


# 검증 전용: np.linalg.eig 결과와 비교한다
def compare_with_numpy_eig(
    matrix: FloatArray, result: PowerIterationResult
) -> EigenComparison:
    """Power Iteration 결과를 np.linalg.eig 결과와 비교한다.

    np.linalg.eig는 이 검증 함수에서만 쓴다 (과제 제약: 고유값 계산 자체에는 쓰지 않음).

    수식: 오차(%) = |λ_power - λ_eig| / |λ_eig| × 100

    Args:
        matrix: (n, n) 정사각 행렬
        result: power_iteration의 결과

    Returns:
        고유값 오차와 고유벡터 코사인을 담은 EigenComparison
    """
    # NumPy로 모든 고유값과 고유벡터를 구한다 (검증 목적)
    numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(matrix)

    # 절댓값이 가장 큰 고유값의 위치를 찾는다
    largest_position = int(np.argmax(np.abs(numpy_eigenvalues)))

    # 최대 고유값을 꺼낸다
    numpy_eigenvalue = float(np.real(numpy_eigenvalues[largest_position]))

    # 대응 고유벡터를 꺼내 부호를 맞춘다
    numpy_eigenvector = align_sign(np.real(numpy_eigenvectors[:, largest_position]))

    # 고유값 상대 오차를 퍼센트로 구한다
    error_percent = (
        abs(result.eigenvalue - numpy_eigenvalue) / abs(numpy_eigenvalue) * PERCENT
    )

    # 두 고유벡터 사이 코사인의 절댓값 (둘 다 길이 1)
    cosine = float(abs(result.eigenvector @ numpy_eigenvector))

    # 비교 결과를 묶어 돌려준다
    return EigenComparison(
        result.eigenvalue, numpy_eigenvalue, numpy_eigenvector, error_percent, cosine
    )


# 저장된 흑백 이미지를 0~1 배열로 읽는다
def load_grayscale_image(image_path: Path = SAMPLE_IMAGE_PATH) -> FloatArray:
    """회색조 PNG를 (높이, 너비) 모양의 0~1 실수 배열로 읽는다.

    Args:
        image_path: PNG 파일 경로

    Returns:
        (높이, 너비) 흑백 이미지 배열
    """
    # PNG를 읽는다 (회색조로 저장해도 RGBA 4채널로 읽힌다)
    image = plt.imread(image_path)

    # 채널이 있으면 첫 채널(R = G = B)만 쓴다
    if image.ndim == 3:
        # 첫 채널만 남긴다
        image = image[..., 0]

    # 실수 배열로 바꿔 돌려준다
    return np.asarray(image, dtype=np.float64)


# 상위 k개 특이값으로 이미지를 압축, 복원한다
def svd_compress(image: FloatArray, rank: int) -> SvdCompression:
    """SVD의 상위 k개 특이값만 남겨 이미지를 압축하고 복원한다.

    수식
        A = U Σ Vᵀ
        A_k = U[:, :k] · diag(σ_1..σ_k) · Vᵀ[:k, :]
        저장 비율 = k(m + n + 1) / (m · n)

    Args:
        image: (m, n) 흑백 이미지
        rank: 남길 특이값 개수 k (rank보다 크면 rank로 자른다)

    Returns:
        복원 이미지, 실제 k, 상대 오차, 저장 비율을 담은 SvdCompression
    """
    # 특이값 분해를 한다 (U: 왼쪽 특이벡터, singular_values: σ, Vᵀ: 오른쪽 특이벡터)
    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(
        image, full_matrices=False
    )

    # k를 실제 특이값 개수 이하로 자른다 (64x64면 최대 64)
    used_rank = min(rank, singular_values.size)

    # 상위 k개만 써서 복원한다
    reconstructed_image = (
        left_vectors[:, :used_rank]
        @ np.diag(singular_values[:used_rank])
        @ right_vectors_transposed[:used_rank, :]
    )

    # 프로베니우스 노름으로 상대 오차를 구한다
    relative_error = float(
        np.linalg.norm(image - reconstructed_image) / np.linalg.norm(image)
    )

    # 행 수 m과 열 수 n
    row_count, column_count = image.shape

    # 원본 대비 저장해야 할 숫자 개수 비율
    storage_ratio = (
        used_rank * (row_count + column_count + 1) / (row_count * column_count)
    )

    # 결과를 묶어 돌려준다
    return SvdCompression(reconstructed_image, used_rank, relative_error, storage_ratio)


# 원본과 k별 복원 결과를 한 그림에 비교한다
def plot_svd_compression(
    image: FloatArray, ranks: tuple[int, ...], file_name: str
) -> list[SvdCompression]:
    """원본과 k별 복원 이미지를 하나의 Figure에 나란히 그린다.

    Args:
        image: (m, n) 흑백 이미지
        ranks: 비교할 k 값들
        file_name: outputs/에 저장할 파일 이름

    Returns:
        k별 SvdCompression 목록
    """
    # k마다 압축, 복원한다
    compressions = [svd_compress(image, rank) for rank in ranks]

    # 원본 1칸 + k 개수만큼 가로로 칸을 만든다
    figure, axes = plt.subplots(1, len(ranks) + 1, figsize=(3 * (len(ranks) + 1), 3.4))

    # 첫 칸에 원본을 그린다
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)

    # 첫 칸 제목
    axes[0].set_title(f"original {image.shape[0]}x{image.shape[1]}")

    # 나머지 칸에 k별 복원 결과를 그린다
    for axis, rank, compression in zip(axes[1:], ranks, compressions):
        # 복원 이미지를 그린다 (0~1 밖의 값은 잘라서 보여 준다)
        axis.imshow(np.clip(compression.reconstructed_image, 0, 1), cmap="gray")

        # 제목에 k, 실제 k, 오차, 저장 비율을 적는다
        axis.set_title(
            f"k={rank} (used {compression.used_rank})\n"
            f"error {compression.relative_error:.2%}, "
            f"storage {compression.storage_ratio:.0%}"
        )

    # 모든 칸의 눈금을 숨긴다
    for axis in axes:
        # 눈금을 끈다
        axis.axis("off")

    # 그림을 outputs/에 저장한다
    figure.savefig(output_path(file_name), dpi=FIGURE_DPI, bbox_inches="tight")

    # 그림을 닫는다
    plt.close(figure)

    # k별 결과를 돌려준다
    return compressions


# 행렬 변환 3종을 그리고 면적비 검증을 출력한다
def run_transformations() -> None:
    """회전, 스케일링, 전단 시각화를 저장하고 면적비 검증 결과를 출력한다."""
    # (제목, 행렬, 파일 이름) 목록
    transformations = [
        # 45도 회전 (det = 1)
        (
            "Rotation R(45deg)",
            rotation_matrix(ROTATION_ANGLE_RADIANS),
            "transform_rotation.png",
        ),
        # 가로 2배, 세로 0.5배 (det = 1)
        (
            "Scaling S(2, 0.5)",
            scaling_matrix(SCALE_X, SCALE_Y),
            "transform_scaling.png",
        ),
        # x 방향 전단 (det = 1)
        ("Shear Sh(1)", shear_matrix(SHEAR_FACTOR), "transform_shear.png"),
        # 면적이 5배가 되는 일반 행렬 (det = 5)
        ("General A=[[2,1],[1,3]]", GENERAL_MATRIX, "transform_general.png"),
    ]

    # 변환마다 그림을 그리고 결과를 출력한다
    for title, matrix, file_name in transformations:
        # 그림을 저장하고 면적비 검증 결과를 받는다
        area_check = plot_transformation(matrix, title, file_name)

        # 오차가 1% 이내인지 판정한다
        verdict = (
            "통과" if area_check.error_percent < AREA_ERROR_LIMIT_PERCENT else "실패"
        )

        # 결과를 한 줄로 출력한다
        print(
            f"[{title}] det(A)={area_check.determinant:.4f}, "
            f"넓이 {area_check.area_before:.4f} -> {area_check.area_after:.4f}, "
            f"면적비={area_check.area_ratio:.4f}, "
            f"오차={area_check.error_percent:.6f}% (1% 이내 {verdict}) "
            f"-> outputs/{file_name}"
        )


# Power Iteration을 실행하고 np.linalg.eig와 비교한다
def run_power_iteration() -> None:
    """예시 대칭 행렬로 Power Iteration을 실행하고 검증 결과를 출력한다."""
    # Power Iteration을 실행한다
    result = power_iteration(EXAMPLE_SYMMETRIC_MATRIX)

    # np.linalg.eig와 비교한다
    comparison = compare_with_numpy_eig(EXAMPLE_SYMMETRIC_MATRIX, result)

    # 오차가 5% 이내인지 판정한다
    is_within_limit = (
        comparison.eigenvalue_error_percent < EIGENVALUE_ERROR_LIMIT_PERCENT
    )

    # 판정 문구를 정한다
    verdict = "통과" if is_within_limit else "실패"

    # 행렬을 출력한다
    print(f"행렬 A = {EXAMPLE_SYMMETRIC_MATRIX.tolist()} (대칭 행렬)")

    # Power Iteration 결과를 출력한다
    print(
        f"Power Iteration (수렴 {result.iteration_count}회, "
        f"tol={POWER_ITERATION_TOLERANCE}): λ={result.eigenvalue:.4f}, "
        f"v={np.round(result.eigenvector, 4).tolist()}"
    )

    # np.linalg.eig 결과를 출력한다
    print(
        f"np.linalg.eig: λ={comparison.numpy_eigenvalue:.4f}, "
        f"v={np.round(comparison.numpy_eigenvector, 4).tolist()}"
    )

    # 오차와 판정을 출력한다
    print(
        f"고유값 오차: {comparison.eigenvalue_error_percent:.6f}% (5% 이내 {verdict}), "
        f"고유벡터 코사인: {comparison.eigenvector_cosine:.6f}"
    )


# SVD 압축을 실행하고 결과를 출력한다
def run_svd_compression() -> None:
    """64x64 이미지를 k=10, 50, 100으로 압축하고 결과를 출력한다."""
    # 샘플 이미지를 읽는다
    image = load_grayscale_image()

    # 압축 결과 그림을 저장한다
    compressions = plot_svd_compression(image, SVD_RANKS, "svd_compression.png")

    # k별 결과를 출력한다
    for rank, compression in zip(SVD_RANKS, compressions):
        # 한 줄로 출력한다
        print(
            f"k={rank:>3} (실제 {compression.used_rank:>2}): "
            f"상대 오차 {compression.relative_error:.4f}, "
            f"저장 비율 {compression.storage_ratio:.1%}"
        )

    # 저장 위치를 알려준다
    print("-> outputs/svd_compression.png")


# 모듈을 직접 실행하면 선형대수 실험을 모두 수행한다
def main() -> None:
    """행렬 변환, Power Iteration, SVD 압축을 차례로 실행한다."""
    # 1단계 제목
    print("== 1. 행렬 변환과 면적비 검증 ==")

    # 행렬 변환 3종을 실행한다
    run_transformations()

    # 2단계 제목
    print("\n== 2. Power Iteration vs np.linalg.eig ==")

    # Power Iteration을 실행한다
    run_power_iteration()

    # 3단계 제목
    print("\n== 3. SVD 이미지 압축 ==")

    # SVD 압축을 실행한다
    run_svd_compression()


# 파일을 직접 실행했을 때만 아래 코드를 실행한다
if __name__ == "__main__":
    # 스크립트로 실행하면 main을 부른다
    main()
