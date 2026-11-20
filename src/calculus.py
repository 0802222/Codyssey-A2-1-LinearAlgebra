"""미적분 모듈: 중심차분 수치 미분, Gradient, Hessian, Gradient 시각화.

다루는 수식
    중심차분: f'(x) ≈ (f(x + h) - f(x - h)) / 2h   (오차 O(h²))
    Gradient: ∇f = (∂f/∂x, ∂f/∂y)
    Hessian: H_ij = ∂²f / ∂x_i ∂x_j
    f(x, y) = x² + y²   → ∇f = (2x, 2y)
    f(x, y) = x² + 10y² → ∇f = (2x, 20y), H = [[2, 0], [0, 20]]
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src.constants import FIGURE_DPI, FloatArray, output_path

# 중심차분의 기본 간격 h (너무 작으면 반올림 오차, 너무 크면 근사 오차가 커진다)
DEFAULT_STEP_SIZE = 1e-5

# Hessian용 간격 (2차 미분은 h²으로 나누므로 1차보다 크게 잡는다)
HESSIAN_STEP_SIZE = 1e-4

# 과제 수치 미분 검증 지점 x = 3
DERIVATIVE_CHECK_POINT = 3.0

# f(x) = x²의 x = 3 해석 미분값 (2·3)
ANALYTIC_DERIVATIVE_AT_CHECK_POINT = 6.0

# 수치 미분 허용 오차 (과제 기준 1e-4)
DERIVATIVE_TOLERANCE = 1e-4

# 등고선 그림의 범위 (-3 ~ 3)
PLOT_LIMIT = 3.0

# 등고선 그림 격자 한 축의 점 개수
CONTOUR_GRID_POINT_COUNT = 200

# 화살표를 놓을 등고선 반지름들
ARROW_CONTOUR_RADII = (1.0, 2.0, 2.8)

# 등고선 하나에 놓을 화살표 개수
ARROWS_PER_CONTOUR = 12

# 화살표 길이 (방향을 보는 그림이라 Gradient를 길이 1로 맞춘 뒤 이 길이로 그린다)
ARROW_LENGTH = 0.45


# 1변수 함수 f(x) = x²
def square_function(x_value: float) -> float:
    """1변수 함수 f(x) = x²를 계산한다.

    Args:
        x_value: 입력 x

    Returns:
        x²
    """
    # x를 제곱한다
    return x_value**2


# 2변수 함수 f(x, y) = x² + y² (그릇 모양)
def bowl_function(point: FloatArray) -> FloatArray:
    """f(x, y) = x² + y²를 계산한다. 등고선은 원점 중심의 원이다.

    Args:
        point: 첫 성분이 x, 둘째 성분이 y인 배열 (격자 배열도 가능)

    Returns:
        함수값
    """
    # x² + y²를 계산한다
    return np.asarray(point[0] ** 2 + point[1] ** 2)


# x² + y²의 해석 Gradient (2x, 2y)
def bowl_gradient(point: FloatArray) -> FloatArray:
    """f(x, y) = x² + y²의 해석 Gradient ∇f = (2x, 2y)를 계산한다.

    Args:
        point: (2,) 점 (x, y)

    Returns:
        (2,) Gradient
    """
    # 각 성분에 2를 곱한다
    return np.array([2 * point[0], 2 * point[1]])


# 2변수 함수 f(x, y) = x² + 10y² (길쭉한 타원형 그릇)
def elliptic_function(point: FloatArray) -> FloatArray:
    """f(x, y) = x² + 10y²를 계산한다. y 방향이 x 방향보다 10배 가파르다.

    Args:
        point: 첫 성분이 x, 둘째 성분이 y인 배열 (격자 배열도 가능)

    Returns:
        함수값
    """
    # x² + 10y²를 계산한다
    return np.asarray(point[0] ** 2 + 10 * point[1] ** 2)


# x² + 10y²의 해석 Gradient (2x, 20y)
def elliptic_gradient(point: FloatArray) -> FloatArray:
    """f(x, y) = x² + 10y²의 해석 Gradient ∇f = (2x, 20y)를 계산한다.

    Args:
        point: (2,) 점 (x, y)

    Returns:
        (2,) Gradient
    """
    # x 성분은 2배, y 성분은 20배
    return np.array([2 * point[0], 20 * point[1]])


# 중심차분으로 1변수 함수의 미분값을 구한다
def central_difference(
    function: Callable[[float], float],
    x_value: float,
    step_size: float = DEFAULT_STEP_SIZE,
) -> float:
    """중심차분법으로 1변수 함수의 미분값을 근사한다.

    수식: f'(x) ≈ (f(x + h) - f(x - h)) / 2h
    전방차분 (f(x+h) - f(x)) / h의 오차가 O(h)인 데 비해 중심차분은 O(h²)이다.

    Args:
        function: 미분할 1변수 함수
        x_value: 미분할 지점
        step_size: 간격 h

    Returns:
        미분 근사값
    """
    # 앞뒤로 h만큼 떨어진 두 점의 기울기를 구한다
    return (function(x_value + step_size) - function(x_value - step_size)) / (
        2 * step_size
    )


# 중심차분으로 다변수 함수의 Gradient를 구한다
def numerical_gradient(
    function: Callable[[FloatArray], FloatArray],
    point: FloatArray,
    step_size: float = DEFAULT_STEP_SIZE,
) -> FloatArray:
    """각 변수 방향으로 중심차분을 적용해 Gradient를 근사한다.

    수식: ∂f/∂x_i ≈ (f(x + h·e_i) - f(x - h·e_i)) / 2h  (e_i: i번째 단위벡터)

    Args:
        function: 미분할 다변수 함수 (스칼라를 돌려준다)
        point: 미분할 지점
        step_size: 간격 h

    Returns:
        point와 같은 모양의 Gradient
    """
    # 결과를 담을 배열을 만든다
    gradient = np.zeros_like(point, dtype=np.float64)

    # 변수(성분)마다 반복한다
    for variable_position in range(point.size):
        # i번째 성분만 h인 이동 벡터를 만든다
        step_vector = np.zeros_like(point, dtype=np.float64)

        # i번째 성분에 h를 넣는다 (다차원이면 펼친 순서로)
        step_vector.flat[variable_position] = step_size

        # 앞쪽 함수값
        forward_value = float(function(point + step_vector))

        # 뒤쪽 함수값
        backward_value = float(function(point - step_vector))

        # i번째 편미분을 중심차분으로 구한다
        gradient.flat[variable_position] = (forward_value - backward_value) / (
            2 * step_size
        )

    # Gradient를 돌려준다
    return gradient


# 중심차분으로 Hessian(2차 미분 행렬)을 구한다
def numerical_hessian(
    function: Callable[[FloatArray], FloatArray],
    point: FloatArray,
    step_size: float = HESSIAN_STEP_SIZE,
) -> FloatArray:
    """Gradient를 한 번 더 중심차분해 Hessian을 근사한다 (보너스 Newton Method용).

    수식: H_ij ≈ (∂f/∂x_i(x + h·e_j) - ∂f/∂x_i(x - h·e_j)) / 2h

    Args:
        function: 2번 미분할 다변수 함수
        point: (n,) 미분할 지점
        step_size: 간격 h

    Returns:
        (n, n) Hessian 행렬
    """
    # 변수 개수
    variable_count = point.size

    # 결과를 담을 행렬을 만든다
    hessian = np.zeros((variable_count, variable_count))

    # j번째 변수 방향으로 반복한다
    for column_position in range(variable_count):
        # j번째 성분만 h인 이동 벡터를 만든다
        step_vector = np.zeros(variable_count)

        # j번째 성분에 h를 넣는다
        step_vector[column_position] = step_size

        # 앞쪽 Gradient
        forward_gradient = numerical_gradient(function, point + step_vector)

        # 뒤쪽 Gradient
        backward_gradient = numerical_gradient(function, point - step_vector)

        # j번째 열을 중심차분으로 채운다
        hessian[:, column_position] = (forward_gradient - backward_gradient) / (
            2 * step_size
        )

    # 대칭이 되도록 위아래 평균을 낸다 (반올림 오차 정리)
    return np.asarray((hessian + hessian.T) / 2)


# 원 모양 등고선 위에서 Gradient와 접선 사이의 코사인을 구한다
def gradient_tangent_cosines(
    function: Callable[[FloatArray], FloatArray], radius: float, point_count: int
) -> FloatArray:
    """원 모양 등고선 위 점에서 수치 Gradient와 등고선 접선 사이의 코사인을 구한다.

    x² + y² = r²의 접선 방향은 (-sin θ, cos θ)이다.
    코사인이 0이면 두 벡터 사이 각이 90도, 즉 Gradient가 등고선에 수직이다.

    Args:
        function: 등고선이 원인 2변수 함수
        radius: 등고선 반지름 r
        point_count: 확인할 점 개수

    Returns:
        (point_count,) 코사인 배열
    """
    # 원 위의 각도를 고르게 나눈다 (끝점 2π는 0과 같아서 뺀다)
    angles = np.linspace(0, 2 * np.pi, point_count, endpoint=False)

    # 결과를 담을 배열
    cosines = np.zeros(point_count)

    # 점마다 반복한다
    for angle_position, angle in enumerate(angles):
        # 등고선 위의 점 (r cos θ, r sin θ)
        point = radius * np.array([np.cos(angle), np.sin(angle)])

        # 그 점의 수치 Gradient
        gradient = numerical_gradient(function, point)

        # 등고선의 접선 방향 (길이 1)
        tangent = np.array([-np.sin(angle), np.cos(angle)])

        # 코사인 = (∇f · 접선) / |∇f|
        cosines[angle_position] = gradient @ tangent / np.linalg.norm(gradient)

    # 코사인 배열을 돌려준다
    return cosines


# 등고선과 Gradient 화살표를 함께 그린다
def plot_gradient_field(
    function: Callable[[FloatArray], FloatArray], title: str, file_name: str
) -> None:
    """2변수 함수의 등고선을 그리고, 등고선 위 여러 점에 Gradient 화살표를 그린다.

    화살표를 놓을 점은 등고선 f = c 위에 정확히 둔다.
    2차 형식은 f(t·u) = t²·f(u)이다.
    그래서 방향 u마다 t = √(c / f(u))로 잡으면 f(t·u) = c이다.
    화살표는 방향을 보기 위해 길이를 같게 맞춘다 (∇f / |∇f| × ARROW_LENGTH).

    Args:
        function: 그릴 2변수 함수
        title: 그림 제목
        file_name: outputs/에 저장할 파일 이름
    """
    # 한 축의 좌표를 고르게 나눈다
    axis_values = np.linspace(-PLOT_LIMIT, PLOT_LIMIT, CONTOUR_GRID_POINT_COUNT)

    # x, y 격자를 만든다
    grid_x, grid_y = np.meshgrid(axis_values, axis_values)

    # 격자 전체의 함수값을 한 번에 구한다
    grid_values = function(np.array([grid_x, grid_y]))

    # 점 (r, 0)을 지나는 등고선 높이 c = f(r, 0)를 정한다
    contour_levels = sorted(
        float(function(np.array([radius, 0.0]))) for radius in ARROW_CONTOUR_RADII
    )

    # 화살표 방향을 고르게 나눈다
    angles = np.linspace(0, 2 * np.pi, ARROWS_PER_CONTOUR, endpoint=False)

    # 방향마다 길이 1인 벡터 u를 만든다 (행: 방향)
    unit_directions = np.column_stack([np.cos(angles), np.sin(angles)])

    # 방향 u에서의 함수값 f(u)
    direction_values = np.array(
        [float(function(direction)) for direction in unit_directions]
    )

    # 등고선 높이마다 t = √(c / f(u))만큼 늘린 점을 모은다
    arrow_array = np.vstack(
        [
            unit_directions * np.sqrt(level / direction_values)[:, np.newaxis]
            for level in contour_levels
        ]
    )

    # 점마다 수치 Gradient를 구한다
    gradients = np.array([numerical_gradient(function, point) for point in arrow_array])

    # Gradient를 길이 1로 맞춘다 (방향만 남긴다)
    unit_gradients = gradients / np.linalg.norm(gradients, axis=1, keepdims=True)

    # 그림판과 축을 만든다
    figure, axis = plt.subplots(figsize=(6.5, 6))

    # 배경 등고선을 색으로 채워 그린다
    filled_contour = axis.contourf(grid_x, grid_y, grid_values, levels=20, cmap="Blues")

    # 화살표가 놓인 등고선을 검은 선으로 그린다
    axis.contour(grid_x, grid_y, grid_values, levels=contour_levels, colors="black")

    # 등고선 위 점에 Gradient 화살표를 그린다 (길이는 축소해서)
    axis.quiver(
        arrow_array[:, 0],
        arrow_array[:, 1],
        unit_gradients[:, 0] * ARROW_LENGTH,
        unit_gradients[:, 1] * ARROW_LENGTH,
        color="tab:red",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
    )

    # 색 막대(함수값 범례)를 붙인다
    figure.colorbar(filled_contour, ax=axis, label="f(x, y)")

    # 가로세로 비율을 1:1로 맞춘다 (직각이 직각으로 보이게)
    axis.set_aspect("equal")

    # 제목을 단다
    axis.set_title(title)

    # x축 이름
    axis.set_xlabel("x")

    # y축 이름
    axis.set_ylabel("y")

    # 그림을 저장한다
    figure.savefig(output_path(file_name), dpi=FIGURE_DPI, bbox_inches="tight")

    # 그림을 닫는다
    plt.close(figure)


# 모듈을 직접 실행하면 미적분 실험을 모두 수행한다
def main() -> None:
    """수치 미분 검증, Gradient 수직 검증, Gradient 시각화를 실행한다."""
    # 1단계 제목
    print("== 1. 수치 미분 (중심차분) ==")

    # f(x) = x²의 x = 3 수치 미분값
    derivative = central_difference(square_function, DERIVATIVE_CHECK_POINT)

    # 해석해와의 오차
    derivative_error = abs(derivative - ANALYTIC_DERIVATIVE_AT_CHECK_POINT)

    # 허용 오차 이내인지 판정한다
    verdict = "통과" if derivative_error < DERIVATIVE_TOLERANCE else "실패"

    # 결과를 출력한다
    print(
        f"f(x)=x², x=3: 수치 미분={derivative:.10f}, 해석 미분=6, "
        f"오차={derivative_error:.2e} (1e-4 이내 {verdict})"
    )

    # 2단계 제목
    print("\n== 2. Gradient와 등고선의 수직 검증 ==")

    # 반지름마다 코사인의 최댓값을 확인한다
    for radius in ARROW_CONTOUR_RADII:
        # 등고선 위 점에서 코사인을 구한다
        cosines = gradient_tangent_cosines(bowl_function, radius, ARROWS_PER_CONTOUR)

        # 코사인 절댓값의 최댓값
        max_cosine = float(np.max(np.abs(cosines)))

        # 각도로 바꾼다 (코사인이 0이면 90도)
        angle_degrees = float(np.degrees(np.arccos(max_cosine)))

        # 결과를 출력한다
        print(
            f"반지름 {radius} 등고선: |cos| 최댓값={max_cosine:.2e} "
            f"-> Gradient와 접선 사이 각 {angle_degrees:.4f}도"
        )

    # 3단계 제목
    print("\n== 3. Gradient 시각화 ==")

    # x² + y²의 등고선과 화살표를 그린다
    plot_gradient_field(
        bowl_function,
        "f(x, y) = x² + y²\ngradient arrows are perpendicular to contours",
        "gradient_bowl.png",
    )

    # x² + 10y²도 같은 방식으로 그린다 (최적화 비교용)
    plot_gradient_field(
        elliptic_function,
        "f(x, y) = x² + 10y²\ngradient arrows are perpendicular to contours",
        "gradient_elliptic.png",
    )

    # 저장 위치를 알려준다
    print("-> outputs/gradient_bowl.png, outputs/gradient_elliptic.png")


# 파일을 직접 실행했을 때만 아래 코드를 실행한다
if __name__ == "__main__":
    # 스크립트로 실행하면 main을 부른다
    main()
