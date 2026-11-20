"""최적화 모듈: VanillaGD, Momentum, (보너스) Adam, NewtonMethod와 수렴 경로 시각화.

다루는 수식 (θ: 파라미터, g = ∇f(θ): Gradient, η: 학습률)
    Vanilla GD: θ ← θ - η·g
    Momentum:   v ← β·v + g,  θ ← θ - η·v
    Adam:       m ← β1·m + (1-β1)·g,  s ← β2·s + (1-β2)·g²
                m̂ = m / (1-β1ᵗ),  ŝ = s / (1-β2ᵗ),  θ ← θ - η·m̂ / (√ŝ + ε)
    Newton:     θ ← θ - η·H⁻¹·g   (H: Hessian)

안정 조건 (2차 함수 f = Σ a_i·x_i², 곡률 L_i = 2a_i)
    Vanilla GD의 한 걸음은 x_i ← (1 - η·L_i)·x_i 이다.
    |1 - η·L_i| < 1, 즉 η < 2/L_max 이어야 수렴한다.
        x² + y²: η < 1 (η = 0.5면 한 번에 원점, η = 1이면 제자리 진동, η > 1이면 발산)
        x² + 10y²: η < 0.1 (y 방향이 가팔라서 한계가 10배 낮다)
    Momentum의 안정 조건은 η < 2(1 + β)/L_max 로 넓다 (β = 0.9면 GD의 1.9배).
        x² + 10y²: η < 0.19
    대신 Momentum은 반복마다 거리가 최대 √β배씩만 줄어드는 한계가 있다.
    (β = 0.9면 약 0.949배)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src.calculus import (
    bowl_function,
    bowl_gradient,
    elliptic_function,
    elliptic_gradient,
    numerical_hessian,
)
from src.constants import FIGURE_DPI, FloatArray, output_path

# 모든 실험의 시작점 (과제 결과 예시와 같다)
START_POINT = np.array([5.0, 5.0])

# 과제에서 정한 반복 횟수
ITERATION_COUNT = 100

# 과제에서 정한 기본 학습률
BASE_LEARNING_RATE = 0.1

# 수렴 판정 반경 (과제: 원점 반경 0.1 이내)
CONVERGENCE_RADIUS = 0.1

# Momentum 계수 (과제: beta = 0.9)
MOMENTUM_BETA = 0.9

# 작은 학습률 (Momentum의 이점이 드러나는 경우)
SMALL_LEARNING_RATE = 0.01

# 타원형 함수에서 Vanilla GD가 y 방향으로 지그재그하는 학습률 (0.05 < η < 0.1)
ZIGZAG_LEARNING_RATE = 0.09

# 발산을 보여 줄 (함수 이름, 학습률) 조합
DIVERGENCE_CASES = (
    # 기준: 천천히 수렴
    ("x²+y²", 0.1),
    # 1 - 2·0.5 = 0 이라 한 번에 원점
    ("x²+y²", 0.5),
    # 1 - 2·1.0 = -1 이라 제자리 진동
    ("x²+y²", 1.0),
    # 1 - 2·1.1 = -1.2 라 발산
    ("x²+y²", 1.1),
    # y 방향 1 - 20·0.5 = -9 라 빠르게 발산
    ("x²+10y²", 0.5),
)

# 발산 그래프의 반복 횟수 (발산은 금방 숫자가 커지므로 짧게)
DIVERGENCE_ITERATION_COUNT = 20

# 진동 판정 폭 (마지막 거리가 처음 거리의 ±1% 안이면 진동으로 본다)
OSCILLATION_BAND = 0.01

# 로그 축 그래프에서 0 대신 쓰는 아주 작은 값 (log 0을 막는다)
LOG_PLOT_FLOOR = 1e-12

# 발산 경로 그림에 넣을 걸음 수
DIVERGENCE_PATH_STEPS = 11

# 발산 경로 그림의 범위
DIVERGENCE_PLOT_LIMIT = 15.0

# 학습률 비교표의 학습률 (x²+10y²에서 0.1, 0.15는 GD 한계 이상, Momentum 한계 미만)
COMPARISON_LEARNING_RATES = (0.01, 0.02, 0.05, 0.09, 0.1, 0.15)

# 보너스 비교에서 Momentum의 학습률 (Momentum은 0.01~0.09에서 수렴 반복이 비슷하다)
BONUS_MOMENTUM_LEARNING_RATE = 0.05

# 보너스 Adam 학습률 (시뮬레이션으로 고른 값, 0.3~0.5에서 빠르다)
ADAM_LEARNING_RATE = 0.5

# 보너스 비교의 반복 횟수
BONUS_ITERATION_COUNT = 200

# 보너스 비교의 엄격한 수렴 반경
STRICT_CONVERGENCE_RADIUS = 1e-3

# Adam 분모가 0이 되는 것을 막는 작은 값
ADAM_EPSILON = 1e-8

# Adam 1차 모멘트 계수 (원 논문 기본값)
ADAM_BETA1 = 0.9

# Adam 2차 모멘트 계수 (원 논문 기본값)
ADAM_BETA2 = 0.999

# 등고선 그림의 범위
PLOT_LIMIT = 6.0

# 등고선 그림 격자 한 축의 점 개수
CONTOUR_GRID_POINT_COUNT = 200

# 등고선 높이 (0.5 ~ 300을 로그 간격 14개로)
CONTOUR_LEVELS = np.geomspace(0.5, 300, 14)

# 함수 이름 → (함수, Gradient 함수)
TEST_FUNCTIONS = {
    # 둥근 그릇 모양
    "x²+y²": (bowl_function, bowl_gradient),
    # 길쭉한 타원형 그릇 모양
    "x²+10y²": (elliptic_function, elliptic_gradient),
}


class Optimizer(ABC):
    """모든 옵티마이저의 공통 틀. 한 걸음(step) 이동 규칙만 서로 다르다."""

    def __init__(self, learning_rate: float) -> None:
        """학습률을 저장한다.

        Args:
            learning_rate: 학습률 η (한 걸음의 크기)
        """
        # 학습률 η
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, parameters: FloatArray, gradient: FloatArray) -> FloatArray:
        """현재 위치와 Gradient로 다음 위치를 계산한다.

        Args:
            parameters: 현재 위치 θ
            gradient: 현재 위치의 Gradient g

        Returns:
            다음 위치
        """

    def reset(self) -> None:
        """새 경로를 시작하기 전에 내부 상태(속도 등)를 지운다.

        상태가 없는 옵티마이저는 할 일이 없다.
        """


class VanillaGD(Optimizer):
    """가장 기본 경사하강법. 수식: θ ← θ - η·g."""

    def step(self, parameters: FloatArray, gradient: FloatArray) -> FloatArray:
        """Gradient 반대 방향으로 η만큼 이동한다.

        Args:
            parameters: 현재 위치 θ
            gradient: 현재 위치의 Gradient g

        Returns:
            다음 위치 θ - η·g
        """
        # 내리막(Gradient 반대) 방향으로 한 걸음 간다
        return parameters - self.learning_rate * gradient


class Momentum(Optimizer):
    """관성을 더한 경사하강법. 수식: v ← β·v + g,  θ ← θ - η·v.

    v는 지난 Gradient의 지수 가중 합이다. 같은 방향이 이어지면 v가 커져 빨라지고,
    방향이 번갈아 바뀌면(지그재그) 서로 상쇄돼 흔들림이 줄어든다.
    """

    def __init__(self, learning_rate: float, beta: float = MOMENTUM_BETA) -> None:
        """학습률, 관성 계수, 속도를 초기화한다.

        Args:
            learning_rate: 학습률 η
            beta: 관성 계수 β (지난 속도를 얼마나 남길지, 0.9면 90%)
        """
        # 부모 클래스에 학습률을 넘긴다
        super().__init__(learning_rate)

        # 관성 계수 β
        self.beta = beta

        # 속도 v (첫 step에서 Gradient 모양으로 만든다)
        self.velocity: FloatArray | None = None

    def reset(self) -> None:
        """속도 v를 지운다."""
        # 속도를 비운다
        self.velocity = None

    def step(self, parameters: FloatArray, gradient: FloatArray) -> FloatArray:
        """속도를 갱신하고 속도 방향으로 이동한다.

        Args:
            parameters: 현재 위치 θ
            gradient: 현재 위치의 Gradient g

        Returns:
            다음 위치 θ - η·v
        """
        # 첫 걸음이면 속도를 0으로 만든다
        if self.velocity is None:
            # Gradient와 같은 모양의 0 벡터
            self.velocity = np.zeros_like(gradient)

        # 지난 속도의 β배에 이번 Gradient를 더한다
        self.velocity = self.beta * self.velocity + gradient

        # 속도 방향으로 η만큼 이동한다
        return parameters - self.learning_rate * self.velocity


class Adam(Optimizer):
    """(보너스) Momentum과 RMSProp을 합친 Adam.

    수식
        m ← β1·m + (1-β1)·g          (1차 모멘트: Momentum 역할)
        s ← β2·s + (1-β2)·g²         (2차 모멘트: RMSProp 역할)
        m̂ = m / (1-β1ᵗ), ŝ = s / (1-β2ᵗ)   (초반 0 쪽 치우침 보정)
        θ ← θ - η·m̂ / (√ŝ + ε)
    좌표마다 Gradient 크기로 나누므로
    가파른 방향과 완만한 방향의 걸음 크기가 비슷해진다.
    """

    def __init__(
        self,
        learning_rate: float,
        beta1: float = ADAM_BETA1,
        beta2: float = ADAM_BETA2,
        epsilon: float = ADAM_EPSILON,
    ) -> None:
        """학습률과 두 모멘트 계수를 저장하고 상태를 초기화한다.

        Args:
            learning_rate: 학습률 η
            beta1: 1차 모멘트 계수 β1
            beta2: 2차 모멘트 계수 β2
            epsilon: 0으로 나누기를 막는 작은 값 ε
        """
        # 부모 클래스에 학습률을 넘긴다
        super().__init__(learning_rate)

        # 1차 모멘트 계수
        self.beta1 = beta1

        # 2차 모멘트 계수
        self.beta2 = beta2

        # 0 나누기 방지 값
        self.epsilon = epsilon

        # 1차 모멘트 m
        self.first_moment: FloatArray | None = None

        # 2차 모멘트 s
        self.second_moment: FloatArray | None = None

        # 지금까지 걸은 횟수 t (치우침 보정에 쓴다)
        self.step_count = 0

    def reset(self) -> None:
        """모멘트와 걸음 횟수를 지운다."""
        # 1차 모멘트를 비운다
        self.first_moment = None

        # 2차 모멘트를 비운다
        self.second_moment = None

        # 걸음 횟수를 0으로 되돌린다
        self.step_count = 0

    def step(self, parameters: FloatArray, gradient: FloatArray) -> FloatArray:
        """두 모멘트를 갱신하고 좌표별로 크기를 맞춘 걸음으로 이동한다.

        Args:
            parameters: 현재 위치 θ
            gradient: 현재 위치의 Gradient g

        Returns:
            다음 위치
        """
        # 첫 걸음이면 두 모멘트를 0으로 만든다
        if self.first_moment is None or self.second_moment is None:
            # 1차 모멘트 0 벡터
            self.first_moment = np.zeros_like(gradient)

            # 2차 모멘트 0 벡터
            self.second_moment = np.zeros_like(gradient)

        # 걸음 횟수를 1 늘린다
        self.step_count += 1

        # 1차 모멘트를 갱신한다 (Gradient의 이동 평균)
        self.first_moment = self.beta1 * self.first_moment + (1 - self.beta1) * gradient

        # 2차 모멘트를 갱신한다 (Gradient 제곱의 이동 평균)
        self.second_moment = (
            self.beta2 * self.second_moment + (1 - self.beta2) * gradient**2
        )

        # 1차 모멘트의 초반 치우침을 보정한다
        corrected_first = self.first_moment / (1 - self.beta1**self.step_count)

        # 2차 모멘트의 초반 치우침을 보정한다
        corrected_second = self.second_moment / (1 - self.beta2**self.step_count)

        # 좌표별 크기를 맞춘 걸음으로 이동한다
        return parameters - self.learning_rate * corrected_first / (
            np.sqrt(corrected_second) + self.epsilon
        )


class NewtonMethod(Optimizer):
    """(보너스) Hessian(2차 미분)을 쓰는 Newton Method. 수식: θ ← θ - η·H⁻¹·g.

    곡률(H)로 걸음 크기를 방향마다 자동으로 맞춘다.
    2차 함수에서는 η = 1이면 한 번에 최소점에 간다.
    대신 n개 변수면 H가 n×n이라 계산과 저장 비용이 크다.
    """

    def __init__(
        self,
        hessian_function: Callable[[FloatArray], FloatArray],
        learning_rate: float = 1.0,
    ) -> None:
        """Hessian 함수와 학습률을 저장한다.

        Args:
            hessian_function: 위치를 받아 Hessian 행렬을 돌려주는 함수
            learning_rate: 학습률 η (기본 1.0)
        """
        # 부모 클래스에 학습률을 넘긴다
        super().__init__(learning_rate)

        # Hessian을 계산하는 함수
        self.hessian_function = hessian_function

    def step(self, parameters: FloatArray, gradient: FloatArray) -> FloatArray:
        """H·Δ = g를 풀어 Newton 방향 Δ = H⁻¹·g로 이동한다.

        Args:
            parameters: 현재 위치 θ
            gradient: 현재 위치의 Gradient g

        Returns:
            다음 위치 θ - η·H⁻¹·g
        """
        # 현재 위치의 Hessian을 구한다
        hessian = self.hessian_function(parameters)

        # 역행렬 대신 연립방정식 H·Δ = g를 푼다 (더 정확하고 빠르다)
        newton_direction = np.linalg.solve(hessian, gradient)

        # Newton 방향으로 이동한다
        return np.asarray(parameters - self.learning_rate * newton_direction)


@dataclass
class LearningRateComparison:
    """한 학습률에서 Vanilla GD와 Momentum을 비교한 결과."""

    # 함수 이름
    function_name: str

    # 학습률 η
    learning_rate: float

    # Vanilla GD의 100회 뒤 원점 거리
    vanilla_final_distance: float

    # Momentum의 100회 뒤 원점 거리
    momentum_final_distance: float

    # Vanilla GD가 반경 0.1 안에 머물기 시작한 반복 (못 하면 None)
    vanilla_converge_step: int | None

    # Momentum이 반경 0.1 안에 머물기 시작한 반복 (못 하면 None)
    momentum_converge_step: int | None


# 옵티마이저를 정해진 횟수만큼 돌려 경로를 기록한다
def run_optimizer(
    optimizer: Optimizer,
    gradient_function: Callable[[FloatArray], FloatArray],
    start_point: FloatArray,
    iteration_count: int,
) -> FloatArray:
    """시작점에서 옵티마이저를 반복 적용하고 지나간 점을 모두 기록한다.

    Args:
        optimizer: 사용할 옵티마이저
        gradient_function: 위치를 받아 Gradient를 돌려주는 함수
        start_point: 시작 위치
        iteration_count: 반복 횟수

    Returns:
        (iteration_count + 1, 변수 개수) 경로. 0행은 시작점이다.
    """
    # 지난 실행의 속도, 모멘트를 지운다
    optimizer.reset()

    # 현재 위치를 시작점 복사본으로 둔다
    parameters = np.array(start_point, dtype=np.float64)

    # 경로 목록에 시작점을 넣는다
    path = [parameters.copy()]

    # 정해진 횟수만큼 반복한다
    for _ in range(iteration_count):
        # 한 걸음 이동한다
        parameters = optimizer.step(parameters, gradient_function(parameters))

        # 이동한 위치를 기록한다
        path.append(parameters.copy())

    # 경로를 배열로 바꿔 돌려준다
    return np.array(path)


# 반경 안에 들어와 계속 머물기 시작한 반복 번호를 구한다
def steps_to_converge(path: FloatArray, radius: float) -> int | None:
    """원점 반경 안에 들어온 뒤 끝까지 머무는 첫 반복 번호를 구한다.

    Momentum처럼 원점을 지나쳤다가 돌아오는 경로는 잠깐 반경 안을 지나갈 수 있다.
    그래서 "처음 들어온 때"가 아니라 "그 뒤로 계속 머무는 첫 때"를 수렴 시점으로 본다.

    Args:
        path: (반복 + 1, 변수 개수) 경로
        radius: 수렴 반경

    Returns:
        반복 번호. 끝까지 머물지 못하면 None
    """
    # 각 지점의 원점 거리
    distances = np.linalg.norm(path, axis=1)

    # 반경 밖에 있던 지점의 번호들
    outside_positions = np.flatnonzero(distances >= radius)

    # 한 번도 밖에 없었으면 시작부터 수렴이다
    if outside_positions.size == 0:
        # 0번째부터 수렴
        return 0

    # 마지막으로 밖에 있던 번호 다음이 수렴 시점이다
    converge_step = int(outside_positions[-1]) + 1

    # 그 번호가 경로 안에 있으면 돌려주고, 아니면 수렴하지 못한 것이다
    return converge_step if converge_step < len(path) else None


# 등고선 위에 여러 옵티마이저의 경로를 점선으로 그린다
def plot_paths_on_contour(
    axis: plt.Axes,
    function: Callable[[FloatArray], FloatArray],
    paths: dict[str, FloatArray],
    title: str,
    plot_limit: float = PLOT_LIMIT,
) -> None:
    """한 축(axis)에 함수의 등고선과 옵티마이저 경로(점선)를 겹쳐 그린다.

    Args:
        axis: 그림을 그릴 Matplotlib 축
        function: 등고선을 그릴 2변수 함수
        paths: 이름 → 경로 사전
        title: 축 제목
        plot_limit: 그림 범위 (-plot_limit ~ plot_limit)
    """
    # 한 축의 좌표를 고르게 나눈다
    axis_values = np.linspace(-plot_limit, plot_limit, CONTOUR_GRID_POINT_COUNT)

    # x, y 격자를 만든다
    grid_x, grid_y = np.meshgrid(axis_values, axis_values)

    # 격자 전체의 함수값을 구한다
    grid_values = function(np.array([grid_x, grid_y]))

    # 등고선을 회색으로 그린다 (원점 근처도 보이도록 높이를 로그 간격으로 잡는다)
    axis.contour(
        grid_x,
        grid_y,
        grid_values,
        levels=CONTOUR_LEVELS,
        colors="gray",
        linewidths=0.6,
    )

    # 경로마다 점선과 점으로 그린다
    for path_name, path in paths.items():
        # 그림 범위를 벗어나는 점은 잘라서 그린다 (발산 경로용)
        clipped_path = np.clip(path, -plot_limit, plot_limit)

        # 점선(--)과 작은 점으로 경로를 그린다
        axis.plot(
            clipped_path[:, 0],
            clipped_path[:, 1],
            "--o",
            markersize=2.5,
            linewidth=1.2,
            label=path_name,
        )

    # 시작점을 검은 별로 표시한다
    axis.plot(*START_POINT, "k*", markersize=12, label="start (5, 5)")

    # 최소점(원점)을 검은 가위표로 표시한다
    axis.plot(0, 0, "kx", markersize=10)

    # x 범위를 고정한다
    axis.set_xlim(-plot_limit, plot_limit)

    # y 범위를 고정한다
    axis.set_ylim(-plot_limit, plot_limit)

    # 가로세로 비율을 1:1로 맞춘다
    axis.set_aspect("equal")

    # 제목을 단다
    axis.set_title(title)

    # 범례를 표시한다
    axis.legend(loc="lower left", fontsize=8)


# 여러 축을 가로로 놓은 그림을 저장한다
def save_figure(figure: plt.Figure, file_name: str) -> None:
    """그림을 outputs/에 저장하고 닫는다.

    Args:
        figure: 저장할 그림
        file_name: 파일 이름
    """
    # 여백을 정리한다
    figure.tight_layout()

    # 그림을 저장한다
    figure.savefig(output_path(file_name), dpi=FIGURE_DPI, bbox_inches="tight")

    # 그림을 닫는다
    plt.close(figure)


# 경로의 마지막 점과 원점 거리를 문자열로 만든다
def describe_path(path: FloatArray) -> str:
    """경로의 마지막 점과 원점 거리를 한 줄 문자열로 만든다.

    Args:
        path: 옵티마이저 경로

    Returns:
        "(x, y), 거리 d" 형식 문자열
    """
    # 마지막 점
    final_point = path[-1]

    # 원점 거리
    final_distance = float(np.linalg.norm(final_point))

    # 사람이 읽기 좋은 문자열을 만든다
    return (
        f"({final_point[0]:.4f}, {final_point[1]:.4f}), 원점 거리 {final_distance:.2e}"
    )


# 1. Vanilla GD 기본 실험 (lr=0.1, 100회)
def run_vanilla_gd() -> None:
    """x² + y²에 Vanilla GD(lr=0.1)를 100번 적용하고 경로를 저장한다."""
    # 경로를 구한다
    path = run_optimizer(
        VanillaGD(BASE_LEARNING_RATE), bowl_gradient, START_POINT, ITERATION_COUNT
    )

    # 마지막 점의 원점 거리
    final_distance = float(np.linalg.norm(path[-1]))

    # 반경 0.1 이내인지 판정한다
    verdict = "통과" if final_distance < CONVERGENCE_RADIUS else "실패"

    # 결과를 출력한다
    print(
        f"Vanilla GD lr=0.1, 100회 뒤: {describe_path(path)} (반경 0.1 이내 {verdict})"
    )

    # 그림판과 축을 만든다
    figure, axis = plt.subplots(figsize=(6, 6))

    # 등고선 위에 경로를 점선으로 그린다
    plot_paths_on_contour(
        axis,
        bowl_function,
        {"Vanilla GD (lr=0.1)": path},
        "f = x²+y², Vanilla GD 100 steps",
    )

    # 그림을 저장한다
    save_figure(figure, "gd_path.png")

    # 저장 위치를 알려준다
    print("-> outputs/gd_path.png")


# 2. 학습률이 크면 발산하는 것을 확인한다
def run_divergence() -> None:
    """학습률별 원점 거리 변화(로그 축)와 발산 경로를 한 그림에 그린다."""
    # 왼쪽: 경로, 오른쪽: 원점 거리 그래프
    figure, (path_axis, distance_axis) = plt.subplots(1, 2, figsize=(13, 6))

    # 경로 그림에 넣을 경로 모음
    divergence_paths = {}

    # (함수, 학습률) 조합마다 반복한다
    for function_name, learning_rate in DIVERGENCE_CASES:
        # 이 함수의 Gradient 함수를 꺼낸다
        gradient_function = TEST_FUNCTIONS[function_name][1]

        # 짧게 경로를 구한다
        path = run_optimizer(
            VanillaGD(learning_rate),
            gradient_function,
            START_POINT,
            DIVERGENCE_ITERATION_COUNT,
        )

        # 반복마다 원점 거리를 구한다
        distances = np.linalg.norm(path, axis=1)

        # 원점 거리를 로그 축에 그린다 (0이면 로그가 안 되므로 아주 작은 값으로 바꾼다)
        distance_axis.semilogy(
            np.maximum(distances, LOG_PLOT_FLOOR),
            "-o",
            markersize=3,
            label=f"{function_name}, lr={learning_rate}",
        )

        # 마지막 거리
        final_distance = float(distances[-1])

        # 처음보다 멀어졌으면 발산, 같으면 진동, 가까워졌으면 수렴
        if final_distance > distances[0] * (1 + OSCILLATION_BAND):
            # 발산
            status = "발산"
        elif final_distance > distances[0] * (1 - OSCILLATION_BAND):
            # 진동
            status = "진동 (거리 그대로)"
        else:
            # 수렴
            status = "수렴"

        # 결과를 출력한다
        print(
            f"{function_name}, lr={learning_rate}: "
            f"{DIVERGENCE_ITERATION_COUNT}회 뒤 원점 거리 {final_distance:.2e} "
            f"-> {status}"
        )

        # x²+y²의 lr≥1 경로만 왼쪽에 넣는다 (x²+10y²는 너무 빨리 커져 오른쪽만)
        if function_name == "x²+y²" and learning_rate >= 1.0:
            # 처음 10걸음을 넣는다
            divergence_paths[f"lr={learning_rate}"] = path[:DIVERGENCE_PATH_STEPS]

    # 왼쪽 축에 x²+y² 등고선과 진동, 발산 경로를 그린다
    plot_paths_on_contour(
        path_axis,
        bowl_function,
        divergence_paths,
        "f = x²+y², lr=1.0 oscillates, lr=1.1 diverges (10 steps)",
        plot_limit=DIVERGENCE_PLOT_LIMIT,
    )

    # lr=1.1 경로의 점마다 반복 번호를 붙인다 (원점 양쪽을 오가며 멀어지는 모습)
    for step_number, point in enumerate(divergence_paths["lr=1.1"]):
        # 그림 범위 안의 점에만 번호를 단다
        if np.all(np.abs(point) < DIVERGENCE_PLOT_LIMIT):
            # 점 오른쪽 아래에 번호를 쓴다
            path_axis.annotate(
                str(step_number),
                point,
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=8,
            )

    # 오른쪽 축 제목
    distance_axis.set_title("distance to origin (log scale)")

    # 오른쪽 축 x 이름
    distance_axis.set_xlabel("iteration")

    # 오른쪽 축 y 이름
    distance_axis.set_ylabel("||(x, y)||")

    # 보조 격자를 켠다
    distance_axis.grid(True, which="both", alpha=0.3)

    # 범례를 표시한다
    distance_axis.legend()

    # 그림을 저장한다
    save_figure(figure, "gd_divergence.png")

    # 저장 위치를 알려준다
    print("-> outputs/gd_divergence.png")


# 3. 두 함수에서 Vanilla GD와 Momentum 경로를 겹쳐 비교한다
def run_momentum_comparison() -> None:
    """함수마다 두 학습률에서 Vanilla GD와 Momentum 경로를 한 Figure에 겹쳐 그린다."""
    # (함수 이름, 학습률 두 개, 파일 이름)
    comparison_settings = (
        # 둥근 함수: 과제 기준 lr=0.1과 작은 lr=0.01
        ("x²+y²", (BASE_LEARNING_RATE, SMALL_LEARNING_RATE), "momentum_vs_gd_bowl.png"),
        # 타원형 함수: 작은 lr=0.01과 GD가 지그재그하는 lr=0.09
        (
            "x²+10y²",
            (SMALL_LEARNING_RATE, ZIGZAG_LEARNING_RATE),
            "momentum_vs_gd_elliptic.png",
        ),
    )

    # 함수마다 반복한다
    for function_name, learning_rates, file_name in comparison_settings:
        # 함수와 Gradient 함수를 꺼낸다
        function, gradient_function = TEST_FUNCTIONS[function_name]

        # 학습률 두 개를 나란히 그릴 그림판
        figure, axes = plt.subplots(1, 2, figsize=(13, 6.5))

        # 학습률마다 반복한다
        for axis, learning_rate in zip(axes, learning_rates):
            # Vanilla GD 경로
            vanilla_path = run_optimizer(
                VanillaGD(learning_rate),
                gradient_function,
                START_POINT,
                ITERATION_COUNT,
            )

            # Momentum 경로
            momentum_path = run_optimizer(
                Momentum(learning_rate, MOMENTUM_BETA),
                gradient_function,
                START_POINT,
                ITERATION_COUNT,
            )

            # 두 경로를 한 축에 겹쳐 그린다
            plot_paths_on_contour(
                axis,
                function,
                {"Vanilla GD": vanilla_path, "Momentum (beta=0.9)": momentum_path},
                f"f = {function_name}, lr={learning_rate}, 100 steps",
            )

            # 결과를 출력한다
            print(
                f"[{function_name}, lr={learning_rate}] "
                f"Vanilla GD: {describe_path(vanilla_path)}"
            )

            # 결과를 출력한다
            print(
                f"[{function_name}, lr={learning_rate}] "
                f"Momentum:   {describe_path(momentum_path)}"
            )

        # 그림을 저장한다
        save_figure(figure, file_name)

        # 저장 위치를 알려준다
        print(f"-> outputs/{file_name}")


# 학습률별 Vanilla GD와 Momentum 비교표를 만든다
def compare_learning_rates() -> list[LearningRateComparison]:
    """두 함수와 여러 학습률에서 100회 뒤 거리와 수렴 시점을 비교한다.

    Returns:
        (함수, 학습률)마다 LearningRateComparison 목록
    """
    # 결과를 담을 목록
    comparisons = []

    # 함수마다 반복한다
    for function_name, (_, gradient_function) in TEST_FUNCTIONS.items():
        # 학습률마다 반복한다
        for learning_rate in COMPARISON_LEARNING_RATES:
            # Vanilla GD 경로
            vanilla_path = run_optimizer(
                VanillaGD(learning_rate),
                gradient_function,
                START_POINT,
                ITERATION_COUNT,
            )

            # Momentum 경로
            momentum_path = run_optimizer(
                Momentum(learning_rate, MOMENTUM_BETA),
                gradient_function,
                START_POINT,
                ITERATION_COUNT,
            )

            # 비교 결과를 목록에 더한다
            comparisons.append(
                LearningRateComparison(
                    function_name,
                    learning_rate,
                    float(np.linalg.norm(vanilla_path[-1])),
                    float(np.linalg.norm(momentum_path[-1])),
                    steps_to_converge(vanilla_path, CONVERGENCE_RADIUS),
                    steps_to_converge(momentum_path, CONVERGENCE_RADIUS),
                )
            )

    # 비교 결과를 돌려준다
    return comparisons


# 비교표를 출력한다
def print_learning_rate_table() -> None:
    """학습률별 비교표를 Markdown 표 형식으로 출력한다."""
    # 표 머리글
    print(
        "| 함수 | lr | GD 100회 뒤 거리 | Momentum 100회 뒤 거리 "
        "| GD 수렴 반복 | Momentum 수렴 반복 |"
    )

    # 표 구분선
    print("|---|---|---|---|---|---|")

    # 행마다 출력한다
    for comparison in compare_learning_rates():
        # 수렴하지 못했으면 "-"로 적는다
        vanilla_step = (
            comparison.vanilla_converge_step
            if comparison.vanilla_converge_step is not None
            else "-"
        )

        # 수렴하지 못했으면 "-"로 적는다
        momentum_step = (
            comparison.momentum_converge_step
            if comparison.momentum_converge_step is not None
            else "-"
        )

        # 한 행을 출력한다
        print(
            f"| {comparison.function_name} | {comparison.learning_rate} | "
            f"{comparison.vanilla_final_distance:.2e} | "
            f"{comparison.momentum_final_distance:.2e} | "
            f"{vanilla_step} | {momentum_step} |"
        )


# (보너스) 네 옵티마이저의 수렴 속도를 비교한다
def run_bonus_comparison() -> None:
    """x² + 10y²에서 GD, Momentum, Adam, Newton Method의 수렴 속도를 비교한다."""

    # Newton Method가 쓸 Hessian 함수
    def elliptic_hessian(point: FloatArray) -> FloatArray:
        """x² + 10y²의 수치 Hessian을 돌려준다."""
        # 중심차분으로 Hessian을 구한다
        return numerical_hessian(elliptic_function, point)

    # 비교할 옵티마이저들 (같은 함수, 같은 시작점)
    optimizers = {
        # Vanilla GD (지그재그 직전의 큰 학습률)
        "Vanilla GD (lr=0.09)": VanillaGD(ZIGZAG_LEARNING_RATE),
        # Momentum
        "Momentum (lr=0.05)": Momentum(BONUS_MOMENTUM_LEARNING_RATE, MOMENTUM_BETA),
        # Adam
        f"Adam (lr={ADAM_LEARNING_RATE})": Adam(ADAM_LEARNING_RATE),
        # Newton Method
        "Newton (lr=1)": NewtonMethod(elliptic_hessian),
    }

    # 오른쪽 거리 그래프와 왼쪽 경로 그림
    figure, (path_axis, distance_axis) = plt.subplots(1, 2, figsize=(13, 6))

    # 경로 모음
    paths = {}

    # 옵티마이저마다 반복한다
    for optimizer_name, optimizer in optimizers.items():
        # 경로를 구한다
        path = run_optimizer(
            optimizer, elliptic_gradient, START_POINT, BONUS_ITERATION_COUNT
        )

        # 경로를 모은다
        paths[optimizer_name] = path

        # 원점 거리를 로그 축에 그린다
        distance_axis.semilogy(
            np.maximum(np.linalg.norm(path, axis=1), LOG_PLOT_FLOOR),
            label=optimizer_name,
        )

        # 수렴 시점 (반경 0.1, 반경 0.001)
        loose_step = steps_to_converge(path, CONVERGENCE_RADIUS)

        # 엄격한 수렴 시점
        strict_step = steps_to_converge(path, STRICT_CONVERGENCE_RADIUS)

        # 결과를 출력한다
        print(
            f"{optimizer_name:<22}: 반경 0.1 수렴 {loose_step}회, "
            f"반경 0.001 수렴 {strict_step}회"
        )

    # 왼쪽에 경로를 그린다
    plot_paths_on_contour(
        path_axis, elliptic_function, paths, "f = x²+10y², bonus optimizers"
    )

    # 오른쪽 제목
    distance_axis.set_title("distance to origin (log scale)")

    # 오른쪽 x 이름
    distance_axis.set_xlabel("iteration")

    # 보조 격자를 켠다
    distance_axis.grid(True, which="both", alpha=0.3)

    # 범례를 표시한다
    distance_axis.legend()

    # 그림을 저장한다
    save_figure(figure, "bonus_optimizers.png")

    # 저장 위치를 알려준다
    print("-> outputs/bonus_optimizers.png")


# 모듈을 직접 실행하면 최적화 실험을 모두 수행한다
def main() -> None:
    """Vanilla GD, 발산, Momentum 비교, 학습률 비교표, 보너스 비교를 차례로 실행한다."""
    # 1단계 제목
    print("== 1. Vanilla GD (x²+y², lr=0.1, 100회) ==")

    # 기본 실험을 실행한다
    run_vanilla_gd()

    # 2단계 제목
    print("\n== 2. 큰 학습률에서의 발산 ==")

    # 발산 실험을 실행한다
    run_divergence()

    # 3단계 제목
    print("\n== 3. Vanilla GD vs Momentum 경로 비교 ==")

    # 경로 비교를 실행한다
    run_momentum_comparison()

    # 4단계 제목
    print(
        "\n== 4. 학습률별 비교표 (수렴 반복: 반경 0.1 안에 계속 머물기 시작한 반복) =="
    )

    # 비교표를 출력한다
    print_learning_rate_table()

    # 5단계 제목
    print("\n== 5. (보너스) Adam, Newton Method 비교 (x²+10y²) ==")

    # 보너스 비교를 실행한다
    run_bonus_comparison()


# 파일을 직접 실행했을 때만 아래 코드를 실행한다
if __name__ == "__main__":
    # 스크립트로 실행하면 main을 부른다
    main()
