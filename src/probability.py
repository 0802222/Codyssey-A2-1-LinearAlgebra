"""확률/통계 모듈: 확률 분포, 소프트맥스, 손실 함수, (보너스) 정보 이론.

다루는 수식
    정규분포 PDF: p(x) = 1/√(2πσ²) · exp(-(x - μ)² / 2σ²)
    베르누이 PMF: P(X = k) = pᵏ (1 - p)¹⁻ᵏ,  k ∈ {0, 1}
    소프트맥스: softmax(z)_i = exp(z_i - max z) / Σ_j exp(z_j - max z)
    MSE = (1/n) Σ (y_i - ŷ_i)²
    BCE = -(1/n) Σ [y_i log ŷ_i + (1 - y_i) log(1 - ŷ_i)]
    엔트로피 H(p) = -Σ p_i log p_i
    KL(p || q) = Σ p_i log(p_i / q_i)
    Cross-Entropy H(p, q) = -Σ p_i log q_i = H(p) + KL(p || q)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.container import BarContainer

from src.constants import FIGURE_DPI, FloatArray, output_path

# log(0)을 막기 위해 확률을 자르는 작은 값
PROBABILITY_EPSILON = 1e-12

# 정규분포 비교: (평균 μ, 분산 σ²) 두 개 (과제: N(0, 1), N(2, 0.5))
NORMAL_DISTRIBUTIONS = ((0.0, 1.0), (2.0, 0.5))

# 베르누이 비교: 성공 확률 p 두 개 (과제: B(0.3), B(0.7))
BERNOULLI_PROBABILITIES = (0.3, 0.7)

# 정규분포 그림의 x 범위
NORMAL_PLOT_RANGE = (-4.0, 6.0)

# 정규분포 그림의 점 개수
NORMAL_PLOT_POINT_COUNT = 500


# 정규분포의 확률 밀도
def normal_pdf(values: FloatArray, mean: float, variance: float) -> FloatArray:
    """정규분포 N(μ, σ²)의 확률 밀도 함수(PDF) 값을 계산한다.

    수식: p(x) = 1/√(2πσ²) · exp(-(x - μ)² / 2σ²)

    Args:
        values: 확률 밀도를 구할 x 값들
        mean: 평균 μ
        variance: 분산 σ² (표준편차의 제곱)

    Returns:
        x마다의 확률 밀도
    """
    # 정규화 상수 1/√(2πσ²)
    normalizer = 1.0 / np.sqrt(2 * np.pi * variance)

    # 지수 부분 exp(-(x - μ)² / 2σ²)를 곱한다
    return np.asarray(normalizer * np.exp(-((values - mean) ** 2) / (2 * variance)))


# 베르누이 분포의 확률 질량
def bernoulli_pmf(outcomes: npt.ArrayLike, success_probability: float) -> FloatArray:
    """베르누이 분포 B(p)의 확률 질량 함수(PMF) 값을 계산한다.

    수식: P(X = k) = pᵏ (1 - p)¹⁻ᵏ  (k = 1이면 p, k = 0이면 1 - p)

    Args:
        outcomes: 결과값들 (0 또는 1)
        success_probability: 성공(1) 확률 p

    Returns:
        결과마다의 확률
    """
    # 결과를 실수 배열로 바꾼다
    outcome_array = np.asarray(outcomes, dtype=np.float64)

    # pᵏ (1 - p)¹⁻ᵏ를 계산한다
    return np.asarray(
        success_probability**outcome_array
        * (1 - success_probability) ** (1 - outcome_array)
    )


# 소프트맥스: 점수(로짓)를 확률로 바꾼다
def softmax(logits: FloatArray) -> FloatArray:
    """로짓 벡터를 합이 1인 확률 벡터로 바꾼다.

    수식: softmax(z)_i = exp(z_i - max z) / Σ_j exp(z_j - max z)
    최댓값을 빼도 결과는 같지만 exp가 너무 커지는 overflow를 막는다.

    Args:
        logits: 로짓 벡터 z

    Returns:
        같은 길이의 확률 벡터
    """
    # 최댓값을 빼서 exp가 넘치지 않게 한다
    shifted_logits = logits - np.max(logits)

    # 지수를 취한다
    exponentials = np.exp(shifted_logits)

    # 전체 합으로 나눠 확률로 만든다
    return np.asarray(exponentials / np.sum(exponentials))


# 평균 제곱 오차 (MSE)
def mean_squared_error(targets: FloatArray, predictions: FloatArray) -> float:
    """평균 제곱 오차(MSE)를 계산한다.

    수식: MSE = (1/n) Σ (y_i - ŷ_i)²

    Args:
        targets: 정답 y
        predictions: 예측 ŷ

    Returns:
        MSE 값
    """
    # 오차 제곱의 평균을 구한다
    return float(np.mean((targets - predictions) ** 2))


# 정규분포 가정의 음의 로그우도 (평균)
def gaussian_negative_log_likelihood(
    targets: FloatArray, predictions: FloatArray, variance: float
) -> float:
    """정답이 y ~ N(ŷ, σ²)를 따를 때 평균 음의 로그우도(NLL)를 계산한다.

    수식: NLL = (1/n) Σ [(y_i - ŷ_i)² / 2σ² + ½ log(2πσ²)]
         = MSE / 2σ² + 상수  → σ가 고정이면 MSE를 줄이는 것과 같다

    Args:
        targets: 관측값 y
        predictions: 예측 평균 ŷ
        variance: 고정 분산 σ²

    Returns:
        평균 NLL
    """
    # 데이터마다 -log p(y | ŷ)를 구한다
    negative_log_likelihoods = (targets - predictions) ** 2 / (
        2 * variance
    ) + 0.5 * np.log(2 * np.pi * variance)

    # 평균을 돌려준다
    return float(np.mean(negative_log_likelihoods))


# 이진 교차 엔트로피 (BCE)
def binary_cross_entropy(targets: FloatArray, probabilities: FloatArray) -> float:
    """이진 교차 엔트로피(BCE)를 계산한다. 베르누이 분포의 평균 음의 로그우도와 같다.

    수식: BCE = -(1/n) Σ [y_i log ŷ_i + (1 - y_i) log(1 - ŷ_i)]

    Args:
        targets: 정답 y (0 또는 1)
        probabilities: 예측 확률 ŷ (1일 확률)

    Returns:
        BCE 값
    """
    # log(0)이 되지 않게 확률을 (ε, 1 - ε)로 자른다
    clipped = np.clip(probabilities, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)

    # 데이터마다 손실을 구해 평균을 낸다
    return float(
        -np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped))
    )


# 다중 분류 교차 엔트로피
def categorical_cross_entropy(
    one_hot_target: FloatArray, predicted_probabilities: FloatArray
) -> float:
    """원-핫 정답과 예측 확률 사이의 Cross-Entropy를 계산한다 (카테고리 분포의 NLL).

    수식: CE = -Σ_k y_k log ŷ_k  (원-핫이면 -log ŷ_정답)

    Args:
        one_hot_target: 원-핫 정답 벡터
        predicted_probabilities: 예측 확률 벡터 (예: softmax 출력)

    Returns:
        CE 값
    """
    # log(0)을 막도록 확률을 자른다
    clipped = np.clip(predicted_probabilities, PROBABILITY_EPSILON, 1.0)

    # 정답 위치의 -log 확률을 더한다
    return float(-np.sum(one_hot_target * np.log(clipped)))


# (보너스) 엔트로피
def entropy(probabilities: FloatArray) -> float:
    """분포의 엔트로피(불확실성)를 계산한다. 단위는 nat(자연로그)이다.

    수식: H(p) = -Σ p_i log p_i  (p_i = 0인 항은 0으로 본다)

    Args:
        probabilities: 확률 분포 p

    Returns:
        엔트로피
    """
    # 0이 아닌 확률만 남긴다 (0 · log 0 = 0)
    nonzero = probabilities[probabilities > 0]

    # -Σ p log p를 구한다
    return float(-np.sum(nonzero * np.log(nonzero)))


# (보너스) KL-Divergence
def kl_divergence(
    true_probabilities: FloatArray, model_probabilities: FloatArray
) -> float:
    """실제 분포 p와 모델 분포 q 사이의 KL-Divergence를 계산한다.

    수식: KL(p || q) = Σ p_i log(p_i / q_i) ≥ 0, p = q일 때만 0

    Args:
        true_probabilities: 실제 분포 p
        model_probabilities: 모델 분포 q

    Returns:
        KL 값
    """
    # p가 0인 항은 0이므로 제외할 위치를 고른다
    nonzero_mask = true_probabilities > 0

    # p > 0인 항의 p
    true_nonzero = true_probabilities[nonzero_mask]

    # 같은 위치의 q (0이면 ε으로 자른다)
    model_nonzero = np.clip(model_probabilities[nonzero_mask], PROBABILITY_EPSILON, 1.0)

    # Σ p log(p / q)를 구한다
    return float(np.sum(true_nonzero * np.log(true_nonzero / model_nonzero)))


# (보너스) Cross-Entropy
def cross_entropy(
    true_probabilities: FloatArray, model_probabilities: FloatArray
) -> float:
    """실제 분포 p를 모델 분포 q로 부호화할 때의 Cross-Entropy를 계산한다.

    수식: H(p, q) = -Σ p_i log q_i = H(p) + KL(p || q)

    Args:
        true_probabilities: 실제 분포 p
        model_probabilities: 모델 분포 q

    Returns:
        Cross-Entropy 값
    """
    # log(0)을 막도록 q를 자른다
    clipped = np.clip(model_probabilities, PROBABILITY_EPSILON, 1.0)

    # -Σ p log q를 구한다
    return float(-np.sum(true_probabilities * np.log(clipped)))


# 정규분포 PDF 두 개를 한 그림에 비교한다
def plot_normal_pdfs(file_name: str = "normal_pdf.png") -> None:
    """N(0, 1)과 N(2, 0.5)의 PDF 곡선을 하나의 Figure에 그린다.

    Args:
        file_name: outputs/에 저장할 파일 이름
    """
    # x 값을 고르게 나눈다
    x_values = np.linspace(*NORMAL_PLOT_RANGE, NORMAL_PLOT_POINT_COUNT)

    # 그림판과 축을 만든다
    figure, axis = plt.subplots(figsize=(8, 4.5))

    # 분포마다 곡선을 그린다
    for mean, variance in NORMAL_DISTRIBUTIONS:
        # PDF 값을 구한다
        densities = normal_pdf(x_values, mean, variance)

        # 곡선을 그린다
        axis.plot(
            x_values,
            densities,
            linewidth=2,
            label=f"N({mean:g}, {variance:g}): μ={mean:g}, σ={np.sqrt(variance):.3f}",
        )

        # 곡선 아래를 옅게 칠한다
        axis.fill_between(x_values, densities, alpha=0.15)

        # 평균 위치에 점선을 긋는다
        axis.axvline(mean, linestyle=":", color="gray")

    # 제목을 단다
    axis.set_title(
        "Normal distribution PDF: N(0, 1) vs N(2, 0.5)  (second number = variance σ²)"
    )

    # x축 이름
    axis.set_xlabel("x")

    # y축 이름
    axis.set_ylabel("probability density p(x)")

    # 범례를 표시한다
    axis.legend()

    # 그림을 저장한다
    figure.savefig(output_path(file_name), dpi=FIGURE_DPI, bbox_inches="tight")

    # 그림을 닫는다
    plt.close(figure)


# 베르누이 PMF 두 개를 한 그림에 비교한다
def plot_bernoulli_pmfs(file_name: str = "bernoulli_pmf.png") -> None:
    """B(0.3)과 B(0.7)의 PMF를 하나의 Figure에 막대그래프로 그린다.

    Args:
        file_name: outputs/에 저장할 파일 이름
    """
    # 결과값 0과 1
    outcomes = np.array([0, 1])

    # Seaborn에 넘길 열 데이터: 결과값, 확률, 분포 이름
    outcome_column, probability_column, distribution_column = [], [], []

    # 분포마다 확률을 모은다
    for success_probability in BERNOULLI_PROBABILITIES:
        # 결과값을 더한다
        outcome_column.extend(outcomes.tolist())

        # 확률을 더한다
        probability_column.extend(bernoulli_pmf(outcomes, success_probability).tolist())

        # 분포 이름을 더한다
        distribution_column.extend([f"B({success_probability})"] * outcomes.size)

    # 그림판과 축을 만든다
    figure, axis = plt.subplots(figsize=(7, 4.5))

    # 분포별 색을 나눠 막대를 나란히 그린다 (Seaborn)
    sns.barplot(
        x=outcome_column, y=probability_column, hue=distribution_column, ax=axis
    )

    # 막대 위에 확률 값을 적는다
    for container in axis.containers:
        # 막대 묶음(BarContainer)에만 값을 단다
        if isinstance(container, BarContainer):
            # 소수점 1자리로 적는다
            axis.bar_label(container, fmt="%.1f")

    # 제목을 단다
    axis.set_title("Bernoulli PMF: B(0.3) vs B(0.7)")

    # x축 이름
    axis.set_xlabel("outcome k")

    # y축 이름
    axis.set_ylabel("P(X = k)")

    # y 범위를 0~1로 고정한다
    axis.set_ylim(0, 1)

    # 그림을 저장한다
    figure.savefig(output_path(file_name), dpi=FIGURE_DPI, bbox_inches="tight")

    # 그림을 닫는다
    plt.close(figure)
