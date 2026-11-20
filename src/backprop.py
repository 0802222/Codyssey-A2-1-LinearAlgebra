"""역전파 모듈: 2층 신경망(입력 2 → 은닉 2 → 출력 1)의 순전파와 역전파를 계산한다.

노트북 notebooks/backprop_derivation.ipynb의 손계산 결과를 검증하는 코드이다.
코드 이름과 수식 기호의 대응
    x  = input_vector             (2,)
    W1 = hidden_weights           (2, 2)    b1 = hidden_bias    (2,)
    z1 = hidden_pre_activation    (2,)      z1 = W1 @ x + b1
    a1 = hidden_activation        (2,)      a1 = σ(z1)
    W2 = output_weights           (2,)      b2 = output_bias    스칼라
    z2 = output_pre_activation    스칼라    z2 = W2 @ a1 + b2
    ŷ  = prediction               스칼라    ŷ = σ(z2)
    L  = loss                     스칼라    L = -(y log ŷ + (1 - y) log(1 - ŷ))

역전파 (연쇄 법칙)
    ∂L/∂ŷ  = -y/ŷ + (1 - y)/(1 - ŷ)
    ∂L/∂z2 = ∂L/∂ŷ · σ'(z2) = ∂L/∂ŷ · ŷ(1 - ŷ) = ŷ - y
    ∂L/∂W2 = ∂L/∂z2 · a1,  ∂L/∂b2 = ∂L/∂z2
    ∂L/∂a1 = ∂L/∂z2 · W2
    ∂L/∂z1 = ∂L/∂a1 ⊙ σ'(z1) = ∂L/∂a1 ⊙ a1(1 - a1)
    ∂L/∂W1 = outer(∂L/∂z1, x),  ∂L/∂b1 = ∂L/∂z1
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.calculus import numerical_gradient
from src.constants import FloatArray
from src.probability import binary_cross_entropy


@dataclass
class NetworkParameters:
    """2층 신경망의 가중치와 편향."""

    # W1: (2, 2), 행 = 은닉 뉴런, 열 = 입력
    hidden_weights: FloatArray

    # b1: (2,)
    hidden_bias: FloatArray

    # W2: (2,)
    output_weights: FloatArray

    # b2: 스칼라
    output_bias: float


@dataclass
class FixedExample:
    """과제 필수 고정 예제의 입력, 파라미터, 정답."""

    # x: (2,)
    input_vector: FloatArray

    # W1, b1, W2, b2
    parameters: NetworkParameters

    # 정답 y_true
    target: float


@dataclass
class ForwardResult:
    """순전파 중간값 (역전파에서 다시 쓴다)."""

    # x: (2,)
    input_vector: FloatArray

    # z1 = W1 @ x + b1: (2,)
    hidden_pre_activation: FloatArray

    # a1 = σ(z1): (2,)
    hidden_activation: FloatArray

    # z2 = W2 @ a1 + b2: 스칼라
    output_pre_activation: float

    # ŷ = σ(z2): 스칼라
    prediction: float

    # BCE 손실 L: 스칼라
    loss: float


@dataclass
class Gradients:
    """손실 L을 각 값으로 미분한 결과. 이름은 "무엇으로 미분했는지"이다."""

    # ∂L/∂ŷ: 스칼라
    prediction: float

    # ∂L/∂z2: 스칼라
    output_pre_activation: float

    # ∂L/∂W2: (2,)
    output_weights: FloatArray

    # ∂L/∂b2: 스칼라
    output_bias: float

    # ∂L/∂a1: (2,)
    hidden_activation: FloatArray

    # ∂L/∂z1: (2,)
    hidden_pre_activation: FloatArray

    # ∂L/∂W1: (2, 2)
    hidden_weights: FloatArray

    # ∂L/∂b1: (2,)
    hidden_bias: FloatArray


# 과제 필수 고정 예제를 만든다
def make_fixed_example() -> FixedExample:
    """과제에서 정한 고정 예제를 돌려준다.

    x = [1, 0], W1 = [[0.1, 0.2], [0.3, 0.4]], b1 = [0, 0],
    W2 = [0.5, 0.6], b2 = 0, y_true = 1 (과제 결과 예시를 따른다)

    Returns:
        입력, 파라미터, 정답을 담은 FixedExample
    """
    # 파라미터를 묶는다
    parameters = NetworkParameters(
        # W1: (2, 2)
        hidden_weights=np.array([[0.1, 0.2], [0.3, 0.4]]),
        # b1: (2,)
        hidden_bias=np.array([0.0, 0.0]),
        # W2: (2,)
        output_weights=np.array([0.5, 0.6]),
        # b2: 스칼라
        output_bias=0.0,
    )

    # 입력 x = [1, 0], 정답 y = 1로 예제를 만든다
    return FixedExample(np.array([1.0, 0.0]), parameters, 1.0)


# 시그모이드 활성화 함수
def sigmoid(values: FloatArray) -> FloatArray:
    """시그모이드 σ(z) = 1 / (1 + e⁻ᶻ)를 계산한다. 출력은 0~1 사이이다.

    Args:
        values: 입력 z

    Returns:
        σ(z)
    """
    # 1 / (1 + e^(-z))
    return np.asarray(1.0 / (1.0 + np.exp(-values)))


# 시그모이드의 미분
def sigmoid_derivative(values: FloatArray) -> FloatArray:
    """시그모이드의 미분 σ'(z) = σ(z)(1 - σ(z))를 계산한다.

    Args:
        values: 입력 z

    Returns:
        σ'(z)
    """
    # σ(z)를 먼저 구한다
    activated = sigmoid(values)

    # σ(z)(1 - σ(z))
    return np.asarray(activated * (1 - activated))


# 순전파: 입력에서 손실까지 계산한다
def forward_pass(
    input_vector: FloatArray, parameters: NetworkParameters, target: float
) -> ForwardResult:
    """입력 → 은닉층 → 출력층 → 손실 순서로 계산하고 중간값을 모두 기록한다.

    수식: z1 = W1 x + b1, a1 = σ(z1), z2 = W2·a1 + b2, ŷ = σ(z2), L = BCE(y, ŷ)

    Args:
        input_vector: 입력 x (2,)
        parameters: W1, b1, W2, b2
        target: 정답 y

    Returns:
        z1, a1, z2, ŷ, L을 담은 ForwardResult
    """
    # 은닉층 선형 결합 z1 = W1 @ x + b1: (2,2) @ (2,) → (2,)
    hidden_pre_activation = (
        parameters.hidden_weights @ input_vector + parameters.hidden_bias
    )

    # 은닉층 활성화 a1 = σ(z1): (2,)
    hidden_activation = sigmoid(hidden_pre_activation)

    # 출력층 선형 결합 z2 = W2 @ a1 + b2: (2,) @ (2,) → 스칼라
    output_pre_activation = float(
        parameters.output_weights @ hidden_activation + parameters.output_bias
    )

    # 출력 확률 ŷ = σ(z2): 스칼라
    prediction = float(sigmoid(np.array(output_pre_activation)))

    # BCE 손실 L: 스칼라
    loss = binary_cross_entropy(np.array([target]), np.array([prediction]))

    # 중간값을 모두 묶어 돌려준다
    return ForwardResult(
        input_vector,
        hidden_pre_activation,
        hidden_activation,
        output_pre_activation,
        prediction,
        loss,
    )


# 역전파: 출력층에서 입력층 방향으로 연쇄 법칙을 적용한다
def backward_pass(
    forward_result: ForwardResult, parameters: NetworkParameters, target: float
) -> Gradients:
    """연쇄 법칙으로 손실을 각 파라미터와 중간값으로 미분한다.

    지름길 ∂L/∂z2 = ŷ - y를 바로 쓰지 않고,
    ∂L/∂ŷ · σ'(z2)로 한 단계씩 곱해 연쇄 법칙을 보인다.

    Args:
        forward_result: 순전파 중간값
        parameters: W1, b1, W2, b2
        target: 정답 y

    Returns:
        모든 기울기를 담은 Gradients
    """
    # 예측값 ŷ
    prediction = forward_result.prediction

    # 1. ∂L/∂ŷ = -y/ŷ + (1 - y)/(1 - ŷ): 스칼라
    gradient_prediction = -target / prediction + (1 - target) / (1 - prediction)

    # 2. ∂L/∂z2 = ∂L/∂ŷ · σ'(z2), σ'(z2) = ŷ(1 - ŷ): 스칼라
    gradient_output_pre_activation = gradient_prediction * prediction * (1 - prediction)

    # 3. ∂L/∂W2 = ∂L/∂z2 · a1 (z2 = W2·a1 이므로 ∂z2/∂W2 = a1): (2,)
    gradient_output_weights = (
        gradient_output_pre_activation * forward_result.hidden_activation
    )

    # 3-1. ∂L/∂b2 = ∂L/∂z2 (∂z2/∂b2 = 1): 스칼라
    gradient_output_bias = gradient_output_pre_activation

    # 4. ∂L/∂a1 = ∂L/∂z2 · W2 (∂z2/∂a1 = W2): (2,)
    gradient_hidden_activation = (
        gradient_output_pre_activation * parameters.output_weights
    )

    # 5. ∂L/∂z1 = ∂L/∂a1 ⊙ σ'(z1) (원소별 곱): (2,)
    gradient_hidden_pre_activation = gradient_hidden_activation * sigmoid_derivative(
        forward_result.hidden_pre_activation
    )

    # 6. ∂L/∂W1 = outer(∂L/∂z1, x), ∂z1_i/∂W1_ij = x_j 이므로: (2, 2)
    gradient_hidden_weights = np.outer(
        gradient_hidden_pre_activation, forward_result.input_vector
    )

    # 6-1. ∂L/∂b1 = ∂L/∂z1: (2,)
    gradient_hidden_bias = gradient_hidden_pre_activation

    # 모든 기울기를 묶어 돌려준다
    return Gradients(
        prediction=gradient_prediction,
        output_pre_activation=gradient_output_pre_activation,
        output_weights=gradient_output_weights,
        output_bias=gradient_output_bias,
        hidden_activation=gradient_hidden_activation,
        hidden_pre_activation=gradient_hidden_pre_activation,
        hidden_weights=gradient_hidden_weights,
        hidden_bias=gradient_hidden_bias,
    )


# 수치 미분으로 ∂L/∂W1을 구한다 (역전파 검증용)
def numerical_hidden_weight_gradient(
    input_vector: FloatArray, parameters: NetworkParameters, target: float
) -> FloatArray:
    """W1의 각 원소를 조금씩 바꿔 가며 중심차분으로 ∂L/∂W1을 구한다.

    역전파(연쇄 법칙) 결과가 맞는지 다른 방법으로 확인하는 Gradient Check이다.

    Args:
        input_vector: 입력 x
        parameters: W1, b1, W2, b2
        target: 정답 y

    Returns:
        (2, 2) 수치 ∂L/∂W1
    """

    # W1만 바꿔 손실을 계산하는 함수
    def loss_for_hidden_weights(hidden_weights: FloatArray) -> FloatArray:
        """W1을 받아 손실 L을 돌려준다."""
        # W1만 바꾼 파라미터를 만든다
        changed_parameters = replace(parameters, hidden_weights=hidden_weights)

        # 순전파로 손실을 구한다
        return np.asarray(forward_pass(input_vector, changed_parameters, target).loss)

    # 중심차분으로 W1과 같은 모양의 Gradient를 구한다
    return numerical_gradient(loss_for_hidden_weights, parameters.hidden_weights)
