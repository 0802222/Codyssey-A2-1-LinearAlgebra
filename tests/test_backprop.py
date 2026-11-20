"""역전파 모듈(src/backprop.py)의 단위 테스트."""

import unittest

import numpy as np

from src.backprop import (
    backward_pass,
    forward_pass,
    make_fixed_example,
    numerical_hidden_weight_gradient,
    sigmoid,
)


class TestFixedExample(unittest.TestCase):
    """과제 필수 고정 예제의 손계산 값과 NumPy 결과를 비교한다."""

    def setUp(self) -> None:
        """고정 예제로 순전파와 역전파를 한 번 실행한다."""
        # 과제에서 정한 입력, 가중치, 정답을 불러온다
        self.example = make_fixed_example()

        # 순전파를 실행한다
        self.forward_result = forward_pass(
            self.example.input_vector, self.example.parameters, self.example.target
        )

        # 역전파를 실행한다
        self.gradients = backward_pass(
            self.forward_result, self.example.parameters, self.example.target
        )

    def test_forward_matches_hand_calculation(self) -> None:
        """z1, a1, z2, y_pred가 손계산과 소수점 4자리까지 같다."""
        # z1 = W1 @ x + b1 = [0.1, 0.3]
        np.testing.assert_allclose(
            self.forward_result.hidden_pre_activation, [0.1, 0.3], atol=5e-5
        )

        # a1 = sigmoid(z1) = [0.5250, 0.5744]
        np.testing.assert_allclose(
            self.forward_result.hidden_activation, [0.5250, 0.5744], atol=5e-5
        )

        # z2 = W2 @ a1 + b2 = 0.607155 → 0.6072 (a1을 4자리로 먼저 반올림하면 0.6071)
        self.assertAlmostEqual(
            self.forward_result.output_pre_activation, 0.6072, delta=5e-5
        )

        # y_pred = sigmoid(z2) = 0.6473
        self.assertAlmostEqual(self.forward_result.prediction, 0.6473, delta=5e-5)

    def test_backward_matches_hand_calculation(self) -> None:
        """역전파 6개 값이 손계산과 소수점 4자리까지 같다."""
        # ∂L/∂y_pred = -1/0.6473 = -1.5449
        self.assertAlmostEqual(self.gradients.prediction, -1.5449, delta=5e-5)

        # ∂L/∂z2 = y_pred - y_true = -0.3527
        self.assertAlmostEqual(
            self.gradients.output_pre_activation, -0.3527, delta=5e-5
        )

        # ∂L/∂W2 = ∂L/∂z2 · a1 = [-0.1852, -0.2026]
        np.testing.assert_allclose(
            self.gradients.output_weights, [-0.1852, -0.2026], atol=5e-5
        )

        # ∂L/∂a1 = ∂L/∂z2 · W2 = [-0.1764, -0.2116]
        np.testing.assert_allclose(
            self.gradients.hidden_activation, [-0.1764, -0.2116], atol=5e-5
        )

        # ∂L/∂z1 = ∂L/∂a1 ⊙ σ'(z1) = [-0.0440, -0.0517]
        np.testing.assert_allclose(
            self.gradients.hidden_pre_activation, [-0.0440, -0.0517], atol=5e-5
        )

        # ∂L/∂W1 = outer(∂L/∂z1, x) = [[-0.0440, 0], [-0.0517, 0]]
        np.testing.assert_allclose(
            self.gradients.hidden_weights,
            [[-0.0440, 0.0], [-0.0517, 0.0]],
            atol=5e-5,
        )

    def test_shapes(self) -> None:
        """각 값의 shape이 과제 표기와 같다."""
        # z1, a1은 (2,)
        self.assertEqual(self.forward_result.hidden_activation.shape, (2,))

        # ∂L/∂W2는 W2와 같은 (2,)
        self.assertEqual(self.gradients.output_weights.shape, (2,))

        # ∂L/∂W1은 W1과 같은 (2, 2)
        self.assertEqual(self.gradients.hidden_weights.shape, (2, 2))

    def test_chain_rule_matches_numerical_gradient(self) -> None:
        """연쇄 법칙으로 구한 ∂L/∂W1이 수치 미분 결과와 같다."""
        # 중심차분으로 ∂L/∂W1을 구한다
        numerical_gradient = numerical_hidden_weight_gradient(
            self.example.input_vector, self.example.parameters, self.example.target
        )

        # 역전파 결과와 같은지 확인한다
        np.testing.assert_allclose(
            self.gradients.hidden_weights, numerical_gradient, atol=1e-8
        )


class TestSigmoid(unittest.TestCase):
    """Sigmoid 함수를 확인한다."""

    def test_sigmoid_of_zero(self) -> None:
        """sigmoid(0)은 0.5이다."""
        # 0을 넣으면 0.5가 나오는지 확인한다
        self.assertAlmostEqual(float(sigmoid(np.array([0.0]))[0]), 0.5)


# 파일을 직접 실행했을 때만 아래 코드를 실행한다
if __name__ == "__main__":
    # 이 파일을 직접 실행하면 테스트를 돌린다
    unittest.main()
