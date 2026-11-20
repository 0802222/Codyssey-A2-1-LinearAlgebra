"""확률/통계 모듈(src/probability.py)의 단위 테스트."""

import unittest

import numpy as np

from src.probability import (
    bernoulli_pmf,
    binary_cross_entropy,
    categorical_cross_entropy,
    cross_entropy,
    entropy,
    gaussian_negative_log_likelihood,
    kl_divergence,
    mean_squared_error,
    normal_pdf,
    softmax,
)


class TestDistributions(unittest.TestCase):
    """정규분포 PDF와 베르누이 PMF를 확인한다."""

    def test_normal_pdf_peak_and_area(self) -> None:
        """N(0,1)의 꼭대기는 1/√(2π)이고 전체 넓이는 1이다."""
        # x=0에서의 확률 밀도를 확인한다
        self.assertAlmostEqual(
            float(normal_pdf(np.array([0.0]), 0.0, 1.0)[0]), 1 / np.sqrt(2 * np.pi)
        )

        # -10~10 구간을 잘게 나눈다
        x_values = np.linspace(-10.0, 10.0, 20001)

        # N(2, 0.5)의 넓이를 직사각형 합(높이 × 간격)으로 근사한다
        area = float(
            np.sum(normal_pdf(x_values, 2.0, 0.5)) * (x_values[1] - x_values[0])
        )

        # 넓이가 1인지 확인한다
        self.assertAlmostEqual(area, 1.0, places=6)

    def test_bernoulli_pmf(self) -> None:
        """B(0.3)은 P(0)=0.7, P(1)=0.3이다."""
        # 0과 1의 확률을 구한다
        probabilities = bernoulli_pmf(np.array([0, 1]), 0.3)

        # 결과가 [0.7, 0.3]인지 확인한다
        np.testing.assert_allclose(probabilities, [0.7, 0.3])


class TestSoftmax(unittest.TestCase):
    """소프트맥스 함수를 확인한다."""

    def test_sum_is_one(self) -> None:
        """출력 합과 1.0의 차이가 1e-6 이내이다."""
        # 소프트맥스를 적용한다
        probabilities = softmax(np.array([2.0, 1.0, 0.1]))

        # 합이 1과 1e-6 이내인지 확인한다
        self.assertLess(abs(float(probabilities.sum()) - 1.0), 1e-6)

    def test_large_logits_do_not_overflow(self) -> None:
        """큰 값을 넣어도 overflow 없이 계산한다 (최댓값을 빼는 이유)."""
        # 매우 큰 로짓에 소프트맥스를 적용한다
        probabilities = softmax(np.array([1000.0, 1000.0]))

        # 결과가 [0.5, 0.5]인지 확인한다
        np.testing.assert_allclose(probabilities, [0.5, 0.5])


class TestLossAndMle(unittest.TestCase):
    """손실 함수와 MLE의 연결을 수치로 확인한다."""

    def test_mse_and_gaussian_nll_share_argmin(self) -> None:
        """MSE를 최소화하는 상수 예측과 정규분포 NLL을 최소화하는 예측이 같다."""
        # 관측값 4개
        targets = np.array([1.0, 2.0, 2.5, 4.5])

        # 후보 예측값을 촘촘히 만든다
        candidates = np.linspace(0.0, 5.0, 501)

        # 후보마다 MSE를 구한다
        mse_values = [
            mean_squared_error(targets, np.full(4, value)) for value in candidates
        ]

        # 후보마다 정규분포 음의 로그우도를 구한다
        nll_values = [
            gaussian_negative_log_likelihood(targets, np.full(4, value), 1.0)
            for value in candidates
        ]

        # 두 최소점이 같은지 확인한다
        self.assertEqual(int(np.argmin(mse_values)), int(np.argmin(nll_values)))

    def test_binary_cross_entropy_example(self) -> None:
        """y=1, ŷ=0.5일 때 BCE는 log 2이다."""
        # BCE를 계산한다
        loss = binary_cross_entropy(np.array([1.0]), np.array([0.5]))

        # log 2와 같은지 확인한다
        self.assertAlmostEqual(loss, np.log(2))

    def test_categorical_cross_entropy_one_hot(self) -> None:
        """원-핫 정답의 CE는 정답 클래스 확률의 -log이다."""
        # 정답이 두 번째 클래스인 원-핫 벡터
        one_hot_target = np.array([0.0, 1.0, 0.0])

        # 모델의 예측 확률
        predicted_probabilities = np.array([0.2, 0.7, 0.1])

        # CE가 -log 0.7과 같은지 확인한다
        self.assertAlmostEqual(
            categorical_cross_entropy(one_hot_target, predicted_probabilities),
            -np.log(0.7),
        )


class TestInformationTheory(unittest.TestCase):
    """보너스: 엔트로피, KL-Divergence, Cross-Entropy를 확인한다."""

    def test_entropy_of_fair_coin(self) -> None:
        """공정한 동전의 엔트로피는 log 2이다."""
        # 엔트로피가 log 2인지 확인한다
        self.assertAlmostEqual(entropy(np.array([0.5, 0.5])), np.log(2))

    def test_cross_entropy_equals_entropy_plus_kl(self) -> None:
        """H(p, q) = H(p) + KL(p || q)가 성립한다."""
        # 실제 분포 p
        true_probabilities = np.array([0.1, 0.6, 0.3])

        # 모델 분포 q
        model_probabilities = np.array([0.3, 0.4, 0.3])

        # 왼쪽 항 H(p, q)
        left_side = cross_entropy(true_probabilities, model_probabilities)

        # 오른쪽 항 H(p) + KL(p || q)
        right_side = entropy(true_probabilities) + kl_divergence(
            true_probabilities, model_probabilities
        )

        # 두 값이 같은지 확인한다
        self.assertAlmostEqual(left_side, right_side)

    def test_kl_is_zero_for_same_distribution(self) -> None:
        """같은 분포 사이의 KL은 0이다."""
        # 같은 분포 p
        probabilities = np.array([0.2, 0.8])

        # KL이 0인지 확인한다
        self.assertAlmostEqual(kl_divergence(probabilities, probabilities), 0.0)


# 파일을 직접 실행했을 때만 아래 코드를 실행한다
if __name__ == "__main__":
    # 이 파일을 직접 실행하면 테스트를 돌린다
    unittest.main()
