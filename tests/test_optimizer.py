"""최적화 모듈(src/optimizer.py)의 단위 테스트."""

import unittest

import numpy as np

from src.calculus import (
    bowl_gradient,
    elliptic_function,
    elliptic_gradient,
    numerical_hessian,
)
from src.optimizer import (
    CONVERGENCE_RADIUS,
    ITERATION_COUNT,
    START_POINT,
    Adam,
    Momentum,
    NewtonMethod,
    VanillaGD,
    run_optimizer,
    steps_to_converge,
)


class TestVanillaGD(unittest.TestCase):
    """Vanilla GD의 수렴과 발산을 확인한다."""

    def test_converges_with_learning_rate_0_1(self) -> None:
        """lr=0.1, 100회 뒤 원점 반경 0.1 이내에 도달한다."""
        # x²+y²에 Vanilla GD를 100번 적용한다
        path = run_optimizer(
            VanillaGD(0.1), bowl_gradient, START_POINT, ITERATION_COUNT
        )

        # 경로 길이가 시작점 포함 101개인지 확인한다
        self.assertEqual(path.shape, (ITERATION_COUNT + 1, 2))

        # 마지막 점이 원점 반경 0.1 이내인지 확인한다
        self.assertLess(float(np.linalg.norm(path[-1])), CONVERGENCE_RADIUS)

    def test_diverges_when_learning_rate_too_large(self) -> None:
        """lr이 안정 한계를 넘으면 원점에서 멀어진다."""
        # x²+y²는 lr>1에서 발산한다 (1 - 2·lr의 절댓값이 1보다 커진다)
        bowl_path = run_optimizer(VanillaGD(1.1), bowl_gradient, START_POINT, 20)

        # 처음보다 원점에서 더 멀어졌는지 확인한다
        self.assertGreater(np.linalg.norm(bowl_path[-1]), np.linalg.norm(START_POINT))

        # x²+10y²는 lr=0.5에서 발산한다 (1 - 20·lr = -9)
        ellipse_path = run_optimizer(VanillaGD(0.5), elliptic_gradient, START_POINT, 20)

        # 처음보다 원점에서 더 멀어졌는지 확인한다
        self.assertGreater(
            np.linalg.norm(ellipse_path[-1]), np.linalg.norm(START_POINT)
        )


class TestMomentum(unittest.TestCase):
    """Momentum의 이점을 확인한다."""

    def test_momentum_beats_vanilla_on_ellipse_with_small_lr(self) -> None:
        """x²+10y², lr=0.01에서 Momentum이 원점에 더 가깝다."""
        # Vanilla GD 경로를 구한다
        vanilla_path = run_optimizer(
            VanillaGD(0.01), elliptic_gradient, START_POINT, ITERATION_COUNT
        )

        # Momentum(beta=0.9) 경로를 구한다
        momentum_path = run_optimizer(
            Momentum(0.01, beta=0.9), elliptic_gradient, START_POINT, ITERATION_COUNT
        )

        # Momentum의 마지막 점이 원점에 더 가까운지 확인한다
        self.assertLess(
            np.linalg.norm(momentum_path[-1]), np.linalg.norm(vanilla_path[-1])
        )

    def test_reset_clears_velocity(self) -> None:
        """초기화(reset) 뒤에는 같은 입력에 같은 결과를 낸다."""
        # Momentum 옵티마이저를 만든다
        momentum = Momentum(0.1, beta=0.9)

        # 첫 경로를 구한다
        first_path = run_optimizer(momentum, bowl_gradient, START_POINT, 10)

        # 두 번째 경로를 구한다 (run_optimizer가 처음에 reset을 부른다)
        second_path = run_optimizer(momentum, bowl_gradient, START_POINT, 10)

        # 두 경로가 같은지 확인한다
        np.testing.assert_allclose(first_path, second_path)


class TestBonusOptimizers(unittest.TestCase):
    """보너스 옵티마이저(Adam, Newton Method)를 확인한다."""

    def test_adam_reaches_origin(self) -> None:
        """Adam은 x²+10y²에서 원점 반경 0.1 이내에 도달한다."""
        # Adam으로 200번 반복한다
        path = run_optimizer(Adam(0.5), elliptic_gradient, START_POINT, 200)

        # 마지막 점이 원점 반경 0.1 이내인지 확인한다
        self.assertLess(float(np.linalg.norm(path[-1])), 0.1)

    def test_newton_solves_quadratic_in_one_step(self) -> None:
        """2차 함수에서 Newton Method는 한 번에 최소점에 간다."""

        # 수치 Hessian을 돌려주는 함수
        def hessian_function(point: np.ndarray) -> np.ndarray:
            """x²+10y²의 수치 Hessian을 돌려준다."""
            # 해당 점의 Hessian을 계산한다
            return numerical_hessian(elliptic_function, point)

        # Newton Method로 한 번만 이동한다
        path = run_optimizer(
            NewtonMethod(hessian_function), elliptic_gradient, START_POINT, 1
        )

        # 한 번 만에 원점에 도달했는지 확인한다
        np.testing.assert_allclose(path[-1], [0.0, 0.0], atol=1e-4)

    def test_steps_to_converge(self) -> None:
        """반경 안에 들어온 뒤 계속 머무는 첫 반복 번호를 돌려준다."""
        # 원점 거리가 4, 0.5(잠깐 들어옴), 2(다시 나감), 0.5, 0.3인 경로
        path = np.array([[4.0, 0], [0.5, 0], [2.0, 0], [0.5, 0], [0.3, 0]])

        # 반경 1 안에 계속 머물기 시작한 번호는 1이 아니라 3이다
        self.assertEqual(steps_to_converge(path, 1.0), 3)

        # 반경 0.1 안에는 못 들어왔으므로 None이다
        self.assertIsNone(steps_to_converge(path, 0.1))


# 파일을 직접 실행했을 때만 아래 코드를 실행한다
if __name__ == "__main__":
    # 이 파일을 직접 실행하면 테스트를 돌린다
    unittest.main()
