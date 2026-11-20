"""미적분 모듈(src/calculus.py)의 단위 테스트."""

import unittest

import numpy as np

from src.calculus import (
    DERIVATIVE_TOLERANCE,
    bowl_function,
    bowl_gradient,
    central_difference,
    elliptic_function,
    elliptic_gradient,
    gradient_tangent_cosines,
    numerical_gradient,
    numerical_hessian,
    square_function,
)


class TestNumericalDerivative(unittest.TestCase):
    """중심차분 수치 미분을 해석해와 비교한다."""

    def test_square_derivative_at_three(self) -> None:
        """f(x)=x²의 x=3 미분은 해석해 6과 1e-4 이내로 같다."""
        # 중심차분으로 미분값을 구한다
        derivative = central_difference(square_function, 3.0)

        # 해석해 6과의 차이가 1e-4보다 작은지 확인한다
        self.assertLess(abs(derivative - 6.0), DERIVATIVE_TOLERANCE)

    def test_gradient_matches_analytic(self) -> None:
        """수치 Gradient는 두 함수 모두 해석 Gradient와 같다."""
        # 확인할 점 (1, -2)
        point = np.array([1.0, -2.0])

        # x²+y²의 수치 Gradient가 (2x, 2y)와 같은지 확인한다
        np.testing.assert_allclose(
            numerical_gradient(bowl_function, point), bowl_gradient(point), atol=1e-6
        )

        # x²+10y²의 수치 Gradient가 (2x, 20y)와 같은지 확인한다
        np.testing.assert_allclose(
            numerical_gradient(elliptic_function, point),
            elliptic_gradient(point),
            atol=1e-6,
        )

    def test_hessian_of_elliptic_function(self) -> None:
        """x²+10y²의 Hessian은 [[2, 0], [0, 20]]이다."""
        # 점 (3, 1)에서 Hessian을 구한다
        hessian = numerical_hessian(elliptic_function, np.array([3.0, 1.0]))

        # 해석해와 같은지 확인한다
        np.testing.assert_allclose(hessian, [[2.0, 0.0], [0.0, 20.0]], atol=1e-3)


class TestGradientPerpendicular(unittest.TestCase):
    """Gradient가 등고선에 수직인지 수치로 확인한다."""

    def test_gradient_is_perpendicular_to_contour(self) -> None:
        """등고선 접선과 Gradient 사이 코사인이 0에 가깝다 (90도)."""
        # 반지름 2인 등고선 위 점 16개에서 코사인을 구한다
        cosines = gradient_tangent_cosines(bowl_function, radius=2.0, point_count=16)

        # 모든 코사인의 절댓값이 1e-6보다 작은지 확인한다
        self.assertLess(float(np.max(np.abs(cosines))), 1e-6)


# 파일을 직접 실행했을 때만 아래 코드를 실행한다
if __name__ == "__main__":
    # 이 파일을 직접 실행하면 테스트를 돌린다
    unittest.main()
