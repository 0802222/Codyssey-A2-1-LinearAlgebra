"""선형대수 모듈(src/linear_algebra.py)의 단위 테스트."""

import unittest

import numpy as np

from src.linear_algebra import (
    AREA_ERROR_LIMIT_PERCENT,
    EIGENVALUE_ERROR_LIMIT_PERCENT,
    EXAMPLE_SYMMETRIC_MATRIX,
    check_area_ratio,
    compare_with_numpy_eig,
    load_grayscale_image,
    make_unit_circle,
    polygon_area,
    power_iteration,
    rotation_matrix,
    scaling_matrix,
    shear_matrix,
    svd_compress,
)


class TestTransformationMatrices(unittest.TestCase):
    """회전, 스케일링, 전단 행렬과 면적비 검증을 확인한다."""

    def test_unit_circle_has_radius_one(self) -> None:
        """단위 원의 모든 점은 원점에서 거리가 1이다."""
        # 점 100개로 단위 원을 만든다
        circle_points = make_unit_circle(100)

        # 모양이 (2, 100)인지 확인한다 (행: x, y / 열: 점)
        self.assertEqual(circle_points.shape, (2, 100))

        # 각 점의 원점 거리가 1인지 확인한다
        np.testing.assert_allclose(np.linalg.norm(circle_points, axis=0), 1.0)

    def test_rotation_keeps_area_and_has_det_one(self) -> None:
        """회전 행렬은 행렬식이 1이고 면적을 바꾸지 않는다."""
        # 45도 회전 행렬을 만든다
        rotation = rotation_matrix(np.pi / 4)

        # 행렬식이 1인지 확인한다
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

        # 단위 원에 적용한 면적비를 검증한다
        area_check = check_area_ratio(rotation, make_unit_circle(100))

        # 면적비 오차가 1% 이내인지 확인한다
        self.assertLess(area_check.error_percent, AREA_ERROR_LIMIT_PERCENT)

    def test_scaling_matches_determinant(self) -> None:
        """S(2, 0.5)는 가로 2배, 세로 0.5배이고 면적비는 det=1과 같다."""
        # 스케일링 행렬 S(2, 0.5)를 만든다
        scaling = scaling_matrix(2.0, 0.5)

        # 대각선이 (2, 0.5)인지 확인한다
        np.testing.assert_allclose(np.diag(scaling), [2.0, 0.5])

        # 단위 원에 적용한 면적비를 검증한다
        area_check = check_area_ratio(scaling, make_unit_circle(100))

        # 행렬식이 1인지 확인한다
        self.assertAlmostEqual(area_check.determinant, 1.0)

        # 면적비 오차가 1% 이내인지 확인한다
        self.assertLess(area_check.error_percent, AREA_ERROR_LIMIT_PERCENT)

    def test_shear_moves_only_x(self) -> None:
        """전단 행렬 Sh(k)는 y에 비례해 x만 민다."""
        # k=1 전단 행렬을 만든다
        shear = shear_matrix(1.0)

        # 점 (0, 1)이 (1, 1)로 가는지 확인한다
        np.testing.assert_allclose(shear @ np.array([0.0, 1.0]), [1.0, 1.0])

        # 단위 원에 적용한 면적비 오차가 1% 이내인지 확인한다
        area_check = check_area_ratio(shear, make_unit_circle(100))

        # 면적비 오차가 1% 이내인지 확인한다
        self.assertLess(area_check.error_percent, AREA_ERROR_LIMIT_PERCENT)

    def test_polygon_area_of_unit_square(self) -> None:
        """신발끈 공식으로 구한 단위 정사각형 넓이는 1이다."""
        # 단위 정사각형의 네 꼭짓점 (2, 4)
        square_points = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])

        # 넓이가 1인지 확인한다
        self.assertAlmostEqual(polygon_area(square_points), 1.0)


class TestPowerIteration(unittest.TestCase):
    """Power Iteration 결과를 np.linalg.eig와 비교한다."""

    def test_largest_eigenvalue_of_example_matrix(self) -> None:
        """예시 행렬의 최대 고유값은 (7 + √5) / 2 ≈ 4.6180이다."""
        # Power Iteration을 실행한다
        result = power_iteration(EXAMPLE_SYMMETRIC_MATRIX)

        # 수렴했는지 확인한다
        self.assertTrue(result.is_converged)

        # 해석해 (7 + √5) / 2와 소수점 4자리까지 같은지 확인한다
        self.assertAlmostEqual(result.eigenvalue, (7 + np.sqrt(5)) / 2, places=4)

    def test_error_against_numpy_eig_within_limit(self) -> None:
        """np.linalg.eig와의 고유값 오차는 5% 이내, 고유벡터 방향은 같다."""
        # Power Iteration을 실행한다
        result = power_iteration(EXAMPLE_SYMMETRIC_MATRIX)

        # np.linalg.eig 결과와 비교한다
        comparison = compare_with_numpy_eig(EXAMPLE_SYMMETRIC_MATRIX, result)

        # 고유값 오차가 5% 이내인지 확인한다
        self.assertLess(
            comparison.eigenvalue_error_percent, EIGENVALUE_ERROR_LIMIT_PERCENT
        )

        # 두 고유벡터가 같은 방향(코사인 1)인지 확인한다
        self.assertAlmostEqual(comparison.eigenvector_cosine, 1.0, places=5)

    def test_eigen_equation_holds(self) -> None:
        """구한 값은 고유값 방정식 A v = λ v를 만족한다."""
        # 3x3 대칭 행렬로 일반적인 경우도 확인한다
        matrix = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])

        # Power Iteration을 실행한다
        result = power_iteration(matrix)

        # A v와 λ v가 같은지 확인한다
        np.testing.assert_allclose(
            matrix @ result.eigenvector,
            result.eigenvalue * result.eigenvector,
            atol=1e-4,
        )


class TestSvdCompression(unittest.TestCase):
    """SVD 기반 이미지 압축과 복원을 확인한다."""

    def setUp(self) -> None:
        """테스트마다 64x64 샘플 이미지를 읽는다."""
        # 저장소에 있는 64x64 Grayscale 이미지를 읽는다
        self.image = load_grayscale_image()

    def test_image_is_64_grayscale(self) -> None:
        """샘플 이미지는 64x64 흑백이고 값은 0~1이다."""
        # 크기가 64x64인지 확인한다
        self.assertEqual(self.image.shape, (64, 64))

        # 밝기 값이 0~1 사이인지 확인한다
        self.assertTrue(0.0 <= self.image.min() and self.image.max() <= 1.0)

    def test_error_decreases_as_rank_grows(self) -> None:
        """k가 커질수록 복원 오차가 줄어든다."""
        # k=10, 50, 100으로 압축한 결과의 오차를 모은다
        relative_errors = [
            svd_compress(self.image, rank).relative_error for rank in (10, 50, 100)
        ]

        # 오차가 k 순서대로 줄어드는지 확인한다
        self.assertGreater(relative_errors[0], relative_errors[1])

        # 50보다 100에서 오차가 더 작거나 같은지 확인한다
        self.assertGreaterEqual(relative_errors[1], relative_errors[2])

    def test_rank_above_64_restores_exactly(self) -> None:
        """64x64 이미지의 rank는 최대 64라서 k=100은 64로 잘리고 완전 복원된다."""
        # k=100으로 압축한다
        compression = svd_compress(self.image, 100)

        # 실제로 쓴 rank가 64인지 확인한다
        self.assertEqual(compression.used_rank, 64)

        # 원본과 거의 같게 복원되는지 확인한다
        np.testing.assert_allclose(
            compression.reconstructed_image, self.image, atol=1e-10
        )


# 파일을 직접 실행했을 때만 아래 코드를 실행한다
if __name__ == "__main__":
    # 이 파일을 직접 실행하면 테스트를 돌린다
    unittest.main()
