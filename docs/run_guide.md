# 실행 가이드

[README 1.4 실행 방법](../README.md#14-실행-방법)에서 다루지 않은 상세 내용이다. 시연 스크립트의 단계별 동작, 한 단계씩 직접 실행하는 방법, 코딩 컨벤션 검사 명령어를 정리했다.

<br>
<br>

# 1. 시연 스크립트 `demo.sh`의 단계

| 단계 | 하는 일 |
|---|---|
| 0 | 실행 환경과 라이브러리 버전 출력 |
| 1 | 금지 라이브러리(`torch`, `tensorflow`, `jax`, `sklearn`) `import` 검사, `np.linalg.eig` 사용 위치 출력 |
| 2 | 단위 테스트 37개 실행 |
| 3 | `src/linear_algebra.py` 실행: 행렬 변환 4종, Power Iteration, SVD 압축 |
| 4 | `src/calculus.py` 실행: 수치 미분, Gradient 수직 검증, Gradient 그림 |
| 5 | `src/optimizer.py` 실행: Vanilla GD, 발산, Momentum 비교, 학습률 비교표, 보너스 |
| 6 | 노트북 2개 재실행 후 텍스트 출력 표시 (`--no-nb`면 생략) |
| 7 | `outputs/`의 PNG 목록 출력 |

- 3~5단계는 `MPLBACKEND=Agg`로 그림 창 없이 PNG만 저장한다. 6단계는 노트북 안에 그림이 남도록 이 설정을 빼고 실행한다.

<br>
<br>

# 2. 단계별 직접 실행

```bash
.venv/bin/python -m unittest discover -s tests -v    # 단위 테스트
.venv/bin/python -m src.linear_algebra               # 선형대수 결과 출력, outputs/transform_*.png, svd_compression.png
.venv/bin/python -m src.calculus                     # 미적분 결과 출력, outputs/gradient_*.png
.venv/bin/python -m src.optimizer                    # 최적화 결과 출력, outputs/gd_*.png, momentum_vs_gd_*.png, bonus_optimizers.png
(cd notebooks && ../.venv/bin/python -m nbconvert --to notebook --execute --inplace backprop_derivation.ipynb)
(cd notebooks && ../.venv/bin/python -m nbconvert --to notebook --execute --inplace probability_loss.ipynb)
```

64x64 샘플 이미지 다시 만들기

```bash
# Matplotlib 예제 사진 grace_hopper.jpg를 흑백 64x64로 줄여 data/sample_image_64.png에 저장한다
.venv/bin/python -m data.prepare_image
```

<br>
<br>

# 3. 코딩 컨벤션 검사

코드를 고친 뒤 실행하는 검사 도구 설치와 실행 명령어. 대상은 `src`, `tests`, `data/prepare_image.py`이고, `black`과 `ruff`는 노트북에도 적용한다.

```bash
# 검사 도구 설치 (처음 한 번)
.venv/bin/pip install -r requirements-dev.txt

# 코드 모양 검사 (--check를 빼면 자동 수정)
.venv/bin/black --check src tests data/prepare_image.py notebooks

# import 순서 검사
.venv/bin/isort --check src tests data/prepare_image.py

# PEP8, 미사용 코드, docstring 검사
.venv/bin/ruff check src tests data notebooks

# 타입 힌트 검사
.venv/bin/mypy src tests data/prepare_image.py
```

<br>
<br>

# 4. 환경 참고

- 개발 환경은 `Python 3.11`, macOS이다.
- 과제 조건인 `Python 3.8`(NumPy 1.24, Matplotlib 3.7)에서도 단위 테스트와 모듈 3개 실행을 확인했다.
    - `from __future__ import annotations`로 `int | None` 같은 타입 표기를 3.8에서도 쓸 수 있게 했다.
- 터미널, VS Code, 노트북이 서로 다른 `Python`을 쓰면 `No module named ...` 오류가 난다. 하나로 맞춘다.
