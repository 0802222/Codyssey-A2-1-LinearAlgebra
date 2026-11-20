#!/usr/bin/env bash
# 시연 스크립트: 실행만 하면 과제 결과를 순서대로 출력하고 그림을 outputs/에 저장한다.
# 사용법: ./demo.sh            (전체)
#         ./demo.sh --no-nb    (노트북 재실행 생략)

# 에러가 나면 바로 멈추고, 없는 변수를 쓰면 에러로 본다
set -euo pipefail

# 이 스크립트가 있는 폴더(프로젝트 최상위)로 이동한다
cd "$(dirname "$0")"

# 그림 창을 띄우지 않고 파일로만 저장한다
export MPLBACKEND=Agg

# 가상환경의 Python이 실행 가능하면
if [ -x ".venv/bin/python" ]; then
    # 가상환경의 Python을 쓴다
    PY=".venv/bin/python"
# 없으면
else
    # 시스템 python3를 쓴다
    PY="python3"
fi

# 단계 제목을 보기 좋게 출력하는 함수
step() {
    # 첫 번째 인자를 제목으로 출력한다
    printf '\n========== %s ==========\n' "$1"
}

# 0단계 제목 출력
step "0. 실행 환경"

# Python과 라이브러리 버전을 출력한다
"$PY" -c "import sys, numpy, matplotlib, seaborn; print('Python', sys.version.split()[0], '/ NumPy', numpy.__version__, '/ Matplotlib', matplotlib.__version__, '/ Seaborn', seaborn.__version__)"

# 1단계 제목 출력
step "1. 금지 라이브러리 사용 여부 검사"

# 자동 미분 라이브러리와 scikit-learn import를 찾는다
if grep -rnE "(^|\")\s*(import|from)\s+(torch|tensorflow|jax|sklearn)" src tests data notebooks --include=*.py --include=*.ipynb; then
    # 찾았으면 알려주고
    echo "금지 라이브러리가 발견됐다"

    # 실패로 끝낸다
    exit 1
# 못 찾았으면
else
    # 통과했다고 알려준다
    echo "PyTorch, TensorFlow, JAX, scikit-learn 사용 없음"
fi

# np.linalg.eig가 검증 함수에서만 쓰였는지 보여 준다
echo "np.linalg.eig 사용 위치 (검증 함수 compare_with_numpy_eig 안에서만 써야 한다):"

# 주석을 뺀 실제 호출 줄만 찾는다
grep -n "np.linalg.eig(" src/*.py | grep -v "#"

# 2단계 제목 출력
step "2. 단위 테스트"

# 테스트를 모두 돌린다
"$PY" -m unittest discover -s tests 2>&1 | tail -n 4

# 3단계 제목 출력
step "3. 선형대수 (src/linear_algebra.py)"

# 행렬 변환, Power Iteration, SVD 압축을 실행한다
"$PY" -m src.linear_algebra

# 4단계 제목 출력
step "4. 미적분 (src/calculus.py)"

# 수치 미분, Gradient 수직 검증, Gradient 시각화를 실행한다
"$PY" -m src.calculus

# 5단계 제목 출력
step "5. 최적화 (src/optimizer.py)"

# Vanilla GD, 발산, Momentum 비교, 보너스를 실행한다
"$PY" -m src.optimizer

# --no-nb 옵션이 없으면 노트북도 실행한다
if [ "${1:-}" != "--no-nb" ]; then
    # 6단계 제목 출력
    step "6. 노트북 재실행 (역전파, 확률/통계)"

    # 노트북 두 개를 차례로 처리한다
    for notebook_name in backprop_derivation probability_loss; do
        # 노트북은 그림을 셀 안에 보여야 하므로 MPLBACKEND를 빼고 처음부터 끝까지 실행해 저장한다
        (cd notebooks && env -u MPLBACKEND "../$PY" -m nbconvert --to notebook --execute --inplace "$notebook_name.ipynb" 2>/dev/null)

        # 노트북 이름을 출력한다
        echo "--- notebooks/$notebook_name.ipynb 출력 ---"

        # 코드 셀의 텍스트 출력만 모아 보여 준다
        "$PY" -c "import nbformat, sys; notebook = nbformat.read(sys.argv[1], 4); print(''.join(output.get('text', '') for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs))" "notebooks/$notebook_name.ipynb"
    done
fi

# 마지막 단계 제목 출력
step "7. 저장된 그림 (outputs/)"

# 그림 파일 목록을 보여 준다
ls outputs
