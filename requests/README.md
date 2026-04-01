# TurboQuant Request Guide

## 목차
1. [요청 문서 구조](#요청-문서-구조)
2. [작성 원칙](#작성-원칙)
3. [템플릿](#템플릿)
4. [예시](#예시)

---

## 요청 문서 구조

모든 요청 문서는 다음 구조를 따라야 합니다:

```markdown
# [문제/작업 제목]

## 문제 설명

### 현재 상황
- [현재 시스템 상태 설명]

### 문제 증상
1. [증상 1]
2. [증상 2]

### 근본 원인
[문제의 근본 원인 분석]

---

## 수정해야 할 파일

### 1. [파일 경로]

#### 수정 위치: [라인 번호]

**현재 코드:**
```python
# 현재 코드
```

**수정 후 코드:**
```python
# 수정 후 코드
```

**설명:**
[변경 이유 및 동작 설명]

---

## 검증 방법

### 1. [검증 단계 1]
```bash
# 검증 명령어
```

### 2. [검증 단계 2]
```python
# 검증 코드
```

---

## 요약

| 파일 | 수정 위치 | 변경 내용 |
|------|-----------|-----------|
| [파일] | [라인] | [내용] |

**예상 결과:**
- [결과 1]
- [결과 2]
```

---

## 작성 원칙

### 1. 상세성 (Detail)
- **라인 번호 명시**: 모든 수정 위치는 정확한 라인 번호 포함
- **코드 스니펫**: 전체 파일 내용이 아닌 수정할 부분만 표시
- **변경 전/후**: `현재 코드` vs `수정 후 코드` 비교

### 2. 맥락 (Context)
- **현재 상황**: 시스템의 현재 상태 설명
- **문제 증상**: 관찰 가능한 현상 나열
- **근본 원인**: 문제의 원인과 영향 분석

### 3. 검증 가능성 (Verifiability)
- **명확한 검증 방법**: 단계별로 검증 절차 설명
- **예상 결과**: 성공/실패 기준 명시
- **명령어/코드**: 직접 실행 가능한 명령어 제공

### 4. 일관성 (Consistency)
- **표준 형식**: 모든 문서가 동일한 구조 사용
- **일관된 표기법**: 파일 경로, 라인 번호, 코드 형식 통일
- **요약 테이블**: 변경 사항 한눈에 보기

---

## 템플릿

### 기본 템플릿

```markdown
# [제목]

## 문제 설명

### 현재 상황
[상황 설명]

### 문제 증상
1. [증상]
2. [증상]

### 근본 원인
[원인 분석]

---

## 수정해야 할 파일

### 1. [파일 경로]

#### 수정 위치: [라인]

**현재 코드:**
```python
```

**수정 후 코드:**
```python
```

**설명:**
[설명]

---

## 검증 방법

### 1. [단계]
```bash
```

---

## 요약

| 파일 | 위치 | 내용 |
|------|------|------|
| | | |

**예상 결과:**
- [결과]
```

---

## 예시

### 예시 1: CUDA 커널 빌드 경로 수정

#### 문제 설명

**현재 상황:**
- `setup.py` 에서 `.so` 파일을 `cuda_kernels/` 로 복사
- `turboquant_cuda_kernel.py` 에서 `turboquant/cuda_kernels/` 검색

**문제 증상:**
1. CUDA 커널 빌드 성공에도 불구하고 `_CUDA_AVAILABLE = False`
2. TSV 파일에서 모든 값이 NA

**근본 원인:**
- import 경로와 복사 경로 불일치

#### 수정해야 할 파일

**파일:** `methods/turboquant/turboquant_cuda_kernel.py`

**수정 위치:** 라인 30-33

**현재 코드:**
```python
_kernel_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cuda_kernels"
)
```

**수정 후 코드:**
```python
_kernel_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda_kernels"
)
```

**설명:**
- `turboquant_cuda_kernel.py` 는 `methods/turboquant/` 에 위치
- `dirname(dirname(__file__))` 로 `methods/` 까지 올라감
- `cuda_kernels/` 추가 시 `methods/turboquant/cuda_kernels/` 경로 생성

---

## 검증 방법

### 1. CUDA 커널 빌드 검증
```bash
cd methods/turboquant/csrc
python setup.py build_ext --inplace
ls -la ../cuda_kernels/*.so
```

### 2. Import 테스트
```python
from methods.turboquant.turboquant_cuda_kernel import is_cuda_available
print(f"CUDA available: {is_cuda_available()}")
```

---

## 요약

| 파일 | 수정 위치 | 변경 내용 |
|------|-----------|-----------|
| `turboquant_cuda_kernel.py` | 라인 30-33 | `_kernel_dir` 경로 수정 |

**예상 결과:**
- CUDA 커널이 올바르게 로드됨
- TSV 에 실제 값이 표시됨
```

---

## 추가 가이드라인

### 1. 코드 수정 요청 시
- **최소한의 변경**: 불필요한 리팩토링 제외
- **기존 패턴 유지**: 프로젝트 컨벤션 준수
- **타입 안전**: `as any`, `@ts-ignore` 금지

### 2. 테스트 추가 요청 시
- **테스트 ID**: T1, T2, ... 형식
- **명확한 기대값**: PASS/FAIL/WARN 기준
- **검증 방법**: 실행 명령어 포함

### 3. 성능 개선 요청 시
- **현재 상태**: 측정 데이터 포함
- **목표**: 구체적인 수치 목표
- **검증**: 벤치마크 명령어 제공

---

## 주의사항

1. **문서 읽기**: 다른 개발자가 이해할 수 있도록 상세히 작성
2. **코드 검증**: 수정된 코드가 실제로 작동하는지 확인
3. **변경 로그**: 모든 변경 사항을 문서화
4. **백업**: 중요한 수정 전 파일 백업

---

## 파일 위치

- `requests/turboquant_001.txt` - 첫 번째 요청
- `requests/turboquant_002.txt` - 두 번째 요청
- `requests/README.md` - 이 문서
