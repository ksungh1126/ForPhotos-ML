"""모듈 실행 진입점.

`python -m HSemotion` 실행 시 CLI 엔트리포인트(`cli.main`)를 호출합니다.
"""

from .cli import main

if __name__ == "__main__":
    main()