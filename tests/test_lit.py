import subprocess
import pytest
import os

def test_lit():
    result = subprocess.run(['lit', os.path.dirname(os.path.abspath(__file__)) + '/mlir'], capture_output=True, text=True)
    assert result.returncode == 0, f"lit tests failed:\n{result.stdout}\n{result.stderr}"

if __name__ == "__main__":
    pytest.main()
