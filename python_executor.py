import subprocess
import tempfile
import os


class PythonExecutor:

    def execute(self, code, timeout=5):

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False
        ) as f:

            f.write(code)
            filename = f.name

        try:

            result = subprocess.run(
                ["python3", filename],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:

            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timed out.",
                "return_code": -1
            }

        finally:

            os.remove(filename)