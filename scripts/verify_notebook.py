"""Чистый Run All; выполненная копия хранится только в ignored exports/."""

import os
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import sys


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".venv/matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".venv/ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".venv/jupyter-runtime"))


def main():
    notebook = nbformat.read(ROOT / "notebooks/rope.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in notebook.cells if cell.cell_type == "code"), "Исходный notebook содержит outputs"
    assert "widgets" not in notebook.metadata, "В исходнике сохранён widget state"
    # Не регистрируем глобальное ядро: явно запускаем Python текущей .venv.
    kernel_dir = ROOT / ".venv/verification-kernels/ml-visualizations"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "ML Visualizations", "language": "python",
    }))
    manager = KernelManager(kernel_name="ml-visualizations", kernel_spec_manager=KernelSpecManager(
        kernel_dirs=[str(kernel_dir.parent)]))
    client = NotebookClient(notebook, km=manager, timeout=120,
                            resources={"metadata": {"path": str(ROOT / "notebooks")}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    output = ROOT / "exports/rope.executed.ipynb"
    output.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Run All OK: {sum(c.cell_type == 'code' for c in notebook.cells)} code cells; {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
