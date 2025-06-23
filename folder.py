import os

folder_structure = {
    "pandakinetics": {
        ".github": {
            "workflows": [
                "ci.yml", "publish.yml", "docs.yml", "benchmarks.yml"
            ],
            "ISSUE_TEMPLATE": [
                "bug_report.md", "feature_request.md", "performance_issue.md"
            ],
            "files": ["PULL_REQUEST_TEMPLATE.md", "SECURITY.md"]
        },
        "pandakinetics": {
            "core": ["__init__.py", "kinetics.py", "networks.py", "docking.py"],
            "ai": ["__init__.py", "barrier_predictor.py", "sampling.py", "models.py"],
            "simulation": ["__init__.py", "monte_carlo.py", "md_interface.py", "analysis.py"],
            "utils": ["__init__.py", "gpu_utils.py", "io_handlers.py", "validation.py"],
            "visualization": ["__init__.py", "network_plots.py", "kinetic_plots.py"],
            "data": {
                "force_fields": [],
                "models": [],
                "examples": []
            },
            "files": ["__init__.py", "cli.py"]
        },
        "tests": {
            "unit": ["test_core.py", "test_ai.py", "test_simulation.py", "test_utils.py"],
            "integration": ["test_full_pipeline.py", "test_gpu_performance.py"],
            "data": ["test_protein.pdb", "test_ligands.smi"],
            "files": ["__init__.py", "conftest.py"]
        },
        "docs": {
            "source": ["conf.py", "index.rst"],
            "source_dirs": ["api", "tutorials", "examples", "theory"],
            "notebooks": ["quick_start.ipynb", "advanced_usage.ipynb", "benchmarks.ipynb"],
            "images": []
        },
        "scripts": ["setup_dev.py", "download_models.py", "benchmark.py", "validate_install.py"],
        "docker": ["Dockerfile", "Dockerfile.gpu", "docker-compose.yml", "requirements-docker.txt"],
        "conda": ["meta.yaml", "build.sh"],
        "benchmarks": {
            "data": [],
            "results": [],
            "files": ["run_benchmarks.py"]
        },
        "examples": ["basic_prediction.py", "selectivity_analysis.py", "custom_models.py", "batch_processing.py"],
        "files": [
            ".gitignore", ".gitattributes", ".pre-commit-config.yaml", ".codecov.yml",
            ".readthedocs.yml", "pyproject.toml", "setup.py", "requirements.txt",
            "requirements-dev.txt", "requirements-docs.txt", "requirements-test.txt",
            "MANIFEST.in", "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE",
            "AUTHORS.md", "CITATION.cff", "CODE_OF_CONDUCT.md"
        ]
    }
}

def create_structure(base, structure):
    for key, value in structure.items():
        base_path = os.path.join(base, key)
        if isinstance(value, dict):
            os.makedirs(base_path, exist_ok=True)
            for subkey in value:
                if subkey == "files":
                    for file in value[subkey]:
                        open(os.path.join(base_path, file), 'a').close()
                elif subkey.endswith("_dirs"):
                    for dir_name in value[subkey]:
                        os.makedirs(os.path.join(base_path, dir_name), exist_ok=True)
                else:
                    create_structure(base_path, {subkey: value[subkey]})
        elif isinstance(value, list):
            os.makedirs(base_path, exist_ok=True)
            for file in value:
                open(os.path.join(base_path, file), 'a').close()

create_structure(".", folder_structure)
