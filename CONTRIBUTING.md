# =============================================================================
# CONTRIBUTING.md
# =============================================================================

# Contributing to PandaKinetics

We welcome contributions to PandaKinetics! This document provides guidelines for contributing to the project.

## Getting Started

### Development Environment Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pandakinetics.git
   cd pandakinetics
   ```

2. **Set up development environment:**
   ```bash
   python scripts/setup_dev.py
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

3. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Standards

### Code Style
- We use **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run formatting before committing:
```bash
black pandakinetics tests
isort pandakinetics tests
```

### Code Quality
- Write comprehensive docstrings for all public functions
- Add type hints to function signatures
- Include unit tests for new functionality
- Maintain test coverage above 80%

### Documentation
- Update docstrings for any API changes
- Add examples for new features
- Update README.md if necessary

## Testing

### Running Tests
```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# GPU tests (requires CUDA)
pytest tests/ -m "gpu"

# With coverage
pytest --cov=pandakinetics
```

### Writing Tests
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies

Example test structure:
```python
import pytest
from pandakinetics import KineticSimulator

class TestKineticSimulator:
    def test_initialization_default_params(self):
        simulator = KineticSimulator()
        assert simulator.temperature == 310.0
    
    def test_initialization_custom_params(self):
        simulator = KineticSimulator(temperature=300.0)
        assert simulator.temperature == 300.0
```

## Pull Request Process

1. **Ensure all tests pass:**
   ```bash
   pytest tests/
   ```

2. **Check code quality:**
   ```bash
   black --check pandakinetics tests
   flake8 pandakinetics tests
   mypy pandakinetics
   ```

3. **Update documentation** if needed

4. **Write descriptive PR title and description**

5. **Link to relevant issues**

6. **Request review** from maintainers

### PR Requirements
- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog updated (for significant changes)
- [ ] No decrease in test coverage

## Issue Reporting

### Bug Reports
Use the bug report template and include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
- Error messages and stack traces

### Feature Requests
Use the feature request template and include:
- Clear description of the desired feature
- Use case and motivation
- Proposed API (if applicable)
- Willingness to contribute

## Development Guidelines

### Performance Considerations
- Profile performance-critical code
- Use GPU acceleration where possible
- Minimize memory allocations in hot paths
- Consider batch processing for efficiency

### GPU Development
- Always check CUDA availability
- Handle out-of-memory errors gracefully
- Use CuPy for NumPy-like operations
- Profile GPU memory usage

### Scientific Accuracy
- Validate algorithms against literature
- Include references in docstrings
- Test against known benchmarks
- Handle edge cases properly

### API Design
- Follow Python conventions (PEP 8)
- Use descriptive names
- Provide sensible defaults
- Make interfaces consistent
- Consider backward compatibility

## Release Process

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Create release tag:**
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```
4. **GitHub Actions** will automatically build and publish to PyPI

## Communication

- **Discussions:** Use GitHub Discussions for questions and ideas
- **Issues:** Use GitHub Issues for bugs and feature requests
- **Email:** contact@pandakinetics.org for sensitive matters

## Recognition

Contributors will be acknowledged in:
- AUTHORS.md file
- Release notes
- Documentation credits

Thank you for contributing to PandaKinetics! 🐼
