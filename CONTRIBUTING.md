# Contributing to OFI Market Making Strategy

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)

---

## Code of Conduct

This project adheres to academic integrity and professional conduct standards. By participating, you agree to:

- Provide constructive feedback
- Respect differing viewpoints
- Focus on what is best for the project
- Show empathy towards other contributors

---

## Getting Started

### Prerequisites

- Python 3.13 or higher
- Git
- Basic understanding of market making and order flow
- Familiarity with NumPy, Pandas, and Pytest

### First Contribution

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ofi-marketmaking-strat.git
   cd ofi-marketmaking-strat
   ```
3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks (if configured)
pre-commit install
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=maker --cov=src --cov-report=html

# Run specific test file
pytest tests/test_engine.py -v

# Run specific test
pytest tests/test_engine.py::test_reservation_price -v
```

### Code Quality Checks

```bash
# Format code
black maker/ src/ tests/

# Check types
mypy maker/ src/

# Lint code
pylint maker/ src/

# Check imports
isort maker/ src/ tests/
```

---

## Code Style

### Python Style Guide

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 88)
- Use type hints for all functions
- Write docstrings for all public functions/classes

### Docstring Format

Use Google-style docstrings:

```python
def compute_ofi(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
    """Compute Order Flow Imbalance.
    
    Args:
        bid: Best bid price
        ask: Best ask price
        bid_size: Size at best bid
        ask_size: Size at best ask
    
    Returns:
        OFI value (positive = buying pressure, negative = selling pressure)
    
    Raises:
        ValueError: If prices or sizes are negative
    
    Example:
        >>> compute_ofi(100.0, 100.1, 500, 300)
        200.0
    """
    # Implementation
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `OFICalculator`)
- **Functions**: snake_case (e.g., `compute_reservation_price`)
- **Constants**: UPPER_CASE (e.g., `MAX_SPREAD`)
- **Private methods**: Leading underscore (e.g., `_validate_input`)

### File Organization

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from maker.features import OFICalculator
from src.ofi_utils import load_nbbo_data
```

---

## Testing

### Writing Tests

**Requirements**:
- All new features must have tests
- Aim for >90% code coverage
- Test edge cases and error conditions
- Use descriptive test names

**Test Structure**:
```python
import pytest
import numpy as np
from maker.engine import AvellanedaStoikovEngine

class TestAvellanedaStoikovEngine:
    """Test suite for Avellaneda-Stoikov quoting engine."""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for tests."""
        return AvellanedaStoikovEngine(
            risk_aversion=0.1,
            fill_intensity=1.0
        )
    
    def test_reservation_price_zero_inventory(self, engine):
        """Test reservation price equals mid when inventory is zero."""
        mid_price = 100.0
        inventory = 0
        volatility = 0.01
        
        result = engine.compute_reservation_price(
            mid_price=mid_price,
            inventory=inventory,
            volatility=volatility
        )
        
        assert np.isclose(result, mid_price, rtol=1e-6)
    
    def test_reservation_price_long_inventory(self, engine):
        """Test reservation price decreases when long inventory."""
        mid_price = 100.0
        inventory = 100  # Long position
        volatility = 0.01
        
        result = engine.compute_reservation_price(
            mid_price=mid_price,
            inventory=inventory,
            volatility=volatility
        )
        
        assert result < mid_price, "Reservation price should be below mid when long"
```

### Running Specific Tests

```bash
# Run all tests in a class
pytest tests/test_engine.py::TestAvellanedaStoikovEngine -v

# Run a specific test
pytest tests/test_engine.py::TestAvellanedaStoikovEngine::test_reservation_price_zero_inventory -v

# Run tests matching a pattern
pytest tests/ -k "reservation_price" -v

# Run tests with markers
pytest tests/ -m "slow" -v
```

---

## Pull Request Process

### Before Submitting

1. **Update tests**: Ensure all tests pass
2. **Update documentation**: Add/update docstrings and README if needed
3. **Run code quality checks**: Black, mypy, pylint
4. **Update CHANGELOG**: Add your changes (if applicable)
5. **Rebase on main**: Ensure your branch is up-to-date

### PR Checklist

- [ ] Code follows project style guide
- [ ] All tests pass (`pytest tests/`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No merge conflicts with main branch
- [ ] Commit messages are clear and descriptive

### PR Description Template

```markdown
## Description
Brief description of the change and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Code refactoring

## Testing
Describe how you tested your changes.

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code formatted with Black
- [ ] No linting errors

## Related Issues
Closes #[issue number]
```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add OFI spread widening to engine
fix: Correct microprice calculation for zero sizes
docs: Update architecture documentation
test: Add edge cases for fill simulation
refactor: Simplify volatility estimation code
```

Prefix types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance

---

## Project Structure

```
ofi-marketmaking-strat/
├── maker/              # Core market making modules
│   ├── features.py     # OFI, volatility, microprice
│   ├── engine.py       # Avellaneda-Stoikov quoting
│   └── __init__.py
├── src/                # Utilities and data loading
│   ├── ofi_utils.py    # Data loading, OFI calculation
│   └── __init__.py
├── scripts/            # Execution scripts
│   ├── run_single_backtest.py
│   └── run_batch_backtests.py
├── tests/              # Test suite
│   ├── test_features.py
│   ├── test_engine.py
│   └── conftest.py     # Pytest configuration
├── docs/               # Documentation
│   ├── ARCHITECTURE.md
│   ├── FINAL_REPORT.md
│   └── ...
├── configs/            # Strategy configurations
├── data/               # Market data (gitignored)
├── results/            # Backtest results (gitignored)
├── figures/            # Plots and visualizations
├── README.md           # Main documentation
├── requirements.txt    # Python dependencies
└── .gitignore
```

---

## Areas for Contribution

### High Priority

1. **Multi-level OFI**: Extend OFI to deeper order book levels
2. **Adaptive Parameters**: Dynamic adjustment of γ, κ based on market regime
3. **Transaction Costs**: Explicit fee and rebate modeling
4. **Visualization**: Interactive dashboards for backtest results

### Medium Priority

1. **Performance Optimization**: Vectorize computations
2. **Additional Strategies**: VWAP, TWAP, aggressive taking
3. **Risk Metrics**: VaR, CVaR, tail risk measures
4. **Data Pipeline**: Automated data loading and preprocessing

### Low Priority

1. **Documentation**: Expand tutorials and examples
2. **Code Cleanup**: Refactor legacy code
3. **Testing**: Increase coverage in edge cases
4. **CI/CD**: Set up automated testing pipeline

---

## Questions?

- **General questions**: Open an [Issue](https://github.com/xecuterisaquant/ofi-marketmaking-strat/issues)
- **Discussion**: Use [Discussions](https://github.com/xecuterisaquant/ofi-marketmaking-strat/discussions)
- **Bugs**: File a [Bug Report](https://github.com/xecuterisaquant/ofi-marketmaking-strat/issues/new?template=bug_report.md)
- **Features**: Submit a [Feature Request](https://github.com/xecuterisaquant/ofi-marketmaking-strat/issues/new?template=feature_request.md)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to advancing market making research! 🚀**
