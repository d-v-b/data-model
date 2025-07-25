# CI/CD Documentation

This directory contains the GitHub Actions workflows and configuration for the eopf-geozarr project.

## Workflows

### 1. CI Workflow (`ci.yml`)
**Triggers:** Push to main, Pull requests to main

**Jobs:**
- **pre-commit**: Runs pre-commit hooks for code quality
- **test**: Runs tests on Python 3.11 and 3.12 on Ubuntu
- **test-network**: Runs network-dependent tests (only on main branch pushes)
- **docs**: Builds documentation
- **security**: Runs security scans with bandit and safety

**Features:**
- Code coverage reporting with Codecov
- Artifact uploads for documentation and security reports
- Separate network tests to avoid CI failures from external dependencies

### 2. Code Quality Workflow (`code-quality.yml`)
**Triggers:** Pull requests to main

**Jobs:**
- **lint**: Runs black, isort, flake8, and mypy
- **test-coverage**: Runs tests with coverage requirements (80% minimum)
- **security-scan**: Runs bandit and safety security checks

**Purpose:** Ensures code quality standards before merging PRs

### 3. Release Workflow (`release.yml`)
**Triggers:** GitHub releases

**Jobs:**
- **build**: Builds the Python package and validates it
- **publish**: Publishes to PyPI using trusted publishing

**Features:**
- Automated PyPI publishing on releases
- Package validation before publishing
- Uses GitHub's trusted publishing for secure PyPI uploads

### 4. Documentation Workflow (`docs.yml`)
**Triggers:** Push to main, Pull requests to main

**Jobs:**
- **build-docs**: Builds Sphinx documentation
- **deploy**: Deploys to GitHub Pages (main branch only)

**Features:**
- Automatic documentation deployment
- Documentation artifacts for PR previews

## Configuration Files

### Pre-commit Configuration (`.pre-commit-config.yaml`)
Defines hooks for:
- Code formatting (black, isort)
- Linting (flake8, mypy)
- Security scanning (bandit)
- General file checks (trailing whitespace, YAML validation, etc.)

### Dependabot Configuration (`dependabot.yml`)
Automated dependency updates for:
- Python packages (weekly)
- GitHub Actions (weekly)
- Assigned to `developmentseed/data-model-developers` team

## Setup Instructions

### For Developers

1. **Install pre-commit hooks:**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

2. **Run code quality checks locally:**
   ```bash
   black --check eopf_geozarr/
   isort --check-only eopf_geozarr/
   flake8 eopf_geozarr/
   mypy eopf_geozarr/
   ```

3. **Run tests:**
   ```bash
   pytest eopf_geozarr/tests/ -v --tb=short -m "not network"
   ```

### For Repository Maintainers

1. **Enable GitHub Pages:**
   - Go to repository Settings > Pages
   - Set source to "GitHub Actions"

2. **Configure PyPI Publishing:**
   - Set up trusted publishing on PyPI
   - No secrets needed with trusted publishing

3. **Configure Codecov:**
   - Add repository to Codecov
   - No additional configuration needed

## Code Quality Standards

- **Code Coverage:** Minimum 80% required
- **Code Style:** Black formatting with 100 character line length
- **Import Sorting:** isort with black profile
- **Type Checking:** mypy with strict settings
- **Security:** bandit security scanning
- **Documentation:** All public APIs must be documented

## Testing Strategy

- **Unit Tests:** Fast tests without external dependencies
- **Integration Tests:** Tests with real data processing
- **Network Tests:** Tests requiring internet access (marked with `@pytest.mark.network`)
- **End-to-End Tests:** CLI testing with real workflows

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a GitHub release with tag
4. Package is automatically built and published to PyPI

## Troubleshooting

### Common Issues

1. **Pre-commit failures:** Run `pre-commit run --all-files` to fix issues
2. **Test failures:** Check if network tests are failing due to external dependencies
3. **Coverage failures:** Add tests or exclude lines with `# pragma: no cover`
4. **Security scan failures:** Review bandit reports and add exclusions if needed

### Getting Help

- Check workflow logs in the Actions tab
- Review the specific job that failed
- Common fixes are usually code formatting or missing tests
