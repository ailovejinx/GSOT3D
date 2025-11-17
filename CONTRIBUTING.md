# Contributing to GSOT3D

Thank you for your interest in contributing to GSOT3D! This document provides guidelines for contributing to the project.

## Git Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for our commit messages. This leads to more readable messages that are easy to follow when looking through the project history.

### Commit Message Format

Each commit message consists of a **header**, a **body** (optional), and a **footer** (optional). The header has a special format that includes a **type**, a **scope** (optional), and a **subject**:

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

The **header** is mandatory and the **scope** of the header is optional.

Any line of the commit message cannot be longer than 100 characters! This allows the message to be easier to read on GitHub as well as in various git tools.

### Type

Must be one of the following:

* **feat**: A new feature
* **fix**: A bug fix
* **docs**: Documentation only changes
* **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
* **refactor**: A code change that neither fixes a bug nor adds a feature
* **perf**: A code change that improves performance
* **test**: Adding missing tests or correcting existing tests
* **build**: Changes that affect the build system or external dependencies
* **ci**: Changes to our CI configuration files and scripts
* **chore**: Other changes that don't modify src or test files
* **revert**: Reverts a previous commit

### Scope

The scope should be the name of the module affected (as perceived by the person reading the changelog generated from commit messages).

The following is the list of supported scopes:

* **models**: Changes related to model architecture
* **datasets**: Changes related to data loading and processing
* **configs**: Changes related to configuration files
* **evaluation**: Changes related to evaluation metrics and toolkit
* **training**: Changes related to training pipeline
* **utils**: Changes related to utility functions
* **deps**: Changes related to dependencies

### Subject

The subject contains a succinct description of the change:

* use the imperative, present tense: "change" not "changed" nor "changes"
* don't capitalize the first letter
* no dot (.) at the end

### Body

Just as in the **subject**, use the imperative, present tense: "change" not "changed" nor "changes".
The body should include the motivation for the change and contrast this with previous behavior.

### Footer

The footer should contain any information about **Breaking Changes** and is also the place to reference GitHub issues that this commit **Closes**.

**Breaking Changes** should start with the word `BREAKING CHANGE:` with a space or two newlines. The rest of the commit message is then used for this.

### Examples

#### Example 1: New Feature
```
feat(models): add transformer-based attention mechanism

Implement multi-head attention for improved feature extraction
in the PROT3D tracker. This enhances the spatial-temporal 
matching capabilities.

Closes #123
```

#### Example 2: Bug Fix
```
fix(datasets): correct point cloud normalization

Fix incorrect normalization that caused tracking failures 
on sequences with sparse point clouds.
```

#### Example 3: Documentation
```
docs(readme): update installation instructions

Add detailed steps for installing dependencies on Ubuntu 22.04
and clarify CUDA version requirements.
```

#### Example 4: Refactoring
```
refactor(utils): simplify bounding box transformation

Reduce code complexity by removing redundant coordinate
transformation steps.
```

#### Example 5: Performance Improvement
```
perf(training): optimize data loading pipeline

Implement multi-threaded data loading to reduce training
time by 30%.
```

## Code Style

* Follow PEP 8 guidelines for Python code
* Use meaningful variable and function names
* Add docstrings to classes and functions
* Keep functions focused and modular

## Pull Request Process

1. Ensure your code follows the project's coding standards
2. Update the README.md with details of changes if applicable
3. Add tests for new features
4. Ensure all tests pass before submitting
5. Write a clear and descriptive PR title and description
6. Reference any related issues in your PR description

## Questions or Issues?

If you have any questions about contributing, please open an issue or contact us via email.

Thank you for contributing to GSOT3D!
