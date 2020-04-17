# Kaczmarz Algorithms

[![PyPI Version](https://img.shields.io/pypi/v/kaczmarz-algorithms.svg)](https://pypi.org/project/kaczmarz-algorithms/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/kaczmarz-algorithms.svg)](https://pypi.org/project/kaczmarz-algorithms/)
[![Build Status](https://github.com/jdmoorman/kaczmarz-algorithms/workflows/CI/badge.svg)](https://github.com/jdmoorman/kaczmarz-algorithms/actions)
[![Code Coverage](https://codecov.io/gh/jdmoorman/kaczmarz-algorithms/branch/master/graph/badge.svg)](https://codecov.io/gh/jdmoorman/kaczmarz-algorithms)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![DOI](https://zenodo.org/badge/255942132.svg)](https://zenodo.org/badge/latestdoi/255942132)

Variants of the Kaczmarz algorithm for solving linear systems in Python.

---

## Installation
**Stable Release:** `pip install kaczmarz-algorithms`<br>
**Development Head:** `pip install git+https://github.com/jdmoorman/kaczmarz-algorithms.git`

## Quick Start

## Citing
If you use our work in an academic setting, please cite our paper:


## Documentation
TODO: readthedocs
For more information, read the docs.


## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.


#### Additional Optional Setup Steps:
* Make sure the github repository initialized correctly at
    * `https://github.com:jdmoorman/kaczmarz-algorithms.git`
* Add branch protections to `master`
    * To protect from just anyone pushing to `master`
    * Go to your [GitHub repository's settings and under the `Branches` tab](https://github.com/jdmoorman/kaczmarz-algorithms/settings/branches), click `Add rule` and select the
    settings you believe best.
    * _Recommendations:_
      * _Require pull request reviews before merging_
      * _Require status checks to pass before merging_

#### Suggested Git Branch Strategy
1. `master` is for the most up-to-date development, very rarely should you directly commit to this branch. It is recommended to commit to development
branches and make pull requests to master.
3. Your day-to-day work should exist on branches separate from `master`. Even if it is just yourself working on the
repository, make a PR from your working branch to `master` so that you can ensure your commits don't break the
development head. GitHub Actions will run on every push to any branch or any pull request from any branch to any other
branch.
4. It is recommended to use "Squash and Merge" commits when committing PR's. It makes each set of changes to `master`
atomic and as a side effect naturally encourages small well defined PR's.
