# Contributing

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, particularly new selection strategies.


## Get Started!
Ready to contribute? Here's how to set up `kaczmarz-algorithms` for local development.

1. Fork the `kaczmarz-algorithms` repo on GitHub.

2. Clone your fork locally:

    ```bash
    git clone git@github.com:{your_name_here}/kaczmarz-algorithms.git
    ```

3. Install the project in editable mode. (It is also recommended to work in a virtualenv or anaconda environment):

    ```bash
    cd kaczmarz-algorithms/
    pip install -e .[dev]
    ```

4. Create a branch for local development:

    ```bash
    git checkout -b {your_development_type}/short-description
    ```

    Ex: feature/read-tiff-files or bugfix/handle-file-not-found<br>
    Now you can make your changes locally.

5. When you're done making changes, check that your changes pass linting and
   tests, and that the docs still build:

    ```bash
    $ tox -e lint
    $ tox -e py38
    $ tox -e docs
    ```

6. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Resolves gh-###. Your detailed description of your changes."
    git push origin {your_development_type}/short-description
    ```

7. Submit a pull request through the GitHub website.

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

```bash
$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

Next, on GitHub, create a release from the version tag you have just created.

This will release a new package version on Git + GitHub and publish to PyPI.
