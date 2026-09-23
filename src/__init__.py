"""Project packages.

Present so that `src` is a package in its own right: every subpackage below
already had an __init__.py, which let mypy resolve the same file under both
`api.auth` and `src.api.auth` and stop before checking anything. Code, tests,
the Makefile and the DAG all import through `src.`, so this makes that the
single resolvable path.
"""
