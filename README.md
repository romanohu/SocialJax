# SocialJax for uv
This project is a fork of [SocialJax](https://github.com/cooperativex/SocialJax)
While the original repository used **poetry** for Python dependency and package management, this implementation will use **uv** instead.

## Setting
Creating a Python 3.10 Environment
```
uv python install 3.10
uv venv --python 3.10
```
Dependency resolution and installation
```
uv lock
uv sync --no-install-project --group dev
```