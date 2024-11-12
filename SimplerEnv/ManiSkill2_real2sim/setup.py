from setuptools import find_packages, setup

long_description = """ManiSkill2-Real2Sim is a suite of simulation manipulation evaluation environments for real robot setups powered by [SAPIEN](https://sapien.ucsd.edu/).

It serves as a stand-alone component of ManiSkill2 (https://haosulab.github.io/ManiSkill2)."""


def read_requirements():
    with open("requirements.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]
    install_requires = list(filter(None, lines))
    return install_requires


setup(
    name="mani_skill2_real2sim",
    version="0.5.3",
    description="ManiSkill2-Real2Sim: Simulated Manipulation Evaluation Environments build on ManiSkill2 and SAPIEN.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Xuanlin Li et al.",
    url="https://github.com/simpler-env/ManiSkill2_real2sim",
    packages=find_packages(
        include=["mani_skill2_real2sim*"],
    ),
    python_requires=">=3.8",
    setup_requires=["setuptools>=62.3.0"],
    install_requires=read_requirements(),
    # Glob patterns do not automatically match dotfiles
    package_data={
        "mani_skill2_real2sim": ["assets/**"],
    },
    exclude_package_data={"": ["*.convex.stl"]},
    extras_require={
        "tests": ["pytest", "black", "isort"],
        "docs": [
            # Note that currently sphinx 7 does not work, so we must use v6.2.1. See https://github.com/kivy/kivy/issues/8230 which tracks this issue. Once fixed we can use a later version
            "sphinx==6.2.1",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
            # Copy button for code snippets
            "sphinx_copybutton",
            # Markdown parser
            "myst-parser",
            "sphinx-subfigure",
        ],
    },
)
