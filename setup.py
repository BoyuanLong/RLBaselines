import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="lbyrl",
    version="0.1.1",
    description="Ongoing toy RL baselines implemented by boyuan",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/BoyuanLong/RLBaselines",
    author="Boyuan Long",
    author_email="boyuanlo@usc.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    # install_requires=["feedparser", "html2text"],
    entry_points={
        "console_scripts": [
            "lbybaseline=main:main",
        ]
    },
)