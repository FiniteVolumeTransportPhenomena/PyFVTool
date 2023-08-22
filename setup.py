from codecs import open
from setuptools import setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="PyFVTool",
    version="0.1",
    author="Ali A. Eftekhari",
    author_email="e.eftekhari@gmail.com",
    description="A finite volume tool in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simulkade/PyFVTool",
    license="MIT",
    packages=['pyfvtool'],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy>=1.8.0",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Research",
        "License :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
)