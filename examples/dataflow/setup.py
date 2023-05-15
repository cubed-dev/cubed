from setuptools import find_packages, setup

setup(
    name="cubed-dataflow-examples",
    version="1.0.0",
    url="https://github.com/tomwhite/cubed-dataflow-examples",
    author="Tom White",
    author_email="tom.e.white@gmail.com",
    description="Examples for running Cubed on Dataflow",
    packages=find_packages(),
    install_requires=["cubed", "gcsfs"],
)
