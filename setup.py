from setuptools import setup, find_packages

# Listed explicitly so stray __init__.py dirs (baselines, tests) are not installed.
setup(
    name="voxbind",
    author="Pedro O. Pinheiro",
    version="0.1",
    packages=find_packages(include=["voxbind", "voxbind.*", "dataset", "dataset.*",
                                    "pretrain", "pretrain.*"]),
)
