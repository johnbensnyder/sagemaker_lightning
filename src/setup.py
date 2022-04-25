#!/usr/bin/env python
from setuptools import setup, find_packages

wds_version = "0.1.103"

install_requires = [f"webdataset=={wds_version}",
                    "pytorch-lightning",
                    "lightning-bolts",
                    "torch_tb_profiler"]

setup(
    name="sm_resnet",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/johnbensnyder/sagemaker_lightning",
    description="Resnet test",
    packages=find_packages(),
    install_requires=install_requires,
)