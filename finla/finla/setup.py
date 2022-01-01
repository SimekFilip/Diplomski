#!/usr/bin/env python

import os
import shutil
import subprocess
from distutils.command.build import build as _build
from distutils.command.build_py import build_py as _build_py

from setuptools import setup


class BuildOracleLib(_build):
  
  def run(self) -> None:
    # Compile oracle lib. The output is liboracle_trading.so ()
    subprocess.call(["./make"], cwd="libs/oracle")
    _build.run(self)


class CopyOracleLibToBuildDirectory(_build_py):
  
  def run(self) -> None:
    _build_py.run(self)  # Creates build/lib/finla directory
    
    # Move the liboracle_trading to the finla package.
    print(os.listdir("libs/oracle"))
    compiled_files = list(filter(lambda x: "liboracle_trading" in x, os.listdir("libs/oracle")))
    if len(compiled_files) != 1:
      raise RuntimeError("Can not decide which shared library to use: {}".format(compiled_files))
    
    src = os.path.join("libs/oracle", compiled_files[0])
    dest = os.path.join("build/lib/finla", compiled_files[0])
    
    if os.path.exists(dest):
      os.remove(dest)
    shutil.move(src, dest)


requirements = ["numpy"]

setup(
  name='finla',
  version='1.0',
  description='Collection of python tools for labeling financial series.',
  author='Fredi Šarić',
  author_email='fredi.saric@fer.hr',
  requires=requirements,
  packages=['finla'],
  cmdclass={
    'build'   : BuildOracleLib,
    "build_py": CopyOracleLibToBuildDirectory
  },
)
