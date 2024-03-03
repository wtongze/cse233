#!/bin/bash
git submodule init
git submodule update

cd cage-challenge-2
cd CybORG
pip install -e .
cd ..
cd ..
