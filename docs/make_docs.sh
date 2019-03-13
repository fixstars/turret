#!/bin/sh

sphinx-apidoc -f -o apis ../turret && make html
