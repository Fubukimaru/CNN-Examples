#!/bin/bash

#pandoc -t beamer -V CJKmainfont=IPAexMincho slides.md -o slides.pdf


pandoc -t beamer --self-contained --smart -f markdown -V CJKmainfont=IPAexMincho --latex-engine=xelatex -o slides.pdf slides.md
