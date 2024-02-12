#!/bin/bash

set -ex

mkdir -p dist
cp -R static dist/static
npx --yes @marp-team/marp-cli@latest slides.md -o dist/index.html

mkdir -p dist/samples/housing
cp samples/housing/index.html dist/samples/housing/index.html
cp samples/housing/model.onnx dist/samples/housing/model.onnx
pipenv run jupyter nbconvert --to html --output-dir dist/samples/housing --execute samples/housing/training.ipynb

mkdir -p dist/samples/sentiment
cp samples/sentiment/index.html dist/samples/sentiment/index.html
cp samples/sentiment/model.onnx dist/samples/sentiment/model.onnx
pipenv run jupyter nbconvert --to html --output-dir dist/samples/sentiment --execute samples/sentiment/training.ipynb

mkdir -p dist/samples/imaging
cp samples/imaging/index.html dist/samples/imaging/index.html
cp samples/imaging/model.onnx dist/samples/imaging/model.onnx
cp samples/imaging/imagenet_class_index.json dist/samples/imaging/imagenet_class_index.json
pipenv run jupyter nbconvert --to html --output-dir dist/samples/imaging --execute samples/imaging/training.ipynb
