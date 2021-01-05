#!/bin/bash

clear

rm -R i72mumam

unzip i72mumam.zip

cd i72mumam

mkdir build

cd build

cmake ..

make

# ./color_balance manos_sin_balance1.jpg . manos_retocada.jpg . -p=0

./color_balance manos_sin_balance1.jpg  manos_retocada.jpg .. -p=100
