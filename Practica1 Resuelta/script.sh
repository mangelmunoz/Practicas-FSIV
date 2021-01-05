#!/bin/bash

clear

rm -R i72mumam

unzip i72mumam.zip

cd i72mumam

mkdir build

cd build

cmake ..

make

./Realzar_primer_plano manos_sin_balance1.jpg .. ciclista_copia.jpg ..
