README

He realizado la parte obligatoria, la cual consta de seleccionar una zona rectangular de la imagen a partir de 2 puntos que nosotros elegimos, y el punto 1 de la parte opcional, que consiste en seleccionar una zona rectangular mediante la seleccion de un click y arrastrando el ratón por la imagen hasta donde queramos colocar el segundo punto.

- Para compilar, escribimos en la linea de comandos las siguientes lineas:
mkdir build  
cd build 
cmake ..
make

- Para ejecutar el programa, escribimos en la línea de comandos la siguiente linea:

./Realzar_primer_plano <imagen_entrada> <imagen_salida>

Una vez ejecutado el programa, aparecerá un menú, en el cual aparecen 3 opciones:

Opcion 1: Selección con dos clicks
Opcion 2: Seleccion interactiva arrastrando el raton
Opcion 0: Salir

La opción 1 es la opcion donde seleccionamos 2 clicks, y la opcion 2 es la opción donde seleccionamos 1 click y arrastramos. La opción 0 nos permite salir del programa