LEEME

He realizado Los 3 primeros puntos especificados en la práctica, los cuales hacen referencia a:

	- Extracción de características: implementando un descriptor LBP.

	- Ciclo de entrenamiento/validación: implementando diferentes clasificadores y entrenandolos para obtener la mejor configuración de sus hiper-parámetros

	- Test: Con el entrenamiento realizado anteriormente, testeamos el clasificador con un conjunto reducido de muestras, y comprobamos la eficacia de detección

He realizado además los 3 clasificadores especificados en la práctica, cada uno con sus determinados hiper-parametros.

K-NN

Para ejecutar el clasificador K-NN, escribimos la siguiente línea en el terminal:

- Si ejecutamos el proceso de validación/entrenamiento del clasificador:

	./train_test_clf fsiv_gtscb/ models/modelKNN.yml -v -ncells=10x10 -knn_K=1

- Si ejecutamos el testing del clasificador:

	./train_test_clf fsiv_gtscb/ models/modelKNN.yml -t -ncells=10x10 -knn_K=1

Para ejecutar el clasificador SVM, escribimos la siguiente línea en el terminal:

- Si ejecutamos el proceso de validación/entrenamiento del clasificador:

	./train_test_clf fsiv_gtscb/ models/modelSVMLinear.yml -v -class=1 -ncells=1x1

- Si ejecutamos el testing del clasificador:

	./train_test_clf fsiv_gtscb/ models/modelSVMLinear.yml -t -class=1 -ncells=1x1

Para ejecutar el clasificador TRees, escribimos la siguiente línea en el terminal

- Si ejecutamos el proceso de validación/entrenamiento del clasificador:

	./train_test_clf fsiv_gtscb/ models/modelRTrees.yml -v -class=2 -cells=1x1 -rtrees_V=5 -rtrees_T=10000000 -rtrees_E=0.001

- Si ejecutamos el testing del clasificador:

	./train_test_clf fsiv_gtscb/ models/modelRTrees.yml -t -class=2 -cells=1x1 -rtrees_V=5 -rtrees_T=10000000 -rtrees_E=0.001