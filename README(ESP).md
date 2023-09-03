# Pfoil

El presente programa se ha realizado como Trabajo Fin de Grado de Guillermo Peña Martínez, alumno de la Escuela de Ingenierías de la Universidad de León (ULE), para el Grado en Ingeniería Aeroespacial.

La información y ecuaciones de los distintos métodos de los paneles empleados se ha obtenido del libro Fundamentals of Aerodynamics (Anderson) y de la documentación del programa XFOIL.

Requiere especial mención el programa Mfoil, de código libre, creado por el profesor Krzysztof J. Fidkowski pues ha servido de inspiración y como debugger comparador del programa creado. La linealización de las ecuaciones y las distintas mejoras a la convergencia se han obtenido de su documentación.

Se presentan dos métodos de resolución de flujo potencial sustentador, el primero de ellos con un único vórtice que genera la circulación y el segundo con una distribución lineal de vórtices en cada panel. El apartado viscoso, el cual ha sido obtenido de la documentación del profesor Krzysztof J. Fidkowski, emplea este último método para el acoplamiento de las soluciones potencial y viscosa.

El siguiente código no se presenta como una sustitución del XFOIL, puesto que este ha demostrado su efectividad y velocidad en numerosas ocasiones, sino como una implementación en un lenguaje de más alto nivel como es Python. El uso es completamente libre adecuándose a las restricciones de la licencia presentada en GitHub.

Guillermo Peña Martínez
