# Reserva en un observatorio

La herramienta reservar_cupo_observatorio es la única de riesgo alto, porque una reserva real compromete cupos y dinero del turista. Hoy es simulada: no está conectada a ningún sistema de reservas y su respuesta siempre empieza con "RESERVA SIMULADA".

Antes de pedir aprobación valida los datos: entre 1 y 20 personas, una fecha real en formato AAAA-MM-DD y un observatorio de la lista (Mamalluca, Cancana, Alfa Aldea, Cruz del Sur, Collowara y Del Pangue). Si algo no cuadra, no pide aprobación.

Con datos válidos, un supervisor humano debe aprobar en la consola. Si nadie responde dentro del tiempo límite (300 segundos por defecto), la reserva queda no aprobada. Cada decisión, aprobada o no, queda registrada.
