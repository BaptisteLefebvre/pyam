'testmovie kept stack aligned.tif': Original file (i.e. data)


# TODO :
 - [?] Générer une vidéo sans les basses fréquences.
 - [?] Générer une vidéo sans les hautes fréquences.
 - [?] Séparer hautes, moyennes et basses fréquences.
 - [ ] Générer deux vidéos (amplitude & phase) pour chacun des filtres de la
       pyramide.


# Questions :
 - Combien de temps à durée l'enregistrement de 'testmovie kept stack aligned.tif' ?
 - Quelle est l'activité des neurones pendant l'enregistrement ?
   Quelle sont les événements neuronaux qui ont eu lieu pendant l'enregistrement ?
   (activité spontanée, nombres de potentiels d'actions, nombre de "frames" par potentiel d'action)
   Peut-on utiliser un marqueur pour avoir plus d'informations sur l'activité des neurones ?
 
 - Comment sont "flashés" les neurones (puissance, laser, ...) ?
 - Quel est la dye calcique utilisée (i.e. constante de temps de 30 s) ?
 - Pourquoi imager à 20 Hz et non à 60 Hz (constante de temps des variations mécaniques) ?
 - Expériences avec activité spontanée (i.e. sans flash) ?


# Réponses :
 - Les tâches noires qui se déplacent le long des neurones sont probablement des vésicules.
 - La vidéo a été enregistrée à 50 frame par seconde.
 - Pour voir la propagation d'un potentiel d'action (1 mètre par seconde) il faudrait augmenter la fréquence d'échantillonage (i.e. utiliser une caméra à grande vitesse).
 - Pour pouvoir quantifier les modifications de volumes potentiellement détectées, adapter l'algorithme de magnification à la présence de débris + vérifier que la résolution de la vidéo permet de détecter ces modifications.
 - Pour pouvoir attester des événements neuronaux qui ont eu lieu, des mesures optogénétiques vont être effectuées.
 - La but ultime (graâl), observer la propagation des potentiels d'actions au sein d'un réseau de neurones.

# Idées :
 - Utiliser des algorithmes de biologie pour enlever les débris (probablement classique).
 - Utiliser des algorithmes de super-résolution en pré-processing.
 - Adapter l'algorithme de magnification eulérienne pour son application en présence de débris.

# Notes
 - 1 m/s pour la propagation électrique dans l'axone.
