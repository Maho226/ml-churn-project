# ML Churn Project

Industrialisation d'un modèle de Machine Learning de prédiction du churn (départ des clients) d'un opérateur télécom.

Le modèle est exposé par une API FastAPI, testé automatiquement (Pytest, Selenium, JMeter), versionné avec Git, conteneurisé avec Docker, intégré dans une chaîne CI/CD GitHub Actions, déployé sur un serveur et placé sous observabilité (ELK pour les logs, Prometheus et Grafana pour les métriques).

**Auteur :** Mohamed KABORE

---

## Sommaire

1. [Architecture](#1-architecture)
2. [Arborescence du projet](#2-arborescence-du-projet)
3. [Technologies et versions](#3-technologies-et-versions)
4. [Guide d'installation](#4-guide-dinstallation)
5. [Données et modèle](#5-données-et-modèle)
6. [API](#6-api)
7. [Tests](#7-tests)
8. [Docker et Docker Compose](#8-docker-et-docker-compose)
9. [Intégration continue (GitHub Actions)](#9-intégration-continue-github-actions)
10. [Tests de charge (JMeter)](#10-tests-de-charge-jmeter)
11. [Observabilité](#11-observabilité)
12. [Déploiement en production](#12-déploiement-en-production)
13. [Workflow Git](#13-workflow-git)
14. [Dépannage](#14-dépannage)

---

## 1. Architecture

```
                         UTILISATEUR
                              │
                              ▼
                    ┌───────────────────┐
                    │  Interface Web    │   /ui
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │  FastAPI          │   /predict  /health  /model/info
                    │  (3 workers)      │   /metrics
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │  Random Forest    │   models/model.pkl
                    └───────────────────┘

   Logs    : API ──(logs/api.log, volume partagé)──► Logstash ──► Elasticsearch ──► Kibana
   Métriques : API ──(/metrics)──► Prometheus ──► Grafana
   CI/CD   : git push ──► GitHub Actions ──► tests ──► build Docker ──► smoke test
```

Tous les services tournent dans un même réseau Docker Compose. Les services se joignent par leur nom (`api`, `elasticsearch`, `prometheus`).

---

## 2. Arborescence du projet

```
ml-churn-project/
├── api/main.py                          API FastAPI (prédiction, logs JSON, métriques)
├── src/generate_data.py                 Génération du dataset
├── src/train.py                         Entraînement et comparaison des modèles
├── notebooks/exploration.py             Exploration des données
├── data/raw/churn.csv                   Dataset (2000 clients)
├── models/model.pkl                     Modèle entraîné (Random Forest)
├── models/metadata.json                 Version, date, variables et métriques du modèle
├── frontend/index.html                  Interface web minimale
├── tests/test_api.py                    Tests unitaires et API (Pytest)
├── tests/test_selenium.py               Tests de bout en bout (Selenium)
├── jmeter/churn_load_test.jmx           Plan de test de charge
├── docker/logstash/pipeline/            Pipeline Logstash
├── monitoring/prometheus/               Configuration Prometheus
├── monitoring/grafana/                  Source de données et tableau de bord provisionnés
├── .github/workflows/ci.yml             Pipeline CI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                     Dépendances d'exécution (image Docker)
├── requirements-dev.txt                 Dépendances de développement et de test
└── pytest.ini                           Configuration Pytest
```

---

## 3. Technologies et versions

| Domaine | Outil | Version |
|---|---|---|
| Système | Ubuntu | 26.04 LTS |
| Langage | Python | 3.14 |
| Machine Learning | scikit-learn, pandas, numpy | 1.7.2, 2.3.3, 2.3.4 |
| API | FastAPI, Uvicorn, Pydantic | 0.120.0, 0.38.0, 2.12.3 |
| Tests | Pytest, Selenium, JMeter | 8.4.2, 4.38.0, 5.6.3 |
| Conteneurs | Docker, Docker Compose | 29, v2 |
| Logs | Elasticsearch, Logstash, Kibana | 8.15.3 |
| Métriques | Prometheus, Grafana | 2.55.1, 11.3.0 |
| CI/CD | GitHub Actions | — |

La même version de Python (3.14) est utilisée sur le poste, dans l'image Docker et dans la CI : un modèle enregistré avec une version de Python et de scikit-learn peut ne pas se recharger avec une autre.

---

## 4. Guide d'installation

### 4.1 Prérequis

- Ubuntu 26.04 (ou distribution Linux équivalente), 4 cœurs, 6 Go de RAM, 15 Go de disque libre pour la stack complète
- Python 3.14, Git
- Docker, Docker Compose et Buildx :

```bash
sudo apt update
sudo apt install docker.io docker-compose-v2 docker-buildx -y
sudo systemctl enable --now docker
sudo usermod -aG docker $USER      # puis se déconnecter et se reconnecter
docker run hello-world
```

- Réglage système requis par Elasticsearch :

```bash
sudo sysctl -w vm.max_map_count=262144
echo "vm.max_map_count=262144" | sudo tee /etc/sysctl.d/99-elasticsearch.conf
```

### 4.2 Récupérer le projet

```bash
git clone https://github.com/Maho226/ml-churn-project.git
cd ml-churn-project
```

### 4.3 Environnement Python

```bash
sudo apt install python3-venv -y
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 4.4 Lancer l'application

```bash
docker compose up -d --build       # stack complète
docker compose up -d api           # API seule
docker compose ps
```

| Service | URL | Identifiants |
|---|---|---|
| API | http://localhost:8000 | — |
| Swagger | http://localhost:8000/docs | — |
| Interface web | http://localhost:8000/ui/ | — |
| Kibana | http://localhost:5601 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

Pour un accès depuis une autre machine, remplacer `localhost` par l'adresse IP du serveur.

---

## 5. Données et modèle

### 5.1 Dataset

Le dataset est généré par `src/generate_data.py` (graine aléatoire fixée à 42, donc reproductible) : 2000 clients, 6 variables explicatives, aucune valeur manquante.

| Variable | Description |
|---|---|
| `age` | Âge du client (18 à 69 ans) |
| `anciennete` | Ancienneté en mois (1 à 119) |
| `revenu` | Revenu (FCFA) |
| `nb_reclamations` | Nombre de réclamations |
| `montant_mensuel` | Montant mensuel facturé (FCFA) |
| `retards_paiement` | Nombre de retards de paiement |
| `churn` | **Variable cible** : 1 = le client part, 0 = il reste |

Répartition : 1139 clients restent (57 %), 861 partent (43 %).

```bash
python src/generate_data.py
python notebooks/exploration.py     # statistiques + graphiques dans data/
```

### 5.2 Entraînement

`src/train.py` sépare les données en 80 % d'entraînement et 20 % de test (stratifié sur `churn`), puis compare trois modèles :

| Modèle | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Régression logistique (avec StandardScaler) | 0,625 | 0,582 | 0,454 | 0,510 |
| Arbre de décision (profondeur 5) | 0,633 | 0,583 | 0,512 | 0,545 |
| **Random Forest (200 arbres)** — retenu | 0,605 | 0,552 | 0,436 | 0,487 |

Le Random Forest est retenu conformément au sujet. Sur ce jeu de données, généré par une formule logistique simple, les modèles simples obtiennent de meilleurs scores ; la forêt, sans limite de profondeur, apprend aussi le bruit des données.

```bash
python src/train.py
```

Le script produit :

- `models/model.pkl` : le modèle (compressé, environ 2 Mo) ;
- `models/metadata.json` : nom, version, date d'entraînement, variables et métriques. L'API lit ce fichier pour `/model/info` et pour la version indiquée dans chaque réponse et chaque log.

---

## 6. API

### 6.1 Lancement en développement

```bash
uvicorn api.main:app --reload                 # accessible sur la machine locale
uvicorn api.main:app --reload --host 0.0.0.0  # accessible depuis le réseau
```

### 6.2 Routes

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/` | Nom, version et état de l'application |
| GET | `/health` | État du service (utilisé par le healthcheck Docker) |
| GET | `/model/info` | Modèle, version, date d'entraînement, variables, métriques |
| POST | `/predict` | Prédiction du churn d'un client |
| GET | `/metrics` | Métriques au format Prometheus |
| GET | `/docs` | Documentation Swagger |
| GET | `/ui/` | Interface web |

### 6.3 Exemple de prédiction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"anciennete":5,"revenu":450000,"nb_reclamations":2,"montant_mensuel":25000,"retards_paiement":2}'
```

```json
{
  "request_id": "2cae628b-4c40-4d8e-ae97-35ffcaead176",
  "prediction": "CHURN",
  "probability": 0.67,
  "model_version": "1.0.0",
  "duration_ms": 40.49
}
```

Les données sont validées par Pydantic (âge entre 18 et 100, valeurs positives). Une requête invalide ou incomplète est rejetée avec le code **422**.

---

## 7. Tests

### 7.1 Tests unitaires et API (Pytest)

```bash
pytest -v
```

Six tests : `/`, `/health`, `/model/info`, `/predict` (prédiction valide), rejet d'une requête invalide (422) et exposition des métriques. Résultat attendu : `6 passed`.

`pytest.ini` exclut par défaut les tests Selenium (marqueur `e2e`), qui nécessitent une API démarrée et un navigateur.

### 7.2 Tests de bout en bout (Selenium)

Prérequis : l'API tourne (`docker compose up -d api`), Google Chrome et `chromedriver` sont installés.

```bash
# Google Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb -y

# chromedriver de la même version majeure que Chrome (exemple via pip)
pip install chromedriver-py==156.0.8078.80
sudo install -m 755 "$(python -c 'import chromedriver_py; print(chromedriver_py.binary_path)')" /usr/local/bin/chromedriver

pytest -m e2e -v
BASE_URL=http://192.168.20.253:8000 pytest -m e2e -v    # contre un autre serveur
```

Deux tests, exécutés en mode headless :

- `test_swagger_docs` : la page Swagger s'ouvre ;
- `test_prediction_from_ui` : saisie des données dans l'interface web, clic sur « Prédire », vérification du résultat affiché.

---

## 8. Docker et Docker Compose

### 8.1 Image de l'API

```bash
docker build -t ml-churn-api:1.0 .
docker run -d --name ml-churn-api -p 8000:8000 ml-churn-api:1.0
curl http://localhost:8000/health
docker stop ml-churn-api && docker rm ml-churn-api
```

Caractéristiques de l'image :

- base `python:3.14-slim`, dépendances d'exécution uniquement (`requirements.txt`) ;
- exécution sous un utilisateur non-root (`appuser`) ;
- `HEALTHCHECK` sur `/health` ;
- **3 workers Uvicorn** (`WEB_CONCURRENCY=3`) pour utiliser plusieurs cœurs ; les métriques des workers sont agrégées grâce à `PROMETHEUS_MULTIPROC_DIR` ;
- `.dockerignore` excluant `.venv/`, `.git/`, les tests et la configuration de monitoring.

Le modèle doit être entraîné (`models/model.pkl` présent) avant la construction de l'image.

### 8.2 Docker Compose

```bash
docker compose up -d --build        # construire et lancer
docker compose ps                   # état des services
docker compose logs api             # logs de l'API
docker compose down                 # arrêter (les volumes sont conservés)
```

| Service | Image | Port | Rôle |
|---|---|---|---|
| `api` | ml-churn-api:1.0 | 8000 | API de prédiction |
| `elasticsearch` | elasticsearch:8.15.3 | 9200 | Stockage et recherche des logs |
| `logstash` | logstash:8.15.3 | — | Collecte de `api.log` |
| `kibana` | kibana:8.15.3 | 5601 | Exploration des logs |
| `prometheus` | prometheus:v2.55.1 | 9090 | Collecte des métriques |
| `grafana` | grafana:11.3.0 | 3000 | Tableaux de bord |

Volumes : `api-logs` (logs de l'API, partagé avec Logstash), `es-data` (données Elasticsearch), `grafana-data`.

---

## 9. Intégration continue (GitHub Actions)

Le pipeline `.github/workflows/ci.yml` se déclenche à chaque `push` et à chaque *pull request* sur `main` et `develop`.

| Job | Étapes |
|---|---|
| `test` | Installation de Python 3.14 et des dépendances → génération des données → entraînement → `pytest` |
| `build` (si `test` réussit) | Construction de l'image Docker → démarrage du conteneur → appel de `/health` et `/predict` (smoke test) |

Les résultats sont visibles dans l'onglet **Actions** du dépôt GitHub. Durée d'une exécution : environ 2 minutes.

---

## 10. Tests de charge (JMeter)

### 10.1 Installation

```bash
sudo apt install default-jre-headless -y
wget https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.6.3.tgz
tar xzf apache-jmeter-5.6.3.tgz
export PATH="$HOME/apache-jmeter-5.6.3/bin:$PATH"
```

### 10.2 Exécution

Le plan `jmeter/churn_load_test.jmx` envoie des requêtes `POST /predict`, avec une assertion sur le code HTTP 200. Il est paramétrable en ligne de commande. Les tests se lancent en mode non graphique (`-n`), recommandé pour les mesures :

```bash
for u in 100 500 1000; do
  jmeter -n -t jmeter/churn_load_test.jmx -Jthreads=$u -Jrampup=10 -Jloops=10 \
    -l jmeter/results/${u}u.jtl -e -o jmeter/report/${u}u
done
```

Le rapport HTML de chaque palier se trouve dans `jmeter/report/<N>u/index.html`.

### 10.3 Résultats (VM 4 cœurs, 5,5 Go de RAM)

**Optimisation réalisée** après un premier test à 100 utilisateurs (3,3 req/s, 93 % d'erreurs, un seul cœur utilisé) :

- un seul appel au modèle par requête (`predict_proba` au lieu de `predict` + `predict_proba`) ;
- trois workers Uvicorn au lieu d'un.

| Utilisateurs | Moyenne | Médiane | P90 | P95 | Débit | Erreurs |
|---|---|---|---|---|---|---|
| 100 | 12,3 s | 11,3 s | 18,8 s | 20,8 s | 7,0 req/s | 0 % |
| 500 | 29,8 s | 30,0 s | 30,1 s | 30,1 s | 16,1 req/s* | 97,4 % |
| 1000 | 30,1 s | 30,0 s | 30,1 s | 30,2 s | 31,8 req/s* | 100 % |

\* À 500 et 1000 utilisateurs, le débit inclut les requêtes abandonnées au bout du délai de 30 s : il ne correspond pas à des prédictions servies.

**Conclusion :** la capacité de l'API sur cette machine est d'environ 7 prédictions par seconde. La charge de 100 utilisateurs simultanés est supportée sans erreur ; le point de rupture se situe entre 100 et 500 utilisateurs.

---

## 11. Observabilité

### 11.1 Logs (ELK)

L'API écrit une ligne JSON par événement, sur la console et dans `logs/api.log` :

```json
{"timestamp": "2026-11-03T12:00:09+0000", "level": "INFO", "logger": "churn_api",
 "message": "Prediction done", "event_type": "prediction",
 "request_id": "2cae628b-...", "client_ip": "127.0.0.1", "operation": "predict",
 "model": "Random Forest", "model_version": "1.0.0", "prediction": "CHURN",
 "probability": 0.67, "response_time_ms": 40.49}
```

| Question | Champ |
|---|---|
| Quand ? | `timestamp` |
| Qui ? | `client_ip`, `request_id` |
| Quelle opération ? | `operation`, `event_type` |
| Quel modèle ? | `model`, `model_version` |
| Quel résultat ? | `prediction`, `probability` |
| Combien de temps ? | `response_time_ms` |
| Quelle erreur ? | `level` = `ERROR`, `error` |

Logstash (`docker/logstash/pipeline/logstash.conf`) lit le fichier depuis le volume partagé `api-logs` et indexe les événements dans Elasticsearch (index `ml-churn-logs-AAAA.MM.JJ`).

**Dans Kibana :** *Stack Management → Data Views → Create data view*, motif `ml-churn-logs-*`, champ de temps `@timestamp`. Exemples de recherches dans *Discover* :

| Recherche | Requête KQL |
|---|---|
| Prédictions | `event_type : "prediction"` |
| Erreurs | `level : "ERROR"` |
| Version du modèle | `model_version : "1.0.0"` |
| Réponses lentes | `response_time_ms > 100` |

### 11.2 Métriques (Prometheus et Grafana)

Prometheus interroge `/metrics` toutes les 5 secondes (`monitoring/prometheus/prometheus.yml`). Métriques disponibles :

| Métrique | Origine | Contenu |
|---|---|---|
| `http_requests_total` | Instrumentator | Requêtes par route, méthode et code HTTP |
| `http_request_duration_seconds` | Instrumentator | Temps de réponse (histogramme) |
| `churn_predictions_total` | API | Prédictions par résultat (CHURN / NO_CHURN) et version |
| `churn_prediction_errors_total` | API | Erreurs du modèle |
| `churn_prediction_duration_seconds` | API | Durée d'inférence du modèle |

Grafana est configuré automatiquement au démarrage (*provisioning*) : source de données Prometheus et tableau de bord **ML Churn → ML Churn API**, avec :

- nombre de requêtes, de prédictions et d'erreurs ;
- taux d'erreur ;
- requêtes par seconde par route ;
- temps de réponse de `/predict` (médiane, p95, p99) ;
- prédictions par minute (CHURN / NO_CHURN) ;
- erreurs par seconde.

---

## 12. Déploiement en production

Environnement : un serveur Ubuntu distinct de la machine de développement (`vm-tp-prod`, 192.168.20.253).

```bash
# 1. Prérequis : Docker, Compose, Buildx et vm.max_map_count (voir 4.1)

# 2. Récupérer la version validée (branche main)
git clone https://github.com/Maho226/ml-churn-project.git
cd ml-churn-project

# 3. Construire et lancer
docker compose build
docker compose up -d
docker compose ps

# 4. Vérifier depuis une autre machine
curl http://192.168.20.253:8000/health
curl -X POST http://192.168.20.253:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"anciennete":5,"revenu":450000,"nb_reclamations":2,"montant_mensuel":25000,"retards_paiement":2}'
```

Le modèle (`models/model.pkl`) fait partie du dépôt : aucun réentraînement n'est nécessaire sur le serveur. Le tableau de bord Grafana est disponible dès le démarrage. Dans Kibana, la vue de données `ml-churn-logs-*` est à créer une fois.

Mise à jour : `git pull` puis `docker compose up -d --build`.

---

## 13. Workflow Git

| Branche | Rôle |
|---|---|
| `main` | Version stable, déployée en production |
| `develop` | Intégration des développements |
| `feature/*` | Développement d'une fonctionnalité (ex. `feature/model`) |

Cycle : `feature/*` → fusion dans `develop` → *pull request* `develop` → `main`, validée par la CI avant fusion.

---

## 14. Dépannage

| Problème | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'api'` avec `pytest` | Racine du projet absente du chemin Python | `pytest.ini` contient `pythonpath = .` |
| Selenium : `Unable to obtain driver for chrome` | Selenium Manager ne peut pas télécharger `chromedriver` | Installer `chromedriver` localement (voir 7.2) |
| Elasticsearch ne démarre pas | `vm.max_map_count` trop faible | `sudo sysctl -w vm.max_map_count=262144` |
| `git push` refusé pour `.github/workflows` | Jeton GitHub sans le droit `workflow` | Ajouter le droit `workflow` au jeton |
| Grafana affiche « No data » alors que la cible Prometheus est UP | Horloge de la VM décalée (après une mise en veille) | `sudo date -s "$(curl -sI https://www.google.com \| grep -i '^date:' \| cut -d' ' -f2-)"`, puis `docker compose restart prometheus` |
| Téléchargement des images Docker interrompu | Connexion instable | Réglage `max-concurrent-downloads: 1` dans `/etc/docker/daemon.json`, ou transfert depuis une autre machine : `docker save <images> \| ssh user@serveur docker load` |
| `jmeter -e -o` refuse d'écrire le rapport | Dossier de rapport non vide | `rm -rf jmeter/report/<N>u` avant de relancer |

> **Sécurité :** la configuration est prévue pour un environnement de TP. En production réelle, il faudrait activer l'authentification d'Elasticsearch (`xpack.security.enabled=true`), changer le mot de passe Grafana et restreindre l'ouverture des ports.
