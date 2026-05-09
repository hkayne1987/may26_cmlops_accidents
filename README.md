# MLOps Project Template

⚠️ **Ce dépôt est un TEMPLATE** — adaptez-le à votre projet ML réel. Il fournit une structure de base et des exemples pour chaque phase du cycle de vie MLOps.

---

## 📚 Méthodologie MLOps

Cette méthodologie décrit un parcours structuré pourIndustrialiser un projet ML, en passant par 4 phases principales. L'objectif est d'aboutir à un système de productionrobuste, monitoré et maintenable.

### 🧭 Cadrage (Étape préliminaire)

Avant toute implémentation technique, il est impératif de cadrer le projet :

| Tâche | Description |
|-------|-------------|
| **Définir le problème métier** | Quelle décision le modèle influence-t-il ? |
| **Identifier les métriques** | Métriques ML (accuracy, recall, etc.) + Métriques métier (revenus, économie, etc.) |
| **Établir un baseline** | Modèle simple ou heuristique à battre |
| **Évaluer la faisabilité** | Données disponibles ? Suffisantes ? Labélisées ? |
| **Identifier les contraintes** | Latence,硬件 (CPU/GPU/edge), réglementations |

**Fichier à compléter :** `configs/problem.yaml`

```bash
# Premier réflexe : remplir le cadrage
code configs/problem.yaml
```

---

### 📋 Phases du Projet

#### Phase 1 : Fondations & Containerisation
**Deadline : 30 janvier**

| Tâche principale | Tâches parallèles |
|------------------|-------------------|
| Mettre en place l'environnement de développement | Préparer la CI/CD de base |
| Collecter et prétraiter les données | Rédiger la documentation data |
| Construire le modèle ML de base | Définir les contrats de données (Great Expectations) |
| Implémenter des tests unitaires | Préparer le versioning (DVC) |
| Implémenter l'API d'inférence | Mettre en place le monitoring basique |

**Livrables :** Modèle entraîné, API fonctionnelle, tests green, Dockerfiles valides

#### Phase 2 : Microservices, Suivi & Versioning
**Deadline : 6 février**

| Tâche principale | Tâches parallèles |
|------------------|-------------------|
| Configurer MLflow pour le suivi des expériences | Préparer les pipelines d'entraînement |
| Implémenter le versioning des données et modèles | Définir les métriques de validation |
| Décomposer en microservices | Rédiger l'API spec (OpenAPI) |
| Concevoir l'orchestration simple | Préparer la scalabilité |

**Livrables :** MLflow opérationnel, versioning actif, architecture microservices définie

#### Phase 3 : Orchestration & Déploiement
**Deadline : 13 février**

| Tâche principale | Tâches parallèles |
|------------------|-------------------|
| Finaliser l'orchestration bout en bout | Optimiser les performances API |
| Créer le pipeline CI complet | Sécuriser l'API (auth, rate limiting) |
| Implémenter la scalabilité (K8s) | Préparer le monitoring avancé |
| Déployer en staging/production | Rédiger les runbooks |

**Livrables :** Pipeline CI/CD opérationne, API optimisée et sécurisée, déploiement K8s

#### Phase 4 : Monitoring & Maintenance
**Deadline : 20 février**

| Tâche principale | Tâches parallèles |
|------------------|-------------------|
| Configurer Prometheus/Grafana | Définir les alertes |
| Implémenter la détection de dérive (Evidently) | Mettre en place les tests de charge |
| Automatiser les mises à jour du modèle | Rédiger la documentation technique |
| Finaliser la documentation | Former les équipes.ops |

**Livrables :** Dashboards monitoring, pipeline de retraining, documentation complète

#### 📅 Soutenance : 23/24 février

---

## 🔄 Transitions entre Phases

Chaque phase débouche sur une **revue** avant passage à la suivante :

```
Phase 1 ──► Revue (modèle + API validés) ──► Phase 2 ──► Revue ──► Phase 3 ──► Revue ──► Phase 4
   │                                                    │                              │
   ▼                                                    ▼                              ▼
Validation technique                              Validation infra                  Validation prod
```

### Critères de passage :

- **Phase 1 → Phase 2 :** Tests OK, API fonctionnelle, Dockerfile validé
- **Phase 2 → Phase 3 :** Expériences tracées, versioning opérationnel, artefacts versionnés
- **Phase 3 → Phase 4 :** CI/CD opérationnelle, rollback possible,监控 en place

---

## 👥 Travail en Équipe (Tâches Parallèles)

Pour optimiser le temps, certaines tâches peuvent être menées en **parallèle** :

| Tâche A | Tâche B | Équipe |
|---------|---------|--------|
| Entraînement modèle | Développement API | ML Eng + MLE |
| Feature engineering | Définition data contracts | Data Eng |
| Mise en place MLflow | Infrastructure Docker | MLE + DevOps |
| Tests unitaires | Documentation | QA + Tech Writer |
| Monitoring | CI/CD | SRE + DevOps |

---

## 🛠️ Technologies MLOps

Ce template utilise certaines technologies par défaut, mais d'autres options sont possibles selon vos besoins :

### Orchestration de Pipelines

| Techno | Quand l'utiliser | Installation |
|--------|------------------|--------------|
| **Prefect** (défaut) | Petit à moyen projet, simplicité, monitoring cloud | `uv pip install prefect` |
| **Airflow** | Gros projet, écosystème robuste, nombreuses intégrations | `pip install apache-airflow` |
| **Dagster** | Projet moderne, bonnes pratiques software engineering | `pip install dagster` |

**Prefect (par défaut) :**
```python
# Exemple de pipeline Prefect
from prefect import flow, task

@task
def preprocess_data():
    # votre code
    pass

@flow
def train_pipeline():
    data = preprocess_data()
    model = train_model(data)
    return model
```

**Airflow (alternative) :**
```bash
# Lancer Airflow
airflow standalone
```

```python
# Exemple de DAG Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG('ml_pipeline', start_date=datetime(2024,1,1)) as dag:
    preprocess = PythonOperator(task_id='preprocess', python_callable=preprocess)
    train = PythonOperator(task_id='train', python_callable=train)
    preprocess >> train
```

### Versioning des Données

| Techno | Quand l'utiliser | Installation |
|--------|------------------|--------------|
| **DVC** (défaut) | Versioning fichier, pipeline déclaratif | `pip install dvc` |
| **lakeFS** | Data lake avec versioning, environnement dev/prod | Docker Compose |
| **Delta Lake** | Format Parquet avec transactions (Spark) | Spark dependency |

**DVC :**
```bash
# Initialiser DVC
dvc init

# Ajouter des données
dvc add data/raw/

# Versionner
git add data/raw.dvc
git commit -m "Add raw data v1"

# Récupérer une version
dvc checkout
```

**Commandes DVC utiles :**
```bash
dvc repro          # Relancer le pipeline
dvc metrics show   # Afficher les métriques
dvc diff           # Voir les changements
dvc queue start    # Queue de retraining
```

### Suivi des Expériences

| Techno | Quand l'utiliser | Installation |
|--------|------------------|--------------|
| **MLflow** (défaut) | Tracking ouvert, multi-langue | `pip install mlflow` |
| **Weights & Biases** | Interface utilisateur élégante, collaboration | `pip install wandb` |
| **Neptune.ai** | Plateforme complète, metadata richness | `pip install neptune` |

**MLflow :**
```bash
# Démarrer le serveur
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
```

```python
import mlflow

mlflow.set_experiment("mon_experience")
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_params({"n_estimators": 100})
    mlflow.sklearn.log_model(model, "model")
```

### API d'Inférence

| Techno | Quand l'utiliser | Installation |
|--------|------------------|--------------|
| **FastAPI** (défaut) | Performance, validation Pydantic,async natif | `pip install fastapi uvicorn` |
| **Flask** | Simplicité, petit projet | `pip install flask` |
| **BentoML** | Framework spécialisé inference, packaging simple | `pip install bentoml` |

**FastAPI (par défaut) :**
```bash
# Lancer le serveur
uvicorn api.main:app --reload

# Tester
curl http://localhost:8000/health
```

### Monitoring & Drift Detection

| Techno | Quand l'utiliser | Installation |
|--------|------------------|--------------|
| **Prometheus + Grafana** (défaut) | Métriques custom, visualisations puissantes | docker-compose |
| **Evidently** (défaut) | Data/model drift detection | `pip install evidently` |
| **Arize** | Plateforme complète ML monitoring | pip install arize-ai |

**Evidently :**
```python
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(reference_data=df_ref, current_data=df_current)
dashboard.save("reports/drift.html")
```

---

## 🚀 Installation et Utilisation Rapide

### Prérequis
```bash
# Installer UV (gestionnaire de packages)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer Docker et Docker Compose
# Voir https://docs.docker.com/
```

### Setup du projet
```bash
# Cloner le template
git clone https://github.com/votre-repo/mlops-project.git
cd mlops-project

# Installer les dépendances
uv sync

# Installer les dépendances dev
uv sync --extra dev
```

### Étape 1 : Cadrage
```bash
# Remplir le fichier de problème
code configs/problem.yaml
```

### Lancer les services
```bash
# API seule (Phase 1)
uvicorn api.main:app --reload

# Tous les services (Phase 2-4)
docker-compose up -d
```

### Services disponibles
| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | Inference endpoint |
| API Doc | http://localhost:8000/docs | Swagger UI |
| MLflow | http://localhost:5000 | Tracking |
| Prometheus | http://localhost:9090 | Métriques |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| MinIO | http://localhost:9000 | Stockage objets |

---

## 📁 Structure du Projet

```
mlops_project/
├── .github/workflows/     # CI/CD
├── api/                   # API FastAPI
├── configs/               #Configurations Hydra
│   └── problem.yaml      # Cadrage projet
├── data/                  # Données
│   ├── raw/              # Données brutes
│   ├── processed/        # Données traitées
│   └── features/         # Features engineering
├── models/                # Modèles entraînés
├── notebooks/             # Jupyter exploration
├── pipelines/            # Pipelines Prefect
├── reports/               # Rapports drift
├── src/                   # Code source
│   └── ml_project/       # Package principal
├── tests/                 # Tests
├── docker-compose.yml     # Orchestration
├── Dockerfile.api        # Container API
├── Dockerfile.train      # Container training
├── pyproject.toml        # Dépendances
└── dvc.yaml              # Pipeline DVC
```

---

## ✅ Checklist par Phase

### Phase 1 : Fondations
- [ ] `configs/problem.yaml` rempli
- [ ] Environnement reproductible (uv sync)
- [ ] Données collectées et explorées
- [ ] Modèle baseline entraîné
- [ ] Tests unitaires ajoutés
- [ ] API d'inférence fonctionnelle

### Phase 2 : Microservices
- [ ] MLflow opérationnel
- [ ] Données versionnées (DVC)
- [ ] Modèles versionnés (MLflow)
- [ ] Architecture microservices définie

### Phase 3 : Orchestration
- [ ] Pipeline CI/CD fonctionnel
- [ ] API sécurisée (auth, rate limiting)
- [ ] Déploiement Docker/K8s
- [ ] Rollback testé

### Phase 4 : Monitoring
- [ ] Prometheus/Grafana configurés
- [ ] Alertes définies
- [ ] Drift detection opérationnelle
- [ ] Retraining automatisé
- [ ] Documentation complète

---

## 🔧 Personnalisation du Template

Pour adapter ce template à votre projet :

1. **Remplacer `problem.yaml`** avec votre cas d'usage
2. **Modifier les configs Hydra** dans `configs/`
3. **Implémenter votre modèle** dans `src/ml_project/`
4. **Adapter l'API** dans `api/main.py`
5. **Configurer les métriques** dans `prometheus.yml`
6. **Ajouter vos tests** dans `tests/`

---

## 📚 Ressources Complémentaires

- [Livre "Engineering MLOps"](https://www.amazon.com/Engineering-MLOps-Immutable-pipeline-production-ebook/dp/B09XQX1JKF)
- [Awesome MLOps](https://github.com/visenger/awesome-mlops)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [DVC Documentation](https://dvc.org/doc)