# Modèles composites — packager une chaîne de modèles derrière une seule signature

> ⚠️ **Analyse initiale — à revoir**
>
> Ce document est une **première lecture théorique** du sujet, rédigée pour ouvrir la
> discussion. Il n'a pas été validé contre un cas d'usage client concret. Les patterns,
> les trade-offs et les recommandations qu'il contient sont à itérer ensemble avant toute
> mise en application. À traiter comme un point de départ de conversation, pas comme un
> guide arrêté.

Cas d'usage avancé : vous avez plusieurs modèles qui collaborent (chaînes, ensembles,
routage par segment) et vous souhaitez les **packager en un seul artefact MLflow**, exposé
derrière **une signature unique**. L'app source on-prem fait un seul `/predict` ;
le wrapper pyfunc orchestre les sous-modèles en interne.

Ce document complète le [README](README.md) du kit. Il n'a pas vocation à être runnable —
c'est un guide de conception. Un exemple concret peut être ajouté dans `examples/composite/`
dans un second temps, une fois le pattern validé avec votre équipe.

---

## Le problème

Aujourd'hui, supposons que votre prévision repose sur 6 modèles :

```
                    ┌── A1 ──→ A2 ┐
                    │             │
   Input météo  ────┼── B1 ──→ B2 ├──→ logique d'agrégation ──→ Prévision finale
                    │             │
                    └── C1 ──→ C2 ┘
```

Chaque sous-modèle a son propre cycle de vie indépendant :

- 6 signatures MLflow à maintenir
- 6 alias `@champion` à promouvoir
- 6 risques de désalignement (A1.v3 entraîné avec une feature engineering incompatible
  avec A2.v7)
- L'app source on-prem doit orchestrer 6 appels `/predict` (ou consommer 6 endpoints
  Databricks Model Serving), plus une logique d'agrégation côté client

Le contrat d'interface explose en complexité au fur et à mesure que la chaîne s'étend.

---

## L'idée

Un seul **wrapper pyfunc** encapsule tous les sous-modèles. Une seule signature côté
on-prem. Un seul alias `@champion` à promouvoir. La complexité interne du modèle est
totalement invisible pour le consommateur.

```
   ┌─────────────────────────────────────────────────────────┐
   │  Artefact MLflow `forecast_T` (un seul fichier MLmodel) │
   │                                                         │
   │   ┌──────┐    ┌──────┐    ┌──────┐                     │
   │   │  A1  │──→ │  A2  │──┐ │      │                     │
   │   └──────┘    └──────┘  │ │      │                     │
   │   ┌──────┐    ┌──────┐  ├→│ aggr │──→ output           │
   │   │  B1  │──→ │  B2  │──┤ │      │                     │
   │   └──────┘    └──────┘  │ │      │                     │
   │   ┌──────┐    ┌──────┐  │ │      │                     │
   │   │  C1  │──→ │  C2  │──┘ │      │                     │
   │   └──────┘    └──────┘    └──────┘                     │
   │                                                         │
   └─────────────────────────────────────────────────────────┘
       Une signature en entrée, une prédiction en sortie
```

L'app source RTE pousse un payload structuré, reçoit une prédiction (ou un tableau de
prédictions selon la signature retenue). Tout le reste se passe à l'intérieur de
l'artefact.

---

## Le wrapper en pratique

Le squelette du `CombinedForecaster` :

```python
import mlflow.pyfunc
import joblib

class CombinedForecaster(mlflow.pyfunc.PythonModel):
    """Encapsule N pipelines sklearn + logique d'agrégation."""

    def load_context(self, context):
        # Chargé UNE FOIS au boot du container — pas à chaque /predict
        self.A1 = joblib.load(context.artifacts["A1"])
        self.A2 = joblib.load(context.artifacts["A2"])
        self.B1 = joblib.load(context.artifacts["B1"])
        self.B2 = joblib.load(context.artifacts["B2"])
        self.C1 = joblib.load(context.artifacts["C1"])
        self.C2 = joblib.load(context.artifacts["C2"])

        # Logique d'orchestration embarquée via code_paths
        import orchestration as _orch
        self._orch = _orch

    def predict(self, context, model_input):
        # 1. Sépare le payload en sous-inputs par branche
        in_A, in_B, in_C = self._orch.split(model_input)

        # 2. Chaîne chaque branche
        out_A = self.A2.predict(self.A1.predict(in_A))
        out_B = self.B2.predict(self.B1.predict(in_B))
        out_C = self.C2.predict(self.C1.predict(in_C))

        # 3. Agrège (logique métier, autre modèle, moyenne pondérée, etc.)
        return self._orch.aggregate(out_A, out_B, out_C)
```

Et le `log_model` côté training :

```python
mlflow.pyfunc.log_model(
    python_model=CombinedForecaster(),
    artifacts={
        "A1": dump_to_disk(pipeline_A1),
        "A2": dump_to_disk(pipeline_A2),
        "B1": dump_to_disk(pipeline_B1),
        "B2": dump_to_disk(pipeline_B2),
        "C1": dump_to_disk(pipeline_C1),
        "C2": dump_to_disk(pipeline_C2),
    },
    code_paths=["orchestration.py", "features.py"],
    signature=infer_signature(sample_input, sample_output),
    pip_requirements=runtime_lock,
)
```

Un seul artefact MLflow contient les 6 pipelines + le code d'orchestration. L'on-prem
fait un seul `pull_artifact`, charge un seul `pyfunc.load_model`, appelle un seul
`/predict`.

---

## Comment structurer le payload

Trois formats viables pour le `model_input` de `predict()`. Le choix dépend de ce que
votre app source produit déjà nativement.

### Option 1 — Payload nommé (recommandé pour la lisibilité)

```json
{
  "branch_A": {"window": [{"timestamp": "...", "u10": 10, ...}, ...]},
  "branch_B": {"window": [{"timestamp": "...", "u10": 11, ...}, ...]},
  "branch_C": {"window": [{"timestamp": "...", "u10": 12, ...}, ...]}
}
```

Auto-documenté. La signature MLflow type chaque branche explicitement. L'agrégation
sait exactement quelle sortie correspond à quel sous-modèle.

### Option 2 — Liste plate avec colonne de routage

```json
{
  "items": [
    {"branch": "A", "timestamp": "...", "u10": 10, ...},
    {"branch": "B", "timestamp": "...", "u10": 11, ...},
    {"branch": "C", "timestamp": "...", "u10": 12, ...}
  ]
}
```

Plus tabulaire. Pratique si votre app source produit déjà des lignes uniformes (par exemple
un export d'une table avec une colonne discriminante). Le wrapper fait un `groupby` pour router.

### Option 3 — Array of arrays (compact mais ordre implicite)

```json
{
  "windows": [
    [{...}, {...}, ...],
    [{...}, {...}, ...],
    [{...}, {...}, ...]
  ]
}
```

L'ordre des sous-arrays porte le sens (premier = A, deuxième = B, troisième = C).
Moins robuste qu'un payload nommé — un changement d'ordre côté app source casse silencieusement
l'inférence. À documenter explicitement dans la signature.

---

## Les 4 trade-offs à arbitrer

### 1. Entraînement coordonné vs indépendant

**Indépendant** — chaque sous-modèle est entraîné sur sa propre cible métier. Le wrapper
compose à l'inférence mais n'impose pas de dépendance d'entraînement. Plus simple, mais
ne permet pas le **stacking** (utiliser les prédictions de A1 comme features de A2).

**Coordonné (stacking)** — A2 utilise les prédictions de A1 comme features.
Plus puissant mais nécessite des prédictions out-of-fold de A1 pendant le training de A2
pour éviter le leakage. Le notebook de training devient plus subtil.

### 2. Re-entraînement partiel

**Si A1 dérive et nécessite un ré-entraînement**, vous devez **re-loger le composite T entier**
(avec le nouveau A1 + les anciens A2/B/C inchangés). C'est plus lourd qu'un simple
`set_alias` sur A1 individuel.

**Contrepoid** — c'est aussi plus sûr. Impossible d'oublier de re-valider A2 après le
changement de A1. La promotion atomique de T garantit que la chaîne est cohérente
versionnée ensemble.

**Variante hybride** — garder les sous-modèles versionnés individuellement dans UC
(pour le tracking et l'audit), et reconstruire automatiquement T à chaque promotion d'un
sous-modèle via un job Databricks. T devient une **vue assemblée** des derniers `@champion`
individuels. Plus de plomberie mais plus de souplesse.

### 3. Latence

Chargement initial — N modèles à `joblib.load` au boot du container, donc N × mémoire.
Si chaque modèle fait 50 MB en joblib, on est à 300 MB de RAM résident pour 6 modèles.
Acceptable pour un container de prod, à dimensionner.

Inférence — les N inférences se font **séquentiellement** par défaut. Si chaque
sous-modèle prend ~10 ms, on est à 60 ms par requête pour 6 modèles. Pour de la prévision
toutes les 16 minutes, complètement acceptable. Pour du temps réel sub-seconde, à
mesurer.

Si la latence devient un problème, le wrapper peut paralléliser via
`concurrent.futures.ThreadPoolExecutor` — mais le gain est marginal sur des modèles
sklearn légers, et ajoute de la complexité.

### 4. Audit OIV

Avec **un seul artefact composite**, vous avez :
- Une seule version dans UC à auditer
- Un seul `run_id` MLflow qui trace la chaîne complète
- Un seul hash SHA-256 à signer

Mais vous perdez la **traçabilité fine** des sous-modèles individuels (sauf si vous les
loggez aussi séparément dans UC pour le tracking, en parallèle du composite).

Pour OIV, le pattern recommandé est de **logger les sous-modèles individuellement** (audit
trail complet) **+** **assembler un composite T** (artefact opérationnel). Deux objets UC
distincts, avec une référence croisée explicite dans les méta-données du composite.

---

## Quand l'utiliser, quand ne pas

### Utiliser un modèle composite quand

- Plusieurs sous-modèles sont **toujours appelés ensemble** par l'app source
- La **cohérence des versions** entre sous-modèles est critique (risque de désalignement
  si versionnés indépendamment)
- L'app source ne veut pas porter de **logique d'agrégation** (priorité au contrat
  d'interface simple)
- Vous voulez **un seul cycle de promotion** (un seul `@champion` à coordonner avec les
  équipes Ops)

### Garder les modèles indépendants quand

- Les sous-modèles servent des **cas d'usage différents** consommés par des apps
  différentes (un modèle de prévision, un modèle de détection d'anomalie, un modèle de
  classification — pas la même chaîne)
- Les sous-modèles ont des **cycles de vie très différents** (re-entraînement quotidien
  pour l'un, trimestriel pour l'autre — le composite force un cycle aligné, coûteux)
- Vous voulez la **flexibilité de combiner différemment** selon le contexte (parfois
  A+B, parfois A+C, parfois B seul)

---

## Pour aller plus loin

Si vous voulez prototyper avec votre cas réel, partagez :
- Le nombre de sous-modèles et leurs dépendances (chaîne, parallèle, hybride)
- Les signatures actuelles de chaque sous-modèle (colonnes attendues, types)
- La logique d'agrégation entre sorties (moyenne pondérée, vote, modèle final, logique
  métier)
- La cadence de re-entraînement attendue par sous-modèle

Sur cette base, on peut produire un exemple runnable adapté à votre cas dans
`examples/composite/`, plutôt qu'un exemple générique qui imposerait un pattern
qui ne correspondrait pas à votre architecture.
