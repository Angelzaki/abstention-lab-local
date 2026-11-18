<div align="center">

# abstention-lab-local

**Calibración de abstención** · Angel Salazar

`min_score × min_overlap → accuracy`

</div>

---

### `$ whoami`

El RAG también debe saber callarse. Calibro el umbral con casos answer/abstain en local.

---

### `$ ./lab.sh`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pytest
python -m abstention_lab
```

Luego aplicas el best en `local-rag-pipeline` (`MIN_SCORE`) y en la lógica de overlap.

---

Hecho en Lima · Angel Salazar · CIP 404730
