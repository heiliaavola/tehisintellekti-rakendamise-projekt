# TÜ Ainete Soovitaja

Tartu Ülikooli õppeainete semantiline otsimootor. Tudeng kirjeldab, mida ta õppida soovib ja rakendus leiab ÕISist sobivaimad ained — ka siis, kui kirjeldus ei kattu aine nimega.

## Probleem

ÕISi otsing töötab ainult nimepõhiselt. Kui tudeng kirjutab "tahan õppida masinõpet", aga otsitav aine kandub hoopis "Statistical Machine Learning" nime all, ei pruugi ta kunagi seda leida. See rakendus lahendab selle semantilise lõhe.

## Lahendus

RAG-arhitektuuril põhinev soovitaja:

1. Iga aine kohta koostatakse kahekeelne tekstitükk (`rag_text`) koos kirjelduse, eesmärkide, õpiväljundite ja metaandmetega
2. Tekstitükid vektoriseeritakse mitmekeelse sentence-transformers mudeliga
3. Vektorid salvestatakse ChromaDB-sse
4. Kasutaja päringutest tehakse sama embeddinguga semantiline otsing
5. Tulemused kuvatakse Streamlit veebirakenduses koos otselingiga ÕISi aineleheküljele

## Failistruktuur

```
├── andmetega_tutvumine.ipynb       # EDA – andmestikuga tutvumine (kodutöö 1)
├── andmete_ettevalmistus.ipynb     # Andmete puhastamine ja rag_text ehitamine
├── build_vectorstore.py            # ChromaDB vektorandmebaasi ehitamine
├── app.py                          # Streamlit veebirakendus
├── environment.yml                 # Conda keskkond (oisi_projekt, Python 3.10)
├── projektiplaan.md                # CRISP-DM projektiplaaan
├── ideas.md                        # Ideede märkmed
└── data/
    ├── toorandmed_aasta.csv              # Toore ÕISi andmestik (ei ole gitis)
    ├── rag_courses.parquet               # Puhastatud andmestik (3 156 ainet)
    ├── rag_courses_filtered.parquet      # RAG-i sisend (2 113 ainet, filtrite järel)
    ├── rag_text_length_distribution.png  # Tekstitükkide pikkuste histogram
    └── chroma_db/                        # ChromaDB vektorandmebaas
```

## Kiire alustamine

### 1. Loo Conda keskkond

```bash
conda env create -f environment.yml
conda activate oisi_projekt
```

### 2. Valmista andmed ette

Käivita märkmik `andmete_ettevalmistus.ipynb` — see loeb `data/toorandmed_aasta.csv`, puhastab andmed ja salvestab `data/rag_courses_filtered.parquet`.

### 3. Ehita vektorandmebaas

```bash
conda run -n oisi_projekt python build_vectorstore.py
```

Embeddib 2 113 ainet ja salvestab ChromaDB `data/chroma_db/` alla. Esimesel käivitusel laadib ~118 MB mudeli alla (üks kord).

### 4. Käivita rakendus

```bash
conda run -n oisi_projekt streamlit run app.py
```

Ava brauser aadressil `http://localhost:8501`.

## Tehniline ülevaade

| Komponent | Valik | Põhjus |
|-----------|-------|--------|
| Embeddingu mudel | `paraphrase-multilingual-MiniLM-L12-v2` | Mitmekeelne (ET + EN), 118 MB, lokaalne, MIT litsents |
| Vektorandmebaas | ChromaDB | Lokaalne, ei vaja serverit, Python-native |
| Kasutajaliides | Streamlit | Kiire prototüüpimine, ei vaja JS-i |
| Andmestik | TÜ ÕIS (3 156 → 2 113 ainet) | Täiendõpe ja kaitsmisained filtreeritud välja |

## Piirangud

- Andmestik on staatiline koopia — ei uuene automaatselt
- LLM-põhist kokkuvõtet ei ole veel lisatud — rakendus kuvab otsitulemused ilma loomuliku keele selgituseta
- Väga lühikesed või üldised päringud (nt "aine") annavad nõrgemaid sarnasuse skoore
