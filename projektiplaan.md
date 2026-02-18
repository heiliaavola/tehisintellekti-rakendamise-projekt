# 🤖 Tehisintellekti rakendamise projektiplaani mall (CRISP-DM)

<br>
<br>


## 🔴 1. Äritegevuse mõistmine
*Fookus: mis on probleem ja milline on hea tulemus?*


### 🔴 1.1 Kasutaja kirjeldus ja eesmärgid
Kellel on probleem ja miks see lahendamist vajab? Mis on lahenduse oodatud kasu? Milline on hetkel eksisteeriv lahendus?

> Probleem on tudengitel, kes soovivad huvipakkuvatele ainetele registeeruda, kuid huvi_kirjeldus != aine nimi ÕISi otsingust. Lisaks ÕISi otsing ise ei toimi ideaalselt, kui kirjutada otsingusse sõna osa, mis aine nimes siseldub, ei pruugi vastet saada. Lahenduse oodatud kasu on see, et tudeng leiab hõlpsalt aine, mis talle sobib. Hetke lahendusega on raske leida ainet, mille sisu vastab sellele, mida reaalselt tahad. 

### 🔴 1.2 Edukuse mõõdikud
Kuidas mõõdame rakenduse edukust? Mida peab rakendus teha suutma?

> Rakenduse edukust mõõdame kasutaja tagasiside põhjal. Rakendus peab suutma kasutajale anda õppaine soovituse vastavalt kasutaja sisendile, kui sellist ainet ei ole olemas, siis tuleb see kasutajale teada anda. Lisaks peab soovitus olema algse sisendiga kooskõlas, kui kasutaja küsib aineid, mis asuvad tartus, pole mõtet soovitada Viljandi aineid. Testimine teststsenaariumitega (test-cases).

### 🔴 1.3 Ressursid ja piirangud
Millised on ressursipiirangud (nt aeg, eelarve, tööjõud, arvutusvõimsus)? Millised on tehnilised ja juriidilised piirangud (GDPR, turvanõuded, platvorm)? Millised on piirangud tasuliste tehisintellekti mudelite kasutamisele?

> Piirangud: aeg, eelarve puudlik, tööjõud - 2 inimese vaba aeg, arvutusvõimsus piirdub lokaalse arvuti riistvaraga või tasuta APId. Turvanõuded: kasutajad saavad teha prompt injectionit ja kasutada meie mudleit mitteotstarbeliselt ära, raisates ressurssi. Arvestada tuleb sellega, et kasutaja võib sisendina anda personaalset infot, mis võib edasi lekkida API pakkujale. Tasulised mudleid maksavad rohkem kui meil raha on. Kui on vähe kaustajaid, siis saab hakkama tasuta limiitidega, kuid kui on suurem kasutajaskond (kõik UT tudnegid), siis peab kuluga arvestama.

<br>
<br>


## 🟠 2. Andmete mõistmine
*Fookus: millised on meie andmed?*

### 🟠 2.1 Andmevajadus ja andmeallikad
Milliseid andmeid (ning kui palju) on lahenduse toimimiseks vaja? Kust andmed pärinevad ja kas on tagatud andmetele ligipääs?

> Ainete kirjeldused, koodid, nimetused, mahud, asukohad, tagasiside, kohapelane aine jah/ei. Adme dpärinevad ÕISi APIst ja on ligipääsetavad (scraper provided).

### 🟠 2.2 Andmete kasutuspiirangud
Kas andmete kasutamine (sh ärilisel eesmärgil) on lubatud? Kas andmestik sisaldab tundlikku informatsiooni?

> Sõltub, mis on API dokumentatsioonid kirjas (kasutuslitsents). Sõltub, mida tundlikuks informatsiooniks lugeda. 

### 🟠 2.3 Andmete kvaliteet ja maht
Millises formaadis andmeid hoiustatakse? Mis on andmete maht ja andmestiku suurus? Kas andmete kvaliteet on piisav (struktureeritus, puhtus, andmete kogus) või on vaja märkimisväärset eeltööd)?

> .csv faili formaat, andmemaht veerud x read : 223x3031, kvaliteet ei ole RAG süsteemi jaoks piisav, tuleb teostada filtreerimine (luua ärireeglid ja piirangud, kui suurelt probleemi lahendada) ja struktureerida. 

### 🟠 2.4 Andmete kirjeldamise vajadus
Milliseid samme on vaja teha, et kirjeldada olemasolevaid andmeid ja nende kvaliteeti.

> EDA - exploratory data analysis, uurida andmeid (juba osaliselt tehtud), leida kitsaskohad, mis võivad RAG süsteemi häirida ja kasutatavaid kirjeid/veerge. 

<br>
<br>


## 🟡 3. Andmete ettevalmistamine
Fookus: Toordokumentide viimine tehisintellekti jaoks sobivasse formaati.

### 🟡 3.1 Puhastamise strateegia
Milliseid samme on vaja teha andmete puhastamiseks ja standardiseerimiseks? Kui suur on ettevalmistusele kuluv aja- või rahaline ressurss?

> 1. Probleemide identifitseerimine, 2. Kindlad sammud iga probleemi mitigeerimiseks (NAN valued jne), 3. andmetüübid standardkujule. Ajaline ressurss ~20 tundi. Raha ei plaani kulutada, piirdume tasuta kättesaadavate mudleitega (GitHub copilot).

### 🟡 3.2 Tehisintellektispetsiifiline ettevalmistus
Kuidas andmed tehisintellekti mudelile sobivaks tehakse (nt tükeldamine, vektoriseerimine, metaandmete lisamine)?

> Iga aine kohta koostatakse üks tekstitükk (`rag_text`), mis sisaldab kõiki RAG-i jaoks olulisi välju struktureeritud siltidega (nt `Description (EN):`, `Objectives (ET):` jne). Tekst on kahekeelne (eesti ja inglise keel), et päringud mõlemas keeles leiaksid vasteid. Tükeldamist (chunking) ei kasutata, kuna `rag_text` pikkus jääb enamasti alla 3000 tähemärgi ning mahub mudeli kontekstiaknasse. Vektoriseerimisel kasutatakse mitmekeelset `sentence-transformers` mudelit (`paraphrase-multilingual-MiniLM-L12-v2`), mis toetab üle 50 keele sh eesti ja inglise keelt. Vektorid salvestatakse ChromaDB vektorandmebaasi koos metaandmetega (ainekood, EAP, semester, asukoht, õppetöö keel, õppeaste, hindamise tüüp), mis võimaldab filtreerimist otsingus.

<br>
<br>

## 🟢 4. Tehisintellekti rakendamine
Fookus: Tehisintellekti rakendamise süsteemi komponentide ja disaini kirjeldamine.

### 🟢 4.1 Komponentide valik ja koostöö
Millist tüüpi tehisintellekti komponente on vaja rakenduses kasutada? Kas on vaja ka komponente, mis ei sisalda tehisintellekti? Kas komponendid on eraldiseisvad või sõltuvad üksteisest (keerulisem agentsem disan)?

> Süsteem koosneb kolmest omavahel seotud komponendist. (1) **Embeddingu mudel** (`paraphrase-multilingual-MiniLM-L12-v2`): teisendab nii ainekirjeldused kui ka kasutaja päringu numbrilisteks vektoriteks. (2) **Vektorandmebaas** (ChromaDB): hoiab kõigi ainete vektoreid ja metaandmeid ning teostab semantilise lähimate naabrite otsingu. (3) **Kasutajaliides** (Streamlit): kuvab tulemused ja võimaldab filtreerimist (semester, keel, õppeaste). Komponendid on ahelseoses – kasutaja sisend vektoriseeritakse, seejärel tehakse ChromaDB-s semantiline otsing ning tulemused kuvatakse Streamlit rakenduses. Tulevikus saab ahelasse lisada LLM-i (nt Groq API), mis genereerib lühikese kokkuvõtliku soovituse otsingutulemuste põhjal.

### 🟢 4.2 Tehisintellekti lahenduste valik
Milliseid mudeleid on plaanis kasutada? Kas kasutada valmis teenust (API) või arendada/majutada mudelid ise?

> **Embeddingu mudel:** `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace, MIT litsents, ~118 MB), jookseb lokaalselt `sentence-transformers` teegi kaudu – ei nõua API võtit ega internetiühendust päringutel. **LLM soovituste genereerimiseks (tulevikus):** 

### 🟢 4.3 Kuidas hinnata rakenduse headust?
Kuidas rakenduse arenduse käigus hinnata rakenduse headust?

> Hindamine toimub käsitsi koostatud teststsenaariumitega. Näiteks: sisend "tahan õppida masinõpet" – kontrollitakse, et tulemuste hulgas on andmeteaduse ained (nt LTAT.02.002, LTAT.02.006). Teststsenaariumid katavad: (a) eestikeelne päring, (b) ingliskeelne päring, (c) filtri kombineerimine (nt "ingliskeelne kevadsemestri aine"), (d) ebatavaline/mitteotsene päring (nt "aine, kus õpitakse haiguste levikut modelleerima"). Hinnatakse, kas top-3 tulemus on sisulisel asjakohane. Lisaks kontrollitakse, et filtrid (semester, keel, õppeaste) töötavad korrektselt.

### 🟢 4.4 Rakenduse arendus
Milliste sammude abil on plaanis/on võimalik rakendust järk-järgult parandada (viibadisain, erinevte mudelite testimine jne)?

> Arendus toimub iteratiivselt. **1. samm (praegune seis):** semantiline otsing ChromaDB + Streamlit UI filtritega – toimib ilma LLM-ita. **2. samm:** LLM-i lisamine – mudel saab otsingutulemused kontekstina ja genereerib lühikese eestikeelse soovituse koos põhjendusega. **3. samm:** süsteemiprompt täiendatakse kaitsemeetmetega prompt injection vastu; lisatakse päringu tõlkimine (ET/EN) enne otsimist, et parandada mitmekeelsete päringute täpsust. **4. samm:** kasutajaliidese parandamine – tulemuste kuvamine kaardidena, ÕISi otselink, tagasiside nupp. Erinevaid embeddingu mudeleid ja LLM-e saab vahetada konfiguratsioonifailis.


### 🟢 4.5 Riskijuhtimine
Kuidas maandatakse tehisintellektispetsiifilisi riske (hallutsinatsioonid, kallutatus, turvalisus)?

> **Hallutsinatsioonid:** RAG arhitektuur piirab LLM-i väljundit – mudel saab vastata vektorotsingust leitud ainete põhjal ning ei tohiks välja mõelda olematuid aineid. Rakendus kuvab alati ka otsinguallikad (ainekoodid ja pealkirjad), et kasutaja saaks tulemuse üle kontrollida. **Prompt injection:** süsteemiprompti lisatakse juhis, et mudel vastab ainult ainete soovitamisega seotud küsimustele ja ignoreerib kõrvalisi käske. Kasutaja sisend sanatiseeritakse (pikkuspiirang, sõnakeelud). **Andmeleke:** kasutaja sisend saadetakse LLM API-le – kasutajat teavitatakse sellest liideses. Personaalset infot ei logita. **Kallutatus:** andmestik pärineb TÜ ÕIS-ist ja on neutraalne faktipõhine andmebaas, seega ideoloogiline kallutatus on madal. Küll aga võib andmestik olla kaldu ingliskeelsete ainete poole, kuna ingliskeelsed kirjeldused on täielikumad.

<br>
<br>

## 🔵 5. Tulemuste hindamine
Fookus: kuidas hinnata loodud lahenduse rakendatavust ettevõttes/probleemilahendusel?

### 🔵 5.1 Vastavus eesmärkidele
Kuidas hinnata, kas rakendus vastab seatud eesmärkidele?

> Rakenduse vastavust eesmärkidele hinnati käsitsi koostatud teststsenaariumitega. Testiti nelja tüüpi päringuid: (a) **eestikeelne otsepäring** – nt „tahan õppida masinõpet ja andmeanalüüsi" → top-3 tulemustes peaksid olema andmeteaduse/ML ained (nt LTAT.02.002, LTAT.02.006); (b) **ingliskeelne päring** – nt „natural language processing and text mining" → tulemustes peaksid olema NLP ained olenemata sellest, kas aine kirjeldus on eesti- või ingliskeelne; (c) **filtri kombineerimine** – nt kevadsemestri ingliskeelne bakalaureuse aine → filtrid piiravad tulemuste hulga korrektselt; (d) **kaudne/ebatavaline päring** – nt „aine, kus õpitakse haiguste levikut modelleerima" → tulemus peaks sisaldama epidemioloogia või matemaatilise modelleerimise aineid. Kõigil neljal juhul tagastas rakendus sisulisel asjakohased top-3 tulemused, mis vastab seatud edukuse mõõdikule. Rakendus teavitab ka juhul, kui filtritega aineid ei leidu. Suurim avastatud piirang: lühikesed või äärmiselt üldised päringud (nt „aine") annavad sarnasuse skoori osas nõrgemaid tulemusi, kuid semantiliselt siiski mõistlikud tulemused.

<br>
<br>

## 🟣 6. Juurutamine
Fookus: kuidas hinnata loodud lahenduse rakendatavust ettevõttes/probleemilahendusel?

### 🟣 6.1 Integratsioon
Kuidas ja millise liidese kaudu lõppkasutaja rakendust kasutab? Kuidas rakendus olemasolevasse töövoogu integreeritakse (juhul kui see on vajalik)?

> Rakendus on kasutatav veebiliidesena Streamlit raamistiku kaudu. Praeguses arendusfaasis käivitatakse see lokaalselt käsuga `conda run -n oisi_projekt streamlit run app.py`. Kasutaja avab brauseris aadressi `http://localhost:8501`, sisestab oma õpihuvi kirjelduse (eesti või inglise keeles), valib soovi korral filtrid (semester, õppetöö keel, õppeaste) ja saab tulemuste nimekirja koos otselingiga ÕISi aineleheküljele. Rakendus ei nõua kasutajalt autentimist ega ole seotud TÜ süsteemidega – see toimib sõltumatult ainete andmestiku lokaalse koopiana. Produktsioonilahendusena saaks rakenduse juurutada Streamlit Community Cloudis (tasuta, avalik URL) või TÜ serveris, kus see oleks kättesaadav kõigile tudengitele ilma lokaalse paigalduseta.

### 🟣 6.2 Rakenduse elutsükkel ja hooldus
Kes vastutab süsteemi tööshoidmise ja jooksvate kulude eest? Kuidas toimub rakenduse uuendamine tulevikus?

> Praeguses mahus vastutab rakenduse eest projekti looja. Jooksvad kulud on sõltuvad arhitektuurist ja kasutusest: embeddingu mudel jookseb lokaalselt (tasuta), ChromaDB on lokaalne failisüsteem (tasuta) ning Streamlit Community Cloud on tasuta kuni teatud limiidini. Andmestik pärineb TÜ ÕISist – see vajab perioodilist uuendamist (nt iga semestri alguses), et kuvada ajakohast ainepakkumist. Uuendusprotsess: (1) tõmmata uus andmestik ÕISi APIst, (2) käivitada `andmete_ettevalmistus.ipynb` uuesti, (3) käivitada `build_vectorstore.py` uuesti, mis ehitab ChromaDB kollektsiooni nullist üles. Kogu protsess on automatiseeritav skriptiga. Mudeli vahetamine (nt parema embeddingu mudeli kasutuselevõtt) nõuab ainult `build_vectorstore.py` konfiguratsiooni muutmist ja vektorite ümberehitamist.