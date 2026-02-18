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

> csv -> json -> wordtovec vektoriseerimine -> andmebaas

<br>
<br>

## 🟢 4. Tehisintellekti rakendamine
Fookus: Tehisintellekti rakendamise süsteemi komponentide ja disaini kirjeldamine.

### 🟢 4.1 Komponentide valik ja koostöö
Millist tüüpi tehisintellekti komponente on vaja rakenduses kasutada? Kas on vaja ka komponente, mis ei sisalda tehisintellekti? Kas komponendid on eraldiseisvad või sõltuvad üksteisest (keerulisem agentsem disan)?

> ...

### 🟢 4.2 Tehisintellekti lahenduste valik
Milliseid mudeleid on plaanis kasutada? Kas kasutada valmis teenust (API) või arendada/majutada mudelid ise?

> ...

### 🟢 4.3 Kuidas hinnata rakenduse headust?
Kuidas rakenduse arenduse käigus hinnata rakenduse headust?

> ...

### 🟢 4.4 Rakenduse arendus
Milliste sammude abil on plaanis/on võimalik rakendust järk-järgult parandada (viibadisain, erinevte mudelite testimine jne)?

> ...


### 🟢 4.5 Riskijuhtimine
Kuidas maandatakse tehisintellektispetsiifilisi riske (hallutsinatsioonid, kallutatus, turvalisus)?

> ...

<br>
<br>

## 🔵 5. Tulemuste hindamine
Fookus: kuidas hinnata loodud lahenduse rakendatavust ettevõttes/probleemilahendusel?

### 🔵 5.1 Vastavus eesmärkidele
Kuidas hinnata, kas rakendus vastab seatud eesmärkidele?

> ...

<br>
<br>

## 🟣 6. Juurutamine
Fookus: kuidas hinnata loodud lahenduse rakendatavust ettevõttes/probleemilahendusel?

### 🟣 6.1 Integratsioon
Kuidas ja millise liidese kaudu lõppkasutaja rakendust kasutab? Kuidas rakendus olemasolevasse töövoogu integreeritakse (juhul kui see on vajalik)?

> ...

### 🟣 6.2 Rakenduse elutsükkel ja hooldus
Kes vastutab süsteemi tööshoidmise ja jooksvate kulude eest? Kuidas toimub rakenduse uuendamine tulevikus?

> ...