# 2026-02-16
tavatekst, kui versioonis rohkem info siis versioon

ee vs eng?

System prompt tõlgib päringu kahte keelde ja teeb selle põhjal otsingu. 

elfiltrid:
viskame imelikud ained välja? mitmeosalised jne, duration_in semesters >1, Viskamien minema sessioonõppe ja täiendõppe (tehtud uues toorandmete failis), kaitsmine hindamine

kodutöös kui aeg hakkab otsa saama, siis rohkem ei lisa veerge

Jäta need, mis vaja, mitte nice to heav feature'd

descrbition, overview, code

n-tähtsat metaandmee filtrit: 
EAP'd, eristav/mitteeristav hindamine, keel, kevad/sügis, asukoht, eksam /jah/ei <- hindamiskriteeriumid, study level õppeaste

kirjeldus:
-eeldusained
-description
-õpiväljundid

Ise otsustad täpsed välja ja keele
võta võib välja jätta

Dokuemnteeri, mis sa sisse ja välja jätad oma rakendusse

Samm 6 iga aine kohta, kui mitu tähemärki on /et näha, ka son vaj achunkida)


) Nagu üks tudeng raporteeris, meil oli sügisene 6EAP Sissejuhatus andmeteadusesse aine CSV failist puudu. Minu tabelite ühendamise skripti süü. Nüüd uuendasin CSV faili moodle'is ja see peaks olema korras.
2) Väljad
2.1) "version__target__language__en" ei sisalda seda infot, mida me tahaks. "version__overview__study_languages" tundub pigem olema see, mida me õppetöö keele mõistmiseks tahame. 
2.2) "overview__learning_outcomes_text_et" ja "overview__objectives_text_et" on pooltel juhtudel tühjad. Tuleb välja, et ainuke usaldusväärne allikas eesmärkide ja väljundite jaoks on JSON formaadis väljad ( 'overview__objectives', 'version__overview__objectives') ja   ('overview__learning_outcomes', 'version__overview__learning_outcomes'). See tähendab tüütut JSONitest info välja võtmist. Keelemudelid aitavad.

JSONite väljundite kohta. 
Minu enda eeltöötluse skriptis on järgmised funktsioonid JSON väljade töötlemiseks

extract_prerequisites()    #tagastab teksti, mis sisaldab JA ja VÕI klausleid ning (ainekood, nimi) paare, näiteks: "(LOFY.04.073 (Kvantmehaanika) VÕI LTFY.04.001 (Kvantmehaanika))"  Seda teksti kasutab RAG, vast saab aru küll.
extract_languages_et()  # tagastab keelte loetelu eesti keeles, näiteks "eesti keel, inglise keel" või "norra keel, rootsi keel, taani keel"
extract_study_level() # tagastab loetelu õppeastetest tekstina, näiteks "magistriõpe, doktoriõpe"
extract_et_outcomes() # tagastab tekstina selle, mis oli "et" märkega väljadel)
extract_et_objectives() # tagastab tekstina selle, mis oli "et" märkega

Praegu on mu väljund ühe aine jaoks selline (mitte kenasti kuvatud)
[['OIEO.06.046' 'Rahvusvaheline eraõigus' 'Private International Law' 6.0 'kevad' 'Eristav (A, B, C, D, E, F, mi)' 'eesti keel' 'Tartu linn' 'magistriõpe' 'põimõpe' '' 'Kursuse raames käsitletakse rahvusvahelise eraõiguse põhiteemasid, terminoloogiat ja kujunemist, tutvustatakse ja õpetatakse praktikas kasutama Eestis kehtivaid rahvusvahelise kohtualluvuse ja kohaldatava õiguse kindlaksmääramise instrumente ja nende olulisemaid norme (Haagi konventsioonid, muud Eesti Vabariigile siduvaid välislepingud, EL õigusaktid ja riigisisene õigus).' 'Õppeaine üldiseks eesmärgiks on õpetada rahvusvahelise eraõiguse teoreetilisi alusteadmisi; selgitada rahvusvahelise eraõiguse põhimõtete ja põhiinstituutide kujunemist, nende olemust ja vastastikuseid seoseid; anda ülevaade rahvusvahelise eraõiguse põhiteemadest ja terminoloogiast; õpetada üliõpilasi kasutama rahvusvahelise eraõiguse instrumente, tegemaks kindlaks vaidluse rahvusvaheline kohtualluvus ja sellele kohaldatav õigus ning hindamaks välisriigi kohtulahendite tunnustamise ja täitmise võimalusi Eestis.' 'Kursuse lõpuks peab üliõpilane olema võimeline:, -\tselgitama rahvusvahelise eraõiguse põhiinstituutide ja üldpõhimõtete olemust ning nende vastastikuseid seoseid; , -\tmõistma rahvusvahelise eraõiguse erinevaid valdkondi ning neis kehtestatud normide kohaldamise spetsiifikat;, -\tkohaldama rahvusvahelise eraõiguse norme vastavate instituutidega seotud vaidluste lahendamisel (esitades korrektse õigusliku arvamuse);, -\tkasutama olemasolevat õiguskirjandust ja kohtupraktikat kaasuste lahendamisel;, -\trakendama erinevaid tõlgendusmeetodeid rahvusvahelise eraõiguse normi sisu ja eesmärkide selgitamiseks.' '100% kirjalik eksam.']]

Leppisime kokku järgmised asjad veergude osas:

Jätsime täpsed veerud lahtiseks, igaüks saab vastavalt oma tahtmisele teha täpse valiku, näiteks:
oma valik, kas teed jätad ainult eestikeelsed väljad või valid mõlemad keeled
oma valik, millisest väljast täpselt vajaliku info võtad (nt kirjeldus/description on erinevas osas)
oma valik, millised metatunnused tahad sisse jätta
Allpool on miinimum, mis võiks järgmiseks korraks ette valmistatud olla.
Eelfiltrid (millised ained üldse meie andmestikust välja jätta):
võtta ainult päevaõpe (see on juba tehtud andmestikus toorandmed_aasta.csv)
eemaldada ained, mille kestus on rohkem kui 1 semester
eemaldada kaitsmise ained
Hard-filtrid (mille abil teha ridade eelfiltreerimist enne RAGi otsingut):
eristav/mitteeristav
kevad/sügis
õppeaste (baka/magister jne)
keel (võib ka esialgu mitte teha, oma valik)
asukoht (võib ka esialgu mitte teha, oma valik)
hindamisest lugeda välja, kas eksam toimub (suht nišš, võib mitte teha)
Kirjeldusse lisatav info:
aine kirjeldus
eeldusained
õpiväljundid
...