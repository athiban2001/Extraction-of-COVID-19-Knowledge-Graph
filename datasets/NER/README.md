# Named Entity Recognition
Named Entity Recognition refers to finding named entities such as person's name, location and company name using classification techniques.

## **Datasets**
Named Entity Recognition datasets are presented in BIO scheme. B for Beginning of entity, I for Inside of entity and O for Outside of entity. Each word in the sentence is classified into one of B, I and O.

### **1. Disease Named Entity Recognition - NCBI Disease**

The NCBI disease corpus is fully annotated at the mention and concept level to serve as a research resource for the biomedical natural language processing community. Two-annotators are assigned per document (randomly paired) and annotations are checked for corpus-wide consistency of annotations. The available tags are B-Disease, I-Disease and O. The NCBI Disease article can be seen [here.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3951655/)

### **2. Chemical Named Entity Recognition - CHEMDNER**

The abstracts of the CHEMDNER corpus were selected to be representative for all major chemical disciplines. Each of the chemical entity mentions was manually labeled according to its structure-associated chemical entity mention (SACEM) class: abbreviation, family, formula, identifier, multiple, systematic and trivial. The available tags are B-Chemical, I-Chemical and O. The CHEMDNER article is given [here.](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2)

### **3. Protein Named Entity Recognition - JNLPBA**

The data came from the GENIA version 3.02 corpus. This was formed from a controlled search on MEDLINE using the MeSH terms human, blood cells and transcription factors. The available tags are B-Protein, I-Protein, B-DNA, I-DNA, B-RNA, I-RNA, B-cell_line, I-cell_line, B-cell_type, I-cell_type, O. The JNLPBA article is given [here.](https://dl.acm.org/doi/pdf/10.5555/1567594.1567610)

## **References**

<a id="1">[1]</a> 
Doğan RI, Leaman R, Lu Z. (2014) NCBI disease corpus: a resource for disease name recognition and concept normalization. J BiomedInform. 2014 Feb;47:1-10. doi: 10.1016/j.jbi.2013.12.006. Epub2014 Jan 3. PMID: 24393765; PMCID: PMC3951655

<a id="2">[2]</a> 
Krallinger M, Rabal O, Leitner F, Vazquez M, Salgado D, LuZ, Leaman R, Lu Y, Ji D, Lowe DM, Sayle RA, Batista-NavarroRT, Rak R, Huber T, Rocktäschel T, Matos S, Campos D, TangB, XuH, Munkhdalai T, Ryu KH, Ramanan SV, Nathan S, Žitnik S, BajecM, Weber L, Irmer M, Akhondi SA, Kors JA, Xu S, An X, Sikdar UK, Ekbal A, Yoshioka M, Dieb TM, Choi M, Verspoor K, KhabsaM, Giles CL, Liu H, Ravikumar KE, Lamurias A, Couto FM, Dai HJ, Tsai RT, Ata C, Can T, Usié A, Alves R, Segura-Bedmar I, MartínezP, Oyarzabal J, Valencia A. (2015) The CHEMDNER corpus of chemicals and drugs and its annotation principles. J Cheminform. 2015Jan19

<a id="3">[3]</a> 
Nigel Collier and Jin-Dong Kim. (2004) Introduction to the bio-entity recognition task at jnlpba. NLPBA/BioNLP.