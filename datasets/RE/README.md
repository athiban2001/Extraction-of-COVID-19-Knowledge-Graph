# Relation Extraction
Relation Extraction refers to finding relations between certain objects in a sentence such as chemical cures disease, gene altered by chemical etc.,

## **Datasets**
Relation Extraction datasets consists of overall text, entities from the text, relationships between entities and type of relationships. They are mostly split into training and testing themselves.

### **1. Chemical Disease Relation - BC5CDR**

Chemicals, diseases, and their relations are among the most searched topics by PubMed users worldwide (1-3) as they play central roles in many areas of biomedical research and healthcare such as drug discovery and safety surveillance. Although the ultimate goal in drug discovery is to develop chemicals for therapeutics, recognition of adverse drug reactions between chemicals and diseases. The only relation available is Chemical-Induced-Disease. The BC5CDR article is given [here.](https://pubmed.ncbi.nlm.nih.gov/27161011/)

### **2. Chemical Protein Relation - CHEMPROT**

It is a manually annotated corpus, the CHEMPROT corpus, where domain experts have exhaustively labeled:(a) all chemical and gene mentions, and (b) all binary relationships between them corresponding to a specific set of biologically relevant relation types (CHEMPROT relation classes). The aim of the CHEMPROT track is to promote the development of systems able to extract chemical-protein interactions of relevance for precision medicine, drug discovery as well as basic biomedical research. There are 10 relations in CHEMPROT. The CHEMPROT article is given [here.](https://academic.oup.com/nar/article/39/suppl_1/D367/2508509)

## **References**

<a id="1">[1]</a> 
Li J, Sun Y, Johnson RJ, Sciaky D, Wei CH, Leaman R, Davis AP, Mattingly CJ, Wiegers TC, Lu Z. BioCreative V CDR task corpus: a resource for chemical disease relation extraction. Database (Oxford). 2016 May 9;2016:baw068. doi: 10.1093/database/baw068. PMID: 27161011; PMCID: PMC4860626.

<a id="2">[2]</a> 
Olivier Taboureau, Sonny Kim Nielsen, Karine Audouze, Nils Weinhold, Daniel Edsgärd, Francisco S. Roque, Irene Kouskoumvekaki, Alina Bora, Ramona Curpan, Thomas Skøt Jensen, Søren Brunak, Tudor I. Oprea, ChemProt: a disease chemical biology database, Nucleic Acids Research, Volume 39, Issue suppl_1, 1 January 2011, Pages D367–D372, doi: 10.1093/nar/gkq906