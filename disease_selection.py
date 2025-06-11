import pandas as pd

# Show all rows in terminal
pd.set_option("display.max_rows", None)

# Load the sample metadata
metadata = pd.read_csv("readed_data/sample_metadata.tsv", sep="\t")

# Normalize the phenotype names
metadata["phenotype"] = metadata["phenotype"].str.lower().str.strip()

# Count samples per phenotype
phenotype_counts = metadata["phenotype"].value_counts()

# Print the full list
print("🔍 Full phenotype sample counts:\n")
print(phenotype_counts)

# Save to CSV just in case
phenotype_counts.to_csv("phenotype_sample_counts.csv")


"""
Results:
phenotype
health                                            21090
healthy                                            5460
pouchitis                                          1886
irritable bowel syndrome                           1866
infant, premature                                  1400
migraine disorders                                 1235
lung diseases                                      1228
autoimmune diseases                                1154
enterocolitis, necrotizing                         1135
thyroid (usp)                                      1091
diarrhea                                           1066
obesity                                             906
crc                                                 843
crohn disease                                       747
bipolar disorder                                    697
constipation type 1 2                               683
colitis, ulcerative                                 601
depression                                          599
schizophrenia                                       582
constipation                                        561
colonic diseases                                    544
colorectal neoplasms                                531
normal                                              526
fungal overgrowth                                   482
inflammatory bowel disease                          452
hematologic neoplasms                               420
precursor cell lymphoblastic leukemia-lymphoma      406
attention deficit disorder with hyperactivity       393
clostridium difficile colitis                       388
stomach neoplasms                                   388
diabetes mellitus, type 2                           356
cdi                                                 346
inflammatory bowel diseases                         322
cardiovascular diseases                             310
diarrhea type 5 6 7                                 289
intestinal bacteria overgrow                        269
liver cirrhosis                                     268
diabetes mellitus                                   252
non-alcoholic fatty liver disease                   244
clostridium infections                              240
autism spectrum disorder                            235
arthritis, juvenile                                 233
adenoma                                             223
pregnant                                            207
hepatitis b virus                                   206
celiac disease                                      192
autistic disorder                                   189
liver diseases                                      186
ibs                                                 178
parkinson disease                                   176
ibd                                                 171
small adenoma                                       171
prediabetic state                                   166
kidney diseases                                     163
hypertension                                        156
uveomeningoencephalitic syndrome                    150
anorexia                                            147
asthma                                              145
t1d                                                 142
breast cancer.                                      133
gastroesophageal reflux                             129
hiv                                                 124
melanoma                                            122
infant, low birth weight                            116
epilepsy                                            108
cystic fibrosis                                     105
carcinoma                                           104
gdm                                                  98
spondylitis, ankylosing                              97
arthritis, rheumatoid                                96
helicobacter pylori                                  92
large adenoma                                        87
diabetes mellitus, type 1                            86
clostridium difficile infection                      85
clostridium difficile                                84
gallstones                                           82
hiv-1                                                79
short bowel syndrome                                 78
tuberculosis                                         76
metabolic syndrome                                   72
dermatitis                                           70
overweight                                           66
t2d                                                  63
nonibd                                               55
rotavirus infections                                 50
fussy                                                49
blastocystis infections                              48
severe acute malnutrition                            46
asd                                                  43
premature                                            39
graves disease                                       39
dermatitis, atopic                                   38
cholangitis, sclerosing                              38
recurrent clostridium difficile infection            38
rem sleep behavior disorder                          37
hashimoto disease                                    35
arthritis, reactive                                  32
cough                                                32
psoriasis                                            32
td                                                   31
hiv infections                                       30
thyroid neoplasms                                    30
pandas                                               30
symptomatic atherosclerosis                          27
spondylarthritis                                     27
crohn’s disease                                      26
ibs-d                                                25
behcet's disease                                     24
ulcerative colitis                                   22
clostridium difficile infectio                       20
healthy control for cdi                              20
adenomatous polyps                                   20
ibs-mix                                              18
gingivitis                                           16
moderate chronic periodontitis                       16
renal dialysis                                       15
peritoneal dialysis                                  15
anemia, sickle cell                                  14
alzheimer disease                                    13
phenylketonurias                                     11
fever                                                 9
severe chronic periodontitis                          9
ear infection                                         8
periodontal health                                    8
shiga-toxigenic escherichia coli                      7
fever, fussy                                          6
fever, ear infection, cough                           6
eczema                                                5
cough, eczema                                         5
ibs-c                                                 5
sepsis                                                4
cd                                                    4
ear infection, cough                                  3
fussy, cough                                          3
obesity, morbid                                       2
fever, ear infection                                  2
fever, ear infection, fussy                           2
fever, cough                                          1
1148 del47                                            1
rsv infection, cough, eczema                          1
ear infection, rsv infection, cough                   1
fussy, cough, eczema                                  1
c.808delc p.arg270glu fs 288x                         1
r255x mecp2                                           1
fever, fussy, cough, eczema                           1
fever, fussy, cough                                   1
fever, cough, eczema                                  1
fever, ear infection, fussy, cough                    1
r306c mecp2                                           1
r106w mecp2                                           1
r294x mecp2                                           1
r133c mecp2                                           1
deletion of exons 3,4 mecp2 gene                      1

Contenders:
- Health / healthy
- Metabolic: T2D
- Intestinal: Crohn's
- Mental: Depression
- Liver-related: Liver cirrhosis
"""