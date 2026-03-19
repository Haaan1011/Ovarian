COLS_TO_DROP = [
    "ID",
    "Optime",
    "PRL",
    "T",
    "TC",
    "TG",
    "LDL",
    "HDL",
    "HCY",
    "CA125",
    "Insulin",
    "BUN",
    "FBG",
    "BMI",
    "A.panicillin",
    "A.cepha",
    "NG.DNA",
    "UUorMH.DNA",
    "Rh_neg",
    "Num.pretrigger",
]

PORRM_FEATURES = [
    "AMH",
    "AFC",
    "POIorDOR",
    "FSH",
    "Age",
    "P",
    "Weight",
    "DBP",
    "WBC",
    "ALT",
    "RBC",
    "Duration",
    "LH",
]

HORRM_FEATURES = [
    "AMH",
    "AFC",
    "FSH",
    "Age",
    "LH",
    "POIorDOR",
    "PCOS",
    "PLT",
    "Weight",
    "Duration",
]

OS_INTERVENTIONS = ["Protocol", "Initial.FSH", "Recombinant", "Use.LH"]
CATEGORICAL_COLS = ["POIorDOR", "PCOS", "Protocol", "Recombinant", "Use.LH"]
TARGET_POR = "POR"
TARGET_HOR = "HOR"
