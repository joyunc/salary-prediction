"""
preprocessing.py
Data preprocessing pipeline replicating the original R code logic.
Handles missing value imputation, outlier removal, and feature encoding.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew


# ── Utility functions ────────────────────────────────────────────────────────

def get_mode(series: pd.Series):
    """Return the mode of a Series, ignoring NaN."""
    s = series.dropna()
    return s.mode()[0] if not s.empty else np.nan


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Numeric, |skewness| < 0.5  → mean
      - Numeric, |skewness| >= 0.5 → median
      - Categorical / object        → mode
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            sk = skew(df[col].dropna())
            fill = df[col].mean() if abs(sk) < 0.5 else df[col].median()
            df[col].fillna(round(fill, 2), inplace=True)
        else:
            df[col].fillna(get_mode(df[col]), inplace=True)
    return df


def remove_outliers_iqr(series: pd.Series) -> pd.Series:
    """Replace values outside [Q1-1.5*IQR, Q3+1.5*IQR] with NaN."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series.where(series.between(lower, upper), other=np.nan)


# ── Individual dataset loaders ───────────────────────────────────────────────

def load_parents_2005(path: str) -> pd.DataFrame:
    """
    TEPS 2005 parent survey.
    Keeps: stud_id, Monthly_Income (family monthly income, ordinal 1–6).
    """
    df = pd.read_csv(path, encoding="big5")
    income_col = df.columns[157]          # column index 158 in R (0-based → 157)
    df[income_col] = pd.to_numeric(df[income_col], errors="coerce")
    df[income_col] = df[income_col].where(df[income_col].isin(range(1, 7)))
    out = df[["stud_id", income_col]].copy()
    out = impute_missing(out)
    out.rename(columns={income_col: "Monthly_Income"}, inplace=True)
    out["Monthly_Income"] = out["Monthly_Income"].astype("category")
    return out


def load_student_2007(path: str) -> pd.DataFrame:
    """
    TEPS 2007 student survey.
    Keeps: ability tests, Educational_Expectations, Job_Expectations.
    Re-orders ordinal levels to match thesis coding.
    """
    df = pd.read_csv(path, encoding="big5")

    # Drop students who transferred (w4s391 == 1 in R, index 390)
    flag_col = df.columns[390]
    df = df[df[flag_col] != 1].copy()

    comp_col   = df.columns[15]   # Comprehensive test
    gen_col    = df.columns[18]   # General test
    math_col   = df.columns[21]   # Math test
    edu_col    = df.columns[284]  # Educational expectations
    job_col    = df.columns[291]  # Job expectations

    # Replace 97/99 with NaN for expectation columns (285–318 in R → 284–317)
    for idx in range(284, 318):
        col = df.columns[idx]
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(~df[col].isin([97, 99]))

    # Re-map Educational Expectations: original 1=don't know→ recode to highest
    edu_map = {6: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    df[edu_col] = df[edu_col].map(edu_map)

    # Re-map Job Expectations
    job_map = {1: 5, 2: 2, 3: 4, 4: 3, 5: 6, 6: 1}   # matches R recoding
    df[job_col] = df[job_col].map(job_map)

    # Convert test score columns to numeric (they come in as strings from CSV)
    for c in [comp_col, gen_col, math_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df[["stud_id", comp_col, gen_col, math_col, edu_col, job_col]].copy()
    out[gen_col] = remove_outliers_iqr(out[gen_col])
    out = impute_missing(out)
    out = out.rename(columns={
        comp_col: "Comprehensive_Test",
        gen_col:  "General_Test",
        math_col: "Math_Test",
        edu_col:  "Educational_Expectations",
        job_col:  "Job_Expectations",
    })
    for col in ["Educational_Expectations", "Job_Expectations"]:
        out[col] = out[col].astype("category")
    return out


def load_tepsb_2013(path: str) -> pd.DataFrame:
    """
    TEPS-B 2013 phone survey.
    Keeps: stud_id, Gender, Marital_Status_14.
    """
    df = pd.read_csv(path, encoding="big5")
    df = df[df.iloc[:, 1] == 2].copy()   # completed interviews only

    gender_col  = df.columns[2]
    marital_col = df.columns[7]

    # Gender: original 1=female,2=male → recode to 1=male,2=female
    df[gender_col] = pd.to_numeric(df[gender_col], errors="coerce")
    df[gender_col] = df[gender_col].map({1: 2, 2: 1})

    # Marital status: 1=unmarried,2/3=married, others→NaN
    df[marital_col] = pd.to_numeric(df[marital_col], errors="coerce")
    df[marital_col] = df[marital_col].map({1: 2, 2: 1, 3: 1})

    out = df[["stud_id", gender_col, marital_col]].copy()
    out = impute_missing(out)
    out.rename(columns={gender_col: "Gender", marital_col: "Marital_Status_14"}, inplace=True)
    out["Gender"] = out["Gender"].astype("category")
    out["Marital_Status_14"] = out["Marital_Status_14"].astype("category")
    return out


def _encode_occupation(series: pd.Series) -> pd.Series:
    """
    Convert raw ISCO-style 2-digit codes into 9 thesis categories (0–9).
    Matches R logic: single digit → 0, two-digit → first digit, else NaN.
    Then remap to thesis categories.
    """
    def extract_first_digit(x):
        try:
            x = int(x)
            if x < 10:
                return 0
            elif x < 100:
                return x // 10
            else:
                return np.nan
        except (ValueError, TypeError):
            return np.nan

    s = series.apply(extract_first_digit)
    remap = {1: 7, 2: 8, 3: 6, 4: 2, 5: 4, 6: 0, 7: 3, 8: 5, 9: 1, 0: 9}
    return s.map(remap).astype("category")


def _encode_residence(series: pd.Series, year: int) -> pd.Series:
    """Map region codes → 1=Central, 2=South, 3=East, 4=North, 5=Overseas."""
    if year == 2014:
        mapping = {
            **{k: 4 for k in [1,2,3,4,5,6]},       # North
            **{k: 1 for k in [7,8,9,10,11,12]},     # Central
            **{k: 2 for k in [13,14,15,16,17,18,19,20,23]},  # South
            21: 3, 22: 3,                             # East
            24: 5, 25: 5, 26: 5,                     # Overseas
        }
    else:  # 2019
        mapping = {
            **{k: 4 for k in [1,2,3,4,5,6]},
            **{k: 1 for k in [7,8,9,10,11]},
            12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 1,
            18: 3, 19: 3, 20: 2, 21: 5, 22: 5,
            23: 5, 24: 5,
        }
    return series.map(mapping).astype("category")


def load_tepsb_2014(path: str) -> pd.DataFrame:
    """
    TEPS-B 2014 face-to-face survey → Starting Salary model features + y1.
    Applies all business rules from the R script.
    """
    df = pd.read_csv(path, encoding="big5")

    # Keep only full-time workers (col 1370 != 1) and those who completed (col 3 == 1)
    df = df[df.iloc[:, 1369] != 1].copy()
    df = df[df.iloc[:, 2] == 1].copy()

    col = lambda i: df.columns[i - 1]   # 1-based helper matching R

    keep = {
        "stud_id":                  "stud_id",
        col(68):  "Education_Level_14",
        col(218): "GSAT_Score",
        col(1131):"Average_salary_y1",
        col(1138):"Professional_Knowledge_14",
        col(1302):"Happiness_14",
        col(1343):"School_Type_14",
        col(1347):"Major_Category_14",
        col(1355):"Employer_14",
        col(1357):"Number_of_Employees_14",
        col(1358):"Supervisory_Role_14",
        col(1359):"Occupation_14",
        col(1361):"Work_Hours_14",
        col(9):   "Residence_14",
    }
    df = df[list(keep.keys())].rename(columns=keep)

    # Education: keep only 5–8, recode to 1/2/3
    df["Education_Level_14"] = pd.to_numeric(df["Education_Level_14"], errors="coerce")
    df = df[df["Education_Level_14"].between(5, 8)]
    df["Education_Level_14"] = df["Education_Level_14"].map({5:1, 6:2, 7:3, 8:3})

    # GSAT score: 0–75
    df["GSAT_Score"] = pd.to_numeric(df["GSAT_Score"], errors="coerce")
    df.loc[df["GSAT_Score"] > 75, "GSAT_Score"] = np.nan

    # Salary: filter by min wage 2014=19273, remove outliers
    df["Average_salary_y1"] = pd.to_numeric(df["Average_salary_y1"], errors="coerce")
    df = df[(df["Average_salary_y1"] >= 19273) & (df["Average_salary_y1"] <= 9_999_990)]
    df["Average_salary_y1"] = remove_outliers_iqr(df["Average_salary_y1"])
    df = df.dropna(subset=["Average_salary_y1"])

    # Supervisory role: >1 & <10 → 2
    df["Supervisory_Role_14"] = pd.to_numeric(df["Supervisory_Role_14"], errors="coerce")
    df.loc[df["Supervisory_Role_14"].between(2, 9), "Supervisory_Role_14"] = 2

    # Professional knowledge: reverse scale (1=very unfit→5, 5=very fit→1)
    pk_map = {1:5, 2:4, 3:3, 4:2, 5:1}
    df["Professional_Knowledge_14"] = df["Professional_Knowledge_14"].map(pk_map)

    # Happiness: reverse scale
    hap_map = {1:4, 2:3, 3:2, 4:1}
    df["Happiness_14"] = df["Happiness_14"].map(hap_map)

    # School type recode → 1=public-night,2=private-night,3=private-day,4=public-day
    school_map = {1:4, 2:1, 4:2}
    df["School_Type_14"] = df["School_Type_14"].map(school_map)

    # Major category recode (11–19 raw codes → 1–9 thesis categories)
    major_raw_map = {
        11:6, 12:3, 13:5, 14:4, 15:7, 16:1, 17:8, 18:2, 19:9,
        8:3, 4:5, 3:7, 2:1, 6:8, 5:2, 1:9, 7:9
    }
    df["Major_Category_14"] = pd.to_numeric(df["Major_Category_14"], errors="coerce").map(major_raw_map)

    # Employer recode
    emp_map = {1:4, 2:3, 3:1, 4:5, 5:5, 6:2, 8:2, 9:2, 10:5}
    df["Employer_14"] = pd.to_numeric(df["Employer_14"], errors="coerce").map(emp_map)

    # Number of employees: 1–6 → 1 (small), 7–9 → 2 (large)
    df["Number_of_Employees_14"] = pd.to_numeric(df["Number_of_Employees_14"], errors="coerce")
    ne14 = df["Number_of_Employees_14"]
    ne14 = ne14.where(ne14.between(1, 9))   # keep only valid range, else NaN
    df["Number_of_Employees_14"] = ne14.map(lambda x: (1 if x <= 6 else 2) if pd.notna(x) else np.nan)

    # Occupation
    df["Occupation_14"] = _encode_occupation(df["Occupation_14"])

    # Work hours: max 168, remove outliers
    df["Work_Hours_14"] = pd.to_numeric(df["Work_Hours_14"], errors="coerce")
    df.loc[df["Work_Hours_14"] > 900, "Work_Hours_14"] = np.nan
    df["Work_Hours_14"] = remove_outliers_iqr(df["Work_Hours_14"])

    # Residence
    df["Residence_14"] = pd.to_numeric(df["Residence_14"], errors="coerce")
    df["Residence_14"] = _encode_residence(df["Residence_14"], 2014)

    # Set remaining columns to numeric/category and impute
    num_cols = ["GSAT_Score", "Work_Hours_14"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["stud_id", "Average_salary_y1"]]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    df = impute_missing(df)
    return df


def load_tepsb_2019(path: str, df_2014: pd.DataFrame) -> pd.DataFrame:
    """
    TEPS-B 2019 phone survey → Current Salary model features + y2.
    Merges with 2014 to validate/update Education_Level and Major_Category.
    """
    df = pd.read_csv(path, encoding="big5")
    df = df[df.iloc[:, 5] == 2].copy()   # completed interviews

    col = lambda i: df.columns[i - 1]

    # Residence (col 9)
    df["Residence_19"] = pd.to_numeric(df[col(9)], errors="coerce")
    df = df[~df["Residence_19"].isin([21, 22, 23, 24])]   # remove outlying islands
    df["Residence_19"] = _encode_residence(df["Residence_19"], 2019)

    # Merge with 2014 for Education_Level
    df = df.merge(
        df_2014[["stud_id", "Education_Level_14"]].rename(columns={"Education_Level_14": "edu14"}),
        on="stud_id", how="left"
    )
    df["edu14"] = pd.to_numeric(df["edu14"], errors="coerce")

    edu19_raw = pd.to_numeric(df[col(14)], errors="coerce")
    edu19_raw = edu19_raw.where(edu19_raw.between(5, 8))
    edu19_raw = edu19_raw.map({5:1, 6:2, 7:3, 8:3})
    # Rule: 2019 edu >= 2014 edu
    edu19 = edu19_raw.copy()
    edu19[edu19.isna()] = df.loc[edu19.isna(), "edu14"]
    invalid = edu19 < df["edu14"]
    edu19[invalid] = np.nan
    df["Education_Level_19"] = edu19
    df = df.dropna(subset=["Education_Level_19"])

    # Major category (cols 22 & 16 combined)
    maj22 = pd.to_numeric(df[col(22)], errors="coerce")
    maj16 = pd.to_numeric(df[col(16)], errors="coerce")
    maj_raw = maj22.where(maj22 < 90, maj16)
    major_raw_map = {1:6, 2:3, 3:5, 4:4, 5:7, 6:1, 7:8, 8:2, 9:9}
    df["Major_Category_19"] = maj_raw.map(major_raw_map).astype("category")

    # Marital status (col 35)
    mar = pd.to_numeric(df[col(35)], errors="coerce")
    mar = mar.map({2:1, 3:1}).fillna(
        df[col(34)].map({2: 2})   # if single in 2013 → still single
    )
    df["Marital_Status_19"] = mar.astype("category")

    # Employer (col 75 & 77)
    emp75 = pd.to_numeric(df[col(75)], errors="coerce")
    emp77 = pd.to_numeric(df[col(77)], errors="coerce")
    emp = emp75.copy()
    mask = (emp75 == 1) | (emp75 > 90)
    emp[mask & (emp77 == 1)] = 5
    emp[mask & (emp77 == 2)] = 2
    emp_map = {4:4, 3:3, 2:1, 5:5, 1:2}
    df["Employer_19"] = emp.map(emp_map).astype("category")

    # Number of employees (col 80)
    ne = pd.to_numeric(df[col(80)], errors="coerce")
    ne = ne.where(ne.between(1, 9))
    df["Number_of_Employees_19"] = ne.apply(
        lambda x: 1 if 1 <= x <= 6 else (2 if x > 6 else np.nan) if pd.notna(x) else np.nan
    ).astype("category")

    # Occupation (col 81)
    df["Occupation_19"] = _encode_occupation(df[col(81)])

    # Supervisory role (col 82)
    sr = pd.to_numeric(df[col(82)], errors="coerce")
    df["Supervisory_Role_19"] = sr.map({1:2, 2:1}).astype("category")

    # Work hours (col 85)
    wh = pd.to_numeric(df[col(85)], errors="coerce")
    wh[wh > 900] = np.nan
    df["Work_Hours_19"] = remove_outliers_iqr(wh)

    # Salary (col 86 & 87) — banded → midpoint conversion
    sal_raw = pd.to_numeric(df[col(86)], errors="coerce")
    sal_exact = pd.to_numeric(df[col(87)], errors="coerce")
    def decode_salary(row):
        b, e = row
        if b == 2:   return 5000
        if 3 <= b <= 20: return 12500 + (b - 3) * 5000
        if b == 21:  return e
        return np.nan
    sal = pd.DataFrame({'b': sal_raw, 'e': sal_exact}).apply(lambda r: decode_salary(r.values), axis=1)
    sal = sal.where(sal >= 23100)
    sal = remove_outliers_iqr(sal)
    df["Average_salary_y2"] = sal
    df = df.dropna(subset=["Average_salary_y2"])

    # Professional knowledge (col 88) — reverse
    pk = pd.to_numeric(df[col(88)], errors="coerce").map({1:5, 2:4, 4:2, 5:1})
    df["Professional_Knowledge_19"] = pk.astype("category")

    # Happiness (col 119) — reverse
    hap = pd.to_numeric(df[col(119)], errors="coerce").map({1:4, 2:3, 3:2, 4:1})
    df["Happiness_19"] = hap.astype("category")

    # School type (cols 21 & 23 → col 24 derived)
    # Simplified: use 2014 school type if available, else impute
    df["School_Type_19"] = df_2014.set_index("stud_id")["School_Type_14"].reindex(df["stud_id"].values).values

    final_cols = [
        "stud_id", "Residence_19", "Education_Level_19", "Major_Category_19",
        "School_Type_19", "Marital_Status_19", "Employer_19",
        "Number_of_Employees_19", "Occupation_19", "Supervisory_Role_19",
        "Work_Hours_19", "Average_salary_y2", "Professional_Knowledge_19",
        "Happiness_19",
    ]
    df = df[final_cols].copy()
    df = impute_missing(df)
    return df


def _validate(df: pd.DataFrame, name: str, expected_cols: list = None):
    """
    Print quick sanity-check info after each loading step.
    This does NOT prove correctness — only catches obviously broken results
    (empty dataframe, all-NaN columns, wildly wrong sample size).
    """
    print(f"\n--- Validate: {name} ---")
    print(f"  shape: {df.shape}")
    if df.shape[0] == 0:
        print("  ⚠️  WARNING: 0 rows! Check filtering logic / column indices.")
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"  ⚠️  WARNING: missing expected columns: {missing}")
    na_cols = df.columns[df.isna().all()].tolist()
    if na_cols:
        print(f"  ⚠️  WARNING: fully-NaN columns (likely wrong column index): {na_cols}")
    # Show value ranges for numeric columns — helps spot wrong column indices
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print(df[num_cols].describe().loc[["min", "max", "mean"]].round(1).to_string())


# ── Merge datasets ───────────────────────────────────────────────────────────

def build_datasets(data_dir: str, verbose: bool = True):
    """
    Load all raw CSVs and return two merged DataFrames:
      joinM1 → Starting Salary model
      joinM2 → Current Salary model

    Set verbose=True to print sanity-check stats after each step —
    useful for catching wrong column-index mappings before they
    silently corrupt downstream results.
    """
    p2005 = load_parents_2005(f"{data_dir}/2005_parents.csv")
    if verbose: _validate(p2005, "2005 Parents", ["Monthly_Income"])

    s2007 = load_student_2007(f"{data_dir}/2007_student.csv")
    if verbose: _validate(s2007, "2007 Student", ["Comprehensive_Test", "General_Test", "Math_Test"])

    t2013 = load_tepsb_2013(f"{data_dir}/cp2013.csv")
    if verbose: _validate(t2013, "2013 TEPS-B", ["Gender", "Marital_Status_14"])

    t2014 = load_tepsb_2014(f"{data_dir}/cpn2014.csv")
    if verbose: _validate(t2014, "2014 TEPS-B", ["Average_salary_y1", "Occupation_14"])

    t2019 = load_tepsb_2019(f"{data_dir}/cp2019.csv", t2014)
    if verbose: _validate(t2019, "2019 TEPS-B", ["Average_salary_y2", "Occupation_19"])

    base = p2005.merge(s2007, on="stud_id")
    base = base.merge(t2013[["stud_id", "Gender"]], on="stud_id")
    if verbose: _validate(base, "Base (merged 2005+2007+2013)")

    joinM1 = base.merge(t2014, on="stud_id")
    joinM2 = base.merge(t2014[["stud_id", "GSAT_Score"]], on="stud_id").merge(t2019, on="stud_id")

    if verbose:
        _validate(joinM1, "joinM1 (pre-filter)", ["Average_salary_y1"])
        _validate(joinM2, "joinM2 (pre-filter)", ["Average_salary_y2"])

    for df in [joinM1, joinM2]:
        df.drop(columns=["stud_id"], inplace=True, errors="ignore")

    if verbose:
        print(f"\n{'='*50}\n  FINAL SHAPES\n{'='*50}")
        print(f"  joinM1 (Starting Salary) : {joinM1.shape}")
        print(f"  joinM2 (Current Salary)  : {joinM2.shape}")

    return joinM1, joinM2
