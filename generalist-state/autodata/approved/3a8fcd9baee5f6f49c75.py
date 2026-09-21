###################################################################################################################
# Chi-square tests of independence for global comparison of Emotion distribution followed by pairwise comparisons. 
# P-values for pairwise tests were adjusted using Holm correction.
# #Effect sizes were computed using Cramér’s V.
###################################################################################################################

import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import pandas as pd

# upload files
an = pd.read_excel("/Users/.../AN.xlsx")
bn = pd.read_excel("/Users/.../BN.xlsx")
control = pd.read_excel("/Users/.../Control.xlsx")

an["Corpus"] = "AN"
bn["Corpus"] = "BN"
control["Corpus"] = "Control"

df = pd.concat([an, bn, control], ignore_index=True)

##CHI-SQUARE + CRAMER'S V + residuals
def chi_square_test(data, row_var, col_var):
    table = pd.crosstab(data[row_var], data[col_var])
    chi2, p, dof, expected = chi2_contingency(table)
    
    n = table.sum().sum()
    k = min(table.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1)))
    
    # Standardized residuals
    residuals = (table - expected) / np.sqrt(expected)
    residuals = pd.DataFrame(residuals,
                             index=table.index,
                             columns=table.columns)
    
    return chi2, p, dof, cramers_v, table, residuals


# GLOBAL TEST
chi2_e, p_e, dof_e, v_e, table_e, residuals_e = chi_square_test(df, "Corpus", "Emotion")
print(table_e)
print(f"Chi2={chi2_e:.3f}, df={dof_e}, p={p_e:.5f}, Cramer's V={v_e:.3f}")

print(residuals_e.round(2))

# 4. TEST PAIRWISE
corpora_pairs = [("AN", "BN"),
                 ("AN", "Control"),
                 ("BN", "Control")]

p_values_sent = []
p_values_emot = []

print("\n=== PAIRWISE TESTS ===")

for c1, c2 in corpora_pairs:
    
    subset = df[df["Corpus"].isin([c1, c2])]
    
    # Sentiment
    chi2, p, dof, v, table = chi_square_test(subset, "Corpus", "Sentiment")
    p_values_sent.append(p)
    print(f"\nSentiment: {c1} vs {c2}")
    print(f"Chi2={chi2:.3f}, p={p:.5f}, V={v:.3f}")
    
    # Emotion
    chi2_e, p_e, dof_e, v_e, table_e, residuals_e = chi_square_test(subset, "Corpus", "Emotion")
    p_values_emot.append(p_e)
    print(f"Emotion: {c1} vs {c2}")
    print(f"Chi2={chi2_e:.3f}, p={p_e:.5f}, V={v_e:.3f}")

# =========================
# 5. HOLM CORRECTION
# =========================

print("\n=== HOLM CORRECTION ===")

corrected_sent = multipletests(p_values_sent, method='holm')
corrected_emot = multipletests(p_values_emot, method='holm')

print("Sentiment corrected p-values:", corrected_sent[1])
print("Emotion corrected p-values:", corrected_emot[1])
