import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# שלב 1: טעינת הנתונים
df = pd.read_csv("Bank.csv")

# שלב 2: טיפול בשמות עמודות עם נקודות
df.columns = [col.replace('.', '_') for col in df.columns]

# שלב 3: מיפוי מין לערכים טקסטואליים (0 = Male, 1 = Female)
df['SEX'] = df['SEX'].map({0: 'Male', 1: 'Female'})

# שלב 4: בניית מודל רגרסיה ליניארית
# ננסה להסביר את השכר הנוכחי באמצעות מין, שכר התחלתי, גיל, ניסיון ודרגה
model = smf.ols('SAL_NOW ~ SEX + SAL_START + AGE + WORK_EXP + SENIOR', data=df).fit()

# שלב 5: הדפסת סיכום תוצאות הרגרסיה
print(model.summary())

# שלב 6: גרף רגרסיה של שכר נוכחי מול שכר התחלתי לפי מין
plt.figure(figsize=(10, 6))
sns.lmplot(data=df, x='SAL_START', y='SAL_NOW', hue='SEX', aspect=1.5)
plt.title('שכר נוכחי לעומת שכר התחלתי לפי מין (עם קווי רגרסיה)')
plt.xlabel('שכר התחלתי')
plt.ylabel('שכר נוכחי')
plt.tight_layout()
plt.show()