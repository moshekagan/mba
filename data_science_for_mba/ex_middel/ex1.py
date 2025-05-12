import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# קריאת הקובץ
df = pd.read_csv("Bank.csv")

# שינוי שמות עמודות כך שלא יכילו נקודות
df.columns = [col.replace('.', '_') for col in df.columns]

# המרת ערכים בעמודת המין
df['SEX'] = df['SEX'].map({1: 'Female', 0: 'Male'})

# קיבוץ ממוצע שכר נוכחי לפי מין ודרגה
grouped = df.groupby(['SEX', 'SENIOR'])['SAL_NOW'].mean().reset_index()

# ציור גרף עמודות
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped, x='SENIOR', y='SAL_NOW', hue='SEX')
plt.title('ממוצע שכר נוכחי לפי דרגה ומין')
plt.xlabel('דרגה (SENIOR)')
plt.ylabel('שכר נוכחי ממוצע')
plt.legend(title='מין')
plt.tight_layout()
plt.show()