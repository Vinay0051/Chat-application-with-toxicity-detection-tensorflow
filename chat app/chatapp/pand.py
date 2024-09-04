import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('train.csv')
data=data.drop('id',axis=1)
df=data.drop('comment_text',axis=1)



df.sum().plot(kind='bar', color='#86bf91', figsize=(10, 6))

# Set the title and labels
plt.title('Combined Histogram of Binary Columns')
plt.xlabel('Columns')
plt.ylabel('Frequency')

# Customize the x-axis ticks
plt.xticks(rotation=45, ha='right')

plt.savefig('combined_histogram.png', bbox_inches='tight')
plt.show()