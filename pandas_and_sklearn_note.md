## 需要的套件
```shell
!pip install bs4
!pip install requests
!pip install lxml
!pip install pandas
!pip install sklearn
!pip install numpy
```
雖然好像Colab大部分都有就是了..

## fetch網站資料的範例
```python
import requests
from bs4 import BeautifulSoup
import datetime
import pandas as pd

# 爬某個網頁
response = requests.get('https://某個網頁')

soup = BeautifulSoup(response.text, features='lxml')
table = soup.find('table', {'class': 'HTML類'})
rows = table.find_all('tr')[1:] # 去除表頭

target_data = []

# 假裝這是個並排的表格，讓他跑兩次，雖然有點沒效率就是了=_=
for row in rows:
    cols = row.find_all('td')
    code = cols[1].text.strip().split()[0] # 分割後會得到一個字串列表，我只要第一個元素
    try:
        name = cols[2].text.strip().split()[0]
    except:
        continue
    target_data.append((code, name))

for row in rows:
    cols = row.find_all('td')
    code = cols[5].text.strip().split()[0]
    try:
        name = cols[6].text.strip().split()[0]
    except:
        continue
    target_data.append((code, name))

df = pd.DataFrame(target_data, columns=['代號', '名稱'])
df.to_csv('target_output.csv', index=False, encoding='utf-8-sig')
```

## PD產新特徵欄位、打標籤 + 使用sklearn套用決策樹演算法 + 產圖跟擷取葉片/節點資訊（樹遍歷）
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.tree import _tree
import numpy as np

# 讀取資料
df = pd.read_csv("date_data_output.csv", index_col = 0)

# 增加漲跌百分比特徵欄位
for col in df.columns:
    df["變化百分比"] = df["濃度"].pct_change() * 100
    df["符合條件之標籤"] = (df["變化百分比"] >= 3).astype(int)

df = df.dropna()
display(df)

# 尚未做極值清理

# 選擇特徵和標籤
features = df.filter(regex="變化百分比").dropna()
labels = df["符合條件之標籤"].dropna()

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 建立決策樹模型 
clf = DecisionTreeClassifier(random_state=42)
model = clf.fit(X_train, y_train)

# 產生決策樹圖形（只是用來理解該樹的長相）
fig = plt.figure(figsize = (25, 20))
_ = tree.plot_tree(clf, feature_names = "per", class_names = "label", filled = True)

# 取得決策樹圖形上的資訊
tree_ = clf.tree_

def get_leaf_info(node_id):
    if tree_.children_left[node_id] == tree_.children_right[node_id]: # 遍歷到葉子
        value = tree_.value[node_id]
        samples = np.sum(value)
        class_distribution = value / samples
        return {'samples': samples, 'value':value, 'class_distribution': class_distribution}
    else: # 遍歷到節點
        left_child = tree_.children_left[node_id]
        right_child = tree_.children_right[node_id]

        threshold = tree_.threshold[node_id] # 特徵閥值
        value = tree_.value[node_id]
        samples = np.sum(value)
        class_distribution = value / samples
        #TODO 正負號跟class判斷還沒寫
        return {'threshold': threshold, 'samples': samples, 'value':value, 'class_distribution': class_distribution,'left': get_leaf_info(left_child), 'right': get_leaf_info(right_child)}

root_info = get_leaf_info(0)

def print_leaf_info(node_info, depth=0):
    if 'threshold' not in node_info:
        print(f"{'  ' * depth}Samples: {node_info['samples']},Value: {node_info['value']}, 類別分佈: {node_info['class_distribution']}")
    else:
        print(f"P: {node_info['threshold']}, Samples: {node_info['samples']},Value: {node_info['value']}, 類別分佈: {node_info['class_distribution']}")
        print_leaf_info(node_info['left'], depth + 1)
        print_leaf_info(node_info['right'], depth + 1)

print_leaf_info(root_info)
```
