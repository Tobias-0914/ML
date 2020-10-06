# Titanic 生存模型预测


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
```

# 特征选择
## 数据总览

  首先需要查看数据集的基本信息，即train.csv和test.csv


```python
train_data = pd.read_csv('./titanic/train.csv')
train_data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>



```python
test_data = pd.read_csv('./titanic/test.csv')
test_data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>



```python
print(">> Information of train data:")
train_data.info()
print(">> Information of test data:")
test_data.info()
```

    >> Information of train data:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    >> Information of test data:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  418 non-null    int64  
     1   Pclass       418 non-null    int64  
     2   Name         418 non-null    object 
     3   Sex          418 non-null    object 
     4   Age          332 non-null    float64
     5   SibSp        418 non-null    int64  
     6   Parch        418 non-null    int64  
     7   Ticket       418 non-null    object 
     8   Fare         417 non-null    float64
     9   Cabin        91 non-null     object 
     10  Embarked     418 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB


  由上可见，数据集包含12个属性，1309条数据，其中891条为训练数据，418条为测试数据
- PassengerId 整型变量，标识乘客的ID，递增变量，对预测无帮助
- Survived 整型变量，标识该乘客是否幸存，0表示遇难，1表示幸存
- Pclass 整型变量，标识船舱等级，从1至3等级由高到低
- Name 字符型变量，除包含姓和名以外，还包含Mr. Mrs. Dr.这样的具有西方文化特点的信息
- Sex 字符型变量，标识乘客性别
- Age 整型变量，标识乘客年龄（有缺失值）
- SibSp 整型变量，代表兄弟姐妹及配偶的个数：Sib代表Sibling，即兄弟姐妹，Sp代表Spouse，即配偶
- Parch 整型变量，代表父母或子女的个数：其中Par代表Parent，即父母，Ch代表Child，即子女
- Ticket 字符型变量，代表乘客的船票号
- Fare 数值型，代表乘客的船票价（测试集中有缺失值）
- Cabin 字符型，代表乘客所在的舱位（有缺失值）
- Embarked 字符型，代表乘客登船口岸 C：Cherbourg；Q：Queenstown；S：Southampton（训练集中有缺失值）

## 数据清洗与初步处理

通过总览可知，数据集中的Age、Fare、Cabin、Embarked属性存在缺失值，需要对缺失值进行处理，同时对Sex和Embarked属性进行定性转换的预处理

1. 训练集中的Embarked属性和测试集中的Fare属性中只有1-2个缺失值，可以对缺失值填充均值或众数


```python
# 通过众数对Embarked属性中的缺失值进行赋值
print(train_data['Embarked'].value_counts())
print(test_data['Embarked'].value_counts())
```

    S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64
    S    270
    C    102
    Q     46
    Name: Embarked, dtype: int64



```python
train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].fillna('S')
```


```python
# 通过训练集和测试集中Fare的平均值对测试集中Fare属性中的缺失值进行赋值
sum1, sum2 = train_data['Fare'].sum(), test_data['Fare'].sum()
num1, num2 = train_data['Fare'].count(), test_data['Fare'].count()
test_data['Fare'] = test_data['Fare'].fillna((sum1+sum2)/(num1+num2))
```

2. 训练集和测试集中的Cabin属性均存在较多缺失值，考虑到缺失本身也可能代表着隐含信息（Cabin缺失可能代表没有船舱），暂时将缺失值均赋为“X”


```python
train_data['Cabin'] = train_data['Cabin'].fillna('X')
test_data['Cabin'] = test_data['Cabin'].fillna('X')
```

3. Age在该数据集里是一个较为重要的特征（后面会对此进行分析说明），其中缺失值的填充可能会对分类结果产生较大影响，因此可以使用数据完整的条目作为模型的训练集，以此来预测缺失值的Age数值。

​       对于当前的数据集可以使用随机森林算法进行预测，同时为满足sklearn只能处理数值属性的条件，将需要使用的非数据特征转换为数值特征，选取转换后数据集中的数值属性作为特征进行预测


```python
# 选择用来预测Age缺失值的属性，其中“Sex”，“Embarked”为文本类数据，需要通过factorize()方法进行转化
age_train = train_data[['Age','Pclass','SibSp','Parch','Fare','Sex','Embarked']]
age_test = test_data[['Age','Pclass','SibSp','Parch','Fare','Sex','Embarked']]
# 对Age和Embarked属性进行定性转换
age_train['Sex'] = age_train['Sex'].replace({'male': 0, 'female': 1})
age_test['Sex'] = age_test['Sex'].replace({'male': 0, 'female': 1})
age_train['Embarked'] = age_train['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
age_test['Embarked'] = age_test['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})

# print(age_train.head(10))
# print(age_test.head(10))

# 对照训练集和测试集原数据得到转换前后Sex属性中数据对应关系为  male:0, female:1
# Embarked属性中数据转换前后对应关系为  S:0, C:1, Q:2
```


```python
print(age_train.info())
print(age_test.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Age       714 non-null    float64
     1   Pclass    891 non-null    int64  
     2   SibSp     891 non-null    int64  
     3   Parch     891 non-null    int64  
     4   Fare      891 non-null    float64
     5   Sex       891 non-null    int64  
     6   Embarked  891 non-null    int64  
    dtypes: float64(2), int64(5)
    memory usage: 48.9 KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Age       332 non-null    float64
     1   Pclass    418 non-null    int64  
     2   SibSp     418 non-null    int64  
     3   Parch     418 non-null    int64  
     4   Fare      418 non-null    float64
     5   Sex       418 non-null    int64  
     6   Embarked  418 non-null    int64  
    dtypes: float64(2), int64(5)
    memory usage: 23.0 KB
    None


可以看到两个数据集中除Age之外的属性均完成数据清洗和初步转换（需要转换的前提下），下面通过随机森林算法对Age属性中的缺失值进行预测和填充


```python
# 将训练集和测试集中Age值缺失和未缺失的数据提取出来
age_train_notnull = age_train.loc[(age_train['Age'].notnull())]
age_train_isnull = age_train.loc[(age_train['Age'].isnull())]

age_test_notnull = age_test.loc[(age_test['Age'].notnull())]
age_test_isnull = age_test.loc[(age_test['Age'].isnull())]
```


```python
from sklearn.ensemble import RandomForestRegressor
# 首先对训练集中的Age属性中的缺失值进行预测和填充
X1 = age_train_notnull.values[:,1:]
Y1 = age_train_notnull.values[:,0]
# 通过随机森林模型训练年龄相关的属性数据
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X1,Y1)
predictAges1 = RFR.predict(age_train_isnull.values[:,1:])
age_train.loc[age_train['Age'].isnull(), ['Age']]= predictAges1
```


```python
# 按照相同的方法填充测试集中缺失的Age数据
X2 = age_test_notnull.values[:,1:]
Y2 = age_test_notnull.values[:,0]
RFR.fit(X2,Y2)
predictAges2 = RFR.predict(age_test_isnull.values[:,1:])
age_test.loc[age_test['Age'].isnull(), ['Age']]= predictAges2
```


```python
print(age_train.info())
print(age_test.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Age       891 non-null    float64
     1   Pclass    891 non-null    int64  
     2   SibSp     891 non-null    int64  
     3   Parch     891 non-null    int64  
     4   Fare      891 non-null    float64
     5   Sex       891 non-null    int64  
     6   Embarked  891 non-null    int64  
    dtypes: float64(2), int64(5)
    memory usage: 48.9 KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Age       418 non-null    float64
     1   Pclass    418 non-null    int64  
     2   SibSp     418 non-null    int64  
     3   Parch     418 non-null    int64  
     4   Fare      418 non-null    float64
     5   Sex       418 non-null    int64  
     6   Embarked  418 non-null    int64  
    dtypes: float64(2), int64(5)
    memory usage: 23.0 KB
    None


至此，训练集和测试集的数据均完成清洗与初步处理（在Age预测与填充中未使用到的Name，Ticket，Cabin属性此时均无数据缺失，但属于非结构文本数据，其处理与分析部分放置下面“关系分析与特征选择”部分），为之后的数据分析与进一步的存活预测奠定基础。

## 关系分析与特征选择


```python
# 首先将填充后的Age属性及Sex，Embarked属性填充至原数据集
train_data.drop(['Age','Pclass','SibSp','Parch','Fare','Sex','Embarked'], axis=1,inplace=True)
test_data.drop(['Age','Pclass','SibSp','Parch','Fare','Sex','Embarked'], axis=1,inplace=True)

train_data = train_data.join(age_train)
test_data = test_data.join(age_test)
# print(train_data.head(10), test_data.head(10), train_data.info(), test_data.info())
```

### 总体生存情况


```python
fig = px.bar(train_data['Survived'].value_counts())
fig.show()
```


```python
plt.figure(figsize=(6,6))
train_data['Survived'].value_counts().plot.pie(autopct='%.3f%%')
```

![27_1](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_27_1.png)

从训练集数据中的生存情况来看，存活人数占总人数比例将近四成

### 船舱等级和生存情况的关系分析


```python
fig = px.histogram(train_data, x='Survived', y='Pclass', color='Pclass');
fig.show()
plt.figure(figsize=(6, 5))
sns.barplot(x= 'Pclass', y='Survived', data=train_data)
plt.show()
```

![30_1](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_30_1.png)

只从存活人数来看，船舱等级为3的生存人数最多，但这很大一部分原因是该等级的乘客人数基数较大；

从各等级船舱的生存比例统计的中心趋势估计来看，不同等级船舱的生存率有多区别，船舱等级为1的乘客生存率最高，可见船舱等级对生存率具有一定影响

### 性别与生存情况的关系


```python
print(train_data.groupby(['Sex','Survived'])['Survived'].count())
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
```

    Sex  Survived
    0    0           468
         1           109
    1    0            81
         1           233
    Name: Survived, dtype: int64

![19_0](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_19_0.png)

不同性别的生存比例有较大区别，表现性别对生存情况有较大的影响，这可能与传统的“女士优先”思想有关

### 年龄与生存情况的关系


```python
# 绘制不同年龄下的生存情况的分布图
fig, axis1 = plt.subplots(1,1,figsize=(18,6))
train_data["Age_int"] = train_data["Age"].astype(int)
average_age = train_data[["Age_int", "Survived"]].groupby(['Age_int'],as_index=False).mean()
sns.barplot(x='Age_int', y='Survived', data=average_age)
```


![36_1](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_36_1.png)

```python
train_data['Age'].describe()
```


    count    891.000000
    mean      29.561919
    std       13.736080
    min        0.420000
    25%       21.000000
    50%       28.000000
    75%       37.000000
    max       80.000000
    Name: Age, dtype: float64

从不同年龄下的生存情况的分布情况来看，总体上呈现年幼人群生存概率高的分布

根据年龄数据分布，训练集的891个样本年龄数据的平均值约为30岁，标准差为13.5岁，最小年龄为0.42，最大年龄为80

进一步分析中，参考年龄分布将乘客进行群体划分为五个群体，分析群体的生存情况


```python
gro = [0, 12, 18, 47, 64, 81]
train_data['Age_group'] = pd.cut(train_data['Age'], gro)
by_age = train_data.groupby('Age_group')['Survived'].mean()
print(by_age)
```

    Age_group
    (0, 12]     0.525641
    (12, 18]    0.405063
    (18, 47]    0.359498
    (47, 64]    0.453488
    (64, 81]    0.090909
    Name: Survived, dtype: float64


从年龄群体来看，不同年龄群体的乘客生存率有明显区别，呈现双峰的基本分布，这可能与儿童与长者优先的习惯有关

### 亲属情况与生存情况的关系


```python
# 有无兄弟姐妹和配偶同行对生存情况的影响
sibsp_df = train_data[train_data['SibSp'] != 0]
no_sibsp_df = train_data[train_data['SibSp'] == 0]

plt.figure(figsize=(15,7))
plt.subplot(121)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%.2f%%')
plt.xlabel('SibSp')

plt.subplot(122)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%.2f%%')
plt.xlabel('no_SibSp')

plt.show()
```

![42_0](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_42_0.png)


```python
# 有无父母和子女同行对生存情况的影响
parch_df = train_data[train_data['Parch'] != 0]
no_parch_df = train_data[train_data['Parch'] == 0]

plt.figure(figsize=(15,7))
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('ParCh')

plt.subplot(122)
no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_ParCh')

plt.show()
```

![43_0](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_43_0.png)

有无兄弟姐妹和配偶以及有无父母和子女同行对生存情况存在明显的影响，说明亲属情况对生存情况有明显的影响

进一步分析亲属数量对生存情况的影响


```python
# 分别绘制SibSp与Parch数量与生存情况的关系
plt.figure(figsize=(18, 6))
plt.subplot(121)
sns.barplot(x = 'SibSp', y= 'Survived', data= train_data)
plt.title("SibSp-Survived")
plt.subplot(122)
sns.barplot(x = 'Parch', y= 'Survived', data= train_data)
plt.title("ParCh-Survived")
plt.show()
```

![45_0](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_45_0.png)


```python
# 绘制亲属总数量对生存情况的影响
train_data['Relatives'] = train_data['Parch'] + train_data['SibSp']
plt.figure(figsize=(8, 5))
sns.barplot(x = 'Relatives', y= 'Survived', data= train_data)
```

![46_1](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_46_1.png)

从亲属总数与生存概率的分布中可以得出，没有亲属或亲属数量较多的乘客群体生存率更低

### 票价与生存情况的关系


```python
# 绘制票价总体分布情况
plt.figure(figsize=(10, 5))
sns.distplot(train_data['Fare'])
train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
plt.show()
```

![49_0](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_49_0.png)

![49_1](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_49_1.png)

票价总体分布显示大部分票价位于[0,100]区间，与船舱等级分布相联系，说明船舱等级低的票更多且票价更便宜

进一步考虑生存与否与相应平均票价的关系，以得出票价对生存情况的影响


```python
fare_0 = train_data['Fare'][train_data['Survived'] == 0]
fare_1 = train_data['Fare'][train_data['Survived'] == 1]

average_fare = pd.DataFrame([fare_0.mean(), fare_1.mean()])
std_fare = pd.DataFrame([fare_0.std(), fare_1.std()])
average_fare.plot(yerr=std_fare, kind='bar', legend=False)

plt.show()
```

![51_0](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_51_0.png)

从上图得出，票价与生存与否有一定的相关性，生还者的平均票价要大于未生还者的平均票价。

### 登船港口与生存情况的关系


```python
plt.figure(figsize=(8, 5))
sns.barplot(x= 'Embarked', y = 'Survived', data= train_data)
```

![54_1](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_54_1.png)

泰坦尼克号从英国的南安普顿港出发，途径法国瑟堡和爱尔兰昆士敦，那么在昆士敦之前上船的人，有可能在瑟堡或昆士敦下船，这些人将不会遇到海难

由上图可以看出，在不同的港口上船，生还率不同，C(1)最高，Q(2)次之，S(0)最低

### 姓名、船舱位、船票号与生存情况的关系


```python
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_data['Title'], train_data['Sex'])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sex</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capt</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Col</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Countess</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Don</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Dr</th>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Jonkheer</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Lady</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Major</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>0</td>
      <td>182</td>
    </tr>
    <tr>
      <th>Mlle</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Mme</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>517</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>0</td>
      <td>125</td>
    </tr>
    <tr>
      <th>Ms</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Rev</th>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Sir</th>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
plt.figure(figsize=(8, 5))
train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()
```

![58_2](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_58_2.png)

通过观察名字数据，我们可以看出其中包括对乘客的称呼，如：Mr、Miss、Mrs等，称呼信息可能蕴含乘客的年龄、性别等信息，同时也包含了如社会地位等的称呼，如Dr、Lady、Major、Master等的称呼

不同称呼的乘客的生存情况分布有所不同，但对于大部分乘客通用的称呼，包括Master、Mr、Miss、Mrs，其对于生存情况分布与相应性别的生存情况分布基本吻合，因此在考虑简化特征工程的前提下可以不纳入分析

Ticket（船票号）、Cabin（船舱号）,这些因素的不同可能会影响乘客生存的情况。

Ticket属性值为字母加数字的文本数据，可能包含乘客位置、乘客之间的亲属关系等信息，但即使得出这些信息，也主要是对之前确定的船舱等级与亲属情况的补充，因此在简化特征工程的前提下可以不纳入分析

Cabin属性值为字母加数字的文本数据，同时存在大量缺失值，有效值仅仅有204个，很难分析出不同的船舱和存活的关系，且和Ticket属性类似，从中得出的信息可能是对船舱等级和分布位置的进一步解释，在简化特征工程的前提下也可以不纳入分析

## 特征工程

通过对训练集和测试集中数据属性的分析，最终选取的特征包括'Age','Pclass','SibSp','Parch','Fare','Sex','Embarked'，除去了‘Name’,‘Ticket’,‘Cabin’属性，并以此构建特征工程

### 特征处理

首先除去数据集中不需要的属性


```python
train_data.drop(['Name','Ticket','Cabin','Age_int','Age_group','Title'],axis=1,inplace=True)
test_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
```

在测试集中同样添加‘Relatives’属性


```python
test_data['Relatives'] = test_data['Parch'] + test_data['SibSp']
```


```python
print(train_data.info())
print(test_data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 10 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Age          891 non-null    float64
     3   Pclass       891 non-null    int64  
     4   SibSp        891 non-null    int64  
     5   Parch        891 non-null    int64  
     6   Fare         891 non-null    float64
     7   Sex          891 non-null    int64  
     8   Embarked     891 non-null    int64  
     9   Relatives    891 non-null    int64  
    dtypes: float64(2), int64(8)
    memory usage: 69.7 KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 9 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  418 non-null    int64  
     1   Age          418 non-null    float64
     2   Pclass       418 non-null    int64  
     3   SibSp        418 non-null    int64  
     4   Parch        418 non-null    int64  
     5   Fare         418 non-null    float64
     6   Sex          418 non-null    int64  
     7   Embarked     418 non-null    int64  
     8   Relatives    418 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 29.5 KB
    None


### 特征间相关性分析


```python
# 通过热图查看特征之间的相关性
plt.figure(figsize=(12, 8))
sns.heatmap(train_data.corr(), cmap='Reds', linewidths=1, annot=True, fmt='.3f')
fig=plt.gcf()
plt.show()
```

![70_0](https://github.com/Tobias-0914/Image/blob/main/Titanic/output_70_0.png)

# 模型应用与参数计算

## 决策树模型预测


```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
```


```python
x_train = train_data.drop(['PassengerId', 'Survived'], axis=1)
y_train = train_data['Survived']
x_test = test_data.drop(['PassengerId'], axis=1)
```


```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
pred_0 = model.predict(x_test)

# 再次通过训练集数据计算模型精确度
accu_0 = model.score(x_train, y_train)
print( "Model Prediction Score", (accu_0 * 100).round(3))
```

    Model Prediction Score 98.204

```python
dict = {
    'PassengerId' : test_data['PassengerId'],
    'Survived' : pred_0
        }
submission_DTC = pd.DataFrame(dict)
submission_DTC.to_csv('./submission_0.csv', index=False)
```

## 其他模型预测


```python
# 通过其他模型进行计算和预测
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
```


```python
def model_predict(models):
    score = []
    for mdl, filename in models:
        m = mdl
        m.fit(x_train, y_train)
        pred = m.predict(x_test)
        m_accur = m.score(x_train, y_train)
        score.append((m_accur * 100).round(3))
        
        dict = {
            'PassengerId' : test_data['PassengerId'],
            'Survived' : pred
                }
        new_submission = pd.DataFrame(dict)     
        new_submission.to_csv(filename, index=False)
        
    return score
```


```python
models = [
    (RandomForestClassifier(n_estimators=300, max_depth=20, random_state=5), 'DTC_submission.csv'),
    (RandomForestClassifier(), 'RFC_submission.csv'),
    (LogisticRegression(), 'LR_submission.csv'),
    (LinearSVC(), 'SVC_submission.csv'),
    (GaussianNB(), 'GNB_submission.csv'),
    (SGDClassifier(), 'SGD_submission.csv'),
    (KNeighborsClassifier(), 'KNC_submission.csv')
]

data = model_predict(models)
print("scores are", data)
```

    scores are [98.204, 98.204, 80.471, 76.094, 79.012, 70.146, 80.808]


## 参数调整与测试

对于泰坦尼克号乘客数据一类的数据集，数据量较小，通过设定参数调整范围，利用网格搜索器训练出相对最优的参数。该方法通过贪心算法，拿当前对模型影响最大的参数调优，直到最优化；再拿下一个影响最大的参数调优，如此下去，直到所有的参数调整完毕。


```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

rfc_model = RandomForestClassifier(random_state=45)

rfc_params_grid = {
    'n_estimators' : np.arange(500, 1000, 100),
#     'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],
    'max_depth' : np.arange(6, 8),
    'max_features': ['auto'],
    'criterion': ["gini", "entropy"]
}

gscv_random_classifier = GridSearchCV(estimator=rfc_model, param_grid=rfc_params_grid, cv=5)

gscv_random_classifier.fit(x_train, y_train)

pred = gscv_random_classifier.predict(x_test)

# print(accuracy_score(y_test, pred))
print(gscv_random_classifier.best_estimator_)
print(gscv_random_classifier.best_score_)
print(gscv_random_classifier.best_params_)
```

    RandomForestClassifier(criterion='entropy', max_depth=6, n_estimators=600,
                           random_state=45)
    0.8170924612390937
    {'criterion': 'entropy', 'max_depth': 6, 'max_features': 'auto', 'n_estimators': 600}

```python
# 形成最终的提交数据
m = RandomForestClassifier(criterion='entropy', max_depth=7, n_estimators=600, random_state=45)
m.fit(x_train, y_train)
pred = m.predict(x_test)


dict = {
    'PassengerId' : test_data['PassengerId'],
    'Survived' : pred
}

new_submission = pd.DataFrame(dict)
new_submission.to_csv('submission.csv', index=False)
```
