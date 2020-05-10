
Based on [Pandas DataFrame DataCamp tutorial](https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python) made by Karljin Willems (2019).


```python
import os
import numpy  as np
import pandas as pd
```

## Intro - Structured and record arrays with Numpy


```python
# A structured array
StrucArr = np.ones(3, dtype=([('foo', int), ('bar', float)]))

# Printing full array
print(StrucArr)
print()

# Printing the structured array -- foo values
print(StrucArr['foo'])
print()

# Printing the structured array -- bar values
print(StrucArr['bar'])

```

    [(1, 1.) (1, 1.) (1, 1.)]
    
    [1 1 1]
    
    [1. 1. 1.]



```python
# Create a record array
RecArr = StrucArr.view(np.recarray)

# Printing the record array 
# Access fields of structured arrays by attributes 
# rather than by index
print(RecArr.foo)
```

    [1 1 1]


## Creating Pandas DataFrame


```python
# From a numpy array
data = np.array([[    '', 'Col 1', 'Col 2'],
                 ['Row1',       1,       2],
                 ['Row2',       3,       4]])
print(data)
df1  = pd.DataFrame(data    = data[1:, 1:],
                    index   = data[1:,0],
                    columns = data[0,1:])
print()
display(df1) # only IPython
```

    [['' 'Col 1' 'Col 2']
     ['Row1' '1' '2']
     ['Row2' '3' '4']]
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Row1</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Row2</th>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
# From a 2D numpy array
Arr2D = np.array([[1, 2, 3], [4, 5, 6]])
display(pd.DataFrame(Arr2D))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



```python
# From a dict
Dict = {1 : ['1', '4'], 2 : ['2','5'], 3 : ['3', '6']}
display(pd.DataFrame(Dict))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



```python
# From a DataFrame
df2 = pd.DataFrame(data=[[1,2,3],
                         [4,5,6]], 
                   index=range(0,2),
                   columns=range(0,3))
display(pd.DataFrame(df2))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



```python
# From a pd.Series
series1 = pd.Series({"Belgium" : "Brusseuls",
                     "India"   : "New Delhi",
                     "Algeria" : "Algiers",
                     "USA"     : "Washington"})
display(pd.DataFrame(series1))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Belgium</th>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>India</th>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>Algiers</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>Washington</td>
    </tr>
  </tbody>
</table>
</div>


## DataFrame informations


```python
display(df2)

print("Shape of df2 (h,L)                 : ", df2.shape)
print("Height of df2 using len            :  ", len(df2.index))
print("Height of df2 using count (no NaN) :  ", df2[0].count())
print("Column values                      : ", list(df2.columns.values))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


    Shape of df2 (h,L)                 :  (2, 3)
    Height of df2 using len            :   2
    Height of df2 using count (no NaN) :   2
    Column values                      :  [0, 1, 2]


## Accessing a value


```python
df2.columns = ['A', 'B', 'C']
display(df2)

# Using iloc
print("With iloc : ", df2.iloc[0][2])

# Using loc
print("With loc  : ", df2.loc[0]['C'])

# Using iat
print("With iat  : ", df2.iat[0,2])

# Using at
print("With at   : ", df2.at[0, 'C'])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


    With iloc :  3
    With loc  :  3
    With iat  :  3
    With at   :  3


## Accessing a row or a column


```python
display(df2)

# Selecting first row using iloc
display("1st row using iloc", df2.iloc[0])

# Selecting 'B' column using loc
display("'B' column using loc", df2.loc[:,'B'])
# display("'B' column using loc", df2.loc[:]['B']) # Same result
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



    '1st row using iloc'



    A    1
    B    2
    C    3
    Name: 0, dtype: int64



    "'B' column using loc"



    0    2
    1    5
    Name: B, dtype: int64


## Adding an index to a DataFrame


```python
# Use an existing column as an index
df3 = df2.set_index('A')
display(df3)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>A</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


## Adding rows to a DataFrame


```python
df4 = pd.DataFrame(data=np.array([[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]]),
                   index= [2, 'A', 4],
                   columns=[48, 49, 50])
display(df4)

# Printing loc 2 -> loc look after the labels
print(df4.loc[2],"\n")

# Printing iloc 2 -> iloc look after the index number
print(df4.iloc[2])

# Printing ix 2
#print(df4.ix[2]) # Deprecated for pandas > 1.0.0
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>


    48    1
    49    2
    50    3
    Name: 2, dtype: int64 
    
    48    7
    49    8
    50    9
    Name: 4, dtype: int64



```python
# Adding a 4th row
df4.loc[5] = [2, 3, 4]
display(df4)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Adding a 5th row
df4.loc['Alger'] = [2, 3, 4]
display(df4)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Alger</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## Adding columns to a DataFrame


```python
# From the index
df4[51] = df4.index
display(df4)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Alger</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
    </tr>
  </tbody>
</table>
</div>



```python
# From a list
df4['test'] = [10, 11, 12, 13, 14]
display(df4)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>10</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>13</td>
    </tr>
    <tr>
      <th>Alger</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Using loc
df4.loc[:,53] = [15, 16, 'didi', 17, 18]
display(df4)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>test</th>
      <th>53</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>10</td>
      <td>15</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>11</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
      <td>didi</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>13</td>
      <td>17</td>
    </tr>
    <tr>
      <th>Alger</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>14</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Using a series
series2 =   pd.Series({2       : "Brusseuls",
                       "A"     : "New Delhi",
                       4       : "Algiers",
                       5       : "Washington",
                       "Alger" :  4})
df4.loc[:,54] = series2
display(df4)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>test</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>10</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>11</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
      <td>didi</td>
      <td>Algiers</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>13</td>
      <td>17</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>Alger</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>14</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## Reset the index of a DataFrame


```python
# Using reset index, dropping the index column
df5 = df4.reset_index(level=0, drop=True)
display(df5)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>test</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>10</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>11</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
      <td>didi</td>
      <td>Algiers</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>13</td>
      <td>17</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>14</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## Delete indices, rows or columns from a DataFrame

### Deleting an index


```python
display(df4)
# Can only reset the index
df5 = df4.reset_index(level=0, drop=True)
display(df5)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>test</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>10</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>11</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
      <td>didi</td>
      <td>Algiers</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>13</td>
      <td>17</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>Alger</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>14</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>test</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>10</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>11</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
      <td>didi</td>
      <td>Algiers</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>13</td>
      <td>17</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>14</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
# or remove de index name if there is any
# del df4.index.name

# or remove duplicate index values
df6 = pd.DataFrame(data   = np.array([[ 1,   2,  3],
                                      [ 4,   5,  6],
                                      [ 7,   8,  9],
                                      [40,  50, 60],
                                      [23,  35, 37]]), 
                  index   = [2.5, 12.6, 4.8, 4.8, 2.5], 
                  columns = [48, 49, 50])
display(df6)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2.5</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12.6</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4.8</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4.8</th>
      <td>40</td>
      <td>50</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2.5</th>
      <td>23</td>
      <td>35</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Original DataFrame
display('Original DF',df6)

# Start by resetting index
df7 = df6.reset_index()
display('Reset index',df7)

# Drop DataFrame duplicates (opt. keeping last values)
df7 = df7.drop_duplicates(subset='index', keep='last')
display('Remove index col duplicates', df7)

# Setting the no duplicates index as the new index
df7 = df7.set_index('index')
display('Set new index', df7)
```


    'Original DF'



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2.5</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12.6</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4.8</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4.8</th>
      <td>40</td>
      <td>50</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2.5</th>
      <td>23</td>
      <td>35</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>



    'Reset index'



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.6</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.8</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.8</td>
      <td>40</td>
      <td>50</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.5</td>
      <td>23</td>
      <td>35</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>



    'Remove index col duplicates'



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>12.6</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.8</td>
      <td>40</td>
      <td>50</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.5</td>
      <td>23</td>
      <td>35</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>



    'Set new index'



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12.6</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4.8</th>
      <td>40</td>
      <td>50</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2.5</th>
      <td>23</td>
      <td>35</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>


### Deleting a column from a DataFrame


```python
display(df4)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>test</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>10</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>11</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>12</td>
      <td>didi</td>
      <td>Algiers</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>13</td>
      <td>17</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>Alger</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>14</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Drop the column with label 'test'
# axis = 0 -> row
# axis = 1 -> column
df8 = df4.drop('test', axis=1, inplace=False)
display(df8)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>didi</td>
      <td>Algiers</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>17</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>Alger</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


### Deleting a row from a DataFrame


```python
display(df8)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>didi</td>
      <td>Algiers</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>17</td>
      <td>Washington</td>
    </tr>
    <tr>
      <th>Alger</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Drop duplicates existing in one column
df9 = df8.drop_duplicates([48], keep='last', inplace=False)
display(df9)

# Drop duplicates using an index position -- last one
df10 = df9.drop(df9.index[2], inplace=False)
df10.reset_index(drop=True,inplace=True)
display(df10)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>A</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>didi</td>
      <td>Algiers</td>
    </tr>
    <tr>
      <th>Alger</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## Rename the index or columns of a Pandas DataFrame


```python
display(df10)

# Renaming columns
newCols = {48 : 'Col 1',
           49 : 'Col 2',
           50 : 'Col 3',
           51 : 'Col 4',
           53 : 'Col 5',
           54 : 'Col 6'}

newRows = {0 : 'A',
           1 : 'B',
           2 : 'C'}

df10.rename(columns = newCols, inplace=True)

# Renaming index rows
df10.rename(index = newRows, inplace=True)
display(df10)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>53</th>
      <th>54</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Col 1</th>
      <th>Col 2</th>
      <th>Col 3</th>
      <th>Col 4</th>
      <th>Col 5</th>
      <th>Col 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>15</td>
      <td>Brusseuls</td>
    </tr>
    <tr>
      <th>B</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>A</td>
      <td>16</td>
      <td>New Delhi</td>
    </tr>
    <tr>
      <th>C</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>Alger</td>
      <td>18</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## Format the data in a DataFrame

### Replacing occurences of a string in a DataFrame


```python
# From a numpy array
data_stud = np.array([[        '', 'Stud. 1',    'Stud. 2', 'Stud. 3'],
                      ['Lesson 1',   'Awful\n',      'Awful', 'Perfect+-ZZz'],
                      ['Lesson 2',    'Poor\n',    'Perfect\n',    '+-ZZzOK'],
                      ['Lesson 3',      'OK\n', 'Acceptable', 'Perfect+-ZZz']])

df11  = pd.DataFrame(data    = data_stud[1:, 1:],
                     index   = data_stud[1:,0],
                     columns = data_stud[0,1:])
display(df11)

# Delete unwanted parts from the strings in the Stud. 3 column
df11['Stud. 3'] = df11['Stud. 3'].map(lambda x : x.lstrip('+-Zz').rstrip('+-Zzz'))
display(df11)

# Remove \n using regex option
df11.replace({'\n': ''}, regex=True, inplace=True)
display(df11)

# Replacing strings by numerical values (0 to 4)
df11.replace(['Awful', 'Poor', 'OK', 'Acceptable', 'Perfect'],
             [      0,      1,    2,            3,         4],
             inplace=True)
display(df11)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Stud. 1</th>
      <th>Stud. 2</th>
      <th>Stud. 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lesson 1</th>
      <td>Awful\n</td>
      <td>Awful</td>
      <td>Perfect+-ZZz</td>
    </tr>
    <tr>
      <th>Lesson 2</th>
      <td>Poor\n</td>
      <td>Perfect\n</td>
      <td>+-ZZzOK</td>
    </tr>
    <tr>
      <th>Lesson 3</th>
      <td>OK\n</td>
      <td>Acceptable</td>
      <td>Perfect+-ZZz</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Stud. 1</th>
      <th>Stud. 2</th>
      <th>Stud. 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lesson 1</th>
      <td>Awful\n</td>
      <td>Awful</td>
      <td>Perfect</td>
    </tr>
    <tr>
      <th>Lesson 2</th>
      <td>Poor\n</td>
      <td>Perfect\n</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>Lesson 3</th>
      <td>OK\n</td>
      <td>Acceptable</td>
      <td>Perfect</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Stud. 1</th>
      <th>Stud. 2</th>
      <th>Stud. 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lesson 1</th>
      <td>Awful</td>
      <td>Awful</td>
      <td>Perfect</td>
    </tr>
    <tr>
      <th>Lesson 2</th>
      <td>Poor</td>
      <td>Perfect</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>Lesson 3</th>
      <td>OK</td>
      <td>Acceptable</td>
      <td>Perfect</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Stud. 1</th>
      <th>Stud. 2</th>
      <th>Stud. 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lesson 1</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Lesson 2</th>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Lesson 3</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


### Splitting text in a column into multiple rows in a DataFrame


```python
# From a DataFrame
df12 = pd.DataFrame(data=[[34,  0,          '23:44:55'],
                          [22,  0,          '66:77:88'],
                          [19,  1, '43:68:05 56:34:12']], 
                   index=range(0,3),
                   columns=['Age', 'Plus One', 'Ticket'])
display(pd.DataFrame(df12))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Plus One</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>0</td>
      <td>23:44:55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>0</td>
      <td>66:77:88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19</td>
      <td>1</td>
      <td>43:68:05 56:34:12</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Split the ticket column to separate the tickets
ticket_series = df12['Ticket'].str.split(' ')
display(ticket_series)

# Transform to series -> Gives a double index
ticket_series = ticket_series.apply(pd.Series)
display(ticket_series)

# Stack the index
ticket_series = ticket_series.stack()
display(ticket_series)

# Get rid of the multi level index
ticket_series.index = ticket_series.index.droplevel(-1)
display(ticket_series)

# Make the series a dataframe 
ticketDF = pd.DataFrame(ticket_series)
ticketDF.rename(columns = {0 : 'Ticket'}, inplace=True)
display(ticketDF)

# Delete the `Ticket` column from your DataFrame
df13 = df12.drop('Ticket', axis=1, inplace=False)
display(df13)

# Join the ticket DataFrame to `df`
df13 = df13.join(ticketDF)
display(df13)
```


    0              [23:44:55]
    1              [66:77:88]
    2    [43:68:05, 56:34:12]
    Name: Ticket, dtype: object



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23:44:55</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>66:77:88</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43:68:05</td>
      <td>56:34:12</td>
    </tr>
  </tbody>
</table>
</div>



    0  0    23:44:55
    1  0    66:77:88
    2  0    43:68:05
       1    56:34:12
    dtype: object



    0    23:44:55
    1    66:77:88
    2    43:68:05
    2    56:34:12
    dtype: object



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23:44:55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>66:77:88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43:68:05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56:34:12</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Plus One</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Plus One</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>0</td>
      <td>23:44:55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>0</td>
      <td>66:77:88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19</td>
      <td>1</td>
      <td>43:68:05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19</td>
      <td>1</td>
      <td>56:34:12</td>
    </tr>
  </tbody>
</table>
</div>


### Applying a function to columns or rows of a DataFrame


```python
df14 = pd.DataFrame(data=np.array([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]]),
                   index   = range(0,3),
                   columns = range(0,3))
display(df14)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



```python
doubler = lambda x : 2*x

# Applying function to the 1st column of DataFrame
df14[0] = df14[0].apply(doubler)
display(df14)

# Applying function to the 1st row of DataFrame
df14.loc[0] = df14.loc[0].map(doubler) 
#df14.loc[0] = df14.loc[0].apply(doubler) # map or apply ...
display(df14)

# Applying function to each element of DataFrame
df14 = df14.applymap(doubler)
display(df14)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>8</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>10</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>16</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>


## Creating an empty DataFrame


```python
# Using numpy NaN data
df15 = pd.DataFrame(data    = np.nan,
                    index   = range(5),
                    columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G'])

display(df15)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Using 'dtype' option to select initialized data types
# 1 <int> becomes '1' <str>
df16 = pd.DataFrame(data    = 1,
                    index   = range(5),
                    columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                    dtype   = 'str')

display(df16)

print(type(df16.loc[2][2]))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


    <class 'str'>


## Reshaping pandas DataFrame

### Pivoting


```python
products = pd.DataFrame({'category': ['Cleaning', 'Cleaning', 'Entertainment', 
                                      'Entertainment', 'Tech', 'Tech'],
                        'store' : ['Walmart', 'Dia', 'Walmart', 'Fnac', 'Dia','Walmart'],
                        'price' : [11.42, 23.50, 19.99, 15.95, 55.75, 111.55],
                        'testscore' : [4, 3, 5, 7, 5, 8]})
display(products)

# Pivoting the DataFrame
# Category column -> Index
# No values specified -> Pivot by multiple columns
pivot_products_1 = products.pivot(index='category', columns='store')
display(pivot_products_1)

# Values taken as 'price' specified -> Price is the only value in the DataFrame
pivot_products_2 = products.pivot(index='category', columns='store', values='price')
display(pivot_products_2)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>store</th>
      <th>price</th>
      <th>testscore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cleaning</td>
      <td>Walmart</td>
      <td>11.42</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cleaning</td>
      <td>Dia</td>
      <td>23.50</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Entertainment</td>
      <td>Walmart</td>
      <td>19.99</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entertainment</td>
      <td>Fnac</td>
      <td>15.95</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tech</td>
      <td>Dia</td>
      <td>55.75</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tech</td>
      <td>Walmart</td>
      <td>111.55</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">price</th>
      <th colspan="3" halign="left">testscore</th>
    </tr>
    <tr>
      <th>store</th>
      <th>Dia</th>
      <th>Fnac</th>
      <th>Walmart</th>
      <th>Dia</th>
      <th>Fnac</th>
      <th>Walmart</th>
    </tr>
    <tr>
      <th>category</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cleaning</th>
      <td>23.50</td>
      <td>NaN</td>
      <td>11.42</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Entertainment</th>
      <td>NaN</td>
      <td>15.95</td>
      <td>19.99</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Tech</th>
      <td>55.75</td>
      <td>NaN</td>
      <td>111.55</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>store</th>
      <th>Dia</th>
      <th>Fnac</th>
      <th>Walmart</th>
    </tr>
    <tr>
      <th>category</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cleaning</th>
      <td>23.50</td>
      <td>NaN</td>
      <td>11.42</td>
    </tr>
    <tr>
      <th>Entertainment</th>
      <td>NaN</td>
      <td>15.95</td>
      <td>19.99</td>
    </tr>
    <tr>
      <th>Tech</th>
      <td>55.75</td>
      <td>NaN</td>
      <td>111.55</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Another example with duplicates
products = pd.DataFrame({'category': ['Cleaning', 'Cleaning', 'Cleaning', 'Entertainment', 
                                      'Entertainment', 'Tech', 'Tech'],
                        'store' : ['Walmart', 'Dia', 'Dia', 'Walmart', 'Fnac', 'Dia','Walmart'],
                        'price' : [11.42, 23.50, 29.00, 19.99, 15.95, 55.75, 111.55],
                        'testscore' : [4, 3, 5,6, 7, 5, 8]})
display(products)

# In case of duplicate entries (here 2xCleaning dia prices), pivot with these columns doesnt work
# pivot_products_2 = products.pivot(index='category', columns='store', values='price')
# display(pivot_products_2)

# Use pivot table instead, specifying  an aggregate function to use for duplicates
pivot_products_3 = products.pivot_table(index='category', columns='store', 
                                        values='price', aggfunc='mean')

# Display shows Dia 'cleaning' being taken as the mean of both the Dia 'Cleaning' category
display(pivot_products_3)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>store</th>
      <th>price</th>
      <th>testscore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cleaning</td>
      <td>Walmart</td>
      <td>11.42</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cleaning</td>
      <td>Dia</td>
      <td>23.50</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cleaning</td>
      <td>Dia</td>
      <td>29.00</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Entertainment</td>
      <td>Walmart</td>
      <td>19.99</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Entertainment</td>
      <td>Fnac</td>
      <td>15.95</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tech</td>
      <td>Dia</td>
      <td>55.75</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Tech</td>
      <td>Walmart</td>
      <td>111.55</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>store</th>
      <th>Dia</th>
      <th>Fnac</th>
      <th>Walmart</th>
    </tr>
    <tr>
      <th>category</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cleaning</th>
      <td>26.25</td>
      <td>NaN</td>
      <td>11.42</td>
    </tr>
    <tr>
      <th>Entertainment</th>
      <td>NaN</td>
      <td>15.95</td>
      <td>19.99</td>
    </tr>
    <tr>
      <th>Tech</th>
      <td>55.75</td>
      <td>NaN</td>
      <td>111.55</td>
    </tr>
  </tbody>
</table>
</div>


### Using stack and unstack <br/>

#### Stacking <br/> 

+ **Stacking** a DataFrame makes it taller : moves the innermost column index to become the innermost row index.<br/>
<br/>
+ **Stacking** returns a DataFrame with an index with a new innermost level of row labels.<br/><br/>


#### Unstacking <br/> 

+ Inverse of **stacking** : moves the innermost row index to become the innermost column index. <br/><br/>
+ [More infos about pivoting, stacking, unstacking ...](https://nikgrozev.com/2015/07/01/reshaping-in-pandas-pivot-pivot-table-stack-and-unstack-explained-with-pictures/)<br/><br/>

![Stacking and unstacking](/home/hjuba/ML_learning/git_ml_learning/pandas_notes/stack_unstack.png)



### Using melt <br/>

+ Useful in case there are many columns that are identifiers, while all other columns are measured variables. <br/><br/>

+ Measured variables are all "unpivoted" to the row axis. <br/><br/>

+ DataFrame becomes longer instead of wider.




```python
# The `people` DataFrame
people = pd.DataFrame({'FirstName' : ['John',   'Jane'],
                       'LastName'  : [ 'Doe', 'Austen'],
                       'Age'       : [    23,       27],
                       'BloodType' : [  'A-',     'B+'],
                       'Weight'    : [    90,       64],
                       'Height'    : [   175,      188]})
display(people)

melt_people = pd.melt(people, id_vars=['FirstName', 'LastName'], var_name='Measurements')
display(melt_people)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FirstName</th>
      <th>LastName</th>
      <th>Age</th>
      <th>BloodType</th>
      <th>Weight</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>23</td>
      <td>A-</td>
      <td>90</td>
      <td>175</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jane</td>
      <td>Austen</td>
      <td>27</td>
      <td>B+</td>
      <td>64</td>
      <td>188</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FirstName</th>
      <th>LastName</th>
      <th>Measurements</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>Age</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jane</td>
      <td>Austen</td>
      <td>Age</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>John</td>
      <td>Doe</td>
      <td>BloodType</td>
      <td>A-</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jane</td>
      <td>Austen</td>
      <td>BloodType</td>
      <td>B+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>John</td>
      <td>Doe</td>
      <td>Weight</td>
      <td>90</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Jane</td>
      <td>Austen</td>
      <td>Weight</td>
      <td>64</td>
    </tr>
    <tr>
      <th>6</th>
      <td>John</td>
      <td>Doe</td>
      <td>Height</td>
      <td>175</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jane</td>
      <td>Austen</td>
      <td>Height</td>
      <td>188</td>
    </tr>
  </tbody>
</table>
</div>


## Iterating over a DataFrame


```python
df17 = pd.DataFrame(data=np.array([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]]),
                  columns=['A', 'B', 'C'],
                  index=range(3))
display(df17)

# Iterating  = combination of a 'for' loop and 'iterrows'
# `row` is seen as a `series`
for idx, row in df17.iterrows():
    print(idx, (row['A'], row['B'], row['C']))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>


    0 (1, 2, 3)
    1 (4, 5, 6)
    2 (7, 8, 9)


## Exporting a DataFrame into a file <br/>

Many infos on IO tools (text, csv, HDF5, JSON, ...) in pandas can be found in the [pandas doc](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html).


```python
display(people)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FirstName</th>
      <th>LastName</th>
      <th>Age</th>
      <th>BloodType</th>
      <th>Weight</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>John</td>
      <td>Doe</td>
      <td>23</td>
      <td>A-</td>
      <td>90</td>
      <td>175</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jane</td>
      <td>Austen</td>
      <td>27</td>
      <td>B+</td>
      <td>64</td>
      <td>188</td>
    </tr>
  </tbody>
</table>
</div>


### Export to CSV format <br/>

Many other options can be obtained from pandas *to_csv()* method [doc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html).


```python
# Basic export
people.to_csv('people.csv')

# Change the delimiter
people.to_csv('people_sepTab.csv', sep='\t')

# Use specific character encoding
people.to_csv('people_sepTab_utf8.csv', sep='\t', encoding='utf-8')
```

### Export to Excel format <br/>

Many other options can be obtained from pandas *to_excel()* method [doc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_excel.html).


```python
# Create a writer
writer = pd.ExcelWriter('people.xlsx')
people.to_excel(writer, 'DataFrame')
writer.save()
```
