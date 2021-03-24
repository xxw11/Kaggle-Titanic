# Kaggle Titanic

## 数据读入

```python
import numpy as np
import pandas as pd
df_train = pd.read_csv('./train.csv')
df_test  = pd.read_csv('./test.csv')
df_sub   = pd.read_csv('./gender_submission.csv')
```



![image-20210321101039348](D:\Typora\user-image\image-20210321101039348.png)

可以看到数据存在缺失值

![image-20210321101148643](D:\Typora\user-image\image-20210321101148643.png)

## 数据处理

### 删除部分columns

Name,Ticket,Cabin很难量化

其中Name与是否存活关系不大，Cabin很重要，但是缺失值太多

#### DataFrame.drop()

```python
df_train.drop(['Name','Ticket','Cabin'],axis = 1,inplace=True)
df_test.drop(['Name','Ticket','Cabin'],axis = 1,inplace=True)
```

```python
Signature:
df_train.drop(
    labels=None,
    axis=0,
    index=None,
    columns=None,
    level=None,
    inplace=False,
    errors='raise',
)
Docstring:
Drop specified labels from rows or columns.

Remove rows or columns by specifying label names and corresponding
axis, or by specifying directly index or column names. When using a
multi-index, labels on different levels can be removed by specifying
the level.

Parameters
----------
labels : single label or list-like
    Index or column labels to drop.
    
    
axis : {0 or 'index', 1 or 'columns'}, default 0
    Whether to drop labels from the index (0 or 'index') or columns (1 or 'columns').
    
    
index : single label or list-like
    Alternative to specifying axis (``labels, axis=0``
    is equivalent to ``index=labels``).
    
    
    .. versionadded:: 0.21.0
columns : single label or list-like
    Alternative to specifying axis (``labels, axis=1``
    is equivalent to ``columns=labels``).

    
    .. versionadded:: 0.21.0
level : int or level name, optional
    For MultiIndex, level from which the labels will be removed.
    
    
inplace : bool, default False
    If True, do operation inplace and return None.


errors : {'ignore', 'raise'}, default 'raise'
    If 'ignore', suppress error and only existing labels are
    dropped.
```

丢弃部分数值后的df_train 和 df_test

![image-20210321102718588](D:\Typora\user-image\image-20210321102718588.png)

![image-20210321103518217](D:\Typora\user-image\image-20210321103518217.png)



### 部分数据转换为独热向量

#### pd.get_dummies()

![image-20210321110700691](D:\Typora\user-image\image-20210321110700691.png)

```python
sex = pd.get_dummies(df_train['Sex'])
embark = pd.get_dummies(df_train['Embarked'])
```

```python
pd.get_dummies(
    data,
    prefix=None,
    prefix_sep='_',
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
    dtype=None,
) -> 'DataFrame'
Docstring:
Convert categorical variable into dummy/indicator variables.

Parameters
----------
data : array-like, Series, or DataFrame
    Data of which to get dummy indicators.
    
    
prefix : str, list of str, or dict of str, default None
    String to append DataFrame column names.
    Pass a list with length equal to the number of columns
    when calling get_dummies on a DataFrame. Alternatively, `prefix`
    can be a dictionary mapping column names to prefixes.
    
    
prefix_sep : str, default '_'
    If appending prefix, separator/delimiter to use. Or pass a
    list or dictionary as with `prefix`.
    
    
dummy_na : bool, default False
    Add a column to indicate NaNs, if False NaNs are ignored.
    
    
columns : list-like, default None
    Column names in the DataFrame to be encoded.
    If `columns` is None then all the columns with
    `object` or `category` dtype will be converted.
    
    
sparse : bool, default False
    Whether the dummy-encoded columns should be backed by
    a :class:`SparseArray` (True) or a regular NumPy array (False).
            
drop_first : bool, default False
    Whether to get k-1 dummies out of k categorical levels by removing the
    first level.
    
dtype : dtype, default np.uint8
    Data type for new columns. Only a single dtype is allowed.

    .. versionadded:: 0.23.0

```

### DataFrame/Series合并

#### pd.concat()

![image-20210321112003170](D:\Typora\user-image\image-20210321112003170.png)

```python
df_train.drop(['Sex','Embarked'],axis=1,inplace=True)
df_train = pd.concat([df_train,sex,embark],axis = 1)
```

```python
Signature:
pd.concat(
    objs: Union[Iterable[Union[ForwardRef('DataFrame'), ForwardRef('Series')]], Mapping[Union[Hashable, NoneType], Union[ForwardRef('DataFrame'), ForwardRef('Series')]]],
    axis=0,
    join='outer',
    ignore_index: bool = False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool = True,
) -> Union[ForwardRef('DataFrame'), ForwardRef('Series')]
Docstring:
Concatenate pandas objects along a particular axis with optional set logic
along the other axes.

Can also add a layer of hierarchical indexing on the concatenation axis,
which may be useful if the labels are the same (or overlapping) on
the passed axis number.

Parameters
----------
objs : a sequence or mapping of Series or DataFrame objects
    If a dict is passed, the sorted keys will be used as the `keys`
    argument, unless it is passed, in which case the values will be
    selected (see below). Any None objects will be dropped silently unless
    they are all None in which case a ValueError will be raised.
    
    
axis : {0/'index', 1/'columns'}, default 0
    The axis to concatenate along.
    
    
join : {'inner', 'outer'}, default 'outer'
    How to handle indexes on other axis (or axes).
    
    
ignore_index : bool, default False
    If True, do not use the index values along the concatenation axis. The
    resulting axis will be labeled 0, ..., n - 1. This is useful if you are
    concatenating objects where the concatenation axis does not have
    meaningful indexing information. Note the index values on the other
    axes are still respected in the join.
    
    
keys : sequence, default None
    If multiple levels passed, should contain tuples. Construct
    hierarchical index using the passed keys as the outermost level.
    
    
levels : list of sequences, default None
    Specific levels (unique values) to use for constructing a
    MultiIndex. Otherwise they will be inferred from the keys.
    
    
names : list, default None
    Names for the levels in the resulting hierarchical index.
verify_integrity : bool, default False
    Check whether the new concatenated axis contains duplicates. This can
    be very expensive relative to the actual data concatenation.
    
    
sort : bool, default False
    Sort non-concatenation axis if it is not already aligned when `join`
    is 'outer'.
    This has no effect when ``join='inner'``, which already preserves
    the order of the non-concatenation axis.

    .. versionadded:: 0.23.0
    .. versionchanged:: 1.0.0

       Changed to not sort by default.

    
copy : bool, default True
    If False, do not copy data unnecessarily.
```

对测试集也进行处理

```python
sex = pd.get_dummies(df_test['Sex'])
embark = pd.get_dummies(df_test['Embarked'])
df_test = pd.concat([df_test,sex,embark],axis=1)
df_test.drop(['Sex','Embarked'],axis=1,inplace=True)
```



### 对缺失值的填充

#### DataFrame.fillna()

```python
df_train.fillna(df_train.mean(),inplace=True)
df_test.fillna(df_test.mean(),inplace=True)
```



```python
Signature:
df_train.fillna(
    value=None,
    method=None,
    axis=None,
    inplace=False,
    limit=None,
    downcast=None,
) -> Union[ForwardRef('DataFrame'), NoneType]
Docstring:
Fill NA/NaN values using the specified method.

Parameters
----------
value : scalar, dict, Series, or DataFrame
    Value to use to fill holes (e.g. 0), alternately a
    dict/Series/DataFrame of values specifying which value to use for
    each index (for a Series) or column (for a DataFrame).  Values not
    in the dict/Series/DataFrame will not be filled. This value cannot
    be a list.
    
method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
    Method to use for filling holes in reindexed Series
    pad / ffill: propagate last valid observation forward to next valid
    backfill / bfill: use next valid observation to fill gap.
        
axis : {0 or 'index', 1 or 'columns'}
    Axis along which to fill missing values.
    
inplace : bool, default False
    If True, fill in-place. Note: this will modify any
    other views on this object (e.g., a no-copy slice for a column in a
    DataFrame).
    
limit : int, default None
    If method is specified, this is the maximum number of consecutive
    NaN values to forward/backward fill. In other words, if there is
    a gap with more than this number of consecutive NaNs, it will only
    be partially filled. If method is not specified, this is the
    maximum number of entries along the entire axis where NaNs will be
    filled. Must be greater than 0 if not None.
    
downcast : dict, default is None
    A dict of item->dtype of what to downcast if possible,
    or the string 'infer' which will try to downcast to an appropriate
    equal type (e.g. float64 to int64 if possible).
    
Returns
-------
DataFrame or None
    Object with missing values filled or None if ``inplace=True``.

```



### 数据归一化

#### StandardScaler对象

```python
from sklearn.preprocessing import StandardScaler
Scaler1 = StandardScaler()
df_train = pd.DataFrame(Scaler1.fit_transform(df_train))
```

```python
Init signature: StandardScaler(*, copy=True, with_mean=True, with_std=True)
Docstring:     
Standardize features by removing the mean and scaling to unit variance

The standard score of a sample `x` is calculated as:

    z = (x - u) / s

where `u` is the mean of the training samples or zero if `with_mean=False`,
and `s` is the standard deviation of the training samples or one if
`with_std=False`.

Centering and scaling happen independently on each feature by computing
the relevant statistics on the samples in the training set. Mean and
standard deviation are then stored to be used on later data using
:meth:`transform`.

Standardization of a dataset is a common requirement for many
machine learning estimators: they might behave badly if the
individual features do not more or less look like standard normally
distributed data (e.g. Gaussian with 0 mean and unit variance).

For instance many elements used in the objective function of
a learning algorithm (such as the RBF kernel of Support Vector
Machines or the L1 and L2 regularizers of linear models) assume that
all features are centered around 0 and have variance in the same
order. If a feature has a variance that is orders of magnitude larger
that others, it might dominate the objective function and make the
estimator unable to learn from other features correctly as expected.

This scaler can also be applied to sparse CSR or CSC matrices by passing
`with_mean=False` to avoid breaking the sparsity structure of the data.

Read more in the :ref:`User Guide <preprocessing_scaler>`.

Parameters
----------
copy : bool, default=True
    If False, try to avoid a copy and do inplace scaling instead.
    This is not guaranteed to always work inplace; e.g. if the data is
    not a NumPy array or scipy.sparse CSR matrix, a copy may still be
    returned.

with_mean : bool, default=True
    If True, center the data before scaling.
    This does not work (and will raise an exception) when attempted on
    sparse matrices, because centering them entails building a dense
    matrix which in common use cases is likely to be too large to fit in
    memory.

with_std : bool, default=True
    If True, scale the data to unit variance (or equivalently,
    unit standard deviation).

Attributes
----------
scale_ : ndarray of shape (n_features,) or None
    Per feature relative scaling of the data to achieve zero mean and unit
    variance. Generally this is calculated using `np.sqrt(var_)`. If a
    variance is zero, we can't achieve unit variance, and the data is left
    as-is, giving a scaling factor of 1. `scale_` is equal to `None`
    when `with_std=False`.

    .. versionadded:: 0.17
       *scale_*

mean_ : ndarray of shape (n_features,) or None
    The mean value for each feature in the training set.
    Equal to ``None`` when ``with_mean=False``.

var_ : ndarray of shape (n_features,) or None
    The variance for each feature in the training set. Used to compute
    `scale_`. Equal to ``None`` when ``with_std=False``.

n_samples_seen_ : int or ndarray of shape (n_features,)
    The number of samples processed by the estimator for each feature.
    If there are no missing samples, the ``n_samples_seen`` will be an
    integer, otherwise it will be an array of dtype int. If
    `sample_weights` are used it will be a float (if no missing data)
    or an array of dtype float that sums the weights seen so far.
    Will be reset on new calls to fit, but increments across
    ``partial_fit`` calls.
```

#### StandardScaler.fit_transform()

```python
df_train = pd.DataFrame(Scaler1.fit_transform(df_train))
```



```python
Signature: Scaler1.fit_transform(X, y=None, **fit_params)
Docstring:
Fit to data, then transform it.

Fits transformer to `X` and `y` with optional parameters `fit_params`
and returns a transformed version of `X`.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Input samples.

y :  array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None
    Target values (None for unsupervised transformations).

**fit_params : dict
    Additional fit parameters.

Returns
-------
X_new : ndarray array of shape (n_samples, n_features_new)
    Transformed array.
```

#### 列表头

归一化之前

```python
train_columns = df_train.columns
```

归一化之后

```python
df_train.columns = train_columns
```

## Pytorch



### 初始化张量



```python
trainTorch_x = torch.from_numpy(train_x).type(torch.FloatTensor)
trainTorch_y = torch.from_numpy(train_y).type(torch.LongTensor)
```

### 网络模型



```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(10,128)
        self.fc2 = nn.Linear(128,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,128)
        self.fc5 = nn.Linear(128,2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,x):
        x = F.relu(self.fc1(x)) 
        x = self.dropout(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(self.fc4(x))
        x = self.fc5(x)
        return x
model = Net()
```

