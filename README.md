# Customer-Churn-Prediction
Analysis of a dataset to determine the reasons customers are leaving a particular service.

## Download Dataset
[Kaggle Customer Churn]([url](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))

## Import Libraries
```ruby
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
## Initial Analysis
- Verified dataset shape and column names
- Checked data types and summary statistics
- Identified missing values
- Created initial visual showing churn vs non-churn customers.

