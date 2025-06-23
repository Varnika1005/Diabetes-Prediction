
# # Importing the Dependencies

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


from sklearn.preprocessing import StandardScaler   # for data standardization


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


from sklearn import svm


# In[6]:


from sklearn.metrics import accuracy_score


# # Data collection and Analysis

# In[7]:


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv("diabetes.csv")


# In[8]:


# printing the first five rows of the dataset
diabetes_dataset.head()


# In[10]:


#total number of rows and columns is in this dataset 
diabetes_dataset.shape


# In[11]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# In[13]:


diabetes_dataset['Outcome'].value_counts()


# # 0--> Non-Diabetic
# # 1--> Diabetic

# In[15]:


# Calculates the mean of each feature grouped by the 'Outcome' column (0 = non-diabetic, 1 = diabetic)
diabetes_dataset.groupby('Outcome').mean()  


# In[16]:


# seperating the data and labels
x = diabetes_dataset.drop(columns = 'Outcome', axis=1)
y = diabetes_dataset['Outcome']


# In[17]:


print(x)


# In[18]:


print(y)


# # Data Standardiziation

# In[23]:


scaler = StandardScaler()


# In[24]:


scaler.fit(x)


# In[25]:


standardized_data = scaler.transform(x)


# In[26]:


print(standardized_data)


# In[27]:


x = standardized_data
y = diabetes_dataset['Outcome']


# In[28]:


print(x)
print(y)


# # Train Test Split

# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify = y, random_state = 2)


# In[30]:


print(x.shape, x_train.shape, x_test.shape)


# # Training the Model

# In[31]:


classifier = svm.SVC(kernel = 'linear')


# In[32]:


# training the Support Vector Machine Classifier
classifier.fit(x_train, y_train)


# # Model Evaluation:

# ## Accuracy Score

# In[35]:


# accuracy score on the training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)


# In[36]:


print("Accuracy score of the training data : ", training_data_accuracy)


# In[37]:


# accuracy score on the test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)


# In[38]:


print("Accuracy score of the test data : ", test_data_accuracy)


# # Making a Predictive System

# In[43]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if(prediction[0] == 0):
    print("This person is not Diabetic")
else:
    print("This person is Diabetic")


# In[ ]:




