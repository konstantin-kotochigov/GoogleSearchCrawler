# Script for reading query results and building text classification

import pandas
import sklearn
import os

filelist = os.listdir("/srv/kkotochigov/GoogleSearch/result/")

result_df = pandas.DataFrame(columns=["description","target"])
for filename in filelist:
    
    print("Processing " + filename)
    
    f = open("/srv/kkotochigov/GoogleSearch/result/"+filename,"r")
    attr_descriptions_list = f.readlines()
    f.close()
    
    attr_descriptions_df = pandas.DataFrame.from_dict({"description" : attr_descriptions_list, "target":filename})
    result_df = result_df.append(attr_descriptions_df)

# Modeling

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

pipeline = make_pipeline(CountVectorizer(), TfidfTransformer(), MultinomialNB())
parameters = {'countvectorizer__ngram_range':[(1,1),(1,2)],'tfidftransformer__use_idf':(True,False), 'multinomialnb__alpha':(1e-2,1e-3)}

gcv = GridSearchCV(pipeline, parameters, cv=5)
gcv.fit(result_df.description, result_df.target)

results_dict = catboostCV.cv_results_
results_df = pandas.DataFrame({"params":results_dict['params'], "mean_test_score" : results_dict['mean_test_score'],"mean_fit_time" : results_dict['mean_fit_time']})

