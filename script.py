from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def assignment(list_of_features):
  train_emails = fetch_20newsgroups(categories = list_of_features, subset = 'train', shuffle = True, random_state = 108)
  
  test_emails = fetch_20newsgroups(categories = list_of_features, subset = 'test', shuffle = True, random_state = 108)
  counter = CountVectorizer()
  counter.fit(test_emails.data + train_emails.data)
  
  train_counts = counter.transform(train_emails.data)
  test_counts = counter.transform(test_emails.data)
  
  classifier = MultinomialNB()
  classifier.fit(train_counts, train_emails.target)
  
  accuracy = classifier.score(test_counts, test_emails.target)
  return accuracy

#FirstPart
print(assignment(['rec.sport.baseball', 'rec.sport.hockey']))

#SecondPart
print(assignment(['comp.sys.ibm.pc.hardware','rec.sport.hockey']))

#ThirdPart
print(assignment(['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']))

