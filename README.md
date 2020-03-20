# COMP6237 - Understanding Data

## Your First Text Mining Project with Python in 3 steps

1. Building a corpus
	- using Tweepy to gather sample text data from Twitter's API
2. Analyzing text
	- analyzing the sentiment of a piece of text with our own SDK
3. Visualising results
	- how to use Pandas and matplotlib to see the results of your work

---

### Reference code ###
https://github.com/sebag07/Hierarchical-Clustering/blob/master/HierarchicalClustering.ipynb

---
### Data Handling
#### Data Preprocessing

__> scrap_html()__   
- html 파일로부터 데이터를 불러와서 pandas 패키지의 DataFrame 형태로 저장한다.   
- 각 dir는 하나의 책 파일이므로, books라는 변수에는 ['book_name', 'contents']로 저장됨.   
- books를 반환하며 함수 종료   

__> data visualisation에 사용될 title을 미리 저장한다.__   
- titles 라는 변수에 list 형태로 저장.   
- author도 마찬가지 --> 크게 중요하지 않음.   

__> books의 'contents'에는 Beautifulsoup로 읽어 온 raw data로 저장되어있다. 이를 단어별로 쪼개고, stopwords('i','me','my', et cetra)를 사용해 불필요한 단어수를 줄여주자.__   
- stopwords는 `nltk.corpus.stopwords.words('english')`로 정의 가능
- 동사 및 명사의 복수형, 형태변환을 효과적으로 인식하고자, stemmer를 추출한다(어간).
- nltk.stem.snowball.SnowballStemmer('english')를 사용.

__> tokenize_and_stem(text)__
- input text를 nltk.sent_tokenize(text), nltk.word_tokenize(sent)라는 함수들을 통해 단어별로 쪼갠다.
- 쪼개진 tokens를 re.search('[a-zA-Z]', token)함수를 이용해 알파벳으로 이루어진 단어들만 선별해서 filtered_tokens에 저장한다.(filtered_tokens는 list이므로 append로 저장.)
- filtered_tokens 안에 모든 값들을 stemmer.stem() 함수에 input으로 넣고 그 때의 output을 stems 변수에 저장한다.
- return stems

__> tokenize_only()__
- 위(6) 함수의 간략화 버전.
- 단순히 filtered_tokens 만을 생성한다.

__> nltk.download('punkt')__ ***********************
- 검색요망

__> books['contents'] 항목을 input으로 넣어 output tokens를 얻어낸다.__
- 이 때, stemmed token을 저장하는 변수, tokenized만이 된 값을 저장하는 변수를 따로 지정한다.
- vocab_frame 변수는 ['words'라고 이름 붙은 tokenized만 된 값들, index: stemmed 된 값]으로 매칭 시켜서 table화 한다.

---

#### Feature extraction
__> 전처리된 데이터에서 특성을 추출하기위해(feature extraction) Tf-idf를 사용한다.__
- sklearn.feature_extraction.text.TfidfVectorizer 함수 사용
- max_df, max_features, min_df, stop_words, use_idf, tokenizer 를 input으로 받는다.
- TfidfVectorizer class의 함수중 fit_transform()의 input으로 앞서 선언한 변수를 대입한다.
- 최종적으로 tfidf_matrix가 생성될 것이며, 그 크기를 확인하자.

__> terms 변수에 TfidfVectorizer class의 `get_feature_names()` 함수의 결과 값을 저장.__
- 딱히 중요하게 쓰이진 않음
- sklearn.metrics.pairwise import cosine_similarity
- cosine similarity를 왜 사용하는지 알아야함.
- 그래서 최종 거리값 dist는 1-cosine_simiarity()로 계산.

---

### Data mining methods
#### K-means
__> sklearn.cluster.KMeans__
- 임의의 cluster 개수를 선정하고 KMeans 모델을 생성.
- 앞서 만든 tfidf_matrix 를 input 으로하는 model 생성.
- clusters 변수는 모델 km.labels_.tolist() 로 저장.
- 어떻게 분류가 되었는지 확인 가능.

__> sklearn.externals.joblib__
- 학습 시키는 것이 오래걸리기 때문에 모델을 저장하는데 사용.

__> 어떤 책이 어떤 cluster에 속해있는지 pandas DataFrame 변수를 하나 더 생성(book_frame)하여 print.__

__> `from __future__ import print_function`__
- 이거 뭐하는건지 모르겠음, 확인필요
- 에러남.

---

#### Multi Dimensional Scaling
__> sklearn.manifold.MDS()__
- Multi-dimensional scaling 기법 사용.
- MDS() 실행 <- 왜 이것만 code 입력하는지 document 확인 필요.
- mds = MDS(n_components, dissimilarity, random_state) 함수를 통해 모델 생성(?)
- `pos = mds.fit_transform(dist)`
- 앞서 계산한 tfidf의 거리 dist를 input으로 pos 생성

__> cluster_colours 저장__
- visualisation을 위함.

__> df 변수에 xs, ys clusters, titles 를 변수로 갖는 DataFrame 선언.__
- 위의 input을 dict()으로 받았는데 조사 필요.
- df.groupby('label') <- 조사 필요
- 앞서 다 계산했던 내용들을 plotting 하는 과정.
- 데이터를 visualisation하는 과정에 사용된 함수들을 자세히 살펴볼 필요가 있음.

---

#### TSNE
__> sklearn.manifold.TSNE__
- n_sne = 7000 으로 초기화
- TSNE 모델을 만들고 dist를 사용해 학습
- 학습된 모델을 MDS와 같은 과정을 반복해서 plotting

---

#### Hierarchy Clustering
__> scipy.cluster.hierarchy import ward, dendrogram__
- hierarchy clustering을 사용할 것.
- linkage_matrix = ward(dist)
- 마찬가지로 우리가 추출한 feature인 dist를 input으로 ward() 함수 진행.
- 그 결과 값(matrix)을 바탕으로 plotting

### Marking and Feedback
1. Learning Outcomes
- Solve real-word data-mining, data-indexing and information extraction tasks
- Demonstrate knowledge and understanding of:
	* Key concepts, tools and approaches for data mining on complex unstructured dataset.
	* Theoretical concepts and the motivations behind different data-mining approaches.

2. Marking Scheme
Good working notes papers not only effectively apply techniques and describe results, but also offer critical insight into the findings of the analysis in the context of the underlying data.   
In particular you need to demonstrate that you understand the data, and, in the context of that understanding, that you can rationalise and reflect on why the analytic techniques are giving the results they do.

	- Experimentation (28)
		* Analyse the problem and define suitable preprocessing and feature extraction operations
	- Application of techniques (28)
		* Show ability to apply exploratory data mining techniques
	- Analysis (28)
		* Reflection on what can be understood from the data through the application of exploratory techniques
	- Reporting (16)
		* Clear and professional reporting