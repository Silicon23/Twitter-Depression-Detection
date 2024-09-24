---
title: "Analyzing the Effects of Depression in Twitter"
author: " "
output:
  pdf_document:
    number_sections: yes
    toc: yes
    toc_depth: '4'
  word_document:
    toc: yes
    toc_depth: '4'
  html_document:
    code_folding: show
    highlight: haddock
    number_sections: yes
    theme: lumen
    toc: yes
    toc_depth: 4
    toc_float: yes
urlcolor: blue
editor_options:
  chunk_output_type: inline
abstract: |
  Depression has increasingly become recognized as a pressing issue in today’s society. Currently, the WHO estimates that 4% of the population suffers from depression, and 700,000 people die each year from suicide. In fact, suicide is the fourth leading cause of death in people aged 15-29. Despite the availability of effective treatments for depression, many individuals suffering from depression often go undiagnosed and untreated due to economic inequality and social stigma. Social media presents a promising solution to this issue. Today, social media use is extremely widespread, and users’ data presents a cheap and accessible source of information that can be used to identify signs of depression. Thus, this study aims to predict and identify signs of depression based on a Twitter user’s activity. The data consists of a user’s tweet history, follower count, friend count, and other information. To analyze the data, logistic regression, LASSO, decision trees, random forest, and neural networks are utilized on the numerical data. For the tweet data, BERT classification, bag-of-words, and other textual analysis techniques are applied to predict depression and identify words and phrases that potentially indicate it. Results will be included once they are produced…
---



```{r setup, include=FALSE}
# Load or install pacman package for managing dependencies
if (!require('pacman')) {
  install.packages('pacman')
}

# Use pacman to load the required packages
pacman::p_load(knitr, dplyr, ggplot2, gridExtra, ggrepel)

# Set knitr chunk options
knitr::opts_chunk$set(
  echo = TRUE,
  results = "hide",
  fig.width = 6,
  fig.height = 4,
  cache = TRUE,
  message = FALSE,
  warning = FALSE,
  fig.path = 'figs/',
  cache.path = '_cache/',
  fig.process = function(x) {
    x2 = sub('-\\d+([.][a-z]+)$', '\\1', x)
    if (file.rename(x, x2)) x2 else x
  }
)

# Set global R options
options(
  scipen = 0,
  digits = 3  # controls base R output
)

```

# Introduction

Depression is a critical and socially relevant issue that affects millions of people worldwide, significantly impacting their quality of life and overall well-being. It is a leading cause of disability and can result in severe outcomes, including suicide. Recent events, such as the COVID-19 pandemic, have exacerbated the prevalence of depression, with many individuals facing increased stress, isolation, and uncertainty. This rise in depression rates has been further compounded by the challenges associated with accessing mental health care. Rising healthcare costs, coupled with the necessity of in-person consultations, make it difficult for many individuals to seek diagnosis and treatment, often requiring significant time and resources.

In response to these challenges, social media presents a promising alternative for identifying signs of depression in a more accessible and cost-effective manner. With its pervasive use and the vast amounts of data generated daily, platforms like Twitter offer a unique opportunity to observe and analyze user behavior that may indicate mental health issues. Twitter, in particular, is an ideal site for this research due to its public nature and the brevity of its content, which allows for easier data collection and analysis.

**1.1 Analyzing User Interaction Data**

In the first section of our study, we focus on quantitative metrics such as the number of likes, favorites, retweets, and followers that Twitter users have. These metrics offer insights into user engagement and social connectedness, which can be correlated with mental health status. To analyze these data points, we employ several machine learning techniques. Linear regression is used to model the relationship between a dependent variable (e.g., likelihood of depression) and one or more independent variables (e.g., likes, retweets). It is particularly useful for understanding how each feature contributes to the outcome and for predicting continuous outcomes. Logistic regression, on the other hand, is employed for binary classification problems, such as determining whether a user is likely to be depressed or not based on their interaction data. This method is effective in cases where the outcome variable is categorical.

Additionally, we utilize decision trees, a non-parametric supervised learning method that can model complex relationships between input features and outcomes. Decision trees are interpretable and capable of capturing non-linear interactions between features, making them suitable for identifying patterns in user behavior that may indicate depression. To further enhance the predictive accuracy, random forests are applied. Random forests aggregate the predictions of multiple decision trees, improving accuracy and robustness. This technique reduces the risk of overfitting and enhances the model's ability to generalize from the data, making it a powerful tool for predicting depression based on social media activity.

**1.2 Textual Analysis of Tweets**

The second part of our study focuses on the textual analysis of user tweets to extract information that may indicate depressive states. We use several advanced text mining techniques to analyze the linguistic content of tweets. VADER (Valence Aware Dictionary and sEntiment Reasoner) is employed for sentiment analysis; it is a lexicon and rule-based tool specifically designed to analyze sentiments expressed in social media contexts. VADER is effective in assessing the sentiment polarity (positive, negative, or neutral) of tweets, which can help identify depressive language.

We also use the bag-of-words approach combined with LASSO (Least Absolute Shrinkage and Selection Operator) and relaxed LASSO for feature selection and regularization. The bag-of-words model represents text as an unordered collection of words, and LASSO enhances prediction accuracy by selecting the most relevant features while penalizing less important ones. Relaxed LASSO allows for greater flexibility, improving model performance in identifying pertinent features from text data. In addition to this, n-grams analysis is conducted to capture common phrases and word patterns associated with depression. By examining n-grams, we gain deeper insights into the language used by individuals experiencing depressive symptoms.

Furthermore, we implement BERT (Bidirectional Encoder Representations from Transformers) for sequence classification. BERT is a state-of-the-art model for natural language understanding, and in this study, it is used to analyze the semantic context of tweets. By leveraging deep learning, BERT can capture complex linguistic patterns and contextual nuances that simpler models might overlook, making it highly effective for detecting subtle indicators of depression. By integrating these quantitative and textual analyses, this study aims to develop comprehensive frameworks for detecting depression based on Twitter activity, contributing valuable insights to the field of digital mental health.

\pagebreak

#*Detecting Depression Amongst Twitter Users*


```R
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, ggplot2, car, tm, SnowballC, RColorBrewer, wordcloud, glmnet, rpart, rpart.plot, caret,
               randomForest, ranger, data.table, gridExtra, tree, broom, vader, stargazer, lm.beta, pbapply)
```

    Loading required package: pacman
    


## Exploratory Data Analysis

###Load the Data.
Data from [kaggle](https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media).\
Download and upload as `Mental-Health-Twitter.csv` to the folder `data`.\
\
Using package `data.table` to read and to manipulate data is much faster than using `read.csv()` especially when the dataset is large.

###Subsetting
Let's first take a small piece of it to work through. We could use `fread` to only load `nrows` many rows to avoid loading the entire dataset.


```R
#data.all <- read.csv("yelp_subset.csv", as.is=TRUE) # as.is = T to keep the texts as character
data.all <- fread("/content/data/Mental_Health_Twitter.csv", stringsAsFactors = FALSE)
dim(data.all)
object.size(data.all)
summary(data.all)
str(data.all)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>89171</li><li>8</li></ol>




    19251376 bytes



           V1           user_id                     post_text        
     Min.   :    0   Min.   :               9457   Length:89171      
     1st Qu.:22292   1st Qu.:          469925723   Class :character  
     Median :44585   Median : 727014861327093760   Mode  :character  
     Mean   :44585   Mean   : 568113714782200812                     
     3rd Qu.:66878   3rd Qu.:1171262765597712385                     
     Max.   :89170   Max.   :1338665447697235969                     
         label          followers          friends         favorites     
     Min.   :0.0000   Min.   :      0   Min.   :     0   Min.   :     0  
     1st Qu.:0.0000   1st Qu.:    154   1st Qu.:   211   1st Qu.:  3148  
     Median :0.0000   Median :    425   Median :   468   Median : 10975  
     Mean   :0.4915   Mean   :   5272   Mean   :  1298   Mean   : 27936  
     3rd Qu.:1.0000   3rd Qu.:   1289   3rd Qu.:   982   3rd Qu.: 33264  
     Max.   :1.0000   Max.   :1429030   Max.   :181822   Max.   :441317  
        statuses      
     Min.   :     25  
     1st Qu.:   2804  
     Median :   8972  
     Mean   :  25584  
     3rd Qu.:  25741  
     Max.   :1527058  


    Classes ‘data.table’ and 'data.frame':	89171 obs. of  8 variables:
     $ V1       : int  0 1 2 3 4 5 6 7 8 9 ...
     $ user_id  :integer64 1249839468 1249839468 1249839468 1249839468 1249839468 1249839468 1249839468 1249839468 ... 
     $ post_text: chr  "@kristenevanss14 retweet again 🙃🙃🙃" "Wish I could be with my family and it not be in Zavalla bc this town literally makes me sick to my stomach" "@kaileinicole10 This is so true 😩" "Dalton and I started our trip with the best hike ❤️ https://t.co/h5IkRTEArB" ...
     $ label    : int  0 0 0 0 0 0 0 0 0 0 ...
     $ followers: num  378 378 378 378 378 378 378 378 378 378 ...
     $ friends  : num  151 151 151 151 151 151 151 151 151 151 ...
     $ favorites: num  1350 1350 1350 1350 1350 1350 1350 1350 1350 1350 ...
     $ statuses : num  7207 7207 7207 7207 7207 ...
     - attr(*, ".internal.selfref")=<externalptr> 



```R
data.subset <- data.all
```

###EDA & Sentiment Analysis


```R
#@title Plot a box plot for all quantitative predictors
# Box plot for followers
p1 <- ggplot(data.subset, aes(x = "", y = followers)) +
    coord_cartesian(ylim = c(0, quantile(data.subset$followers, 0.99))) +
    geom_boxplot() +
    labs(title = "Box Plot of Followers",
         x = "",
         y = "Followers") #+ scale_y_log10()

# Box plot for friends
p2 <- ggplot(data.subset, aes(x = "", y = friends)) +
    coord_cartesian(ylim = c(0, quantile(data.subset$friends, 0.99))) +
    geom_boxplot() +
    labs(title = "Box Plot of Friends",
         x = "",
         y = "Friends") #+ scale_y_log10()

# Box plot for favourites
p3 <- ggplot(data.subset, aes(x = "", y = favorites)) +
    coord_cartesian(ylim = c(0, quantile(data.subset$favorites, 0.99))) +
    geom_boxplot() +
    labs(title = "Box Plot of Favorites",
         x = "",
         y = "Favorites") # + scale_y_log10()

# Display the plots
grid.arrange(p1, p2, p3, ncol = 3)
```


    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_9_0.png)
    



```R
#@title Sentiment Analysis with Vader
data.subset <- data.subset %>%
  mutate(sentiment_score = vader_df(post_text)$compound)

```

    Warning message:
    “[1m[22mThere were 20 warnings in `mutate()`.
    The first warning was:
    [1m[22m[36mℹ[39m In argument: `sentiment_score = vader_df(post_text)$compound`.
    Caused by warning in `sentiments[i] <- senti_valence(wpe, i, item)`:
    [33m![39m number of items to replace is not a multiple of replacement length
    [1m[22m[36mℹ[39m Run `dplyr::last_dplyr_warnings()` to see the 19 remaining warnings.”



```R
#@title Incorporate Sentiment into Dataset
average_scores <- data.subset %>%
  group_by(user_id) %>%
  summarize(sentiment_score_average = mean(sentiment_score, na.rm = TRUE))

# Join the average sentiment scores back to the original dataset
data.subset <- data.subset %>%
  left_join(average_scores, by = "user_id")
data.all <- data.subset
```


```R
data.subset <- read.csv("sentiment_analysis.csv")
```


```R
data.quant = data.subset[,c("label", "followers", "friends", "favorites", "statuses", "sentiment_score_average")]
dim(data.quant)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>89171</li><li>6</li></ol>




```R
data.quant <- distinct(data.quant)
dim(data.quant)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1123</li><li>6</li></ol>




```R
colnames(data.quant)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'label'</li><li>'followers'</li><li>'friends'</li><li>'favorites'</li><li>'statuses'</li><li>'sentiment_score_average'</li></ol>




```R
data.quant$log_followers <- log(data.quant$followers+1)
```


```R
hist(data.quant$log_followers, xlab="Log Followers", main="Histogram of Log Followers")
```


    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_17_0.png)
    



```R
data.quant$log_friends <- log(data.quant$friends+1)
```


```R
hist(data.quant$log_friends, xlab="Log Friends", main="Histogram of Log Friends")
```


    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_19_0.png)
    



```R
data.quant$log_favorites <- log(data.quant$favorites+1)
```


```R
hist(data.quant$log_favorites, xlab="Log Favorites", main="Histogram of Log Favorites")
```


    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_21_0.png)
    


##Analysis of Variables

This section seeks to generate models exclusively for the variables followers, friends, and favourites.


```R
dim(data.quant)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1123</li><li>9</li></ol>




```R
#@title Splitting the Dataset
# taking a subset of twitter data
# We could use fread with a specified nrows. We might need to shuffle the data in order to get a random sample.
set.seed(1)
test_indices <- sample(nrow(data.quant), 0.2*nrow(data.quant))
data.subset <- data.quant[-test_indices,]
data.testset <- data.quant[test_indices,]
# Take a small set to work through
# MAKE SURE: you will rerun the analyses later by setting back a larger dataset.
table(data.subset$label)
table(data.testset$label)
# dim(data.subset)
# dim(data.testset)
```


    
      0   1 
    456 443 



    
      0   1 
    114 110 



```R
#@title Fit a linear model
data.subz <- data.subset %>%
  mutate(followers = as.numeric(followers/10000),
  friends = as.numeric(friends/10000),
  favorites = as.numeric(favorites/10000))

fit_lm <- lm(label ~ log_followers + log_friends + log_favorites + sentiment_score_average, data = data.subset)
stargazer(fit_lm, type="text")

# fit_lm_1000 <- lm(label ~ log_followers + log_friends + log_favorites + sentiment_score_average, data = data.subz)
# stargazer(fit_lm_1000, type="text")


# fit_standardized <- lm.beta(fit_lm)
# stargazer(fit_standardized, type="text")
```

    
    ===================================================
                                Dependent variable:    
                            ---------------------------
                                       label           
    ---------------------------------------------------
    log_followers                    -0.038***         
                                      (0.012)          
                                                       
    log_friends                       -0.021           
                                      (0.017)          
                                                       
    log_favorites                    0.047***          
                                      (0.009)          
                                                       
    sentiment_score_average           -0.083           
                                      (0.141)          
                                                       
    Constant                         0.439***          
                                      (0.083)          
                                                       
    ---------------------------------------------------
    Observations                        899            
    R2                                 0.041           
    Adjusted R2                        0.037           
    Residual Std. Error          0.491 (df = 894)      
    F Statistic               9.608*** (df = 4; 894)   
    ===================================================
    Note:                   *p<0.1; **p<0.05; ***p<0.01



```R
#@title Fit Logistic Regression Model

# Fit logistic regression model with log-transformed variables
fit_logistic <- glm(label ~ log_followers + log_friends + log_favorites + sentiment_score_average,
                    data = data.subset,
                    family = binomial(logit))

# Summarize the logistic regression model using stargazer
stargazer(fit_logistic, type = "text")

# Summarize the logistic regression model using summary
summary(fit_logistic)

```

    
    ===================================================
                                Dependent variable:    
                            ---------------------------
                                       label           
    ---------------------------------------------------
    log_followers                    -0.173***         
                                      (0.053)          
                                                       
    log_friends                       -0.085           
                                      (0.074)          
                                                       
    log_favorites                    0.207***          
                                      (0.041)          
                                                       
    sentiment_score_average           -0.352           
                                      (0.595)          
                                                       
    Constant                          -0.287           
                                      (0.357)          
                                                       
    ---------------------------------------------------
    Observations                        899            
    Log Likelihood                   -603.533          
    Akaike Inf. Crit.                1,217.066         
    ===================================================
    Note:                   *p<0.1; **p<0.05; ***p<0.01



    
    Call:
    glm(formula = label ~ log_followers + log_friends + log_favorites + 
        sentiment_score_average, family = binomial(logit), data = data.subset)
    
    Coefficients:
                            Estimate Std. Error z value Pr(>|z|)    
    (Intercept)             -0.28710    0.35732  -0.803  0.42169    
    log_followers           -0.17309    0.05276  -3.281  0.00103 ** 
    log_friends             -0.08526    0.07355  -1.159  0.24640    
    log_favorites            0.20687    0.04090   5.058 4.24e-07 ***
    sentiment_score_average -0.35245    0.59473  -0.593  0.55343    
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for binomial family taken to be 1)
    
        Null deviance: 1246.1  on 898  degrees of freedom
    Residual deviance: 1207.1  on 894  degrees of freedom
    AIC: 1217.1
    
    Number of Fisher Scoring iterations: 4




```R
#@title Decision Tree
# Fit the decision tree model
fit_tree1 <- rpart(label ~ followers + friends + favorites + sentiment_score_average,
                   data = data.subset,
                   control = rpart.control(minsplit = 2))

output_file <- "/content/decision_tree_plot.png"

# Save the plot to a PNG file
png(filename = output_file, width = 800, height = 600)

# Plot the decision tree with customizations
rpart.plot(fit_tree1,
           type = 2,           # Type of plot: 0, 1, 2, 3, 4 (3 is compact)
           extra = 101,        # Display number of observations that follow the path and percentage
           under = TRUE,       # Put the split variable name under the box
           faclen = 0,         # Factor levels as full names
           tweak = 1.3,        # Scale text and boxes
           fallen.leaves = TRUE)  # Put leaves at the bottom of the plot

dev.off()

```


<strong>png:</strong> 2



```R
#@title Random Forest
# Load necessary libraries
# set.seed(100)#21, 60
# Fit the random forest model
fit_rf <- randomForest(label ~ followers + friends + favorites + sentiment_score_average,
                       data = data.subset,
                       mtry = 2,
                       ntree = 228,
                       importance = TRUE)

# Print the detailed summary of the model
# print(fit_rf)

# summary(fit_rf)
# stargazer(fit_rf, type="text")
plot(fit_rf)

# Optional: Confusion matrix to evaluate the model
# confusion_matrix <- confusionMatrix(predict(fit_rf, data.subset), data.subset$label)
# print(confusion_matrix)

```

    Warning message in randomForest.default(m, y, ...):
    “The response has five or fewer unique values.  Are you sure you want to do regression?”



    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_28_1.png)
    



```R
#@title Comparing Models
lm_preds <- predict(fit_lm, data.testset, type="response")
lm_preds <- ifelse(lm_preds > 0.5, 1, 0)
logistic_preds <- predict(fit_logistic, data.testset, type="response")
logistic_preds <- ifelse(logistic_preds > 0.5, 1, 0)
tree_preds <- predict(fit_tree1, data.testset[,c("followers", "friends", "favorites", "sentiment_score_average")])
tree_preds <- ifelse(tree_preds > 0.5, 1, 0)
rf_preds <- predict(fit_rf, data.testset, type="class")
rf_preds <- ifelse(rf_preds > 0.5, 1, 0)

lm_accuracy <- mean(lm_preds==data.testset$label)
logistic_accuracy <- mean(logistic_preds==data.testset$label)
tree_accuracy <- mean(tree_preds==data.testset$label)
rf_accuracy <- mean(rf_preds==data.testset$label)

data.frame(
  Model = c("Linear Model", "Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(lm_accuracy, logistic_accuracy, tree_accuracy, rf_accuracy)
)
```


<table class="dataframe">
<caption>A data.frame: 4 × 2</caption>
<thead>
	<tr><th scope=col>Model</th><th scope=col>Accuracy</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Linear Model       </td><td>0.5982143</td></tr>
	<tr><td>Logistic Regression</td><td>0.5892857</td></tr>
	<tr><td>Decision Tree      </td><td>0.5223214</td></tr>
	<tr><td>Random Forest      </td><td>0.6026786</td></tr>
</tbody>
</table>



#Text Analysis: Kaggle Dataset

## Bag of Words & Term Frequency

###Corpus: a collection of text\
 - `VCorpus()`: create Volatile Corpus\
 - `inspect()`: display detailed info of a corpus


```R
mycorpus1 <- VCorpus(VectorSource(data.subset$post_text))
mycorpus1
typeof(mycorpus1)   ## It is a list
# inspect the first corpus
inspect(mycorpus1[[1]])
# or use `as.character` to extract the text
as.character(mycorpus1[[1]])
```


    <<VCorpus>>
    Metadata:  corpus specific: 0, document level (indexed): 0
    Content:  documents: 16288



'list'


    <<PlainTextDocument>>
    Metadata:  7
    Content:  chars: 140
    
    It's just over 2 years since I was diagnosed with #anxiety and #depression. Today I'm taking a moment to reflect on how far I've come since.



'It\'s just over 2 years since I was diagnosed with #anxiety and #depression. Today I\'m taking a moment to reflect on how far I\'ve come since.'


###Data cleaning using `tm_map()`

Before transforming the text into a word frequency matrix, we should transform the text into a more standard format and clean the text by removing punctuation, numbers and some common words that do not have predictive power (a.k.a. [stopwords](https://kavita-ganesan.com/what-are-stop-words/#:~:text=Stop%20words%20are%20a%20set,on%20the%20important%20words%20instead); e.g. pronouns, prepositions, conjunctions). This can be tedious but is necessary for quality analyses. We use the `tm_map()` function with different available transformations including `removeNumbers()`, `removePunctuation()`, `removeWords()`, `stemDocument()`, and `stripWhitespace()`. It works like the `apply()` class function to apply the function to corpus. Details of each step are in the appendix. These cleaning techniques are not one-size-fits-all, and the techniques appropriate for your data will vary based on context.


```R
# Converts all words to lowercase
mycorpus_clean <- tm_map(mycorpus1, content_transformer(tolower))

# Removes common English stopwords (e.g. "with", "i")
mycorpus_clean <- tm_map(mycorpus_clean, removeWords, stopwords("english"))

# Removes any punctuation
# NOTE: This step may not be appropriate if you want to account for differences
#       on semantics depending on which sentence a word belongs to if you end up
#       using n-grams or k-skip-n-grams.
#       Instead, periods (or semicolons, etc.) can be replaced with a unique
#       token (e.g. "[PERIOD]") that retains this semantic meaning.
mycorpus_clean <- tm_map(mycorpus_clean, removePunctuation)

# Removes numbers
mycorpus_clean <- tm_map(mycorpus_clean, removeNumbers)

# Stem words
mycorpus_clean <- tm_map(mycorpus_clean, stemDocument, lazy = TRUE)
```

###Word frequency matrix

Now we transform each review into a word frequency matrix using the function `DocumentTermMatrix()`.


```R
dtm1 <- DocumentTermMatrix( mycorpus_clean )   ## library = collection of words for all documents
class(dtm1)
inspect(dtm1) # typeof(dtm1)  #length(dimnames(dtm1)$Terms)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'DocumentTermMatrix'</li><li>'simple_triplet_matrix'</li></ol>



    <<DocumentTermMatrix (documents: 16288, terms: 22534)>>
    Non-/sparse entries: 109270/366924522
    Sparsity           : 100%
    Maximal term length: 73
    Weighting          : term frequency (tf)
    Sample             :
           Terms
    Docs    can depress get just know like love now one thank
      10470   0       0   0    0    0    0    0   0   0     0
      10475   0       0   0    0    0    0    0   0   0     0
      13134   0       0   0    0    0    0    0   1   0     0
      4191    0       0   0    0    0    0    0   0   0     0
      4249    0       0   0    1    0    0    0   0   0     0
      4366    0       0   0    0    0    0    0   0   0     0
      4501    0       0   0    0    0    0    0   0   0     0
      6750    0       0   0    0    1    0    0   0   0     0
      7571    0       0   0    0    0    1    0   0   0     0
      7972    0       0   0    0    0    1    0   0   0     0


### Reduce the size of the bag

Many words do not appear nearly as often as others. If your cleaning was done appropriately, it will hopefully not lose much of the information if we drop such rare words. So, we first cut the bag to only include the words appearing at least 1% (or the frequency of your choice) of the time. This reduces the dimension of the features extracted to be analyzed.


```R
threshold <- .01*length(mycorpus_clean)   # 1% of the total documents
words.10 <- findFreqTerms(dtm1, lowfreq=threshold)  # words appearing at least among 1% of the documents
length(words.10)
tail(words.10, n=10)
```


81



<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'use'</li><li>'via'</li><li>'wait'</li><li>'want'</li><li>'watch'</li><li>'way'</li><li>'well'</li><li>'will'</li><li>'work'</li><li>'year'</li></ol>




```R
dtm.10<- DocumentTermMatrix(mycorpus_clean, control = list(dictionary = words.10))
dim(as.matrix(dtm.10))
colnames(dtm.10)[40:50]
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>16288</li><li>81</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'never'</li><li>'new'</li><li>'now'</li><li>'one'</li><li>'overcom'</li><li>'peopl'</li><li>'person'</li><li>'pleas'</li><li>'realdonaldtrump'</li><li>'realli'</li><li>'right'</li></ol>



**`removeSparseTerms()`**:

Anther way to reduce the size of the bag is to use `removeSparseTerms`


```R
dtm.10.2 <- removeSparseTerms(dtm1, 1-.01)  # control sparsity < .99
inspect(dtm.10.2)
# colnames(dtm.10.2)[1:50]
# words that are in dtm.10 but not in dtm.10.2
colnames(dtm.10)[!(colnames(dtm.10) %in% colnames(dtm.10.2))]
```

    <<DocumentTermMatrix (documents: 16288, terms: 74)>>
    Non-/sparse entries: 22745/1182567
    Sparsity           : 98%
    Maximal term length: 15
    Weighting          : term frequency (tf)
    Sample             :
           Terms
    Docs    can depress get just know like love now one thank
      12418   1       0   0    0    0    0    0   0   0     0
      1408    0       1   0    0    0    0    1   0   1     0
      5104    0       0   1    0    0    2    0   0   1     0
      6147    0       0   0    0    0    1    0   0   0     0
      7142    1       0   2    1    0    0    0   0   2     0
      7933    0       0   0    0    1    0    0   0   0     0
      8316    0       0   0    0    0    0    1   0   0     0
      932     1       0   0    0    1    0    0   0   0     0
      955     0       0   1    2    1    0    1   0   2     0
      9608    0       0   2    1    0    0    0   0   0     0



<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'guy'</li><li>'migrain'</li><li>'person'</li><li>'pleas'</li><li>'shit'</li><li>'sleep'</li><li>'tweet'</li></ol>



We end up with two different bags because

- `findFreqTerms()`: counts a word multiple times if it appears multiple times in one document.  
- `removeSparseTerms()`: keep words that appear at least once in X% of documents.  


### One step to get `DTM`

We consolidate all possible processing steps to the following clean `R-chunk`, turning texts (input) into `Document Term Frequency` which is a sparse matrix (output) to be used in the down-stream analyses.

All the `tm_map()` can be called inside `DocumentTermMatrix` under parameter called `control`. Here is how.


```R
# Turn texts to corpus
mycorpus1  <- VCorpus(VectorSource(data.subset$post_text))


# Control list for creating our DTM within DocumentTermMatrix
# Can tweak settings based off if you want punctuation, numbers, etc.
control_list <- list( tolower = TRUE,
                      removePunctuation = TRUE,
                      removeNumbers = TRUE,
                      stopwords = stopwords("english"),
                      stemming = TRUE)
# dtm with all terms:
dtm.10.long  <- DocumentTermMatrix(mycorpus1, control = control_list)
#inspect(dtm.10.long)

# kick out rare words
dtm.10<- removeSparseTerms(dtm.10.long, 1-.01)
inspect(dtm.10)

# look at the document 1 before and after cleaning
# inspect(mycorpus1[[1]])
# after cleaning
# colnames(as.matrix(dtm1[1, ]))[which(as.matrix(dtm1[1, ]) != 0)]
```

    <<DocumentTermMatrix (documents: 16288, terms: 81)>>
    Non-/sparse entries: 24779/1294549
    Sparsity           : 98%
    Maximal term length: 15
    Weighting          : term frequency (tf)
    Sample             :
           Terms
    Docs    can depress dont get just know like now one thank
      12418   1       0    0   0    0    0    0   0   0     0
      1408    0       1    0   0    0    0    0   0   1     0
      3171    0       0    1   0    1    0    0   0   0     0
      6147    0       0    0   0    0    0    1   0   0     0
      7142    1       0    0   2    1    0    0   0   2     0
      7933    0       0    1   0    0    1    0   0   0     0
      8455    0       0    1   0    0    0    0   0   0     0
      932     1       0    1   0    0    1    0   0   0     0
      955     0       0    1   1    2    1    0   0   2     0
      9608    0       0    1   2    1    0    0   0   0     0


## N-grams

Using the `tm` package along with a custom tokenizer, we can now implement n-grams using R! As is, the tokenizer is set to produce bugrams but you should be able to change this by tweaking the variable `n`.


```R
# The 'n' for n-grams
# n=2 is bi-grams
n <- 2

# Our custom tokenizer
# Uses the ngrams function from the NLP package
# Right now this is for bigrams, but you can change it by changing the value of
# the variable n (includes N-grams for any N <= n)
ngram_tokenizer <- function(x, n) {
  unlist(lapply(ngrams(words(x), 1:n), paste, collapse = "_"), use.names = FALSE)
}
```

We next prepare a clean corpus


```R
# Converts all words to lowercase
mycorpus1  <- VCorpus(VectorSource(data.subset$post_text))
mycorpus_clean <- tm_map(mycorpus1, content_transformer(tolower))

# Removes common English stopwords (e.g. "with", "i")
mycorpus_clean <- tm_map(mycorpus_clean, removeWords, stopwords("english"))

# Removes any punctuation
# NOTE: This step may not be appropriate if you want to account for differences
#       on semantics depending on which sentence a word belongs to if you end up
#       using n-grams or k-skip-n-grams.
#       Instead, periods (or semicolons, etc.) can be replaced with a unique
#       token (e.g. "[PERIOD]") that retains this semantic meaning.
mycorpus_clean <- tm_map(mycorpus_clean, removePunctuation)

# Removes numbers
mycorpus_clean <- tm_map(mycorpus_clean, removeNumbers)

# Stem words
mycorpus_clean <- tm_map(mycorpus_clean, stemDocument, lazy = TRUE)
```

Apply the bigram tokenizer to the first review, we see the library or word of bag is enlarged.


```R
inspect(mycorpus_clean[[1]])  # see review 1 again
ngram_tokenizer(mycorpus_clean[[1]], 2) # output bigram of review 1
```

    <<PlainTextDocument>>
    Metadata:  7
    Content:  chars: 78
    
    just year sinc diagnos anxieti depress today take moment reflect far come sinc



<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'just'</li><li>'year'</li><li>'sinc'</li><li>'diagnos'</li><li>'anxieti'</li><li>'depress'</li><li>'today'</li><li>'take'</li><li>'moment'</li><li>'reflect'</li><li>'far'</li><li>'come'</li><li>'sinc'</li><li>'just_year'</li><li>'year_sinc'</li><li>'sinc_diagnos'</li><li>'diagnos_anxieti'</li><li>'anxieti_depress'</li><li>'depress_today'</li><li>'today_take'</li><li>'take_moment'</li><li>'moment_reflect'</li><li>'reflect_far'</li><li>'far_come'</li><li>'come_sinc'</li></ol>



While we won't run any analyses with the n-grams, we encourage you to play around with some of the settings to see how the DTM changes, as well as how the analysis might be improved (or worsened) by including the higher order n-grams.


```R
# use ngram_tokenizer()
control_list_ngram <- list(tokenize = function(x) ngram_tokenizer(x, 2))

dtm_ngram <- DocumentTermMatrix(mycorpus_clean, control_list_ngram)
# kick out rare words
dtm_ngram.10 <- removeSparseTerms(dtm_ngram, 1-.01)
inspect(dtm_ngram.10)
```

    <<DocumentTermMatrix (documents: 16288, terms: 80)>>
    Non-/sparse entries: 24004/1279036
    Sparsity           : 98%
    Maximal term length: 17
    Weighting          : term frequency (tf)
    Sample             :
           Terms
    Docs    can depress get just know like love now one thank
      12418   1       0   0    0    0    0    0   0   0     0
      1408    0       1   0    0    0    0    1   0   1     0
      14336   0       0   0    0    0    0    0   1   0     1
      14337   0       0   0    0    0    0    0   1   0     1
      14351   0       0   0    0    0    0    0   1   0     1
      14496   0       0   0    2    0    0    0   1   0     1
      14498   0       0   0    2    0    0    0   1   0     1
      7142    1       0   2    1    0    0    0   0   2     0
      7933    0       0   0    0    1    0    0   0   0     0
      955     0       0   1    2    1    0    1   0   2     0


##Analysis

Once we have turned a text into a vector, we can then apply any methods suitable for the settings. In our case we will use logistic regression models and LASSO to explore the relationship between `label` and `text`.

### Data preparation
The following chunk output a data frame called data2, combining everything and all the tf's of each word.


```R
names(data.subset)
# Combine the original data with the text matrix
data1.temp <- data.frame(data.subset,as.matrix(dtm_ngram.10) )
dim(data1.temp)
names(data1.temp)[1:30]
#str(data1.temp)
# data2 consists of date, label and all the top 1% words
data2 <- data1.temp[, c(7:ncol(data1.temp))]
#data2 <- data1.temp
names(data2)[1:20]
dim(data2)  ### remember we have only run 1000 rows

#### We have previously run the entire 100,000 rows and output the DTM out.
### if not, run and write as csv
fwrite(data2, "/content/data/Twitter_tm_freq.csv", row.names = FALSE)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'V1'</li><li>'post_text'</li><li>'user_id'</li><li>'followers'</li><li>'friends'</li><li>'favorites'</li><li>'statuses'</li><li>'label'</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>16288</li><li>88</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'V1'</li><li>'post_text'</li><li>'user_id'</li><li>'followers'</li><li>'friends'</li><li>'favorites'</li><li>'statuses'</li><li>'label'</li><li>'actual'</li><li>'amp'</li><li>'ask'</li><li>'azarkansero'</li><li>'back'</li><li>'best'</li><li>'can'</li><li>'come'</li><li>'day'</li><li>'depress'</li><li>'depress_treatment'</li><li>'even'</li><li>'feel'</li><li>'follow'</li><li>'follow_twitter'</li><li>'friend'</li><li>'fuck'</li><li>'genevieveverso'</li><li>'get'</li><li>'god'</li><li>'good'</li><li>'got'</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'statuses'</li><li>'label'</li><li>'actual'</li><li>'amp'</li><li>'ask'</li><li>'azarkansero'</li><li>'back'</li><li>'best'</li><li>'can'</li><li>'come'</li><li>'day'</li><li>'depress'</li><li>'depress_treatment'</li><li>'even'</li><li>'feel'</li><li>'follow'</li><li>'follow_twitter'</li><li>'friend'</li><li>'fuck'</li><li>'genevieveverso'</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>16288</li><li>82</li></ol>



### Splitting data

Let's first read in the processed data with text being a vector.


```R
data2 <- fread("/content/data/Twitter_tm_freq.csv")  #dim(data2)
names(data2)[1:20] # notice that user_id, stars and date are in the data2
dim(data2)
data2 <- data2 %>% mutate(label=as.factor(label))# %>% select(-"label_numeric")
table(data2$label)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'statuses'</li><li>'label'</li><li>'actual'</li><li>'amp'</li><li>'ask'</li><li>'azarkansero'</li><li>'back'</li><li>'best'</li><li>'can'</li><li>'come'</li><li>'day'</li><li>'depress'</li><li>'depress_treatment'</li><li>'even'</li><li>'feel'</li><li>'follow'</li><li>'follow_twitter'</li><li>'friend'</li><li>'fuck'</li><li>'genevieveverso'</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>16288</li><li>82</li></ol>




    
       0    1 
    8108 8180 


As one standard machine learning process, we first split data into two sets one training data and the other testing data. We use training data to build models, choose models etc and make final recommendations. We then report the performance using the testing data.

Reserve 100 randomly chosen rows as our test data (`data2.test`) and the remaining 900 as the training data (`data2.train`)


```R
set.seed(1)  # for the purpose of reporducibility
n <- nrow(data2)
test.index <- sample(n, 0.1*n)
# length(test.index)
data2.test <- data2[test.index, -1] # only keep label and the texts
data2.train <- data2[-test.index, -1]
dim(data2.train)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>14660</li><li>81</li></ol>



### Analysis 1: LASSO

We first explore a logistic regression model using LASSO. The following R-chunk runs a LASSO model with $\alpha=.99$. The reason we take an elastic net is to enjoy the nice properties from both `LASSO` (impose sparsity) and `Ridge` (computationally stable).

LASSO takes sparse design matrix as an input. So make sure to extract the sparse matrix first as the input in cv.glm(). It takes about 1 minute to run cv.glm() with sparse matrix or 11 minutes using the regular design matrix.


```R
### or try `sparse.model.matrix()` which is much faster
y <- data2.train$label
X1 <- sparse.model.matrix(label~., data=data2.train)[, -1]
set.seed(2)
result.lasso <- cv.glmnet(X1, y, alpha=.99, family=binomial(logit))
# 1.25 minutes in my MAC
plot(result.lasso)
# this this may take you long time to run, we save result.lasso
saveRDS(result.lasso, file="data/TextMining_lasso.RDS")
# result.lasso can be assigned back by
# result.lasso <- readRDS("data/TextMining_lasso.RDS")

# number of non-zero words picked up by LASSO when using lambda.1se
# coef.1se <- coef(result.lasso, s="lambda.1se")
# lasso.words <- coef.1se@Dimnames[[1]] [coef.1se@i][-1] # non-zero variables without intercept.
# summary(lasso.words)


# or our old way
coef.1se <- coef(result.lasso, s="lambda.1se")
coef.1se <- coef.1se[which(coef.1se !=0),]
lasso.words <- rownames(as.matrix(coef.1se))[-1]
summary(lasso.words)



### cv.glmnt with the non-sparse design matrix takes much longer
# X <- as.matrix(data2.train[, -1]) # we can use as.matrix directly her
#### Be careful to run the following LASSO.
#set.seed(2)
#result.lasso <- cv.glmnet(X, y, alpha=.99, family="binomial")
# 10 minutes in my MAC
#plot(result.lasso)
```


       Length     Class      Mode 
           59 character character 



    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_65_1.png)
    


### Analysis 2: Relaxed LASSO

As an alternative model we will run our relaxed `LASSO`. Input variables are chosen by `LASSO` and we get a regular logistic regression model. Once again it is stored as `result.glm` in `TextMining.RData`.


```R
sel_cols <- c("label", lasso.words)
# use all_of() to specify we would like to select variables in sel_cols
data_sub <- data2.train %>% select(all_of(sel_cols))
result.glm <- glm(label~., family=binomial, data_sub) # takes 3.5 minutes
## glm() returns a big object with unnecessary information
# saveRDS(result.glm,
#      file = "data/TextMining_glm.RDS")

## trim the glm() fat from
## https://win-vector.com/2014/05/30/trimming-the-fat-from-glm-models-in-r/
stripGlmLR = function(cm) {
  cm$y = c()
  cm$model = c()

  cm$residuals = c()
  cm$fitted.values = c()
  cm$effects = c()
  cm$qr$qr = c()
  cm$linear.predictors = c()
  cm$weights = c()
  cm$prior.weights = c()
  cm$data = c()


  cm$family$variance = c()
  cm$family$dev.resids = c()
  cm$family$aic = c()
  cm$family$validmu = c()
  cm$family$simulate = c()
  attr(cm$terms,".Environment") = c()
  attr(cm$formula,".Environment") = c()

  cm
}

#result.glm.small <- stripGlmLR(result.glm)
result.glm.small <- result.glm

saveRDS(result.glm.small,
     file = "/content/data/TextMining_glm_small.RDS")
```

    Warning message:
    “glm.fit: fitted probabilities numerically 0 or 1 occurred”



```R
View(result.glm.small)
```


    
    Call:  glm(formula = label ~ ., family = binomial, data = data_sub)
    
    Coefficients:
        (Intercept)              amp              ask      azarkansero  
           -0.20870         -0.15435          0.45942         15.68073  
               back             best              can              day  
            0.34165         -0.42198          0.06874          0.68845  
            depress             feel           friend   genevieveverso  
            3.08364          0.71367          0.21416         15.16053  
                get              god              got            happi  
            0.69338         -0.92267          0.49489          0.26220  
               help              hey             hope             just  
            0.96364         -1.02911         -0.53614          0.27295  
               life             like              lol             look  
            0.42904          0.05321          0.53441          0.10717  
               love             make              man        misslusyd  
            0.27279          0.47850         -0.93853         15.08569  
               much             need            never              new  
            0.25853          0.09498         -0.24005          0.16968  
         now_follow              one          overcom  realdonaldtrump  
          -16.91854         -0.30769          1.89115         -5.37890  
             realli            right           someon            start  
            0.37095         -0.17319          0.29781          0.39808  
              still             take             talk            thank  
           -0.29167          0.43699          1.31728         -0.44256  
              thing            think             time            today  
            0.15011          0.55579          0.24965          0.85688  
          treatment              tri            trump    twitter_thank  
            3.99057          0.41929         -5.12768               NA  
                use              via             wait             want  
            0.24033          1.19973         -0.19395         -0.18153  
              watch              way             will             work  
           -0.38245          0.59473         -0.28046          0.16581  
    
    Degrees of Freedom: 14659 Total (i.e. Null);  14601 Residual
    Null Deviance:	    20320 
    Residual Deviance: 17220 	AIC: 17340



```R
View(data_sub[1:20])
```


<table class="dataframe">
<caption>A data.table: 20 × 60</caption>
<thead>
	<tr><th scope=col>label</th><th scope=col>amp</th><th scope=col>ask</th><th scope=col>azarkansero</th><th scope=col>back</th><th scope=col>best</th><th scope=col>can</th><th scope=col>day</th><th scope=col>depress</th><th scope=col>feel</th><th scope=col>⋯</th><th scope=col>trump</th><th scope=col>twitter_thank</th><th scope=col>use</th><th scope=col>via</th><th scope=col>wait</th><th scope=col>want</th><th scope=col>watch</th><th scope=col>way</th><th scope=col>will</th><th scope=col>work</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>⋯</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>⋯</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
</tbody>
</table>



### Analysis 3: Word cloud! (Sentiment analysis)

Logistic regression model connects the chance of `being good` given a text/review. What are the `nice` (or positive) words and how much it influence the chance being good? In addition to explore the set of good words we also build word clouds to visualize the correlation between positive words and negative words.

1. Order the glm positive coefficients (positive words). Show them in a word cloud. The size of the words indicates the strength of positive correlation between that word and the chance being a good rating.

2. Order the glm negative coefficients (negative words)

TIME TO PLOT A WORD CLOUD!! Plot the world clouds, the size of the words are prop to the logistic reg coef's


```R
result.glm <- readRDS("data/TextMining_glm_small.RDS")
result.glm.coef <- coef(result.glm)[-1]  # took intercept out
print(coef(result.glm))
```

        (Intercept)             amp             ask     azarkansero            back 
        -0.20869952     -0.15435338      0.45941997     15.68072541      0.34165213 
               best             can             day         depress            feel 
        -0.42197732      0.06874431      0.68844862      3.08364432      0.71366667 
             friend  genevieveverso             get             god             got 
         0.21415932     15.16052578      0.69337746     -0.92266962      0.49489322 
              happi            help             hey            hope            just 
         0.26220126      0.96364244     -1.02910992     -0.53613658      0.27295235 
               life            like             lol            look            love 
         0.42904275      0.05321296      0.53440788      0.10717083      0.27278595 
               make             man       misslusyd            much            need 
         0.47850105     -0.93853027     15.08568524      0.25853356      0.09497677 
              never             new      now_follow             one         overcom 
        -0.24004798      0.16968019    -16.91853634     -0.30769353      1.89115320 
    realdonaldtrump          realli           right          someon           start 
        -5.37889848      0.37094541     -0.17318791      0.29780615      0.39807759 
              still            take            talk           thank           thing 
        -0.29166596      0.43698719      1.31728390     -0.44255891      0.15011281 
              think            time           today       treatment             tri 
         0.55579451      0.24965406      0.85687753      3.99057017      0.41928535 
              trump   twitter_thank             use             via            wait 
        -5.12768106              NA      0.24032660      1.19973005     -0.19395472 
               want           watch             way            will            work 
        -0.18152947     -0.38244561      0.59472733     -0.28046147      0.16580903 



```R
result.glm <- readRDS("data/TextMining_glm_small.RDS")
result.glm.coef <- coef(result.glm)[-1]  # took intercept out
result.glm.coef
hist(result.glm.coef)

# pick up the positive coef's which are positively related to the prob of being depressed
good.glm <- result.glm.coef[which(result.glm.coef > 0)]
good.glm <- good.glm
names(good.glm)  # which words are positively associated with depression

good.fre <- sort(good.glm, decreasing = TRUE) # sort the coef's
round(good.fre, 4) # leading 20 positive words, amazing!
length(good.fre)  # 390 good words

# hist(as.matrix(good.fre), breaks=30, col="red")
good.word <- names(good.fre)  # good words with a decreasing order in the coeff's
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>amp</dt><dd>-0.154353383243174</dd><dt>ask</dt><dd>0.459419973276242</dd><dt>azarkansero</dt><dd>15.6807254133312</dd><dt>back</dt><dd>0.341652127055276</dd><dt>best</dt><dd>-0.421977318849248</dd><dt>can</dt><dd>0.0687443089344538</dd><dt>day</dt><dd>0.68844861664952</dd><dt>depress</dt><dd>3.08364431567679</dd><dt>feel</dt><dd>0.71366666774533</dd><dt>friend</dt><dd>0.214159323799959</dd><dt>genevieveverso</dt><dd>15.1605257820476</dd><dt>get</dt><dd>0.693377459669337</dd><dt>god</dt><dd>-0.922669623554665</dd><dt>got</dt><dd>0.494893219933516</dd><dt>happi</dt><dd>0.262201263953269</dd><dt>help</dt><dd>0.963642436730797</dd><dt>hey</dt><dd>-1.0291099184519</dd><dt>hope</dt><dd>-0.536136580154622</dd><dt>just</dt><dd>0.272952353813912</dd><dt>life</dt><dd>0.429042747809117</dd><dt>like</dt><dd>0.0532129572278479</dd><dt>lol</dt><dd>0.534407877178159</dd><dt>look</dt><dd>0.107170828900866</dd><dt>love</dt><dd>0.272785946677719</dd><dt>make</dt><dd>0.478501053204454</dd><dt>man</dt><dd>-0.938530266681544</dd><dt>misslusyd</dt><dd>15.0856852390427</dd><dt>much</dt><dd>0.258533556692016</dd><dt>need</dt><dd>0.0949767734274871</dd><dt>never</dt><dd>-0.240047981200442</dd><dt>new</dt><dd>0.169680194212623</dd><dt>now_follow</dt><dd>-16.9185363361622</dd><dt>one</dt><dd>-0.307693533146417</dd><dt>overcom</dt><dd>1.89115319539379</dd><dt>realdonaldtrump</dt><dd>-5.37889847834728</dd><dt>realli</dt><dd>0.370945408819479</dd><dt>right</dt><dd>-0.173187905914923</dd><dt>someon</dt><dd>0.297806150896758</dd><dt>start</dt><dd>0.398077593329702</dd><dt>still</dt><dd>-0.291665955373473</dd><dt>take</dt><dd>0.436987188222337</dd><dt>talk</dt><dd>1.317283895156</dd><dt>thank</dt><dd>-0.442558912255612</dd><dt>thing</dt><dd>0.150112811869734</dd><dt>think</dt><dd>0.55579450926683</dd><dt>time</dt><dd>0.249654060151126</dd><dt>today</dt><dd>0.856877526028384</dd><dt>treatment</dt><dd>3.99057016890573</dd><dt>tri</dt><dd>0.419285354292917</dd><dt>trump</dt><dd>-5.12768106421453</dd><dt>twitter_thank</dt><dd>&lt;NA&gt;</dd><dt>use</dt><dd>0.24032660441938</dd><dt>via</dt><dd>1.19973004908238</dd><dt>wait</dt><dd>-0.193954717023049</dd><dt>want</dt><dd>-0.181529474067029</dd><dt>watch</dt><dd>-0.382445610988367</dd><dt>way</dt><dd>0.594727327258758</dd><dt>will</dt><dd>-0.280461467228575</dd><dt>work</dt><dd>0.165809032522599</dd></dl>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'ask'</li><li>'azarkansero'</li><li>'back'</li><li>'can'</li><li>'day'</li><li>'depress'</li><li>'feel'</li><li>'friend'</li><li>'genevieveverso'</li><li>'get'</li><li>'got'</li><li>'happi'</li><li>'help'</li><li>'just'</li><li>'life'</li><li>'like'</li><li>'lol'</li><li>'look'</li><li>'love'</li><li>'make'</li><li>'misslusyd'</li><li>'much'</li><li>'need'</li><li>'new'</li><li>'overcom'</li><li>'realli'</li><li>'someon'</li><li>'start'</li><li>'take'</li><li>'talk'</li><li>'thing'</li><li>'think'</li><li>'time'</li><li>'today'</li><li>'treatment'</li><li>'tri'</li><li>'use'</li><li>'via'</li><li>'way'</li><li>'work'</li></ol>




<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>azarkansero</dt><dd>15.6807</dd><dt>genevieveverso</dt><dd>15.1605</dd><dt>misslusyd</dt><dd>15.0857</dd><dt>treatment</dt><dd>3.9906</dd><dt>depress</dt><dd>3.0836</dd><dt>overcom</dt><dd>1.8912</dd><dt>talk</dt><dd>1.3173</dd><dt>via</dt><dd>1.1997</dd><dt>help</dt><dd>0.9636</dd><dt>today</dt><dd>0.8569</dd><dt>feel</dt><dd>0.7137</dd><dt>get</dt><dd>0.6934</dd><dt>day</dt><dd>0.6884</dd><dt>way</dt><dd>0.5947</dd><dt>think</dt><dd>0.5558</dd><dt>lol</dt><dd>0.5344</dd><dt>got</dt><dd>0.4949</dd><dt>make</dt><dd>0.4785</dd><dt>ask</dt><dd>0.4594</dd><dt>take</dt><dd>0.437</dd><dt>life</dt><dd>0.429</dd><dt>tri</dt><dd>0.4193</dd><dt>start</dt><dd>0.3981</dd><dt>realli</dt><dd>0.3709</dd><dt>back</dt><dd>0.3417</dd><dt>someon</dt><dd>0.2978</dd><dt>just</dt><dd>0.273</dd><dt>love</dt><dd>0.2728</dd><dt>happi</dt><dd>0.2622</dd><dt>much</dt><dd>0.2585</dd><dt>time</dt><dd>0.2497</dd><dt>use</dt><dd>0.2403</dd><dt>friend</dt><dd>0.2142</dd><dt>new</dt><dd>0.1697</dd><dt>work</dt><dd>0.1658</dd><dt>thing</dt><dd>0.1501</dd><dt>look</dt><dd>0.1072</dd><dt>need</dt><dd>0.095</dd><dt>can</dt><dd>0.0687</dd><dt>like</dt><dd>0.0532</dd></dl>




40



    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_74_4.png)
    



```R
names(good.glm)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'ask'</li><li>'azarkansero'</li><li>'back'</li><li>'can'</li><li>'day'</li><li>'depress'</li><li>'feel'</li><li>'friend'</li><li>'genevieveverso'</li><li>'get'</li><li>'got'</li><li>'happi'</li><li>'help'</li><li>'just'</li><li>'life'</li><li>'like'</li><li>'lol'</li><li>'look'</li><li>'love'</li><li>'make'</li><li>'misslusyd'</li><li>'much'</li><li>'need'</li><li>'new'</li><li>'overcom'</li><li>'realli'</li><li>'someon'</li><li>'start'</li><li>'take'</li><li>'talk'</li><li>'thing'</li><li>'think'</li><li>'time'</li><li>'today'</li><li>'treatment'</li><li>'tri'</li><li>'use'</li><li>'via'</li><li>'way'</li><li>'work'</li></ol>



The above chunk shows in detail about the weight for positive words. We only show the positive word-cloud here. One can tell the large positive words are making sense in the way we do expect the collection of large words should have a positive tone towards the restaurant being reviewed.


```R
cor.special <- brewer.pal(8,"Dark2")  # set up a pretty color scheme
wordcloud(good.word, good.fre,  # make a word cloud
          colors=cor.special,
          ordered.colors=F, min.freq=0)
```


    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_77_0.png)
    


###Negative Wordcloud

Similarly to the negative coef's which is positively correlated to the probability of having depression.


```R
bad.glm <- result.glm.coef[which(result.glm.coef < 0)]
# names(bad.glm)[1:50]

cor.special <- brewer.pal(6,"Dark2")
bad.fre <- sort(-bad.glm, decreasing = TRUE)
round(bad.fre, 4)

# hist(as.matrix(bad.fre), breaks=30, col="green")
bad.word <- names(bad.fre)
wordcloud(bad.word, bad.fre,
          color=cor.special, ordered.colors=F, min.freq=0)
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>now_follow</dt><dd>16.9185</dd><dt>realdonaldtrump</dt><dd>5.3789</dd><dt>trump</dt><dd>5.1277</dd><dt>hey</dt><dd>1.0291</dd><dt>man</dt><dd>0.9385</dd><dt>god</dt><dd>0.9227</dd><dt>hope</dt><dd>0.5361</dd><dt>thank</dt><dd>0.4426</dd><dt>best</dt><dd>0.422</dd><dt>watch</dt><dd>0.3824</dd><dt>one</dt><dd>0.3077</dd><dt>still</dt><dd>0.2917</dd><dt>will</dt><dd>0.2805</dd><dt>never</dt><dd>0.24</dd><dt>wait</dt><dd>0.194</dd><dt>want</dt><dd>0.1815</dd><dt>right</dt><dd>0.1732</dd><dt>amp</dt><dd>0.1544</dd></dl>




    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_80_1.png)
    


###Testing

LASSO Prediction


```R
predict.lasso <- ifelse(predict(result.lasso, as.matrix(data2.test[, -1]), type = "response", s="lambda.1se")>.5, "1", "0")
  # output lasso estimates of prob's
#predict.lasso <- predict(result.lasso, as.matrix(data2.test[, -1]), type = "class", s="lambda.1se")
  # output majority vote labels

# LASSO accuracy
mean(data2.test$label == predict.lasso)   #0.64004914004914; 0.637592137592138 with n-grams
```


0.637592137592138


Relaxed LASSO Prediction


```R
predict.glm <- predict(result.glm, data2.test, type = "response")
class.glm <- ifelse(predict.glm > .5, "1", "0")
# length(class.glm)

testaccuracy.glm <- mean(data2.test$label == class.glm)
testaccuracy.glm   #classification accuracy is 0.643120393120393; 0.640663390663391 with n-grams
```


0.640663390663391


#Text Analysis: Safa Dataset

## Bag of Words & Term Frequency

###Corpus: a collection of text
 - `VCorpus()`: create Volatile Corpus
 - `inspect()`: display detailed info of a corpus


```R
data.subset <- fread("/content/data/Mental_Health_Twitter_Big.csv", stringsAsFactors = FALSE)
mycorpus1 <- VCorpus(VectorSource(data.subset$post_text))
mycorpus1
typeof(mycorpus1)   ## It is a list
# inspect the first corpus
inspect(mycorpus1[[1]])
# or use `as.character` to extract the text
as.character(mycorpus1[[1]])
```


    <<VCorpus>>
    Metadata:  corpus specific: 0, document level (indexed): 0
    Content:  documents: 20000



'list'


    <<PlainTextDocument>>
    Metadata:  7
    Content:  chars: 112
    
    The Morning After: The drone that can fly for two hours straight https://t.co/PUvsuAlyLb https://t.co/v1VboCYH4x



'The Morning After: The drone that can fly for two hours straight https://t.co/PUvsuAlyLb https://t.co/v1VboCYH4x'


###Data cleaning using `tm_map()`

Before transforming the text into a word frequency matrix, we should transform the text into a more standard format and clean the text by removing punctuation, numbers and some common words that do not have predictive power (a.k.a. [stopwords](https://kavita-ganesan.com/what-are-stop-words/#:~:text=Stop%20words%20are%20a%20set,on%20the%20important%20words%20instead); e.g. pronouns, prepositions, conjunctions). This can be tedious but is necessary for quality analyses. We use the `tm_map()` function with different available transformations including `removeNumbers()`, `removePunctuation()`, `removeWords()`, `stemDocument()`, and `stripWhitespace()`. It works like the `apply()` class function to apply the function to corpus. Details of each step are in the appendix. These cleaning techniques are not one-size-fits-all, and the techniques appropriate for your data will vary based on context.


```R
# Converts all words to lowercase
mycorpus_clean <- tm_map(mycorpus1, content_transformer(tolower))

# Removes common English stopwords (e.g. "with", "i")
mycorpus_clean <- tm_map(mycorpus_clean, removeWords, stopwords("english"))

# Removes any punctuation
# NOTE: This step may not be appropriate if you want to account for differences
#       on semantics depending on which sentence a word belongs to if you end up
#       using n-grams or k-skip-n-grams.
#       Instead, periods (or semicolons, etc.) can be replaced with a unique
#       token (e.g. "[PERIOD]") that retains this semantic meaning.
mycorpus_clean <- tm_map(mycorpus_clean, removePunctuation)

# Removes numbers
mycorpus_clean <- tm_map(mycorpus_clean, removeNumbers)

# Stem words
mycorpus_clean <- tm_map(mycorpus_clean, stemDocument, lazy = TRUE)
```

###Word frequency matrix

Now we transform each review into a word frequency matrix using the function `DocumentTermMatrix()`.


```R
dtm1 <- DocumentTermMatrix( mycorpus_clean )   ## library = collection of words for all documents
class(dtm1)
inspect(dtm1) # typeof(dtm1)  #length(dimnames(dtm1)$Terms)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'DocumentTermMatrix'</li><li>'simple_triplet_matrix'</li></ol>



    <<DocumentTermMatrix (documents: 20000, terms: 41861)>>
    Non-/sparse entries: 162727/837057273
    Sparsity           : 100%
    Maximal term length: 140
    Weighting          : term frequency (tf)
    Sample             :
           Terms
    Docs    can get just know like love one peopl think will
      13517   0   0    1    0    1    0   0     0     1    0
      1371    0   0    0    0    0    0   0     0     1    0
      16097   0   0    0    0    0    0   0     0     0    0
      16739   0   0    0    0    0    0   1     0     0    0
      3300    0   0    0    0    0    0   0     0     0    0
      5818    0   0    0    0    0    0   0     0     0    0
      6503    0   0    0    0    0    0   0     1     1    1
      761     0   0    0    0    0    0   0     0     0    0
      7644    0   0    0    0    0    0   0     0     0    2
      8205    0   0    1    0    0    0   0     0     0    0


### Reduce the size of the bag

Many words do not appear nearly as often as others. If your cleaning was done appropriately, it will hopefully not lose much of the information if we drop such rare words. So, we first cut the bag to only include the words appearing at least 1% (or the frequency of your choice) of the time. This reduces the dimension of the features extracted to be analyzed.


```R
threshold <- .01*length(mycorpus_clean)   # 1% of the total documents
words.10 <- findFreqTerms(dtm1, lowfreq=threshold)  # words appearing at least among 1% of the documents
length(words.10)
tail(words.10, n=10)
```


90



<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'vote'</li><li>'want'</li><li>'watch'</li><li>'way'</li><li>'well'</li><li>'will'</li><li>'work'</li><li>'yeah'</li><li>'year'</li><li>'yes'</li></ol>




```R
dtm.10<- DocumentTermMatrix(mycorpus_clean, control = list(dictionary = words.10))
dim(as.matrix(dtm.10))
colnames(dtm.10)[40:50]
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>20000</li><li>90</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'look'</li><li>'lot'</li><li>'love'</li><li>'made'</li><li>'make'</li><li>'man'</li><li>'mani'</li><li>'mean'</li><li>'much'</li><li>'need'</li><li>'never'</li></ol>



**`removeSparseTerms()`**:

Anther way to reduce the size of the bag is to use `removeSparseTerms`


```R
dtm.10.2 <- removeSparseTerms(dtm1, 1-.01)  # control sparsity < .99
inspect(dtm.10.2)
# colnames(dtm.10.2)[1:50]
# words that are in dtm.10 but not in dtm.10.2
colnames(dtm.10)[!(colnames(dtm.10) %in% colnames(dtm.10.2))]
```

    <<DocumentTermMatrix (documents: 20000, terms: 87)>>
    Non-/sparse entries: 32985/1707015
    Sparsity           : 98%
    Maximal term length: 6
    Weighting          : term frequency (tf)
    Sample             :
           Terms
    Docs    can get just know like love one peopl think will
      10088   0   7    0    0    0    0   0     0     0    0
      12942   0   3    0    0    0    0   0     0     0    0
      14203   2   1    1    0    1    0   1     1     0    0
      16184   0   0    1    1    0    0   0     0     0    2
      1727    0   0    1    1    4    0   0     1     0    0
      19891   0   0    3    0    2    0   0     0     0    0
      2842    0   0    0    0    1    2   1     2     0    1
      5382    0   0    0    2    1    0   0     0     1    0
      657     0   0    0    0    2    1   0     0     0    1
      7561    0   0    0    1    0    0   0     1     1    0



<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'’ve'</li><li>'tell'</li><li>'vote'</li></ol>



We end up with two different bags because

- `findFreqTerms()`: counts a word multiple times if it appears multiple times in one document.  
- `removeSparseTerms()`: keep words that appear at least once in X% of documents.  


### One step to get `DTM`

We consolidate all possible processing steps to the following clean `R-chunk`, turning texts (input) into `Document Term Frequency` which is a sparse matrix (output) to be used in the down-stream analyses.

All the `tm_map()` can be called inside `DocumentTermMatrix` under parameter called `control`. Here is how.


```R
# Turn texts to corpus
mycorpus1  <- VCorpus(VectorSource(data.subset$post_text))


# Control list for creating our DTM within DocumentTermMatrix
# Can tweak settings based off if you want punctuation, numbers, etc.
control_list <- list( tolower = TRUE,
                      removePunctuation = TRUE,
                      removeNumbers = TRUE,
                      stopwords = stopwords("english"),
                      stemming = TRUE)
# dtm with all terms:
dtm.10.long  <- DocumentTermMatrix(mycorpus1, control = control_list)
#inspect(dtm.10.long)

# kick out rare words
dtm.10<- removeSparseTerms(dtm.10.long, 1-.01)
inspect(dtm.10)

# look at the document 1 before and after cleaning
# inspect(mycorpus1[[1]])
# after cleaning
# colnames(as.matrix(dtm1[1, ]))[which(as.matrix(dtm1[1, ]) != 0)]
```

    <<DocumentTermMatrix (documents: 20000, terms: 92)>>
    Non-/sparse entries: 35148/1804852
    Sparsity           : 98%
    Maximal term length: 6
    Weighting          : term frequency (tf)
    Sample             :
           Terms
    Docs    can get i’m just know like love one peopl will
      10088   0   7   0    0    0    0    0   0     0    0
      12942   0   3   0    0    0    0    0   0     0    0
      14203   2   1   0    1    0    1    0   1     1    0
      15378   0   0   0    2    0    2    0   0     0    0
      15901   0   0   1    0    0    1    1   0     1    0
      16184   0   0   0    1    1    0    0   0     0    2
      1727    0   0   1    1    1    4    0   0     1    0
      19891   0   0   0    3    0    2    0   0     0    0
      2842    0   0   0    0    0    1    2   1     2    1
      7561    0   0   2    0    1    0    0   0     1    0


## N-grams

Using the `tm` package along with a custom tokenizer, we can now implement n-grams using R! As is, the tokenizer is set to produce bugrams but you should be able to change this by tweaking the variable `n`.


```R
# The 'n' for n-grams
# n=2 is bi-grams
n <- 2

# Our custom tokenizer
# Uses the ngrams function from the NLP package
# Right now this is for bigrams, but you can change it by changing the value of
# the variable n (includes N-grams for any N <= n)
ngram_tokenizer <- function(x, n) {
  unlist(lapply(ngrams(words(x), 1:n), paste, collapse = "_"), use.names = FALSE)
}
```

We next prepare a clean copus


```R
# Converts all words to lowercase
mycorpus1  <- VCorpus(VectorSource(data.subset$post_text))
mycorpus_clean <- tm_map(mycorpus1, content_transformer(tolower))

# Removes common English stopwords (e.g. "with", "i")
mycorpus_clean <- tm_map(mycorpus_clean, removeWords, stopwords("english"))

# Removes any punctuation
# NOTE: This step may not be appropriate if you want to account for differences
#       on semantics depending on which sentence a word belongs to if you end up
#       using n-grams or k-skip-n-grams.
#       Instead, periods (or semicolons, etc.) can be replaced with a unique
#       token (e.g. "[PERIOD]") that retains this semantic meaning.
mycorpus_clean <- tm_map(mycorpus_clean, removePunctuation)

# Removes numbers
mycorpus_clean <- tm_map(mycorpus_clean, removeNumbers)

# Stem words
mycorpus_clean <- tm_map(mycorpus_clean, stemDocument, lazy = TRUE)
```

Apply the bigram tokenizer to the first review, we see the library or word of bag is enlarged.


```R
inspect(mycorpus_clean[[1]])  # see review 1 again
ngram_tokenizer(mycorpus_clean[[1]], 2) # output bigram of review 1
```

    <<PlainTextDocument>>
    Metadata:  7
    Content:  chars: 72
    
    morn drone can fli two hour straight httpstcopuvsualylb httpstcovvbocyhx



<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'morn'</li><li>'drone'</li><li>'can'</li><li>'fli'</li><li>'two'</li><li>'hour'</li><li>'straight'</li><li>'httpstcopuvsualylb'</li><li>'httpstcovvbocyhx'</li><li>'morn_drone'</li><li>'drone_can'</li><li>'can_fli'</li><li>'fli_two'</li><li>'two_hour'</li><li>'hour_straight'</li><li>'straight_httpstcopuvsualylb'</li><li>'httpstcopuvsualylb_httpstcovvbocyhx'</li></ol>



While we won't run any analyses with the n-grams, we encourage you to play around with some of the settings to see how the DTM changes, as well as how the analysis might be improved (or worsened) by including the higher order n-grams.


```R
# use ngram_tokenizer()
control_list_ngram <- list(tokenize = function(x) ngram_tokenizer(x, 2))

dtm_ngram <- DocumentTermMatrix(mycorpus_clean, control_list_ngram)
# kick out rare words
dtm_ngram.10 <- removeSparseTerms(dtm_ngram, 1-.01)
inspect(dtm_ngram.10)
```

    <<DocumentTermMatrix (documents: 20000, terms: 87)>>
    Non-/sparse entries: 32985/1707015
    Sparsity           : 98%
    Maximal term length: 6
    Weighting          : term frequency (tf)
    Sample             :
           Terms
    Docs    can get just know like love one peopl think will
      10088   0   7    0    0    0    0   0     0     0    0
      12942   0   3    0    0    0    0   0     0     0    0
      14203   2   1    1    0    1    0   1     1     0    0
      16184   0   0    1    1    0    0   0     0     0    2
      1727    0   0    1    1    4    0   0     1     0    0
      19891   0   0    3    0    2    0   0     0     0    0
      2842    0   0    0    0    1    2   1     2     0    1
      5382    0   0    0    2    1    0   0     0     1    0
      657     0   0    0    0    2    1   0     0     0    1
      7561    0   0    0    1    0    0   0     1     1    0


##Analysis

Once we have turned a text into a vector, we can then apply any methods suitable for the settings. In our case we will use logistic regression models and LASSO to explore the relationship between `label` and `text`.

### Data preparation
The following chunk output a data frame called data2, combining everything and all the tf's of each word.


```R
names(data.subset)
# Combine the original data with the text matrix
data1.temp <- data.frame(data.subset,as.matrix(dtm_ngram.10) )
dim(data1.temp)
names(data1.temp)[1:30]
#str(data1.temp)
# data2 consists of date, label and all the top 1% words
data2 <- data1.temp[, -1]
#data2 <- data1.temp
names(data2)[1:20]
dim(data2)  ### remember we have only run 1000 rows

#### We have previously run the entire 100,000 rows and output the DTM out.
### if not, run and write as csv
fwrite(data2, "/content/data/Twitter_tm_freq.csv", row.names = FALSE)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'post_text'</li><li>'label'</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>20000</li><li>89</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'post_text'</li><li>'label'</li><li>'X.re'</li><li>'actual'</li><li>'also'</li><li>'alway'</li><li>'amp'</li><li>'back'</li><li>'bad'</li><li>'best'</li><li>'better'</li><li>'call'</li><li>'can'</li><li>'come'</li><li>'day'</li><li>'don.t'</li><li>'even'</li><li>'feel'</li><li>'first'</li><li>'follow'</li><li>'friend'</li><li>'fuck'</li><li>'game'</li><li>'get'</li><li>'give'</li><li>'gonna'</li><li>'good'</li><li>'got'</li><li>'guy'</li><li>'happi'</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'label'</li><li>'X.re'</li><li>'actual'</li><li>'also'</li><li>'alway'</li><li>'amp'</li><li>'back'</li><li>'bad'</li><li>'best'</li><li>'better'</li><li>'call'</li><li>'can'</li><li>'come'</li><li>'day'</li><li>'don.t'</li><li>'even'</li><li>'feel'</li><li>'first'</li><li>'follow'</li><li>'friend'</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>20000</li><li>88</li></ol>



### Splitting data

Let's first read in the processed data with text being a vector.


```R
data2 <- fread("/content/data/Twitter_tm_freq.csv")  #dim(data2)
names(data2)[1:20] # notice that user_id, stars and date are in the data2
dim(data2)
data2$label <- as.factor(data2$label)
table(data2$label)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'label'</li><li>'X.re'</li><li>'actual'</li><li>'also'</li><li>'alway'</li><li>'amp'</li><li>'back'</li><li>'bad'</li><li>'best'</li><li>'better'</li><li>'call'</li><li>'can'</li><li>'come'</li><li>'day'</li><li>'don.t'</li><li>'even'</li><li>'feel'</li><li>'first'</li><li>'follow'</li><li>'friend'</li></ol>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>20000</li><li>88</li></ol>




    
        0     1 
    10000 10000 


As one standard machine learning process, we first split data into two sets one training data and the other testing data. We use training data to build models, choose models etc and make final recommendations. We then report the performance using the testing data.

Reserve 100 randomly chosen rows as our test data (`data2.test`) and the remaining 900 as the training data (`data2.train`)


```R
set.seed(1)  # for the purpose of reporducibility
n <- nrow(data2)
test.index <- sample(n, 0.1*n)
# length(test.index)
data2.test <- data2[test.index, ] # only keep label and the texts
data2.train <- data2[-test.index, ]
dim(data2.train)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>18000</li><li>88</li></ol>



### Analysis 1: LASSO

We first explore a logistic regression model using LASSO. The following R-chunk runs a LASSO model with $\alpha=.99$. The reason we take an elastic net is to enjoy the nice properties from both `LASSO` (impose sparsity) and `Ridge` (computationally stable).

LASSO takes sparse design matrix as an input. So make sure to extract the sparse matrix first as the input in cv.glm(). It takes about 1 minute to run cv.glm() with sparse matrix or 11 minutes using the regular design matrix.


```R
### or try `sparse.model.matrix()` which is much faster
y <- data2.train$label
X1 <- sparse.model.matrix(label~., data=data2.train)[, -1]
set.seed(2)
result.lasso <- cv.glmnet(X1, y, alpha=.99, family="binomial")
# 1.25 minutes in my MAC
plot(result.lasso)
# this this may take you long time to run, we save result.lasso
saveRDS(result.lasso, file="data/TextMining_lasso.RDS")
# result.lasso can be assigned back by
# result.lasso <- readRDS("data/TextMining_lasso.RDS")

# number of non-zero words picked up by LASSO when using lambda.1se
coef.1se <- coef(result.lasso, s="lambda.1se")
lasso.words <- coef.1se@Dimnames[[1]] [coef.1se@i][-1] # non-zero variables without intercept.
summary(lasso.words)


# or our old way
#coef.1se <- coef(result.lasso, s="lambda.1se")
#coef.1se <- coef.1se[which(coef.1se !=0),]
#lasso.words <- rownames(as.matrix(coef.1se))[-1]
#summary(lasso.words)



### cv.glmnt with the non-sparse design matrix takes much longer
# X <- as.matrix(data2.train[, -1]) # we can use as.matrix directly her
#### Be careful to run the following LASSO.
#set.seed(2)
#result.lasso <- cv.glmnet(X, y, alpha=.99, family="binomial")
# 10 minutes in my MAC
#plot(result.lasso)
```


       Length     Class      Mode 
           43 character character 



    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_121_1.png)
    


### Analysis 2: Relaxed LASSO

As an alternative model we will run our relaxed `LASSO`. Input variables are chosen by `LASSO` and we get a regular logistic regression model. Once again it is stored as `result.glm` in `TextMining.RData`.


```R
sel_cols <- c("label", lasso.words)
# use all_of() to specify we would like to select variables in sel_cols
data_sub <- data2.train %>% select(all_of(sel_cols))
result.glm <- glm(label~., family=binomial, data_sub) # takes 3.5 minutes
## glm() returns a big object with unnecessary information
# saveRDS(result.glm,
#      file = "data/TextMining_glm.RDS")

## trim the glm() fat from
## https://win-vector.com/2014/05/30/trimming-the-fat-from-glm-models-in-r/
stripGlmLR = function(cm) {
  cm$y = c()
  cm$model = c()

  cm$residuals = c()
  cm$fitted.values = c()
  cm$effects = c()
  cm$qr$qr = c()
  cm$linear.predictors = c()
  cm$weights = c()
  cm$prior.weights = c()
  cm$data = c()


  cm$family$variance = c()
  cm$family$dev.resids = c()
  cm$family$aic = c()
  cm$family$validmu = c()
  cm$family$simulate = c()
  attr(cm$terms,".Environment") = c()
  attr(cm$formula,".Environment") = c()

  cm
}

#result.glm.small <- stripGlmLR(result.glm)
result.glm.small <- result.glm

saveRDS(result.glm.small,
     file = "/content/data/TextMining_glm_small.RDS")
```


```R
View(result.glm.small)
```


    
    Call:  glm(formula = label ~ ., family = binomial, data = data_sub)
    
    Coefficients:
    (Intercept)         X.re        alway         back         best       better  
      -0.007658     0.298520    -0.074021    -0.079673    -0.016224    -0.238516  
            day         even        first       follow       friend         fuck  
       0.112720    -0.062677    -0.007669    -0.203323     0.467575     0.520260  
           game         give        gonna          got        happi         hope  
      -0.437904     0.101845     0.285431     0.030097    -0.162106    -0.003150  
            let         life         like          lol          lot          man  
      -0.206311     0.383762     0.135270     0.110871    -0.074033    -0.099732  
           mani         mean         much        never          one       person  
      -0.408640     0.258361     0.293571    -0.050012    -0.061801     0.053222  
          pleas         read         shit        start        still         sure  
      -0.118377    -0.193374    -0.153857    -0.095358     0.145200    -0.211362  
           take        thank          tri          use          way         well  
      -0.165191     0.017140     0.101179     0.016646     0.014820    -0.213817  
           work         yeah  
       0.030768    -0.293026  
    
    Degrees of Freedom: 17999 Total (i.e. Null);  17956 Residual
    Null Deviance:	    24950 
    Residual Deviance: 24810 	AIC: 24900


### Analysis 3: Word cloud! (Sentiment analysis)

Logistic regression model connects the chance of `being good` given a text/review. What are the `nice` (or positive) words and how much it influence the chance being good? In addition to explore the set of good words we also build word clouds to visualize the correlation between positive words and negative words.

1. Order the glm positive coefficients (positive words). Show them in a word cloud. The size of the words indicates the strength of positive correlation between that word and the chance being a good rating.

2. Order the glm negative coefficients (negative words)

TIME TO PLOT A WORD CLOUD!! Plot the world clouds, the size of the words are prop to the logistic reg coef's


```R
result.glm <- readRDS("data/TextMining_glm_small.RDS")
result.glm.coef <- coef(result.glm)[-1]  # took intercept out
hist(result.glm.coef)

# pick up the positive coef's which are positively related to the prob of being depressed
good.glm <- result.glm.coef[which(result.glm.coef > 0)]
good.glm <- good.glm
names(good.glm)  # which words are positively associated with depression

good.fre <- sort(good.glm, decreasing = TRUE) # sort the coef's
round(good.fre, 4) # leading 20 positive words, amazing!
length(good.fre)  # 390 good words

# hist(as.matrix(good.fre), breaks=30, col="red")
good.word <- names(good.fre)  # good words with a decreasing order in the coeff's
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'X.re'</li><li>'day'</li><li>'friend'</li><li>'fuck'</li><li>'give'</li><li>'gonna'</li><li>'got'</li><li>'life'</li><li>'like'</li><li>'lol'</li><li>'mean'</li><li>'much'</li><li>'person'</li><li>'still'</li><li>'thank'</li><li>'tri'</li><li>'use'</li><li>'way'</li><li>'work'</li></ol>




<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>fuck</dt><dd>0.5203</dd><dt>friend</dt><dd>0.4676</dd><dt>life</dt><dd>0.3838</dd><dt>X.re</dt><dd>0.2985</dd><dt>much</dt><dd>0.2936</dd><dt>gonna</dt><dd>0.2854</dd><dt>mean</dt><dd>0.2584</dd><dt>still</dt><dd>0.1452</dd><dt>like</dt><dd>0.1353</dd><dt>day</dt><dd>0.1127</dd><dt>lol</dt><dd>0.1109</dd><dt>give</dt><dd>0.1018</dd><dt>tri</dt><dd>0.1012</dd><dt>person</dt><dd>0.0532</dd><dt>work</dt><dd>0.0308</dd><dt>got</dt><dd>0.0301</dd><dt>thank</dt><dd>0.0171</dd><dt>use</dt><dd>0.0166</dd><dt>way</dt><dd>0.0148</dd></dl>




19



    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_128_3.png)
    



```R
names(good.glm)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'X.re'</li><li>'day'</li><li>'friend'</li><li>'fuck'</li><li>'give'</li><li>'gonna'</li><li>'got'</li><li>'life'</li><li>'like'</li><li>'lol'</li><li>'mean'</li><li>'much'</li><li>'person'</li><li>'still'</li><li>'thank'</li><li>'tri'</li><li>'use'</li><li>'way'</li><li>'work'</li></ol>



The above chunk shows in detail about the weight for positive words. We only show the positive word-cloud here. One can tell the large positive words are making sense in the way we do expect the collection of large words should have a positive tone towards the restaurant being reviewed.


```R
cor.special <- brewer.pal(8,"Dark2")  # set up a pretty color scheme
wordcloud(good.word, good.fre,  # make a word cloud
          colors=cor.special,
          ordered.colors=F, min.freq=0)
```


    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_131_0.png)
    


###Negative Wordcloud

Similarly to the negative coef's which is positively correlated to the probability of having depression.


```R
bad.glm <- result.glm.coef[which(result.glm.coef < 0)]
# names(bad.glm)[1:50]

cor.special <- brewer.pal(6,"Dark2")
bad.fre <- sort(-bad.glm, decreasing = TRUE)
round(bad.fre, 4)

# hist(as.matrix(bad.fre), breaks=30, col="green")
bad.word <- names(bad.fre)
wordcloud(bad.word, bad.fre,
          color=cor.special, ordered.colors=F, min.freq=0)
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>game</dt><dd>0.4379</dd><dt>mani</dt><dd>0.4086</dd><dt>yeah</dt><dd>0.293</dd><dt>better</dt><dd>0.2385</dd><dt>well</dt><dd>0.2138</dd><dt>sure</dt><dd>0.2114</dd><dt>let</dt><dd>0.2063</dd><dt>follow</dt><dd>0.2033</dd><dt>read</dt><dd>0.1934</dd><dt>take</dt><dd>0.1652</dd><dt>happi</dt><dd>0.1621</dd><dt>shit</dt><dd>0.1539</dd><dt>pleas</dt><dd>0.1184</dd><dt>man</dt><dd>0.0997</dd><dt>start</dt><dd>0.0954</dd><dt>back</dt><dd>0.0797</dd><dt>lot</dt><dd>0.074</dd><dt>alway</dt><dd>0.074</dd><dt>even</dt><dd>0.0627</dd><dt>one</dt><dd>0.0618</dd><dt>never</dt><dd>0.05</dd><dt>best</dt><dd>0.0162</dd><dt>first</dt><dd>0.0077</dd><dt>hope</dt><dd>0.0031</dd></dl>




    
![png](Wharton_Group_5_Data_Science_Project_Twitter_files/Wharton_Group_5_Data_Science_Project_Twitter_134_1.png)
    




###Testing

LASSO Prediction


```R
predict.lasso <- ifelse(predict(result.lasso, as.matrix(data2.test[, -1]), type = "response", s="lambda.1se")>.5, "1", "0")
  # output lasso estimates of prob's
#predict.lasso <- predict(result.lasso, as.matrix(data2.test[, -1]), type = "class", s="lambda.1se")
  # output majority vote labels

# LASSO accuracy
mean(data2.test$label == predict.lasso)   #0.5425; 0.5405 with n-grams
```


0.5405


Relaxed LASSO Prediction


```R
predict.glm <- predict(result.glm, data2.test, type = "response")
class.glm <- ifelse(predict.glm > .5, "1", "0")
# length(class.glm)

testerror.glm <- mean(data2.test$label == class.glm)
testerror.glm   #classification accuracy is 0.5305; 0.534 with n-grams
```


0.534

