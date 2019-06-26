import pandas as pd
import re
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

start = time.time()

#You can insert path of any dataset with column TweetText for the text and Sentiment for the sentiment labels of text

traindata=pd.read_csv("C:/MyData/PythonPractice/twitter_nuclear/nuclear.csv",encoding='ISO-8859-1')  
doc=traindata.TweetText
print(len(doc))
sentidoc=traindata.Sentiment

# Generate universe variables
#   * pos and neg on subjective ranges [0, 1]
#   * op has a range of [0, 10] in units of percentage points
x_p = np.arange(0, 1, 0.1)
x_n = np.arange(0, 1, 0.1)
x_op = np.arange(0, 10, 1)

# Generate fuzzy membership functions
p_lo = fuzz.trimf(x_p, [0, 0, 0.5])
p_md = fuzz.trimf(x_p, [0, 0.5, 1])
p_hi = fuzz.trimf(x_p, [0.5, 1, 1])
n_lo = fuzz.trimf(x_n, [0, 0, 0.5])
n_md = fuzz.trimf(x_n, [0, 0.5, 1])
n_hi = fuzz.trimf(x_n, [0.5, 1, 1])
op_Neg = fuzz.trimf(x_op, [0, 0, 5])  # Scale : Neg Neu Pos
op_Neu = fuzz.trimf(x_op, [0, 5, 10])
op_Pos = fuzz.trimf(x_op, [5, 10, 10])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))
#
ax0.plot(x_p, p_lo, 'b', linewidth=1.5, label='Low')
ax0.plot(x_p, p_md, 'g', linewidth=1.5, label='Medium')
ax0.plot(x_p, p_hi, 'r', linewidth=1.5, label='High')
ax0.set_title('Pos')
ax0.legend()

ax1.plot(x_n, n_lo, 'b', linewidth=1.5, label='Low')
ax1.plot(x_n, n_md, 'g', linewidth=1.5, label='Medium')
ax1.plot(x_n, n_hi, 'r', linewidth=1.5, label='High')
ax1.set_title('Neg')
ax1.legend()

ax2.plot(x_op, op_Pos, 'b', linewidth=1.5, label='Negative')
ax2.plot(x_op, op_Neu, 'g', linewidth=1.5, label='Neutral')
ax2.plot(x_op, op_Neg, 'r', linewidth=1.5, label='Positive')
ax2.set_title('Output')
ax2.legend()

# Turn off top/right axes
#for ax in (ax0, ax1, ax2):
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.get_xaxis().tick_bottom()
#    ax.get_yaxis().tick_left()

#plt.tight_layout()

tweets=[]
senti=[]
sentiment=[]
sentiment_doc=[]

for j in range(len(doc)):
    str1=traindata.TweetText[j]
    str2=str1.lower()
    tweets.append(str2)   # converted into lower case
    senti.append(traindata.Sentiment[j])

def decontracted(phrase):   # text pre-processing 
        # specific
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"@", "" , phrase)         # removal of @
        phrase =  re.sub(r"http\S+", "", phrase)   # removal of URLs
        phrase = re.sub(r"#", "", phrase)          # hashtag processing
    
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    
for k in range(len(doc)):
    tweets[k]=decontracted(tweets[k])
         
sid = SentimentIntensityAnalyzer()

for j in range(len(doc)):
    sentiment_doc.append(senti[j])
    ss = sid.polarity_scores(tweets[j])
    posscore=ss['pos']
    negscore=ss['neg']
    neuscore=ss['neu']
    compoundscore=ss['compound']
   
    print(str(j+1)+" {:-<65} {}".format(tweets[j], str(ss))) 
    
    print("\nPositive Score for each  tweet :")    
    if (posscore==1):
        posscore=0.9 
    else:
        posscore=round(posscore,1)
    print(posscore)

    print("\nNegative Score for each  tweet :")
    if (negscore==1):
        negscore=0.9
    else:
        negscore=round(negscore,1)
    print(negscore)

# We need the activation of our fuzzy membership functions at these values.
    p_level_lo = fuzz.interp_membership(x_p, p_lo, posscore)
    p_level_md = fuzz.interp_membership(x_p, p_md, posscore)
    p_level_hi = fuzz.interp_membership(x_p, p_hi, posscore)
    
    n_level_lo = fuzz.interp_membership(x_n, n_lo, negscore)
    n_level_md = fuzz.interp_membership(x_n, n_md, negscore)
    n_level_hi = fuzz.interp_membership(x_n, n_hi, negscore)
    
    # Now we take our rules and apply them. Rule 1 concerns bad food OR nice.
    # The OR operator means we take the maximum of these two.
    active_rule1 = np.fmin(p_level_lo, n_level_lo)
    active_rule2 = np.fmin(p_level_md, n_level_lo)
    active_rule3 = np.fmin(p_level_hi, n_level_lo)
    active_rule4 = np.fmin(p_level_lo, n_level_md)
    active_rule5 = np.fmin(p_level_md, n_level_md)
    active_rule6 = np.fmin(p_level_hi, n_level_md)
    active_rule7 = np.fmin(p_level_lo, n_level_hi)
    active_rule8 = np.fmin(p_level_md, n_level_hi)
    active_rule9 = np.fmin(p_level_hi, n_level_hi)
    
    # Now we apply this by clipping the top off the corresponding output
    # membership function with `np.fmin`
    
    n1=np.fmax(active_rule4,active_rule7)
    n2=np.fmax(n1,active_rule8)     
    op_activation_lo = np.fmin(n2,op_Neg)
    
    neu1=np.fmax(active_rule1,active_rule5)
    neu2=np.fmax(neu1,active_rule9)     
    op_activation_md = np.fmin(neu2,op_Neu)
    
    p1=np.fmax(active_rule2,active_rule3)
    p2=np.fmax(p1,active_rule6)   
    op_activation_hi = np.fmin(p2,op_Pos)
    
    op0 = np.zeros_like(x_op)
    
    # Aggregate all three output membership functions together
    aggregated = np.fmax(op_activation_lo,
                         np.fmax(op_activation_md, op_activation_hi))
    
    # Calculate defuzzified result
    op = fuzz.defuzz(x_op, aggregated, 'centroid')
    output=round(op,2)

    op_activation = fuzz.interp_membership(x_op, aggregated, op)  # for plot

#     Visualize Aggregated Membership
#    fig, ax0 = plt.subplots(figsize=(8, 3))
#    
#    ax0.plot(x_op, op_Neg, 'b', linewidth=0.5, linestyle='--',label= 'Negative')
#    ax0.plot(x_op, op_Neu, 'g', linewidth=0.5, linestyle='--',label= 'Neutral')
#    ax0.plot(x_op, op_Pos, 'r', linewidth=0.5, linestyle='--',label= 'Positive')
#    ax0.fill_between(x_op, op0, aggregated, facecolor='Orange', alpha=0.7)
#    ax0.plot([op, op], [0, op_activation], 'k', linewidth=1.5, alpha=0.9)
#    ax0.set_title('Aggregated membership and result (line)')
#    ax0.legend()
    
#    # Turn off top/right axes
#    for ax in (ax0,):
#        ax.spines['top'].set_visible(False)
#        ax.spines['right'].set_visible(False)
#        ax.get_xaxis().tick_bottom()
#        ax.get_yaxis().tick_left()
#    
#    plt.tight_layout()
    
    # Visualize Output Membership
    fig, ax0 = plt.subplots(figsize=(8, 3))
    
    ax0.fill_between(x_op, op0, op_activation_lo, facecolor='b', alpha=0.7)
    ax0.plot(x_op, op_Neg, 'b', linewidth=0.5, linestyle='--',label= 'Negative' )
    ax0.fill_between(x_op, op0, op_activation_md, facecolor='g', alpha=0.7)
    ax0.plot(x_op, op_Neu, 'g', linewidth=0.5, linestyle='--', label='Neutral')
    ax0.fill_between(x_op, op0, op_activation_hi, facecolor='r', alpha=0.7)
    ax0.plot(x_op, op_Pos, 'r', linewidth=0.5, linestyle='--', label='Positive')
    ax0.plot([op, op], [0, op_activation], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Output membership activity')
    ax0.legend()
#    
#    # Turn off top/right axes
#    for ax in (ax0,):
#        ax.spines['top'].set_visible(False)
#        ax.spines['right'].set_visible(False)
#        ax.get_xaxis().tick_bottom()
#        ax.get_yaxis().tick_left()
#    
#    plt.tight_layout()       
       
    print("\nFiring Strength of Negative (wneg): "+str(round(n2,4)))
    print("Firing Strength of Neutral (wneu): "+str(round(neu2,4)))
    print("Firing Strength of Positive (wpos): "+str(round(p2,4)))
    
    print("\nResultant consequents MFs:" )
    print("op_activation_low: "+str(op_activation_lo))
    print("op_activation_med: "+str(op_activation_md))
    print("op_activation_high: "+str(op_activation_hi))
    
    print("\nAggregated Output: "+str(aggregated))

    print("\nDefuzzified Output: "+str(output))

# Scale : Neg Neu Pos   
    if 0<(output)<3.33:    # R
        print("\nOutput after Defuzzification: Negative")
        sentiment.append("Negative")
        
    elif 3.34<(output)<6.66:
        print("\nOutput after Defuzzification: Neutral")
        sentiment.append("Neutral")
   
    elif 6.67<(output)<10:
        print("\nOutput after Defuzzification: Positive")
        sentiment.append("Positive")
        
    print("Doc sentiment: " +str(senti[j])+"\n")    
    
count=0
for k in range(len(doc)):
    if(sentiment_doc[k]==sentiment[k]):
        count=count+1       
print("Accuracy is: "+ str(round(count/len(doc)*100,2)))

from sklearn.metrics import f1_score, precision_score, recall_score
y_true = sentiment_doc
y_pred = sentiment

p1=precision_score(y_true, y_pred, average='macro')  

print("Precision score (MACRO): " + str(round((p1*100),2)))

r1=recall_score(y_true, y_pred, average='macro')  

print("Recall score (MACRO): " + str(round((r1*100),2)))

f1=f1_score(y_true, y_pred, average='macro')  
f2=f1_score(y_true, y_pred, average='micro')  

print("F1 score (MACRO): " + str(round((f1*100),2)))
print("F1 score (MICRO): "+ str(round((f2*100),2)))

end = time.time()
print("Execution Time: "+str(round((end - start),3))+" secs")